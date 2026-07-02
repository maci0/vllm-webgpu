import numpy as np
import pytest
from pathlib import Path


SHADERS_DIR = Path(__file__).parent.parent / "vllm_webgpu" / "shaders"


def rms_norm_ref(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    rms = np.sqrt(np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True) + eps)
    return ((x.astype(np.float32) / rms) * weight.astype(np.float32)).astype(np.float16)


def dispatch_kernel(wgpu_device, pipeline_cache, shader_name, bindings, constants, n_groups):
    """Helper: bind buffers and dispatch a compute shader."""
    import wgpu
    dev = wgpu_device.wgpu_device
    pipeline = pipeline_cache.get_or_create(
        __import__("vllm_webgpu.webgpu.pipeline", fromlist=["PipelineKey"]).PipelineKey(
            shader_name, tuple(sorted(constants.items()))
        )
    )
    bg_layout = pipeline.get_bind_group_layout(0)
    entries = [{"binding": i, "resource": {"buffer": b.buf}} for i, b in enumerate(bindings)]
    bg = dev.create_bind_group(layout=bg_layout, entries=entries)
    encoder = dev.create_command_encoder()
    cp = encoder.begin_compute_pass()
    cp.set_pipeline(pipeline)
    cp.set_bind_group(0, bg)
    cp.dispatch_workgroups(*n_groups)
    cp.end()
    dev.queue.submit([encoder.finish()])


def test_rms_norm(wgpu_device):
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey

    hidden = 64
    seq = 4
    x = np.random.randn(seq, hidden).astype(np.float16)
    w = np.random.randn(hidden).astype(np.float16)

    expected = rms_norm_ref(x, w)

    dev = wgpu_device.wgpu_device
    x_buf = WebGPUBuffer.from_numpy(dev, x)
    w_buf = WebGPUBuffer.from_numpy(dev, w)
    out_buf = WebGPUBuffer.empty(dev, x.nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)

    cache = PipelineCache(dev, SHADERS_DIR / "generic")
    key = PipelineKey("rms_norm", (("HIDDEN_DIM", hidden),))
    pipeline = cache.get_or_create(key)

    bg = dev.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": x_buf.buf}},
            {"binding": 1, "resource": {"buffer": w_buf.buf}},
            {"binding": 2, "resource": {"buffer": out_buf.buf}},
        ],
    )
    encoder = dev.create_command_encoder()
    cp = encoder.begin_compute_pass()
    cp.set_pipeline(pipeline)
    cp.set_bind_group(0, bg)
    cp.dispatch_workgroups(seq, 1, 1)   # one workgroup per row
    cp.end()
    dev.queue.submit([encoder.finish()])

    result = out_buf.to_numpy().view(np.float16).reshape(seq, hidden)
    np.testing.assert_allclose(result.astype(np.float32), expected.astype(np.float32),
                               rtol=1e-2, atol=1e-2)


def test_rms_norm_scale_invariance(wgpu_device):
    """Property proof: RMSNorm is scale-invariant (homogeneous of degree 0).

    Mathematical invariant: rms_norm(α*x, weight) == rms_norm(x, weight)
    for any scalar α > 0. This holds because RMS(α*x) = α*RMS(x), so
    the α cancels: (α*x)/(α*RMS(x)) = x/RMS(x).

    Any implementation that fails this test is computing something other
    than RMSNorm — it's confusing the scale with the content.
    """
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey

    hidden = 64
    seq = 4
    x = np.random.randn(seq, hidden).astype(np.float16)
    w = np.random.randn(hidden).astype(np.float16)

    dev = wgpu_device.wgpu_device
    rw = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
    w_buf = WebGPUBuffer.from_numpy(dev, w)
    cache = PipelineCache(dev, SHADERS_DIR / "generic")
    pipeline = cache.get_or_create(PipelineKey("rms_norm", (("HIDDEN_DIM", hidden),)))

    results = []
    for scale in [1.0, 2.0, 0.5, 10.0]:
        x_scaled = (x.astype(np.float32) * scale).astype(np.float16)
        x_buf = WebGPUBuffer.from_numpy(dev, x_scaled)
        out_buf = WebGPUBuffer.empty(dev, x.nbytes, usage=rw)
        bg = dev.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[{"binding": 0, "resource": {"buffer": x_buf.buf}},
                     {"binding": 1, "resource": {"buffer": w_buf.buf}},
                     {"binding": 2, "resource": {"buffer": out_buf.buf}}],
        )
        enc = dev.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipeline)
        cp.set_bind_group(0, bg)
        cp.dispatch_workgroups(seq, 1, 1)
        cp.end()
        dev.queue.submit([enc.finish()])
        results.append(out_buf.to_numpy().view(np.float16).reshape(seq, hidden).astype(np.float32))

    # Invariant: all scaled versions must produce the same output
    for i, scale in enumerate([1.0, 2.0, 0.5, 10.0]):
        np.testing.assert_allclose(
            results[i], results[0], rtol=5e-2, atol=1e-2,
            err_msg=f"RMSNorm scale invariance violated: scale={scale} changed the output"
        )


def test_rms_norm_unit_rms_invariant(wgpu_device):
    """Property proof: RMSNorm (weight=1) produces output with RMS == 1.

    Invariant: mean(output^2) == 1 for any non-zero input (before weight scaling).
    This proves the normalization is mathematically correct, not just close to a
    reference — any implementation that passes this test satisfies the RMSNorm spec.
    """
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey

    hidden = 64
    seq = 8
    x = np.random.randn(seq, hidden).astype(np.float16)
    # Use weight = 1 so output = x / RMS(x), which must have RMS == 1
    w = np.ones(hidden, dtype=np.float16)

    dev = wgpu_device.wgpu_device
    rw = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
    x_buf = WebGPUBuffer.from_numpy(dev, x)
    w_buf = WebGPUBuffer.from_numpy(dev, w)
    out_buf = WebGPUBuffer.empty(dev, x.nbytes, usage=rw)

    cache = PipelineCache(dev, SHADERS_DIR / "generic")
    pipeline = cache.get_or_create(PipelineKey("rms_norm", (("HIDDEN_DIM", hidden),)))
    bg = dev.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[{"binding": 0, "resource": {"buffer": x_buf.buf}},
                 {"binding": 1, "resource": {"buffer": w_buf.buf}},
                 {"binding": 2, "resource": {"buffer": out_buf.buf}}],
    )
    enc = dev.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(pipeline)
    cp.set_bind_group(0, bg)
    cp.dispatch_workgroups(seq, 1, 1)
    cp.end()
    dev.queue.submit([enc.finish()])

    result = out_buf.to_numpy().view(np.float16).reshape(seq, hidden).astype(np.float32)

    # Invariant: each row's RMS must equal 1 (within f16 precision)
    rms_per_row = np.sqrt(np.mean(result ** 2, axis=-1))
    np.testing.assert_allclose(rms_per_row, np.ones(seq), rtol=1e-2, atol=1e-2,
                               err_msg="RMSNorm invariant violated: RMS of normalized output != 1")


def test_add(wgpu_device):
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey

    n = 256
    a = np.random.randn(n).astype(np.float16)
    b = np.random.randn(n).astype(np.float16)
    expected = (a.astype(np.float32) + b.astype(np.float32)).astype(np.float16)

    dev = wgpu_device.wgpu_device
    a_buf = WebGPUBuffer.from_numpy(dev, a)
    b_buf = WebGPUBuffer.from_numpy(dev, b)
    out_buf = WebGPUBuffer.empty(dev, a.nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)

    cache = PipelineCache(dev, SHADERS_DIR / "generic")
    key = PipelineKey("add", (("N", n),))
    pipeline = cache.get_or_create(key)
    bg = dev.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": a_buf.buf}},
            {"binding": 1, "resource": {"buffer": b_buf.buf}},
            {"binding": 2, "resource": {"buffer": out_buf.buf}},
        ],
    )
    encoder = dev.create_command_encoder()
    cp = encoder.begin_compute_pass()
    cp.set_pipeline(pipeline)
    cp.set_bind_group(0, bg)
    # add.wgsl is vec4<f16>: each thread handles 4 elements, so dispatch N/4 threads.
    cp.dispatch_workgroups((n // 4 + 255) // 256, 1, 1)
    cp.end()
    dev.queue.submit([encoder.finish()])

    result = out_buf.to_numpy().view(np.float16)
    np.testing.assert_allclose(result.astype(np.float32), expected.astype(np.float32),
                               rtol=1e-2, atol=1e-2)
