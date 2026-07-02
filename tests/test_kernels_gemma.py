import numpy as np
import pytest
from pathlib import Path

SHADERS_DIR = Path(__file__).parent.parent / "vllm_webgpu" / "shaders"


def logit_softcap_ref(x: np.ndarray, cap: float = 30.0) -> np.ndarray:
    x32 = x.astype(np.float32)
    return (np.tanh(x32 / cap) * cap).astype(np.float16)


def test_logit_softcap(wgpu_device):
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey

    vocab = 64
    x = (np.random.randn(vocab) * 50.0).astype(np.float16)
    expected = logit_softcap_ref(x)

    dev = wgpu_device.wgpu_device
    x_buf = WebGPUBuffer.from_numpy(dev, x)
    out_buf = WebGPUBuffer.empty(dev, x.nbytes,
                                 usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)

    cache = PipelineCache(dev, SHADERS_DIR / "gemma")
    key = PipelineKey("logit_softcap", (("N", vocab),))
    pipeline = cache.get_or_create(key)

    bg = dev.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": x_buf.buf}},
            {"binding": 1, "resource": {"buffer": out_buf.buf}},
        ],
    )
    encoder = dev.create_command_encoder()
    cp = encoder.begin_compute_pass()
    cp.set_pipeline(pipeline)
    cp.set_bind_group(0, bg)
    cp.dispatch_workgroups((vocab + 255) // 256, 1, 1)
    cp.end()
    dev.queue.submit([encoder.finish()])

    result = out_buf.to_numpy().view(np.float16)
    np.testing.assert_allclose(result.astype(np.float32), expected.astype(np.float32),
                               rtol=1e-2, atol=1e-2)


def test_logit_softcap_bounded_invariant(wgpu_device):
    """Property proof: logit_softcap output is bounded by CAP.

    Mathematical invariant: |tanh(x/CAP)*CAP| ≤ CAP for ALL x.
    This follows from |tanh(t)| ≤ 1 for all real t ∈ ℝ.
    ANY input (including infinity, NaN-inducing extremes) must produce
    output with absolute value ≤ CAP. If violated, the shader is wrong.
    """
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey

    cap = 30.0
    # Test with extreme inputs that stress the implementation
    test_inputs = [
        np.array([0.0, 1.0, -1.0, 100.0, -100.0, 29.999, -29.999, 1000.0], dtype=np.float16),
        (np.random.randn(256) * 500.0).astype(np.float16),   # very large random
        np.full(64, np.float16(65504.0)),                     # f16 max value
        np.full(64, np.float16(-65504.0)),                    # f16 min value
    ]

    dev = wgpu_device.wgpu_device
    rw = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
    cache = PipelineCache(dev, SHADERS_DIR / "gemma")

    for x in test_inputs:
        n = len(x)
        x_buf = WebGPUBuffer.from_numpy(dev, x)
        out_buf = WebGPUBuffer.empty(dev, x.nbytes, usage=rw)

        pipeline = cache.get_or_create(PipelineKey("logit_softcap", (("N", n),)))
        bg = dev.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[{"binding": 0, "resource": {"buffer": x_buf.buf}},
                     {"binding": 1, "resource": {"buffer": out_buf.buf}}],
        )
        enc = dev.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipeline)
        cp.set_bind_group(0, bg)
        cp.dispatch_workgroups((n + 255) // 256, 1, 1)
        cp.end()
        dev.queue.submit([enc.finish()])

        result = out_buf.to_numpy().view(np.float16).astype(np.float32)

        # Invariant: |output| ≤ CAP + ε (f16 rounding allowance)
        assert np.all(np.abs(result) <= cap + 0.1), (
            f"Softcap invariant violated: max |output|={np.max(np.abs(result))} > CAP={cap}"
        )
        assert np.all(np.isfinite(result)), "Softcap produced non-finite output"


def test_ple_shaders_compile(wgpu_device):
    """Smoke test: verify all PLE shaders compile without error."""
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey

    dev = wgpu_device.wgpu_device
    cache = PipelineCache(dev, SHADERS_DIR / "gemma")

    shaders = [
        ("per_head_rms_norm_no_weight", (("HEAD_DIM", 64), ("NUM_HEADS", 4), ("WG_SIZE", 128))),
        ("ple_stage1_fuse", (("HIDDEN_DIM", 256), ("PLE_DIM", 16))),
        ("ple_gelu_mul", (("N", 256),)),
        ("ple_skip_scale_add", (("N", 256),)),
    ]

    for shader_name, defines in shaders:
        key = PipelineKey(shader_name, defines)
        pipeline = cache.get_or_create(key)
        assert pipeline is not None, f"{shader_name} failed to compile"
