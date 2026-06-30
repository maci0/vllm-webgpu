import numpy as np
import pytest
from pathlib import Path

SHADERS_DIR = Path(__file__).parent.parent / "vllm_webgpu" / "shaders"


def gelu_mul_ref(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    """SwiGLU: gate * silu(up)."""
    g = gate.astype(np.float32)
    u = up.astype(np.float32)
    silu_u = u * (1.0 / (1.0 + np.exp(-u)))
    return (g * silu_u).astype(np.float16)


def test_gelu_mul(wgpu_device):
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey

    n = 256
    gate = np.random.randn(n).astype(np.float16)
    up = np.random.randn(n).astype(np.float16)
    expected = gelu_mul_ref(gate, up)

    dev = wgpu_device.wgpu_device
    gate_buf = WebGPUBuffer.from_numpy(dev, gate)
    up_buf = WebGPUBuffer.from_numpy(dev, up)
    out_buf = WebGPUBuffer.empty(dev, gate.nbytes,
                                 usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)

    cache = PipelineCache(dev, SHADERS_DIR / "generic")
    key = PipelineKey("gelu_mul", (("N", n),))
    pipeline = cache.get_or_create(key)

    bg = dev.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": gate_buf.buf}},
            {"binding": 1, "resource": {"buffer": up_buf.buf}},
            {"binding": 2, "resource": {"buffer": out_buf.buf}},
        ],
    )
    encoder = dev.create_command_encoder()
    cp = encoder.begin_compute_pass()
    cp.set_pipeline(pipeline)
    cp.set_bind_group(0, bg)
    cp.dispatch_workgroups((n + 255) // 256, 1, 1)
    cp.end()
    dev.queue.submit([encoder.finish()])

    result = out_buf.to_numpy().view(np.float16)
    np.testing.assert_allclose(result.astype(np.float32), expected.astype(np.float32),
                               rtol=1e-2, atol=1e-2)


def test_embedding_lookup(wgpu_device):
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey

    vocab, hidden = 32, 64
    num_tokens = 4
    table = np.random.randn(vocab, hidden).astype(np.float16)
    token_ids = np.array([0, 5, 10, 31], dtype=np.uint32)
    expected = table[token_ids]

    dev = wgpu_device.wgpu_device
    table_buf = WebGPUBuffer.from_numpy(dev, table)
    ids_buf = WebGPUBuffer.from_numpy(dev, token_ids)
    out_buf = WebGPUBuffer.empty(
        dev, num_tokens * hidden * 2,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
    )

    cache = PipelineCache(dev, SHADERS_DIR / "generic")
    key = PipelineKey("embedding_lookup", (("HIDDEN_DIM", hidden),))
    pipeline = cache.get_or_create(key)

    bg = dev.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": table_buf.buf}},
            {"binding": 1, "resource": {"buffer": ids_buf.buf}},
            {"binding": 2, "resource": {"buffer": out_buf.buf}},
        ],
    )
    encoder = dev.create_command_encoder()
    cp = encoder.begin_compute_pass()
    cp.set_pipeline(pipeline)
    cp.set_bind_group(0, bg)
    cp.dispatch_workgroups(num_tokens, 1, 1)
    cp.end()
    dev.queue.submit([encoder.finish()])

    result = out_buf.to_numpy().view(np.float16).reshape(num_tokens, hidden)
    np.testing.assert_allclose(result.astype(np.float32), expected.astype(np.float32),
                               rtol=1e-3, atol=1e-3)
