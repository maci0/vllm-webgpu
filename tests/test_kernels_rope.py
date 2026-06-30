import numpy as np
import pytest
from pathlib import Path

SHADERS_DIR = Path(__file__).parent.parent / "vllm_webgpu" / "shaders"


def rope_ref(x: np.ndarray, positions: np.ndarray, head_dim: int, base: float = 10000.0) -> np.ndarray:
    """Standard RoPE reference. x: [seq, num_heads, head_dim]."""
    seq, num_heads, d = x.shape
    half = d // 2
    theta = 1.0 / (base ** (np.arange(0, d, 2, dtype=np.float32) / d))
    out = x.copy().astype(np.float32)
    for s in range(seq):
        pos = float(positions[s])
        freqs = pos * theta
        cos_f = np.cos(freqs)
        sin_f = np.sin(freqs)
        x1 = out[s, :, :half].copy()
        x2 = out[s, :, half:].copy()
        out[s, :, :half] = x1 * cos_f - x2 * sin_f
        out[s, :, half:] = x2 * cos_f + x1 * sin_f
    return out.astype(np.float16)


def test_rope(wgpu_device):
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey

    seq, num_heads, head_dim = 4, 8, 64
    x = np.random.randn(seq, num_heads, head_dim).astype(np.float16)
    positions = np.arange(seq, dtype=np.uint32)

    expected = rope_ref(x, positions, head_dim)

    dev = wgpu_device.wgpu_device
    x_buf = WebGPUBuffer.from_numpy(dev, x)
    pos_buf = WebGPUBuffer.from_numpy(dev, positions)
    out_buf = WebGPUBuffer.empty(dev, x.nbytes,
                                 usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)

    cache = PipelineCache(dev, SHADERS_DIR / "generic")
    key = PipelineKey("rope", (("HEAD_DIM", head_dim), ("NUM_HEADS", num_heads)))
    pipeline = cache.get_or_create(key)

    bg = dev.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": x_buf.buf}},
            {"binding": 1, "resource": {"buffer": pos_buf.buf}},
            {"binding": 2, "resource": {"buffer": out_buf.buf}},
        ],
    )
    encoder = dev.create_command_encoder()
    cp = encoder.begin_compute_pass()
    cp.set_pipeline(pipeline)
    cp.set_bind_group(0, bg)
    cp.dispatch_workgroups(seq, num_heads, 1)
    cp.end()
    dev.queue.submit([encoder.finish()])

    result = out_buf.to_numpy().view(np.float16).reshape(seq, num_heads, head_dim)
    np.testing.assert_allclose(result.astype(np.float32), expected.astype(np.float32),
                               rtol=1e-2, atol=1e-2)
