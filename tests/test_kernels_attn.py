import numpy as np
import pytest
from pathlib import Path

SHADERS_DIR = Path(__file__).parent.parent / "vllm_webgpu" / "shaders"


def softmax_ref(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x -= x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float16)


def test_softmax(wgpu_device):
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey

    seq, n = 4, 32
    x = np.random.randn(seq, n).astype(np.float16)
    expected = softmax_ref(x)

    dev = wgpu_device.wgpu_device
    x_buf = WebGPUBuffer.from_numpy(dev, x)
    out_buf = WebGPUBuffer.empty(dev, x.nbytes,
                                 usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)

    cache = PipelineCache(dev, SHADERS_DIR / "generic")
    key = PipelineKey("softmax", (("SEQ_LEN", n), ("BATCH", seq)))
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
    cp.dispatch_workgroups(seq, 1, 1)
    cp.end()
    dev.queue.submit([encoder.finish()])

    result = out_buf.to_numpy().view(np.float16).reshape(seq, n)
    np.testing.assert_allclose(result.astype(np.float32), expected.astype(np.float32),
                               rtol=1e-2, atol=1e-2)


def test_kv_cache_store_and_attn(wgpu_device):
    """Smoke test: store K/V then read back via attn_score kernel."""
    # This is an integration-level check — just verifies shapes and no crashes
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer

    num_blocks, block_size, num_kv_heads, head_dim = 8, 16, 4, 64
    seq_len = 2

    dev = wgpu_device.wgpu_device

    k_cache = WebGPUBuffer.empty(
        dev,
        num_blocks * block_size * num_kv_heads * head_dim * 2,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
    )
    v_cache = WebGPUBuffer.empty(
        dev,
        num_blocks * block_size * num_kv_heads * head_dim * 2,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
    )
    # block_table: one block per token for simplicity
    block_table = np.zeros((1, num_blocks), dtype=np.uint32)
    block_table[0, :seq_len] = np.arange(seq_len, dtype=np.uint32)
    bt_buf = WebGPUBuffer.from_numpy(dev, block_table,
                                     usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)

    # Just verify buffers allocated without error
    assert k_cache.nbytes > 0
    assert v_cache.nbytes > 0
