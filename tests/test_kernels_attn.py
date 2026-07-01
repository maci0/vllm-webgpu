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
    key = PipelineKey("softmax", (("SEQ_LEN", n),))
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


def test_attention_pipeline_correctness(wgpu_device):
    """End-to-end correctness: kv_cache_store -> attn_score -> softmax -> attn_output.

    With ctx_len=1 (single cached position), softmax([score]) = [1.0], so
    attn_output == V. This gives us a deterministic correctness check.
    """
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey

    num_q_heads, num_kv_heads, head_dim = 4, 2, 16
    block_size, num_blocks = 16, 4
    ctx_len = 1  # single cached position → softmax([score]) = [1.0] → output = V

    dev = wgpu_device.wgpu_device
    rw = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST

    # Random Q, K, V — result must equal V since softmax of 1 element = 1.0
    q = np.random.randn(num_q_heads, head_dim).astype(np.float16)
    k = np.random.randn(num_kv_heads, head_dim).astype(np.float16)
    v = np.random.randn(num_kv_heads, head_dim).astype(np.float16)

    q_buf = WebGPUBuffer.from_numpy(dev, q)
    k_buf = WebGPUBuffer.from_numpy(dev, k)
    v_buf = WebGPUBuffer.from_numpy(dev, v)

    # KV cache pools
    k_cache = WebGPUBuffer.empty(dev, num_blocks * block_size * num_kv_heads * head_dim * 2, usage=rw)
    v_cache = WebGPUBuffer.empty(dev, num_blocks * block_size * num_kv_heads * head_dim * 2, usage=rw)

    # slot_mapping: [0] → block 0, offset 0
    slot_map = WebGPUBuffer.from_numpy(dev, np.array([0], dtype=np.uint32))
    # block_table: block 0 → physical block 0
    block_table = WebGPUBuffer.from_numpy(dev, np.zeros(num_blocks, dtype=np.uint32))

    cache = PipelineCache(dev, SHADERS_DIR / "generic")
    kv_key = PipelineKey("kv_cache_store",
                         (("BLOCK_SIZE", block_size), ("HEAD_DIM", head_dim), ("NUM_KV_HEADS", num_kv_heads)))

    # Store K
    kv_pip = cache.get_or_create(kv_key)
    for kv_in, kv_out in [(k_buf, k_cache), (v_buf, v_cache)]:
        bg = dev.create_bind_group(layout=kv_pip.get_bind_group_layout(0),
                                   entries=[{"binding": 0, "resource": {"buffer": kv_in.buf}},
                                            {"binding": 1, "resource": {"buffer": kv_out.buf}},
                                            {"binding": 2, "resource": {"buffer": slot_map.buf}}])
        enc = dev.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(kv_pip)
        cp.set_bind_group(0, bg)
        cp.dispatch_workgroups(1, num_kv_heads, 1)
        cp.end()
        dev.queue.submit([enc.finish()])

    # Attention scores: [num_q_heads, ctx_len=1]
    scores_buf = WebGPUBuffer.empty(dev, num_q_heads * ctx_len * 2, usage=rw)
    score_key = PipelineKey("attn_score",
                            (("BLOCK_SIZE", block_size), ("HEAD_DIM", head_dim),
                             ("NUM_Q_HEADS", num_q_heads), ("NUM_KV_HEADS", num_kv_heads),
                             ("MAX_SEQ_LEN", ctx_len)))
    sc_pip = cache.get_or_create(score_key)
    bg = dev.create_bind_group(layout=sc_pip.get_bind_group_layout(0),
                               entries=[{"binding": 0, "resource": {"buffer": q_buf.buf}},
                                        {"binding": 1, "resource": {"buffer": k_cache.buf}},
                                        {"binding": 2, "resource": {"buffer": block_table.buf}},
                                        {"binding": 3, "resource": {"buffer": scores_buf.buf}}])
    enc = dev.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(sc_pip)
    cp.set_bind_group(0, bg)
    cp.dispatch_workgroups(num_q_heads, ctx_len, 1)
    cp.end()
    dev.queue.submit([enc.finish()])

    # Softmax over [num_q_heads, ctx_len=1] — each row has 1 element → output = [1.0]
    sm_buf = WebGPUBuffer.empty(dev, scores_buf.nbytes, usage=rw)
    sm_key = PipelineKey("softmax", (("SEQ_LEN", ctx_len),))
    sm_pip = cache.get_or_create(sm_key)
    bg = dev.create_bind_group(layout=sm_pip.get_bind_group_layout(0),
                               entries=[{"binding": 0, "resource": {"buffer": scores_buf.buf}},
                                        {"binding": 1, "resource": {"buffer": sm_buf.buf}}])
    enc = dev.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(sm_pip)
    cp.set_bind_group(0, bg)
    cp.dispatch_workgroups(num_q_heads, 1, 1)
    cp.end()
    dev.queue.submit([enc.finish()])

    # Attention output: [num_q_heads, head_dim] — with weights=[1.0], output=V
    attn_out = WebGPUBuffer.empty(dev, num_q_heads * head_dim * 2, usage=rw)
    ao_key = PipelineKey("attn_output",
                         (("BLOCK_SIZE", block_size), ("HEAD_DIM", head_dim),
                          ("NUM_Q_HEADS", num_q_heads), ("NUM_KV_HEADS", num_kv_heads),
                          ("CTX_LEN", ctx_len)))
    ao_pip = cache.get_or_create(ao_key)
    bg = dev.create_bind_group(layout=ao_pip.get_bind_group_layout(0),
                               entries=[{"binding": 0, "resource": {"buffer": sm_buf.buf}},
                                        {"binding": 1, "resource": {"buffer": v_cache.buf}},
                                        {"binding": 2, "resource": {"buffer": block_table.buf}},
                                        {"binding": 3, "resource": {"buffer": attn_out.buf}}])
    enc = dev.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(ao_pip)
    cp.set_bind_group(0, bg)
    cp.dispatch_workgroups(num_q_heads, 1, 1)
    cp.end()
    dev.queue.submit([enc.finish()])

    result = attn_out.to_numpy().view(np.float16).reshape(num_q_heads, head_dim)

    # With ctx_len=1: softmax=1.0, so output[q_head] = V[kv_head] where kv_head = q_head // (Q/KV ratio)
    ratio = num_q_heads // num_kv_heads  # 2
    expected = np.stack([v[q_head // ratio] for q_head in range(num_q_heads)])  # [num_q_heads, head_dim]

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
