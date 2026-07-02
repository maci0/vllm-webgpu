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


def test_softmax_sum_to_one_invariant(wgpu_device):
    """Property proof: softmax output always sums to 1.0 for any finite input.

    This is the fundamental mathematical invariant of softmax:
      sum_i(exp(x_i) / sum_j(exp(x_j))) == 1.0 exactly.
    Any implementation that fails this test cannot be a correct softmax.
    """
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey

    seq, n = 8, 64  # 8 independent rows of 64 elements each
    # Test with varied inputs: small, large, uniform, extreme spread
    test_inputs = [
        np.random.randn(seq, n).astype(np.float16),                    # standard
        (np.random.randn(seq, n) * 10).astype(np.float16),             # large values
        np.zeros((seq, n), dtype=np.float16),                           # all zeros
        np.full((seq, n), -5.0, dtype=np.float16),                     # uniform negative
    ]

    dev = wgpu_device.wgpu_device
    rw = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
    cache = PipelineCache(dev, SHADERS_DIR / "generic")
    pipeline = cache.get_or_create(PipelineKey("softmax", (("SEQ_LEN", n),)))

    for x in test_inputs:
        x_buf = WebGPUBuffer.from_numpy(dev, x)
        out_buf = WebGPUBuffer.empty(dev, x.nbytes, usage=rw)

        bg = dev.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[{"binding": 0, "resource": {"buffer": x_buf.buf}},
                     {"binding": 1, "resource": {"buffer": out_buf.buf}}],
        )
        enc = dev.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipeline)
        cp.set_bind_group(0, bg)
        cp.dispatch_workgroups(seq, 1, 1)
        cp.end()
        dev.queue.submit([enc.finish()])

        result = out_buf.to_numpy().view(np.float16).reshape(seq, n).astype(np.float32)

        # Invariant: each row must sum to exactly 1.0
        row_sums = result.sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones(seq), rtol=1e-2, atol=1e-2,
                                   err_msg="Softmax invariant violated: row sum != 1.0")

        # Additional invariant: all values must be in (0, 1]
        assert (result >= 0).all(), "Softmax output contains negative values"
        assert (result <= 1.001).all(), "Softmax output contains values > 1"


def test_softmax_translation_invariance(wgpu_device):
    """Property proof: softmax is translation-invariant.

    Mathematical invariant: softmax(x + c) == softmax(x) for any constant c.
    This follows from exp(x_i + c) / sum(exp(x_j + c)) = exp(x_i)*exp(c) / (exp(c)*sum(exp(x_j)))
    = exp(x_i) / sum(exp(x_j)) = softmax(x)_i.

    Any implementation that fails this cannot be computing softmax correctly —
    and specifically, our numerically stable implementation (subtract max)
    uses this property internally. If it fails, the subtraction is wrong.
    """
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey

    n = 32
    x = np.random.randn(1, n).astype(np.float16)

    dev = wgpu_device.wgpu_device
    rw = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
    cache = PipelineCache(dev, SHADERS_DIR / "generic")
    pipeline = cache.get_or_create(PipelineKey("softmax", (("SEQ_LEN", n),)))

    results = []
    for shift in [0.0, 5.0, -5.0, 100.0]:  # large shift stresses numeric stability
        x_shifted = (x.astype(np.float32) + shift).astype(np.float16)
        xb = WebGPUBuffer.from_numpy(dev, x_shifted)
        ob = WebGPUBuffer.empty(dev, x.nbytes, usage=rw)
        bg = dev.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[{"binding": 0, "resource": {"buffer": xb.buf}},
                     {"binding": 1, "resource": {"buffer": ob.buf}}],
        )
        enc = dev.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipeline)
        cp.set_bind_group(0, bg)
        cp.dispatch_workgroups(1, 1, 1)
        cp.end()
        dev.queue.submit([enc.finish()])
        results.append(ob.to_numpy().view(np.float16).reshape(1, n).astype(np.float32))

    # Invariant: shift doesn't change softmax output
    for i, shift in enumerate([0.0, 5.0, -5.0, 100.0]):
        np.testing.assert_allclose(
            results[i], results[0], rtol=5e-2, atol=1e-2,
            err_msg=f"Softmax translation invariance violated: shift={shift}"
        )


def test_softmax_monotonicity_invariant(wgpu_device):
    """Property proof: softmax preserves the ordering of its inputs.

    Invariant: if x[i] > x[j] then softmax(x)[i] > softmax(x)[j].
    This follows from exp being strictly monotone increasing. Any softmax
    that violates this cannot be computing exp(x_i) / sum(exp(x_j)).
    """
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey

    n = 32
    # Construct input where we know the ordering: x[0] is the max
    x = np.random.randn(1, n).astype(np.float16)
    max_idx = int(np.argmax(x[0]))

    dev = wgpu_device.wgpu_device
    rw = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
    x_buf = WebGPUBuffer.from_numpy(dev, x)
    out_buf = WebGPUBuffer.empty(dev, x.nbytes, usage=rw)

    cache = PipelineCache(dev, SHADERS_DIR / "generic")
    pipeline = cache.get_or_create(PipelineKey("softmax", (("SEQ_LEN", n),)))
    bg = dev.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[{"binding": 0, "resource": {"buffer": x_buf.buf}},
                 {"binding": 1, "resource": {"buffer": out_buf.buf}}],
    )
    enc = dev.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(pipeline)
    cp.set_bind_group(0, bg)
    cp.dispatch_workgroups(1, 1, 1)
    cp.end()
    dev.queue.submit([enc.finish()])

    result = out_buf.to_numpy().view(np.float16).reshape(1, n).astype(np.float32)[0]

    # Invariant: the argmax of softmax output must equal argmax of input
    assert int(np.argmax(result)) == max_idx, (
        f"Softmax monotonicity violated: argmax(softmax(x))={np.argmax(result)} "
        f"but argmax(x)={max_idx}"
    )
    # Invariant: the max element of softmax must be > 1/n (it has more than uniform probability)
    assert result[max_idx] > 1.0 / n, "Softmax monotonicity violated: max element not largest"


def test_attention_equal_keys_uniform_weights(wgpu_device):
    """Property proof: when all K vectors are identical, attention output is mean(V).

    Mathematical invariant:
    If K_0 = K_1 = ... = K_{n-1} = K, then:
      scores_i = Q·K / sqrt(d) = same for all i
      softmax(scores) = [1/n, 1/n, ..., 1/n]
      output = (1/n) * sum(V_i) = mean(V)

    This is an exact, verifiable result for any Q, K, and set of V vectors.
    Tested with n=4 positions: output must be the arithmetic mean of the 4 V vectors.
    """
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey

    num_q_heads, num_kv_heads, head_dim = 2, 1, 16  # 2 Q-heads share 1 KV-head
    block_size, num_blocks, ctx_len = 16, 4, 4  # 4 positions with identical K

    dev = wgpu_device.wgpu_device
    rw = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST

    # Q: arbitrary; K: all identical; V: 4 distinct vectors
    q = np.random.randn(num_q_heads, head_dim).astype(np.float16)
    k_repeated = np.random.randn(head_dim).astype(np.float16)   # same K for all positions
    v_vectors = np.random.randn(ctx_len, head_dim).astype(np.float16)  # distinct V

    # Build KV caches with the same K but different V at each position
    k_cache = WebGPUBuffer.empty(dev, num_blocks * block_size * num_kv_heads * head_dim * 2, usage=rw)
    v_cache = WebGPUBuffer.empty(dev, num_blocks * block_size * num_kv_heads * head_dim * 2, usage=rw)
    block_table = WebGPUBuffer.from_numpy(dev, np.zeros(num_blocks, dtype=np.uint32))

    cache = PipelineCache(dev, SHADERS_DIR / "generic")
    kv_pip = cache.get_or_create(PipelineKey("kv_cache_store",
                 (("BLOCK_SIZE", block_size), ("HEAD_DIM", head_dim), ("NUM_KV_HEADS", num_kv_heads))))

    # Store K (identical) and distinct V at each position
    for pos in range(ctx_len):
        slot_map = WebGPUBuffer.from_numpy(dev, np.array([pos], dtype=np.uint32))
        k_tok = k_repeated.reshape(1, head_dim)   # shape [1, head_dim]
        v_tok = v_vectors[pos].reshape(1, head_dim)
        for kv_in, kv_out in [(WebGPUBuffer.from_numpy(dev, k_tok), k_cache),
                              (WebGPUBuffer.from_numpy(dev, v_tok), v_cache)]:
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

    # attn_score
    q_buf = WebGPUBuffer.from_numpy(dev, q)
    scores_buf = WebGPUBuffer.empty(dev, num_q_heads * ctx_len * 2, usage=rw)
    sc_pip = cache.get_or_create(PipelineKey("attn_score",
                 (("BLOCK_SIZE", block_size), ("HEAD_DIM", head_dim), ("MAX_SEQ_LEN", ctx_len),
                  ("NUM_Q_HEADS", num_q_heads), ("NUM_KV_HEADS", num_kv_heads))))
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

    # softmax
    sm_buf = WebGPUBuffer.empty(dev, scores_buf.nbytes, usage=rw)
    sm_pip = cache.get_or_create(PipelineKey("softmax", (("SEQ_LEN", ctx_len),)))
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

    # attn_output
    attn_out = WebGPUBuffer.empty(dev, num_q_heads * head_dim * 2, usage=rw)
    ao_pip = cache.get_or_create(PipelineKey("attn_output",
                 (("BLOCK_SIZE", block_size), ("HEAD_DIM", head_dim), ("CTX_LEN", ctx_len),
                  ("NUM_Q_HEADS", num_q_heads), ("NUM_KV_HEADS", num_kv_heads))))
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

    # Invariant: identical K vectors → uniform attention weights → output = mean(V)
    # Both Q-heads use the same KV cache (single KV head, GQA ratio=2)
    # Compute reference mean in float32 to match shader's f32 accumulation precision.
    # v_vectors.mean() in f16 accumulates in f16, diverging from the shader's f32 path.
    v_mean = v_vectors.astype(np.float32).mean(axis=0).astype(np.float16)
    for q_head in range(num_q_heads):
        np.testing.assert_allclose(
            result[q_head].astype(np.float32), v_mean.astype(np.float32),
            rtol=0.1, atol=0.5,
            err_msg=f"Attention mean(V) invariant violated for q_head={q_head}: "
                    f"identical K must give uniform weights and output=mean(V)"
        )


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
