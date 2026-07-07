import numpy as np
import pytest
from pathlib import Path


SHADERS_DIR = Path(__file__).parent.parent / "vllm_webgpu" / "shaders"


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

def silu(x: np.ndarray) -> np.ndarray:
    x32 = x.astype(np.float64)
    return x32 * (1.0 / (1.0 + np.exp(-x32)))


def tanh_gelu(x: np.ndarray) -> np.ndarray:
    x32 = x.astype(np.float64)
    c = 0.7978845608  # sqrt(2/pi)
    return 0.5 * x32 * (1.0 + np.tanh(c * (x32 + 0.044715 * x32 ** 3)))


def fused_gate_act_ref(x, gate_W, up_W, gelu=False):
    """Reference: activation(gate_proj(x)) * up_proj(x), f32 accumulation."""
    x32    = x.astype(np.float32)
    gate32 = gate_W.astype(np.float32) @ x32   # [N]
    up32   = up_W.astype(np.float32) @ x32     # [N]
    act    = tanh_gelu(gate32) if gelu else silu(gate32)
    return np.clip(act * up32, -65504.0, 65504.0).astype(np.float16)


def sigmoid_gate_ref(gate, value):
    """Reference: sigmoid(gate) * value, f32 intermediate."""
    g32 = gate.astype(np.float32)
    v32 = value.astype(np.float32)
    sig = 1.0 / (1.0 + np.exp(-g32))
    return np.clip(sig * v32, -65504.0, 65504.0).astype(np.float16)


def fused_qkv_ref(x, q_W, k_W, v_W):
    """Reference: [q_proj(x) | k_proj(x) | v_proj(x)] concatenated."""
    x32 = x.astype(np.float32)
    q = q_W.astype(np.float32) @ x32
    k = k_W.astype(np.float32) @ x32
    v = v_W.astype(np.float32) @ x32
    return np.clip(np.concatenate([q, k, v]), -65504.0, 65504.0).astype(np.float16)


def flash_attn_ref(Q, K, V, scale, num_q_heads, num_kv_heads):
    """Reference: standard softmax attention.

    Q: [num_q_heads, head_dim] f32
    K: [ctx_len, head_dim] f32   (single KV head)
    V: [ctx_len, head_dim] f32
    """
    head_dim = Q.shape[-1]
    ctx_len = K.shape[0]
    out = np.zeros_like(Q, dtype=np.float32)
    for qh in range(num_q_heads):
        kvh = qh * num_kv_heads // num_q_heads
        q = Q[qh]
        # For GQA with num_kv_heads=1 there is only one K/V head.
        scores = (K @ q) * scale    # [ctx_len]
        scores -= scores.max()
        weights = np.exp(scores)
        weights /= weights.sum()
        out[qh] = weights @ V
    return out.astype(np.float16)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pack_weights(W: np.ndarray) -> np.ndarray:
    """Pack [N, K] f16 weight matrix as [N, K/2] u32 (two f16 per u32)."""
    return np.ascontiguousarray(W.astype(np.float16)).view(np.uint32)


def _dispatch(dev, pipeline, bg, groups):
    encoder = dev.create_command_encoder()
    cp = encoder.begin_compute_pass()
    cp.set_pipeline(pipeline)
    cp.set_bind_group(0, bg)
    cp.dispatch_workgroups(*groups)
    cp.end()
    dev.queue.submit([encoder.finish()])


# ---------------------------------------------------------------------------
# fused_gate_act
# ---------------------------------------------------------------------------

def test_fused_gate_act_silu(wgpu_device):
    """Correctness: fused gate+up GEMV with SiLU activation vs numpy reference."""
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey

    K, N = 256, 64
    x      = np.random.randn(K).astype(np.float16)
    gate_W = np.random.randn(N, K).astype(np.float16)
    up_W   = np.random.randn(N, K).astype(np.float16)

    expected = fused_gate_act_ref(x, gate_W, up_W, gelu=False)

    dev = wgpu_device.wgpu_device
    rw = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST

    x_buf      = WebGPUBuffer.from_numpy(dev, x)
    gate_w_buf = WebGPUBuffer.from_numpy(dev, _pack_weights(gate_W))
    up_w_buf   = WebGPUBuffer.from_numpy(dev, _pack_weights(up_W))
    act_buf    = WebGPUBuffer.empty(dev, N * 2, usage=rw)

    cache = PipelineCache(dev, SHADERS_DIR / "generic")
    key = PipelineKey("fused_gate_act", (("K", K), ("N", N), ("GELU", 0)))
    pipeline = cache.get_or_create(key)

    bg = dev.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": x_buf.buf}},
            {"binding": 1, "resource": {"buffer": gate_w_buf.buf}},
            {"binding": 2, "resource": {"buffer": up_w_buf.buf}},
            {"binding": 3, "resource": {"buffer": act_buf.buf}},
        ],
    )
    _dispatch(dev, pipeline, bg, (N, 1, 1))

    result = act_buf.to_numpy().view(np.float16)
    np.testing.assert_allclose(result.astype(np.float32), expected.astype(np.float32),
                               rtol=1e-2, atol=0.05,
                               err_msg="fused_gate_act SiLU mismatch")


def test_fused_gate_act_gelu(wgpu_device):
    """Correctness: fused gate+up GEMV with tanh-GELU activation vs numpy reference."""
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey

    rng = np.random.default_rng(42)
    K, N = 256, 64
    # Use small weights to avoid tanh-GELU overflow (tanh saturates for large inputs)
    x      = (rng.standard_normal(K) * 0.1).astype(np.float16)
    gate_W = (rng.standard_normal((N, K)) * 0.1).astype(np.float16)
    up_W   = (rng.standard_normal((N, K)) * 0.1).astype(np.float16)

    expected = fused_gate_act_ref(x, gate_W, up_W, gelu=True)

    dev = wgpu_device.wgpu_device
    rw = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST

    x_buf      = WebGPUBuffer.from_numpy(dev, x)
    gate_w_buf = WebGPUBuffer.from_numpy(dev, _pack_weights(gate_W))
    up_w_buf   = WebGPUBuffer.from_numpy(dev, _pack_weights(up_W))
    act_buf    = WebGPUBuffer.empty(dev, N * 2, usage=rw)

    cache = PipelineCache(dev, SHADERS_DIR / "generic")
    key = PipelineKey("fused_gate_act", (("K", K), ("N", N), ("GELU", 1)))
    pipeline = cache.get_or_create(key)

    bg = dev.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": x_buf.buf}},
            {"binding": 1, "resource": {"buffer": gate_w_buf.buf}},
            {"binding": 2, "resource": {"buffer": up_w_buf.buf}},
            {"binding": 3, "resource": {"buffer": act_buf.buf}},
        ],
    )
    _dispatch(dev, pipeline, bg, (N, 1, 1))

    result = act_buf.to_numpy().view(np.float16)
    np.testing.assert_allclose(result.astype(np.float32), expected.astype(np.float32),
                               rtol=1e-2, atol=0.05,
                               err_msg="fused_gate_act tanh-GELU mismatch")


def test_fused_gate_act_zero_input(wgpu_device):
    """Property proof: zero input → zero output for both SiLU and GELU.

    silu(0) = 0 * sigma(0) = 0. gelu(0) = 0.5 * 0 * (1 + tanh(0)) = 0.
    So gate_proj(0) = 0 → activation = 0 → output = 0 * up_proj(0) = 0.
    Any implementation that fails this test is computing the wrong function.
    """
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey

    K, N = 64, 16
    x      = np.zeros(K, dtype=np.float16)
    gate_W = np.random.randn(N, K).astype(np.float16)
    up_W   = np.random.randn(N, K).astype(np.float16)

    dev = wgpu_device.wgpu_device
    rw = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST

    x_buf      = WebGPUBuffer.from_numpy(dev, x)
    gate_w_buf = WebGPUBuffer.from_numpy(dev, _pack_weights(gate_W))
    up_w_buf   = WebGPUBuffer.from_numpy(dev, _pack_weights(up_W))

    cache = PipelineCache(dev, SHADERS_DIR / "generic")

    for gelu in [0, 1]:
        act_buf = WebGPUBuffer.empty(dev, N * 2, usage=rw)
        key = PipelineKey("fused_gate_act", (("K", K), ("N", N), ("GELU", gelu)))
        pipeline = cache.get_or_create(key)
        bg = dev.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": x_buf.buf}},
                {"binding": 1, "resource": {"buffer": gate_w_buf.buf}},
                {"binding": 2, "resource": {"buffer": up_w_buf.buf}},
                {"binding": 3, "resource": {"buffer": act_buf.buf}},
            ],
        )
        _dispatch(dev, pipeline, bg, (N, 1, 1))
        result = act_buf.to_numpy().view(np.float16)
        np.testing.assert_array_equal(
            result, np.zeros(N, dtype=np.float16),
            err_msg=f"fused_gate_act zero-input invariant violated (GELU={gelu})",
        )


# ---------------------------------------------------------------------------
# fused_qkv
# ---------------------------------------------------------------------------

def test_fused_qkv(wgpu_device):
    """Correctness: Q+K+V projections in one dispatch vs numpy reference."""
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey

    K, Q_DIM, KV_DIM = 64, 32, 16
    x   = np.random.randn(K).astype(np.float16)
    q_W = np.random.randn(Q_DIM, K).astype(np.float16)
    k_W = np.random.randn(KV_DIM, K).astype(np.float16)
    v_W = np.random.randn(KV_DIM, K).astype(np.float16)

    expected = fused_qkv_ref(x, q_W, k_W, v_W)

    dev = wgpu_device.wgpu_device
    rw = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST

    x_buf   = WebGPUBuffer.from_numpy(dev, x)
    qw_buf  = WebGPUBuffer.from_numpy(dev, _pack_weights(q_W))
    kw_buf  = WebGPUBuffer.from_numpy(dev, _pack_weights(k_W))
    vw_buf  = WebGPUBuffer.from_numpy(dev, _pack_weights(v_W))
    qkv_buf = WebGPUBuffer.empty(dev, (Q_DIM + 2 * KV_DIM) * 2, usage=rw)

    cache = PipelineCache(dev, SHADERS_DIR / "generic")
    key = PipelineKey("fused_qkv", (("K", K), ("Q_DIM", Q_DIM), ("KV_DIM", KV_DIM)))
    pipeline = cache.get_or_create(key)

    bg = dev.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": x_buf.buf}},
            {"binding": 1, "resource": {"buffer": qw_buf.buf}},
            {"binding": 2, "resource": {"buffer": kw_buf.buf}},
            {"binding": 3, "resource": {"buffer": vw_buf.buf}},
            {"binding": 4, "resource": {"buffer": qkv_buf.buf}},
        ],
    )
    _dispatch(dev, pipeline, bg, (Q_DIM + 2 * KV_DIM, 1, 1))

    result = qkv_buf.to_numpy().view(np.float16)
    np.testing.assert_allclose(result.astype(np.float32), expected.astype(np.float32),
                               rtol=1e-2, atol=1e-2,
                               err_msg="fused_qkv output mismatch")


def test_fused_qkv_partition_invariant(wgpu_device):
    """Property proof: Q, K, V sections of the output buffer are correct independently.

    The output layout is [Q | K | V]. If sections overlap or are misrouted,
    at least one section will mismatch the expected projection result.
    This test checks all three sections individually.
    """
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey

    K, Q_DIM, KV_DIM = 64, 32, 16
    x   = np.random.randn(K).astype(np.float16)
    q_W = np.random.randn(Q_DIM, K).astype(np.float16)
    k_W = np.random.randn(KV_DIM, K).astype(np.float16)
    v_W = np.random.randn(KV_DIM, K).astype(np.float16)

    x32 = x.astype(np.float32)
    q_ref = np.clip(q_W.astype(np.float32) @ x32, -65504.0, 65504.0).astype(np.float16)
    k_ref = np.clip(k_W.astype(np.float32) @ x32, -65504.0, 65504.0).astype(np.float16)
    v_ref = np.clip(v_W.astype(np.float32) @ x32, -65504.0, 65504.0).astype(np.float16)

    dev = wgpu_device.wgpu_device
    rw = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST

    x_buf   = WebGPUBuffer.from_numpy(dev, x)
    qw_buf  = WebGPUBuffer.from_numpy(dev, _pack_weights(q_W))
    kw_buf  = WebGPUBuffer.from_numpy(dev, _pack_weights(k_W))
    vw_buf  = WebGPUBuffer.from_numpy(dev, _pack_weights(v_W))
    qkv_buf = WebGPUBuffer.empty(dev, (Q_DIM + 2 * KV_DIM) * 2, usage=rw)

    cache = PipelineCache(dev, SHADERS_DIR / "generic")
    key = PipelineKey("fused_qkv", (("K", K), ("Q_DIM", Q_DIM), ("KV_DIM", KV_DIM)))
    pipeline = cache.get_or_create(key)

    bg = dev.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": x_buf.buf}},
            {"binding": 1, "resource": {"buffer": qw_buf.buf}},
            {"binding": 2, "resource": {"buffer": kw_buf.buf}},
            {"binding": 3, "resource": {"buffer": vw_buf.buf}},
            {"binding": 4, "resource": {"buffer": qkv_buf.buf}},
        ],
    )
    _dispatch(dev, pipeline, bg, (Q_DIM + 2 * KV_DIM, 1, 1))

    result = qkv_buf.to_numpy().view(np.float16)
    q_got = result[:Q_DIM]
    k_got = result[Q_DIM:Q_DIM + KV_DIM]
    v_got = result[Q_DIM + KV_DIM:]

    np.testing.assert_allclose(q_got.astype(np.float32), q_ref.astype(np.float32),
                               rtol=1e-2, atol=1e-2, err_msg="fused_qkv Q section mismatch")
    np.testing.assert_allclose(k_got.astype(np.float32), k_ref.astype(np.float32),
                               rtol=1e-2, atol=1e-2, err_msg="fused_qkv K section mismatch")
    np.testing.assert_allclose(v_got.astype(np.float32), v_ref.astype(np.float32),
                               rtol=1e-2, atol=1e-2, err_msg="fused_qkv V section mismatch")


# ---------------------------------------------------------------------------
# sigmoid_gate
# ---------------------------------------------------------------------------

def test_sigmoid_gate(wgpu_device):
    """Correctness: sigmoid(gate) * value vs numpy reference."""
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey

    N = 256
    gate  = np.random.randn(N).astype(np.float16)
    value = np.random.randn(N).astype(np.float16)

    expected = sigmoid_gate_ref(gate, value)

    dev = wgpu_device.wgpu_device
    rw = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST

    gate_buf  = WebGPUBuffer.from_numpy(dev, gate)
    value_buf = WebGPUBuffer.from_numpy(dev, value)
    out_buf   = WebGPUBuffer.empty(dev, N * 2, usage=rw)

    cache = PipelineCache(dev, SHADERS_DIR / "generic")
    key = PipelineKey("sigmoid_gate", (("N", N),))
    pipeline = cache.get_or_create(key)

    bg = dev.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": gate_buf.buf}},
            {"binding": 1, "resource": {"buffer": value_buf.buf}},
            {"binding": 2, "resource": {"buffer": out_buf.buf}},
        ],
    )
    # Dispatch: ceil(N/4/256) workgroups; N=256: ceil(256/4/256)=1
    _dispatch(dev, pipeline, bg, ((N // 4 + 255) // 256, 1, 1))

    result = out_buf.to_numpy().view(np.float16)
    np.testing.assert_allclose(result.astype(np.float32), expected.astype(np.float32),
                               rtol=1e-2, atol=1e-2,
                               err_msg="sigmoid_gate output mismatch")


def test_sigmoid_gate_bounded_by_value(wgpu_device):
    """Property proof: |sigmoid(gate) * value| <= |value| for all inputs.

    sigmoid is in (0, 1), so multiplying by gate never amplifies value.
    Any implementation that inflates the magnitude is not computing sigmoid.
    """
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey

    N = 256
    # Use large gate values to stress-test: sigmoid saturates near 0 and 1
    gate  = np.array([-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0] * 36 + [-5.0] * 4,
                     dtype=np.float16)[:N]
    value = np.random.randn(N).astype(np.float16)

    dev = wgpu_device.wgpu_device
    rw = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST

    gate_buf  = WebGPUBuffer.from_numpy(dev, gate)
    value_buf = WebGPUBuffer.from_numpy(dev, value)
    out_buf   = WebGPUBuffer.empty(dev, N * 2, usage=rw)

    cache = PipelineCache(dev, SHADERS_DIR / "generic")
    key = PipelineKey("sigmoid_gate", (("N", N),))
    pipeline = cache.get_or_create(key)

    bg = dev.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": gate_buf.buf}},
            {"binding": 1, "resource": {"buffer": value_buf.buf}},
            {"binding": 2, "resource": {"buffer": out_buf.buf}},
        ],
    )
    _dispatch(dev, pipeline, bg, ((N // 4 + 255) // 256, 1, 1))

    result = out_buf.to_numpy().view(np.float16).astype(np.float32)
    value_f32 = value.astype(np.float32)

    # |sigmoid(gate) * value| <= |value| (with a small tolerance for f16 rounding)
    assert np.all(np.abs(result) <= np.abs(value_f32) + 1e-2), (
        "sigmoid_gate amplitude invariant violated: output exceeds input value magnitude"
    )


# ---------------------------------------------------------------------------
# flash_attn_decode (reference test — not wired in production)
# ---------------------------------------------------------------------------

def test_flash_attn_decode(wgpu_device):
    """Reference test: fused QK+softmax+V matches standard softmax attention.

    Uses 1 query head, 1 KV head, context length 8 in a single 8-token block.
    The paged KV cache layout is [num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM].
    """
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey

    NUM_Q_HEADS, NUM_KV_HEADS = 1, 1
    HEAD_DIM, CTX_LEN, BLOCK_SIZE = 128, 8, 8

    np.random.seed(42)
    Q_np = np.random.randn(NUM_Q_HEADS, HEAD_DIM).astype(np.float16)
    # KV cache: [1 block, BLOCK_SIZE tokens, 1 kv_head, HEAD_DIM]
    K_cache_np = np.random.randn(1, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM).astype(np.float16)
    V_cache_np = np.random.randn(1, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM).astype(np.float16)
    # Block 0 holds tokens 0..CTX_LEN-1
    block_table = np.array([0], dtype=np.uint32)

    scale = 1.0 / np.sqrt(HEAD_DIM)

    # Reference: K and V for kv_head=0 in token order
    K_flat = K_cache_np[0, :CTX_LEN, 0, :].astype(np.float32)  # [CTX_LEN, HEAD_DIM]
    V_flat = V_cache_np[0, :CTX_LEN, 0, :].astype(np.float32)
    expected = flash_attn_ref(Q_np.astype(np.float32), K_flat, V_flat,
                              scale, NUM_Q_HEADS, NUM_KV_HEADS)

    dev = wgpu_device.wgpu_device
    rw = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST

    q_buf  = WebGPUBuffer.from_numpy(dev, Q_np)
    kc_buf = WebGPUBuffer.from_numpy(dev, K_cache_np)
    vc_buf = WebGPUBuffer.from_numpy(dev, V_cache_np)
    bt_buf = WebGPUBuffer.from_numpy(dev, block_table)
    out_buf = WebGPUBuffer.empty(dev, NUM_Q_HEADS * HEAD_DIM * 2, usage=rw)

    cache = PipelineCache(dev, SHADERS_DIR / "generic")
    key = PipelineKey("flash_attn_decode", (
        ("NUM_Q_HEADS", NUM_Q_HEADS),
        ("NUM_KV_HEADS", NUM_KV_HEADS),
        ("HEAD_DIM", HEAD_DIM),
        ("CTX_LEN", CTX_LEN),
        ("BLOCK_SIZE", BLOCK_SIZE),
    ))
    pipeline = cache.get_or_create(key)

    bg = dev.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": q_buf.buf}},
            {"binding": 1, "resource": {"buffer": kc_buf.buf}},
            {"binding": 2, "resource": {"buffer": vc_buf.buf}},
            {"binding": 3, "resource": {"buffer": bt_buf.buf}},
            {"binding": 4, "resource": {"buffer": out_buf.buf}},
        ],
    )
    _dispatch(dev, pipeline, bg, (NUM_Q_HEADS, 1, 1))

    result = out_buf.to_numpy().view(np.float16).reshape(NUM_Q_HEADS, HEAD_DIM)
    np.testing.assert_allclose(
        result.astype(np.float32), expected.astype(np.float32),
        rtol=1e-2, atol=1e-2,
        err_msg="flash_attn_decode output differs from standard softmax attention reference",
    )
