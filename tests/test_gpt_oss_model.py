import numpy as np
import pytest
from pathlib import Path

SHADERS_DIR = Path(__file__).parent.parent / "vllm_webgpu" / "shaders"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _f16(dev, rng, shape):
    import wgpu as wgpu_lib
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    rw = (wgpu_lib.BufferUsage.STORAGE
          | wgpu_lib.BufferUsage.COPY_SRC
          | wgpu_lib.BufferUsage.COPY_DST)
    arr = (rng.standard_normal(shape) * 0.02).astype(np.float16)
    return WebGPUBuffer.from_numpy(dev, arr, usage=rw)


def _kv_pool(model, dev, layers, num_blocks, block_size, kv_heads, head_dim):
    import wgpu as wgpu_lib
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    rw = (wgpu_lib.BufferUsage.STORAGE
          | wgpu_lib.BufferUsage.COPY_SRC
          | wgpu_lib.BufferUsage.COPY_DST)
    kv_bytes = num_blocks * block_size * kv_heads * head_dim * 2
    for _ in range(layers):
        k = WebGPUBuffer.empty(dev, kv_bytes, usage=rw)
        v = WebGPUBuffer.empty(dev, kv_bytes, usage=rw)
        model.kv_pool.append((k, v))


class _FakeMeta:
    slot_mapping = [0]
    block_tables = [np.zeros(8, dtype=np.uint32)]
    max_decode_seq_len = 1


# ── Test 1: GptOssForCausalLM full forward ────────────────────────────────────

@pytest.mark.integration
def test_gpt_oss_forward(wgpu_device):
    """2-layer GPT-OSS decode step with 4 experts, top-2, swiglu_limit=7.0,
    attention_bias=True, and alternating sliding/full attention.

    Exercises:
    - attention bias addition (Q/K/V/O) via add.wgsl
    - MoE FFN with mlp.router and mlp.experts key prefix
    - CLAMP_MAX passed to fused_gate_act (swiglu_limit)
    - Per-layer effective context (sliding vs full)
    """
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.gpt_oss import GptOssWebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR as _SD

    hidden      = 64
    layers      = 2
    q_heads     = 4
    kv_heads    = 2
    head_dim    = 16
    inter       = 128
    vocab       = 32
    num_experts = 4
    top_k       = 2
    num_blocks  = 8
    block_size  = 16

    class _Cfg:
        num_hidden_layers       = layers
        num_attention_heads     = q_heads
        num_key_value_heads     = kv_heads
        hidden_size             = hidden
        intermediate_size       = inter
        vocab_size              = vocab
        head_dim                = hidden // q_heads   # 16 — computed to avoid name clash
        max_position_embeddings = 128
        rope_theta              = 10000.0
        sliding_window          = 128
        num_local_experts       = num_experts
        num_experts_per_tok     = top_k
        swiglu_limit            = 7.0
        attention_bias          = True
        layer_types             = ["sliding_attention", "full_attention"]

    dev = wgpu_device.wgpu_device
    cache = PipelineCache(dev, _SD)
    model = GptOssWebGPUModel(_Cfg(), wgpu_device, cache)

    assert model._is_moe
    assert model._attn_bias
    assert model._swiglu_limit == 7.0

    rng = np.random.default_rng(7)
    q_dim  = q_heads * head_dim
    kv_dim = kv_heads * head_dim

    # Shared weights
    model.weights["model.embed_tokens.weight"] = _f16(dev, rng, (vocab, hidden))
    model.weights["model.norm.weight"]          = _f16(dev, rng, (hidden,))

    for i in range(layers):
        p = f"model.layers.{i}"
        model.weights[f"{p}.input_layernorm.weight"]          = _f16(dev, rng, (hidden,))
        model.weights[f"{p}.post_attention_layernorm.weight"] = _f16(dev, rng, (hidden,))
        # Attention projections + biases
        model.weights[f"{p}.self_attn.q_proj.weight"] = _f16(dev, rng, (q_dim,  hidden))
        model.weights[f"{p}.self_attn.k_proj.weight"] = _f16(dev, rng, (kv_dim, hidden))
        model.weights[f"{p}.self_attn.v_proj.weight"] = _f16(dev, rng, (kv_dim, hidden))
        model.weights[f"{p}.self_attn.o_proj.weight"] = _f16(dev, rng, (hidden, q_dim))
        model.weights[f"{p}.self_attn.q_proj.bias"]   = _f16(dev, rng, (q_dim,))
        model.weights[f"{p}.self_attn.k_proj.bias"]   = _f16(dev, rng, (kv_dim,))
        model.weights[f"{p}.self_attn.v_proj.bias"]   = _f16(dev, rng, (kv_dim,))
        model.weights[f"{p}.self_attn.o_proj.bias"]   = _f16(dev, rng, (hidden,))
        # MoE router (mlp.router, not block_sparse_moe.gate)
        model.weights[f"{p}.mlp.router.weight"] = _f16(dev, rng, (num_experts, hidden))
        # Expert weights (mlp.experts, not block_sparse_moe.experts)
        for j in range(num_experts):
            ep = f"{p}.mlp.experts.{j}"
            model.weights[f"{ep}.w1.weight"] = _f16(dev, rng, (inter,  hidden))
            model.weights[f"{ep}.w3.weight"] = _f16(dev, rng, (inter,  hidden))
            model.weights[f"{ep}.w2.weight"] = _f16(dev, rng, (hidden, inter))

    _kv_pool(model, dev, layers, num_blocks, block_size, kv_heads, head_dim)

    result = model.forward(
        np.array([1], dtype=np.uint32),
        np.array([0], dtype=np.uint32),
        _FakeMeta(),
    )

    token_id = int(result[0, 0])
    assert 0 <= token_id < vocab, f"token_id {token_id} out of [0, {vocab})"


# ── Test 2: swiglu_limit clamping via CLAMP_MAX ───────────────────────────────

@pytest.mark.integration
def test_swiglu_limit(wgpu_device):
    """Verify fused_gate_act with CLAMP_MAX=7.0 clips the gate activation.

    Gate input is chosen so that silu(gate_proj(x)) >> 7. With CLAMP_MAX=7.0
    the result should be approximately 7.0 * up_proj(x). Without the clamp it
    would be >> 7.0 * up_proj(x).
    """
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey

    dev = wgpu_device.wgpu_device
    rw = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST

    K, N = 256, 4

    # gate_W: first element of row 0 is large so gate_proj(x)[0] >> 7.
    # silu(gate_proj) will be ~gate_proj for large positive values.
    x = np.zeros(K, dtype=np.float16)
    x[0] = np.float16(1.0)

    gate_W = np.zeros((N, K), dtype=np.float16)
    gate_W[0, 0] = np.float16(20.0)   # gate_proj(x)[0] ≈ 20; silu ≈ 20

    up_W = np.zeros((N, K), dtype=np.float16)
    up_W[0, 0] = np.float16(1.0)      # up_proj(x)[0] = 1.0

    def _pack(W):
        return np.ascontiguousarray(W.astype(np.float16)).view(np.uint32)

    x_buf      = WebGPUBuffer.from_numpy(dev, x)
    gate_w_buf = WebGPUBuffer.from_numpy(dev, _pack(gate_W))
    up_w_buf   = WebGPUBuffer.from_numpy(dev, _pack(up_W))
    act_clamped = WebGPUBuffer.empty(dev, N * 2, usage=rw)
    act_free    = WebGPUBuffer.empty(dev, N * 2, usage=rw)

    cache = PipelineCache(dev, SHADERS_DIR / "generic")

    def _run(act_buf, clamp_max):
        overrides = [("K", K), ("N", N), ("GELU", 0)]
        if clamp_max > 0:
            overrides.append(("CLAMP_MAX", float(clamp_max)))
        key = PipelineKey("fused_gate_act", tuple(overrides))
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
        encoder = dev.create_command_encoder()
        cp = encoder.begin_compute_pass()
        cp.set_pipeline(pipeline)
        cp.set_bind_group(0, bg)
        cp.dispatch_workgroups(N, 1, 1)
        cp.end()
        dev.queue.submit([encoder.finish()])
        dev.queue.on_submitted_work_done_sync()
        return act_buf.to_numpy().view(np.float16)

    result_clamped = _run(act_clamped, clamp_max=7.0)
    result_free    = _run(act_free,    clamp_max=0.0)

    # With CLAMP_MAX=7.0: gate activation clamped to 7.0, result ≈ 7.0 * 1.0 = 7.0
    assert float(result_clamped[0]) <= 7.5, (
        f"Expected clamped result ≤ 7.5, got {result_clamped[0]}"
    )
    # Without clamp: silu(20) ≈ 20, result[0] >> 7
    assert float(result_free[0]) > 10.0, (
        f"Expected unclamped result > 10.0, got {result_free[0]}"
    )
    # Clamped is strictly smaller than unclamped
    assert float(result_clamped[0]) < float(result_free[0]), (
        f"Clamped ({result_clamped[0]}) should be < unclamped ({result_free[0]})"
    )


# ── Test 3: per-expert bias injection (w1/w3/w2 biases) ─────────────────────

@pytest.mark.integration
def test_gpt_oss_expert_bias(wgpu_device):
    """Forward pass with non-zero expert gate, up, and down biases.

    Verifies that the bias-injection path in _dispatch_expert_gate_up and
    _dispatch_expert_down runs without error and produces a valid token id.
    Also checks that the biased result differs from the no-bias result when
    biases are non-zero (non-trivial sanity check).
    """
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.gpt_oss import GptOssWebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR as _SD

    hidden      = 64
    layers      = 1
    q_heads     = 4
    kv_heads    = 2
    head_dim    = 16
    inter       = 128
    vocab       = 32
    num_experts = 4
    top_k       = 2
    num_blocks  = 8
    block_size  = 16

    class _Cfg:
        num_hidden_layers       = layers
        num_attention_heads     = q_heads
        num_key_value_heads     = kv_heads
        hidden_size             = hidden
        intermediate_size       = inter
        vocab_size              = vocab
        head_dim                = hidden // q_heads
        max_position_embeddings = 128
        rope_theta              = 10000.0
        sliding_window          = 128
        num_local_experts       = num_experts
        num_experts_per_tok     = top_k
        swiglu_limit            = 0.0
        attention_bias          = False
        layer_types             = []

    dev = wgpu_device.wgpu_device
    cache = PipelineCache(dev, _SD)
    rng = np.random.default_rng(42)

    def _make_model(with_expert_bias: bool) -> GptOssWebGPUModel:
        model = GptOssWebGPUModel(_Cfg(), wgpu_device, cache)
        model.weights["model.embed_tokens.weight"] = _f16(dev, rng, (vocab, hidden))
        model.weights["model.norm.weight"]          = _f16(dev, rng, (hidden,))
        q_dim  = q_heads * head_dim
        kv_dim = kv_heads * head_dim
        for i in range(layers):
            p = f"model.layers.{i}"
            model.weights[f"{p}.input_layernorm.weight"]          = _f16(dev, rng, (hidden,))
            model.weights[f"{p}.post_attention_layernorm.weight"] = _f16(dev, rng, (hidden,))
            model.weights[f"{p}.self_attn.q_proj.weight"] = _f16(dev, rng, (q_dim,  hidden))
            model.weights[f"{p}.self_attn.k_proj.weight"] = _f16(dev, rng, (kv_dim, hidden))
            model.weights[f"{p}.self_attn.v_proj.weight"] = _f16(dev, rng, (kv_dim, hidden))
            model.weights[f"{p}.self_attn.o_proj.weight"] = _f16(dev, rng, (hidden, q_dim))
            model.weights[f"{p}.mlp.router.weight"] = _f16(dev, rng, (num_experts, hidden))
            for j in range(num_experts):
                ep = f"{p}.mlp.experts.{j}"
                model.weights[f"{ep}.w1.weight"] = _f16(dev, rng, (inter,  hidden))
                model.weights[f"{ep}.w3.weight"] = _f16(dev, rng, (inter,  hidden))
                model.weights[f"{ep}.w2.weight"] = _f16(dev, rng, (hidden, inter))
                if with_expert_bias:
                    model.weights[f"{ep}.w1.bias"] = _f16(dev, rng, (inter,))
                    model.weights[f"{ep}.w3.bias"] = _f16(dev, rng, (inter,))
                    model.weights[f"{ep}.w2.bias"] = _f16(dev, rng, (hidden,))
        _kv_pool(model, dev, layers, num_blocks, block_size, kv_heads, head_dim)
        return model

    input_ids = np.array([1], dtype=np.uint32)
    positions  = np.array([0], dtype=np.uint32)

    model_no_bias   = _make_model(with_expert_bias=False)
    model_with_bias = _make_model(with_expert_bias=True)

    result_no_bias   = model_no_bias.forward(input_ids, positions, _FakeMeta())
    result_with_bias = model_with_bias.forward(input_ids, positions, _FakeMeta())

    token_no_bias   = int(result_no_bias[0, 0])
    token_with_bias = int(result_with_bias[0, 0])

    assert 0 <= token_no_bias   < vocab, f"no-bias token {token_no_bias} out of range"
    assert 0 <= token_with_bias < vocab, f"biased token {token_with_bias} out of range"
    # Non-zero expert biases change the output (not guaranteed for every seed, but
    # extremely unlikely to produce identical logits for random weights and biases).
    assert token_no_bias != token_with_bias, (
        "Expected biased and no-bias outputs to differ; got identical token ids. "
        "Bias injection may be a no-op."
    )
