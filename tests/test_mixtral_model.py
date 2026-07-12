import numpy as np
import pytest


# ── Shared helpers ────────────────────────────────────────────────────────────

def _f16(dev, rng, shape):
    """Return a float16 WebGPUBuffer filled with small random values."""
    import wgpu as wgpu_lib
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    rw = (wgpu_lib.BufferUsage.STORAGE
          | wgpu_lib.BufferUsage.COPY_SRC
          | wgpu_lib.BufferUsage.COPY_DST)
    arr = (rng.standard_normal(shape) * 0.02).astype(np.float16)
    return WebGPUBuffer.from_numpy(dev, arr, usage=rw)


def _inject_llama_weights(model, dev, rng, hidden, q_dim, kv_dim, inter, vocab, layers):
    """Inject standard Llama-style dense weights into model.weights."""
    model.weights["model.embed_tokens.weight"] = _f16(dev, rng, (vocab, hidden))
    model.weights["model.norm.weight"]          = _f16(dev, rng, (hidden,))
    for i in range(layers):
        p = f"model.layers.{i}"
        model.weights[f"{p}.input_layernorm.weight"]          = _f16(dev, rng, (hidden,))
        model.weights[f"{p}.post_attention_layernorm.weight"] = _f16(dev, rng, (hidden,))
        model.weights[f"{p}.self_attn.q_proj.weight"]         = _f16(dev, rng, (q_dim,  hidden))
        model.weights[f"{p}.self_attn.k_proj.weight"]         = _f16(dev, rng, (kv_dim, hidden))
        model.weights[f"{p}.self_attn.v_proj.weight"]         = _f16(dev, rng, (kv_dim, hidden))
        model.weights[f"{p}.self_attn.o_proj.weight"]         = _f16(dev, rng, (hidden, q_dim))
        model.weights[f"{p}.mlp.gate_proj.weight"]            = _f16(dev, rng, (inter,  hidden))
        model.weights[f"{p}.mlp.up_proj.weight"]              = _f16(dev, rng, (inter,  hidden))
        model.weights[f"{p}.mlp.down_proj.weight"]            = _f16(dev, rng, (hidden, inter))


def _kv_pool(model, dev, layers, num_blocks, block_size, kv_heads, head_dim):
    """Allocate KV cache buffers."""
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


# ── Test 1: Mistral (dense, SWA) forward ─────────────────────────────────────

@pytest.mark.integration
def test_mistral_forward(wgpu_device):
    """Mistral dense forward pass with sliding_window=128.

    No MoE. Uses the standard Llama FFN path via MixtralWebGPUModel.
    SWA does not clip the effective ctx for a 1-token context, so the
    attention path is identical to Llama's.
    """
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.mixtral import MixtralWebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    hidden    = 64
    layers    = 2
    q_heads   = 4
    kv_heads  = 2
    head_dim  = 16   # hidden // q_heads
    inter     = 128
    vocab     = 32
    num_blocks = 8
    block_size = 16

    class _MistralCfg:
        num_hidden_layers      = layers
        num_attention_heads    = q_heads
        num_key_value_heads    = kv_heads
        hidden_size            = hidden
        intermediate_size      = inter
        vocab_size             = vocab
        head_dim               = hidden // q_heads  # 16 — avoids name clash with outer var
        max_position_embeddings = 128
        rope_theta             = 10000.0
        sliding_window         = 128   # SWA enabled; won't clip ctx=1
        num_local_experts      = 0     # dense model (no MoE)
        num_experts_per_tok    = 0

    dev = wgpu_device.wgpu_device
    cache = PipelineCache(dev, SHADERS_DIR)
    model = MixtralWebGPUModel(_MistralCfg(), wgpu_device, cache)

    rng = np.random.default_rng(0)
    q_dim  = q_heads  * head_dim
    kv_dim = kv_heads * head_dim
    _inject_llama_weights(model, dev, rng, hidden, q_dim, kv_dim, inter, vocab, layers)
    _kv_pool(model, dev, layers, num_blocks, block_size, kv_heads, head_dim)
    model._batch_matmul_supported = True  # set by load_weights(); bypass guard for direct-inject test

    result = model.forward(
        np.array([1], dtype=np.uint32),
        np.array([0], dtype=np.uint32),
        _FakeMeta(),
    )

    token_id = int(result[0, 0])
    assert 0 <= token_id < vocab, f"token_id {token_id} out of range [0, {vocab})"


# ── Test 2: Mixtral MoE forward ───────────────────────────────────────────────

@pytest.mark.integration
def test_mixtral_moe_forward(wgpu_device):
    """Mixtral sparse-MoE forward pass with 4 experts, top-2 selection.

    Injects f16 weights for all 4 experts (w1/w3/w2) plus the router.
    Exercises the Phase A flush, CPU readback of expert indices, and
    Phase B per-expert dispatch + moe_accumulate.
    """
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.mixtral import MixtralWebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    hidden       = 64
    layers       = 2
    q_heads      = 4
    kv_heads     = 2
    head_dim     = 16
    inter        = 128
    vocab        = 32
    num_experts  = 4
    top_k        = 2
    num_blocks   = 8
    block_size   = 16

    class _MixtralCfg:
        num_hidden_layers      = layers
        num_attention_heads    = q_heads
        num_key_value_heads    = kv_heads
        hidden_size            = hidden
        intermediate_size      = inter
        vocab_size             = vocab
        head_dim               = hidden // q_heads  # 16 — avoids name clash with outer var
        max_position_embeddings = 128
        rope_theta             = 10000.0
        sliding_window         = None
        num_local_experts      = num_experts
        num_experts_per_tok    = top_k

    dev = wgpu_device.wgpu_device
    cache = PipelineCache(dev, SHADERS_DIR)
    model = MixtralWebGPUModel(_MixtralCfg(), wgpu_device, cache)
    assert model._is_moe

    rng = np.random.default_rng(42)
    q_dim  = q_heads  * head_dim
    kv_dim = kv_heads * head_dim

    # Shared weights (embed, norms)
    model.weights["model.embed_tokens.weight"] = _f16(dev, rng, (vocab, hidden))
    model.weights["model.norm.weight"]          = _f16(dev, rng, (hidden,))

    for i in range(layers):
        p = f"model.layers.{i}"
        model.weights[f"{p}.input_layernorm.weight"]          = _f16(dev, rng, (hidden,))
        model.weights[f"{p}.post_attention_layernorm.weight"] = _f16(dev, rng, (hidden,))
        # Attention projections
        model.weights[f"{p}.self_attn.q_proj.weight"]         = _f16(dev, rng, (q_dim,  hidden))
        model.weights[f"{p}.self_attn.k_proj.weight"]         = _f16(dev, rng, (kv_dim, hidden))
        model.weights[f"{p}.self_attn.v_proj.weight"]         = _f16(dev, rng, (kv_dim, hidden))
        model.weights[f"{p}.self_attn.o_proj.weight"]         = _f16(dev, rng, (hidden, q_dim))
        # Router
        moe = f"{p}.block_sparse_moe"
        model.weights[f"{moe}.gate.weight"] = _f16(dev, rng, (num_experts, hidden))
        # Expert weights (w1=gate, w3=up, w2=down)
        for j in range(num_experts):
            ep = f"{moe}.experts.{j}"
            model.weights[f"{ep}.w1.weight"] = _f16(dev, rng, (inter,  hidden))
            model.weights[f"{ep}.w3.weight"] = _f16(dev, rng, (inter,  hidden))
            model.weights[f"{ep}.w2.weight"] = _f16(dev, rng, (hidden, inter))

    _kv_pool(model, dev, layers, num_blocks, block_size, kv_heads, head_dim)
    model._batch_matmul_supported = False  # set by load_weights() for MoE models

    result = model.forward(
        np.array([1], dtype=np.uint32),
        np.array([0], dtype=np.uint32),
        _FakeMeta(),
    )

    token_id = int(result[0, 0])
    assert 0 <= token_id < vocab, f"token_id {token_id} out of range [0, {vocab})"


# ── Test 3: SWA window logic (no GPU execution) ───────────────────────────────

@pytest.mark.integration
def test_mistral_swa_window(wgpu_device):
    """Verify _ctx_window clips correctly at the sliding_window boundary."""
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.mixtral import MixtralWebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    class _Cfg:
        num_hidden_layers      = 2
        num_attention_heads    = 4
        num_key_value_heads    = 2
        hidden_size            = 64
        intermediate_size      = 128
        vocab_size             = 32
        head_dim               = 64 // 4  # 16
        max_position_embeddings = 128
        rope_theta             = 10000.0
        sliding_window         = 16
        num_local_experts      = 0
        num_experts_per_tok    = 0

    dev = wgpu_device.wgpu_device
    cache = PipelineCache(dev, SHADERS_DIR)
    model = MixtralWebGPUModel(_Cfg(), wgpu_device, cache)

    assert model._sw == 16
    # ctx_len > window: clips to window
    assert model._ctx_window(32)[1] == 16
    # ctx_len <= window: returns ctx_len unchanged
    assert model._ctx_window(8)[1] == 8
    # ctx_len == window: no clip
    assert model._ctx_window(16)[1] == 16

    # start_block: skip old blocks so flash_attn_decode reads the newest window.
    # block_size defaults to 16.
    bs = model.block_size
    # ctx_len within window: no offset
    assert model._ctx_window(8)[0] == 0
    assert model._ctx_window(16)[0] == 0
    # ctx_len > window: skip leading blocks
    # 32 tokens, window=16, block_size=16 → start_block = (32-16)//16 = 1
    assert model._ctx_window(32)[0] == (32 - 16) // bs
    # 48 tokens → start_block = (48-16)//16 = 2
    assert model._ctx_window(48)[0] == (48 - 16) // bs
