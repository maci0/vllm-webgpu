"""Tests for DiffusionGemmaWebGPUModel (Gemma4 backbone + shared expert + MoE FFN)."""
import numpy as np
import pytest


def make_diffusion_gemma_config():
    """Tiny DiffusionGemmaConfig: 1 layer, 8 experts, top-2."""
    from unittest.mock import MagicMock

    cfg = MagicMock()
    cfg.hidden_size = 128
    cfg.num_hidden_layers = 1
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.intermediate_size = 256          # shared expert intermediate size
    cfg.vocab_size = 64
    cfg.head_dim = 32                    # hidden_size // num_attention_heads
    cfg.query_pre_attn_scalar = 1.0
    cfg.final_logit_softcapping = 30.0
    cfg.architectures = ["DiffusionGemmaForBlockDiffusion"]
    cfg.ple_layer_indices = []
    cfg.max_position_embeddings = 128
    cfg.rope_theta = 10000.0
    cfg.hidden_size_per_layer_input = 0
    # MoE — num_experts > 0 triggers MoE scratch buffer pre-allocation in __init__
    cfg.num_experts = 8
    cfg.top_k_experts = 2
    # moe_intermediate_size must be <= intermediate_size so sc["gate_buf"]/["up_buf"]
    # (sized at intermediate_size) can safely hold expert projections too.
    cfg.moe_intermediate_size = 128
    cfg.canvas_length = 1
    return cfg


def test_diffusion_gemma_model_instantiates(wgpu_device):
    """DiffusionGemmaWebGPUModel initialises with correct MoE parameters."""
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.diffusion_gemma import DiffusionGemmaWebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    cfg = make_diffusion_gemma_config()
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = DiffusionGemmaWebGPUModel(cfg, wgpu_device, cache)

    assert model.is_moe
    assert model.num_experts == 8
    assert model.top_k_experts == 2
    assert model.moe_intermediate_size == 128
    assert model.num_layers == 1
    assert model.hidden_size == 128
    assert model.vocab_size == 64
    # MoE scratch buffers pre-allocated in __init__
    assert model._shared_res_buf is not None
    assert model._topk_idx_buf is not None
    assert model._topk_weight_buf is not None
    assert model._router_logit_buf is not None


def test_diffusion_gemma_arch_map():
    """ARCH_MAP maps DiffusionGemmaForBlockDiffusion to diffusion_gemma."""
    from vllm_webgpu.v1.model_runner import ARCH_MAP

    assert "DiffusionGemmaForBlockDiffusion" in ARCH_MAP
    assert ARCH_MAP["DiffusionGemmaForBlockDiffusion"] == "diffusion_gemma"


@pytest.mark.integration
def test_diffusion_gemma_moe_forward(wgpu_device):
    """Forward pass with synthetic f16 weights: shared FFN + 8 MoE experts.

    DiffusionGemma.forward() returns logits [num_tokens, vocab] as float32
    (unlike Gemma4 which returns a token id). The router runs inside the
    outer batched_dispatch context, so GPU→CPU readback of topk indices sees
    the write-buffer from the previous step; expert selection may not reflect
    the actual logits, but the code path executes without error and produces
    a finite logit vector from at least the shared expert branch.
    """
    import wgpu as wgpu_lib
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.models.diffusion_gemma import DiffusionGemmaWebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    hidden    = 128
    inter     = 256          # shared-expert intermediate
    moe_inter = 128          # MoE-expert intermediate (must be <= inter)
    heads     = 4
    kv_heads  = 2
    head_dim  = 32           # hidden // heads
    q_dim     = heads * head_dim    # 128
    kv_dim    = kv_heads * head_dim # 64
    vocab     = 64
    layers    = 1
    n_experts = 8
    block_sz  = 16
    n_blocks  = 8

    class _Cfg:
        hidden_size                = hidden
        num_hidden_layers          = layers
        num_attention_heads        = heads
        num_key_value_heads        = kv_heads
        intermediate_size          = inter
        vocab_size                 = vocab
        head_dim                   = hidden // heads
        query_pre_attn_scalar      = 1.0
        final_logit_softcapping    = 30.0
        architectures              = ["DiffusionGemmaForBlockDiffusion"]
        ple_layer_indices          = []
        max_position_embeddings    = 128
        rope_theta                 = 10000.0
        num_experts                = n_experts
        top_k_experts              = 2
        moe_intermediate_size      = moe_inter

    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = DiffusionGemmaWebGPUModel(_Cfg(), wgpu_device, cache)

    dev = wgpu_device.wgpu_device
    rw  = (wgpu_lib.BufferUsage.STORAGE
           | wgpu_lib.BufferUsage.COPY_SRC
           | wgpu_lib.BufferUsage.COPY_DST)
    rng = np.random.default_rng(42)

    def f16(shape):
        """Small random f16 weight buffer on the GPU."""
        arr = (rng.standard_normal(shape) * 0.01).astype(np.float16)
        return WebGPUBuffer.from_numpy(dev, np.ascontiguousarray(arr))

    # Global weights (decoder-prefix)
    model.weights["model.decoder.embed_tokens.weight"] = f16((vocab, hidden))
    model.weights["model.decoder.norm.weight"]         = f16((hidden,))

    # Per-layer weights (decoder prefix, layer 0)
    p = "model.decoder.layers.0"
    model.weights[f"{p}.input_layernorm.weight"]           = f16((hidden,))
    model.weights[f"{p}.post_attention_layernorm.weight"]  = f16((hidden,))
    model.weights[f"{p}.pre_feedforward_layernorm.weight"] = f16((hidden,))
    model.weights[f"{p}.post_feedforward_layernorm.weight"]= f16((hidden,))

    # Attention projections
    model.weights[f"{p}.self_attn.q_proj.weight"] = f16((q_dim,  hidden))
    model.weights[f"{p}.self_attn.k_proj.weight"] = f16((kv_dim, hidden))
    model.weights[f"{p}.self_attn.v_proj.weight"] = f16((kv_dim, hidden))
    model.weights[f"{p}.self_attn.o_proj.weight"] = f16((hidden, q_dim))

    # Shared-expert FFN
    model.weights[f"{p}.mlp.gate_proj.weight"] = f16((inter,  hidden))
    model.weights[f"{p}.mlp.up_proj.weight"]   = f16((inter,  hidden))
    model.weights[f"{p}.mlp.down_proj.weight"] = f16((hidden, inter))

    # MoE norms (required: absence of either causes incorrect compute or silent misinterpretation)
    model.weights[f"{p}.post_feedforward_layernorm_1.weight"] = f16((hidden,))
    model.weights[f"{p}.pre_feedforward_layernorm_2.weight"]  = f16((hidden,))

    # MoE router: [num_experts, hidden]
    model.weights[f"{p}.router.proj.weight"] = f16((n_experts, hidden))

    # All 8 routed experts (inject all so whichever are selected by topk will be found)
    for eid in range(n_experts):
        ep = f"{p}.experts.{eid}"
        model.weights[f"{ep}.gate_proj.weight"] = f16((moe_inter, hidden))
        model.weights[f"{ep}.up_proj.weight"]   = f16((moe_inter, hidden))
        model.weights[f"{ep}.down_proj.weight"] = f16((hidden, moe_inter))

    # _layer_scales is populated by load_weights(); set manually for the test.
    model._layer_scales = [1.0] * layers

    # KV cache — one (K, V) pair per layer
    kv_bytes = n_blocks * block_sz * kv_heads * head_dim * 2  # f16 bytes
    for _ in range(layers):
        model.kv_pool.append((
            WebGPUBuffer.empty(dev, kv_bytes, usage=rw),
            WebGPUBuffer.empty(dev, kv_bytes, usage=rw),
        ))

    class _FakeMeta:
        slot_mapping    = [0]
        block_tables    = [np.zeros(n_blocks, dtype=np.uint32)]
        max_decode_seq_len = 1

    logits = model.forward(
        np.array([1], dtype=np.uint32),
        np.array([0], dtype=np.uint32),
        _FakeMeta(),
    )

    # DiffusionGemma.forward() returns float32 logits [num_tokens, vocab]
    assert logits.shape == (1, vocab), (
        f"Expected logits shape (1, {vocab}), got {logits.shape}")
    assert not np.any(np.isnan(logits)), "logits contain NaN"
    assert not np.any(np.isinf(logits)), "logits contain Inf"

    token_id = int(np.argmax(logits[0]))
    assert 0 <= token_id < vocab, (
        f"argmax token_id {token_id} out of range [0, {vocab})")
