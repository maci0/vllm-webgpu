import numpy as np
import pytest
from unittest.mock import MagicMock


def make_tiny_gemma4_config():
    cfg = MagicMock()
    cfg.hidden_size = 64
    cfg.num_hidden_layers = 2
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.intermediate_size = 128
    cfg.vocab_size = 32
    cfg.head_dim = 16
    cfg.query_pre_attn_scalar = 1.0
    cfg.final_logit_softcapping = 30.0
    cfg.architectures = ["Gemma3ForCausalLM"]
    cfg.ple_layer_indices = []
    return cfg


def test_gemma4_model_instantiates(wgpu_device):
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.gemma4 import Gemma4WebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    cfg = make_tiny_gemma4_config()
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = Gemma4WebGPUModel(cfg, wgpu_device, cache)
    assert model.softcap == 30.0
    assert model.num_layers == 2
    assert model.num_q_heads == 4
    assert model.num_kv_heads == 2
    assert model.head_dim == 16


@pytest.mark.integration
def test_gemma4_gptq_forward(wgpu_device):
    """Smoke test: Gemma4 forward pass with fake GPTQ int4 weights (USE_QUANT=3).

    No quantized Gemma4 checkpoint available locally; this test injects fake i32
    weights directly into model.weights to exercise the quant detection and dispatch
    paths without needing a real GPTQ model on disk.
    """
    import wgpu as wgpu_lib
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.models.gemma4 import Gemma4WebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    # Small but valid dims: hidden=128 >= GROUP_K=128, inter=256 >= 2*GROUP_K.
    hidden = 128
    inter  = 256
    heads  = 4
    kv_heads = 2
    head_dim = hidden // heads   # 32
    q_dim  = heads * head_dim    # 128
    kv_dim = kv_heads * head_dim # 64
    vocab  = 64
    layers = 1
    group_k = 128
    block_size = 16
    num_blocks = 8

    class _FakeConfig:
        hidden_size = hidden
        num_hidden_layers = layers
        num_attention_heads = heads
        num_key_value_heads = kv_heads
        intermediate_size = inter
        vocab_size = vocab
        head_dim = hidden // heads   # 32 — avoid name clash with outer variable
        query_pre_attn_scalar = 1.0
        final_logit_softcapping = 30.0
        architectures = ["Gemma3ForCausalLM"]
        ple_layer_indices = []
        max_position_embeddings = 128
        rope_theta = 10000.0

    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = Gemma4WebGPUModel(_FakeConfig(), wgpu_device, cache)

    dev = wgpu_device.wgpu_device
    rw = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST
    rng = np.random.default_rng(42)

    def f16(shape):
        return WebGPUBuffer.from_numpy(dev,
            rng.standard_normal(shape).astype(np.float16))

    def i32_gptq(n, k):
        # Each i32 packs 8 int4 values. Random packed weights.
        return WebGPUBuffer.from_numpy(dev,
            rng.integers(0, 2**31, size=(n, k // 8), dtype=np.int32))

    def scales_f16(n, k, gk):
        # One f16 scale per (row, group). Shape: (N, K // group_k).
        return WebGPUBuffer.from_numpy(dev,
            np.ones((n, k // gk), dtype=np.float16) * 0.01)

    p = "model.layers.0"
    quant_meta = {}

    def add_gptq(base, n, k):
        key = f"{base}.weight"
        model.weights[key] = i32_gptq(n, k)
        model.weights[f"{key}.scales"] = scales_f16(n, k, group_k)
        quant_meta[base] = {"fmt": "gptq_sym", "group_size": group_k}

    # Shared f16 weights (embed, norms)
    model.weights["model.embed_tokens.weight"] = f16((vocab, hidden))
    model.weights["model.norm.weight"]          = f16((hidden,))
    model.weights[f"{p}.input_layernorm.weight"]          = f16((hidden,))
    model.weights[f"{p}.post_attention_layernorm.weight"] = f16((hidden,))
    model.weights[f"{p}.pre_feedforward_layernorm.weight"]  = f16((hidden,))
    model.weights[f"{p}.post_feedforward_layernorm.weight"] = f16((hidden,))

    # GPTQ quantized projections
    add_gptq(f"{p}.self_attn.q_proj",  q_dim,  hidden)
    add_gptq(f"{p}.self_attn.k_proj",  kv_dim, hidden)
    add_gptq(f"{p}.self_attn.v_proj",  kv_dim, hidden)
    add_gptq(f"{p}.self_attn.o_proj",  hidden, q_dim)
    add_gptq(f"{p}.mlp.gate_proj",     inter,  hidden)
    add_gptq(f"{p}.mlp.up_proj",       inter,  hidden)
    add_gptq(f"{p}.mlp.down_proj",     hidden, inter)

    model.weights["__quant_meta__"] = quant_meta

    # _postprocess_weights tiles q/k norm weights; no norm weights here, so it's a no-op.
    model._postprocess_weights()
    model._layer_scales = [1.0] * layers

    # KV cache
    for _ in range(layers):
        k_buf = WebGPUBuffer.empty(dev, num_blocks * block_size * kv_heads * head_dim * 2, usage=rw)
        v_buf = WebGPUBuffer.empty(dev, num_blocks * block_size * kv_heads * head_dim * 2, usage=rw)
        model.kv_pool.append((k_buf, v_buf))

    class _FakeMeta:
        slot_mapping = [0]
        block_tables = [np.zeros(num_blocks, dtype=np.uint32)]
        max_decode_seq_len = 1

    result = model.forward(
        np.array([1], dtype=np.uint32),
        np.array([0], dtype=np.uint32),
        _FakeMeta(),
    )

    token_id = int(result[0, 0])
    assert 0 <= token_id < vocab, f"token_id {token_id} out of range [0, {vocab})"
