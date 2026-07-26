"""Tests for Olmo2WebGPUModel (post-norm architecture)."""
import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock


def make_tiny_olmo2_config():
    """Tiny OLMo-2 config for unit tests."""
    cfg = MagicMock()
    cfg.hidden_size = 64
    cfg.num_hidden_layers = 2
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.intermediate_size = 128
    cfg.vocab_size = 32
    cfg.max_position_embeddings = 128
    cfg.rope_theta = 10000.0
    cfg.head_dim = 64 // 4   # 16
    cfg.architectures = ["Olmo2ForCausalLM"]
    return cfg


def test_arch_map_includes_olmo2():
    """Olmo2ForCausalLM is registered in ARCH_MAP."""
    from vllm_webgpu.v1.model_runner import ARCH_MAP

    assert "Olmo2ForCausalLM" in ARCH_MAP, (
        f"Olmo2ForCausalLM missing from ARCH_MAP. Keys: {sorted(ARCH_MAP)}"
    )
    assert ARCH_MAP["Olmo2ForCausalLM"] == "olmo2"


def test_olmo2_model_instantiates(wgpu_device):
    """Olmo2WebGPUModel instantiates with correct attributes."""
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.olmo2 import Olmo2WebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    cfg = make_tiny_olmo2_config()
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = Olmo2WebGPUModel(cfg, wgpu_device, cache)

    assert model.num_layers == 2
    assert model.num_q_heads == 4
    assert model.num_kv_heads == 2
    assert model.head_dim == 16
    assert model.hidden_size == 64
    assert model.intermediate_size == 128
    assert model.weights == {}
    assert model.kv_pool == []


def test_olmo2_inherits_llama(wgpu_device):
    """Olmo2WebGPUModel is a LlamaWebGPUModel subclass."""
    from vllm_webgpu.models.olmo2 import Olmo2WebGPUModel
    from vllm_webgpu.models.llama import LlamaWebGPUModel

    assert issubclass(Olmo2WebGPUModel, LlamaWebGPUModel)


def test_olmo2_norm_fusion_disabled(wgpu_device):
    """OLMo-2 disables norm fusion (_norm_fusion=False) because it has no input_layernorm."""
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.olmo2 import Olmo2WebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    cfg = make_tiny_olmo2_config()
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = Olmo2WebGPUModel(cfg, wgpu_device, cache)

    assert model._norm_fusion is False, (
        "Olmo2WebGPUModel._norm_fusion must be False: OLMo-2 has no input_layernorm, "
        "so the Llama final-norm fusion cannot be applied."
    )


def test_olmo2_transformer_layer_method_exists(wgpu_device):
    """Olmo2WebGPUModel overrides _transformer_layer for the post-norm pattern."""
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.olmo2 import Olmo2WebGPUModel
    from vllm_webgpu.models.llama import LlamaWebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    cfg = make_tiny_olmo2_config()
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = Olmo2WebGPUModel(cfg, wgpu_device, cache)

    # Olmo2 must provide its own _transformer_layer override.
    olmo2_layer = type(model)._transformer_layer
    llama_layer  = LlamaWebGPUModel._transformer_layer
    assert olmo2_layer is not llama_layer, (
        "Olmo2WebGPUModel._transformer_layer must be overridden "
        "(not inherited verbatim from LlamaWebGPUModel)"
    )


def test_olmo2_run_decode_dispatches_overridden(wgpu_device):
    """Olmo2WebGPUModel overrides _run_decode_dispatches to skip initial pre-norm."""
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.olmo2 import Olmo2WebGPUModel
    from vllm_webgpu.models.llama import LlamaWebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    cfg = make_tiny_olmo2_config()
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = Olmo2WebGPUModel(cfg, wgpu_device, cache)

    olmo2_rdd  = type(model)._run_decode_dispatches
    llama_rdd  = LlamaWebGPUModel._run_decode_dispatches
    assert olmo2_rdd is not llama_rdd, (
        "Olmo2WebGPUModel._run_decode_dispatches must be overridden "
        "(OLMo-2 has no input_layernorm, so the initial rms_norm dispatch must be skipped)"
    )


def test_arch_map_includes_olmo3():
    """Olmo3ForCausalLM is registered in ARCH_MAP (vLLM treats it as olmo2)."""
    from vllm_webgpu.v1.model_runner import ARCH_MAP

    assert "Olmo3ForCausalLM" in ARCH_MAP, (
        "Olmo3ForCausalLM missing from ARCH_MAP. "
        "vLLM maps Olmo3ForCausalLM to the olmo2 model; the WebGPU plugin must too."
    )
    assert ARCH_MAP["Olmo3ForCausalLM"] == "olmo2"


@pytest.mark.integration
def test_olmo2_decode_forward(wgpu_device):
    """2-layer OLMo-2 decode step returns a valid token id.

    OLMo-2 uses a post-norm architecture: no input_layernorm, post-attention
    and post-feedforward norms applied to branch outputs. This test verifies
    that the overridden _run_decode_dispatches and _transformer_layer run
    end-to-end without error and produce a valid greedy token.
    """
    import wgpu as wgpu_lib
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.models.olmo2 import Olmo2WebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    H          = 64
    layers     = 2
    q_h        = 4
    kv_h       = 2
    hd         = 16
    inter      = 128
    vocab      = 32
    num_blocks = 8
    block_size = 16
    q_dim      = q_h * hd
    kv_dim     = kv_h * hd

    cfg   = make_tiny_olmo2_config()
    dev   = wgpu_device.wgpu_device
    cache = PipelineCache(dev, SHADERS_DIR)
    model = Olmo2WebGPUModel(cfg, wgpu_device, cache)

    rw = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST
    rng = np.random.default_rng(1)

    def f16(*shape):
        return WebGPUBuffer.from_numpy(
            dev, (rng.standard_normal(shape) * 0.01).astype(np.float16), usage=rw)

    model.weights["model.embed_tokens.weight"] = f16(vocab, H)
    model.weights["model.norm.weight"]          = f16(H)

    for i in range(layers):
        p = f"model.layers.{i}"
        # OLMo-2: no input_layernorm; post-norms applied to branch outputs.
        model.weights[f"{p}.post_attention_layernorm.weight"]  = f16(H)
        model.weights[f"{p}.post_feedforward_layernorm.weight"] = f16(H)
        # Per-tensor q/k norms (shape = full q_dim / kv_dim after tiling).
        model.weights[f"{p}.self_attn.q_norm.weight"] = f16(q_dim)
        model.weights[f"{p}.self_attn.k_norm.weight"] = f16(kv_dim)
        model.weights[f"{p}.self_attn.q_proj.weight"] = f16(q_dim, H)
        model.weights[f"{p}.self_attn.k_proj.weight"] = f16(kv_dim, H)
        model.weights[f"{p}.self_attn.v_proj.weight"] = f16(kv_dim, H)
        model.weights[f"{p}.self_attn.o_proj.weight"] = f16(H, q_dim)
        model.weights[f"{p}.mlp.gate_proj.weight"]    = f16(inter, H)
        model.weights[f"{p}.mlp.up_proj.weight"]      = f16(inter, H)
        model.weights[f"{p}.mlp.down_proj.weight"]    = f16(H, inter)

    kv_bytes = num_blocks * block_size * kv_h * hd * 2
    for _ in range(layers):
        model.kv_pool.append((
            WebGPUBuffer.empty(dev, kv_bytes, usage=rw),
            WebGPUBuffer.empty(dev, kv_bytes, usage=rw),
        ))

    model._batch_matmul_supported = False  # sequential fallback; avoids input_layernorm scan

    meta = SimpleNamespace(
        slot_mapping=[0],
        block_tables=[np.zeros(num_blocks, dtype=np.uint32)],
        max_decode_seq_len=1,
    )

    result = model.forward(
        np.array([0], dtype=np.uint32),
        np.array([0], dtype=np.uint32),
        meta,
    )
    tok = int(result[0, 0])
    assert 0 <= tok < vocab, f"token id {tok} out of [0, {vocab})"


@pytest.mark.integration
def test_olmo2_prefill_logits_match_numpy(wgpu_device):
    """2-token OLMo-2 prefill logits match a NumPy post-norm + full-vector q/k-norm ref."""
    import wgpu as wgpu_lib
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.models.olmo2 import Olmo2WebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR
    from ref_transformer import olmo2_layer, prefill_logits

    H, layers, q_h, kv_h, hd = 64, 2, 4, 2, 16
    inter, vocab, num_blocks, block_size = 128, 32, 8, 16
    q_dim, kv_dim = q_h * hd, kv_h * hd

    cfg = make_tiny_olmo2_config()
    dev = wgpu_device.wgpu_device
    cache = PipelineCache(dev, SHADERS_DIR)
    model = Olmo2WebGPUModel(cfg, wgpu_device, cache)

    rw = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST
    rng = np.random.default_rng(11)

    def both(*shape):
        arr = (rng.standard_normal(shape) * 0.01).astype(np.float16)
        return arr, WebGPUBuffer.from_numpy(dev, arr, usage=rw)

    embed_np, embed_buf = both(vocab, H)
    norm_np, norm_buf = both(H)
    model.weights["model.embed_tokens.weight"] = embed_buf
    model.weights["model.norm.weight"] = norm_buf

    layers_w = []
    for i in range(layers):
        p = f"model.layers.{i}"
        w = {}
        for key, shape, np_key in [
            ("post_attention_layernorm.weight", (H,), "post_attention_layernorm"),
            ("post_feedforward_layernorm.weight", (H,), "post_feedforward_layernorm"),
            ("self_attn.q_norm.weight", (q_dim,), "q_norm"),
            ("self_attn.k_norm.weight", (kv_dim,), "k_norm"),
            ("self_attn.q_proj.weight", (q_dim, H), "q_proj"),
            ("self_attn.k_proj.weight", (kv_dim, H), "k_proj"),
            ("self_attn.v_proj.weight", (kv_dim, H), "v_proj"),
            ("self_attn.o_proj.weight", (H, q_dim), "o_proj"),
            ("mlp.gate_proj.weight", (inter, H), "gate_proj"),
            ("mlp.up_proj.weight", (inter, H), "up_proj"),
            ("mlp.down_proj.weight", (H, inter), "down_proj"),
        ]:
            arr, buf = both(*shape)
            model.weights[f"{p}.{key}"] = buf
            w[np_key] = arr
        layers_w.append(w)

    kv_bytes = num_blocks * block_size * kv_h * hd * 2
    for _ in range(layers):
        model.kv_pool.append((
            WebGPUBuffer.empty(dev, kv_bytes, usage=rw),
            WebGPUBuffer.empty(dev, kv_bytes, usage=rw),
        ))

    model._batch_matmul_supported = False
    model._greedy_decode = False

    tokens = np.array([3, 7], dtype=np.uint32)
    T = len(tokens)
    meta = SimpleNamespace(
        slot_mapping=list(range(T)),
        block_tables=[np.arange(num_blocks, dtype=np.uint32)],
        max_decode_seq_len=T,
    )
    gpu = model.forward(tokens, np.arange(T, dtype=np.uint32), meta)[0]

    ref = prefill_logits(
        tokens, embed_np, layers_w, norm_np, embed_np,
        n_q=q_h, n_kv=kv_h, head_dim=hd, layer_fn=olmo2_layer,
    )
    np.testing.assert_allclose(
        gpu, ref, rtol=5e-2, atol=5e-2,
        err_msg="OLMo-2 WebGPU logits diverge from NumPy full-vector-norm reference",
    )
    assert int(gpu.argmax()) == int(ref.argmax())
