"""Tests for SmolLM3WebGPUModel (Llama with NoPE layers)."""
import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock


def make_tiny_smollm3_config(num_layers=8, nope_layers=None):
    """Tiny SmolLM3-like config. nope_layers=None uses the default pattern."""
    cfg = MagicMock()
    cfg.hidden_size = 64
    cfg.num_hidden_layers = num_layers
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.intermediate_size = 128
    cfg.vocab_size = 32
    cfg.max_position_embeddings = 128
    cfg.rope_theta = 10000.0
    cfg.head_dim = 64 // 4   # 16
    cfg.architectures = ["SmolLM3ForCausalLM"]
    if nope_layers is not None:
        cfg.no_rope_layers = nope_layers
    else:
        # MagicMock returns a truthy MagicMock for missing attributes; override
        # to return None so the default pattern triggers.
        del cfg.no_rope_layers
        del cfg.nope_layers
    return cfg


def test_arch_map_includes_smollm3():
    """SmolLM3ForCausalLM is registered in ARCH_MAP."""
    from vllm_webgpu.v1.model_runner import ARCH_MAP

    assert "SmolLM3ForCausalLM" in ARCH_MAP, (
        f"SmolLM3ForCausalLM missing from ARCH_MAP. Keys: {sorted(ARCH_MAP)}"
    )
    assert ARCH_MAP["SmolLM3ForCausalLM"] == "smollm3"


def test_smollm3_model_instantiates_default_nope(wgpu_device):
    """SmolLM3WebGPUModel instantiates with default NoPE pattern (every 4th from 3)."""
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.smollm3 import SmolLM3WebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    cfg = make_tiny_smollm3_config(num_layers=8)
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = SmolLM3WebGPUModel(cfg, wgpu_device, cache)

    assert model.num_layers == 8
    # Default NoPE pattern: 3, 7  (every 4th starting at 3, for 8 layers)
    assert model._nope_layers == frozenset({3, 7})


def test_smollm3_explicit_nope_layers(wgpu_device):
    """SmolLM3WebGPUModel respects explicit no_rope_layers config attribute."""
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.smollm3 import SmolLM3WebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    cfg = make_tiny_smollm3_config(num_layers=8, nope_layers=[2, 5])
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = SmolLM3WebGPUModel(cfg, wgpu_device, cache)

    assert model._nope_layers == frozenset({2, 5})


def test_smollm3_inherits_llama(wgpu_device):
    """SmolLM3WebGPUModel is a LlamaWebGPUModel subclass."""
    from vllm_webgpu.models.smollm3 import SmolLM3WebGPUModel
    from vllm_webgpu.models.llama import LlamaWebGPUModel

    assert issubclass(SmolLM3WebGPUModel, LlamaWebGPUModel)


def test_smollm3_nope_layer_skips_rope(wgpu_device):
    """_attn_block routes NoPE layers to _attn_block_nope (confirmed by method dispatch)."""
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.smollm3 import SmolLM3WebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    cfg = make_tiny_smollm3_config(num_layers=4, nope_layers=[1, 3])
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = SmolLM3WebGPUModel(cfg, wgpu_device, cache)

    # NoPE layer 1 should be in _nope_layers.
    assert 1 in model._nope_layers
    assert 0 not in model._nope_layers  # standard layer

    # Verify that the _attn_block_nope method exists.
    assert callable(getattr(model, "_attn_block_nope", None)), (
        "_attn_block_nope method missing from SmolLM3WebGPUModel"
    )


def test_smollm3_empty_nope_layers(wgpu_device):
    """SmolLM3WebGPUModel with no NoPE layers still instantiates correctly."""
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.smollm3 import SmolLM3WebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    cfg = make_tiny_smollm3_config(num_layers=4, nope_layers=[])
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = SmolLM3WebGPUModel(cfg, wgpu_device, cache)

    assert model._nope_layers == frozenset()


def test_smollm3_flags_format_nope_layers(wgpu_device):
    """no_rope_layers in flags format (0=NoPE, 1=RoPE) is converted to indices.

    The actual SmolLM3-3B config supplies no_rope_layers as a per-layer
    boolean array rather than an explicit index list. For 8 layers the
    real format is [1,1,1,0,1,1,1,0], meaning layers 3 and 7 are NoPE.
    """
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.smollm3 import SmolLM3WebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    # Flags format: 8 entries, 0 at positions 3 and 7 (every 4th from layer 3).
    flags = [1, 1, 1, 0, 1, 1, 1, 0]
    cfg = make_tiny_smollm3_config(num_layers=8, nope_layers=flags)
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = SmolLM3WebGPUModel(cfg, wgpu_device, cache)

    # Should yield indices {3, 7}, not frozenset({0, 1}) (the bug).
    assert model._nope_layers == frozenset({3, 7}), (
        f"Expected NoPE indices {{3, 7}} from flags list but got {model._nope_layers}. "
        "Flags format (0=NoPE, 1=RoPE) must be converted to indices."
    )


def test_smollm3_batch_prefill_routes_to_sequential_for_nope(wgpu_device):
    """_prefill_batch_forward falls back to sequential when NoPE layers are present.

    The inherited Llama batch prefill dispatches RoPE for every layer
    unconditionally and never calls _attn_block. For NoPE layers that means
    wrong (rotated) Q/K are stored in the KV cache and used in attention.
    SmolLM3 must override _prefill_batch_forward to call _prefill_sequential_fallback
    instead, which goes through _transformer_layer -> _attn_block -> _attn_block_nope.
    """
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.smollm3 import SmolLM3WebGPUModel
    from vllm_webgpu.models.llama import LlamaWebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    # Model with NoPE layers (layers 3 and 7).
    cfg = make_tiny_smollm3_config(num_layers=8)
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = SmolLM3WebGPUModel(cfg, wgpu_device, cache)

    assert model._nope_layers, "Expected non-empty _nope_layers for this test"

    # SmolLM3 must override _prefill_batch_forward.
    smollm3_pbf = type(model)._prefill_batch_forward
    llama_pbf   = LlamaWebGPUModel._prefill_batch_forward
    assert smollm3_pbf is not llama_pbf, (
        "SmolLM3WebGPUModel must override _prefill_batch_forward. "
        "The Llama batch prefill applies RoPE to all layers unconditionally; "
        "NoPE layers would receive rotated Q/K, silently corrupting the KV cache."
    )


def test_smollm3_batch_prefill_passthrough_without_nope(wgpu_device):
    """_prefill_batch_forward delegates to Llama when no NoPE layers are configured.

    With an empty _nope_layers set, all layers use RoPE and the Llama batch
    prefill path is correct. The SmolLM3 override must not add unnecessary
    overhead by always forcing the sequential path.
    """
    from unittest.mock import MagicMock, patch
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.smollm3 import SmolLM3WebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    cfg = make_tiny_smollm3_config(num_layers=4, nope_layers=[])
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = SmolLM3WebGPUModel(cfg, wgpu_device, cache)

    assert model._nope_layers == frozenset(), "Expected empty _nope_layers"

    # When _nope_layers is empty, _prefill_batch_forward should call super(),
    # not _prefill_sequential_fallback. Verify by patching both and checking
    # which one gets invoked.
    fallback_called = []
    super_called    = []

    original_fallback = model._prefill_sequential_fallback
    original_super_pbf = SmolLM3WebGPUModel.__mro__[1]._prefill_batch_forward  # LlamaWebGPUModel

    with patch.object(model, "_prefill_sequential_fallback",
                      side_effect=lambda *a, **kw: fallback_called.append(True)):
        with patch.object(
            SmolLM3WebGPUModel.__mro__[1],
            "_prefill_batch_forward",
            side_effect=lambda *a, **kw: super_called.append(True),
        ):
            try:
                model._prefill_batch_forward(None, None, None, 1)
            except Exception:
                pass  # errors from mock args are expected; call routing is what matters

    assert not fallback_called, (
        "_prefill_sequential_fallback was called even though _nope_layers is empty. "
        "SmolLM3 should delegate to the Llama batch prefill when no NoPE layers are present."
    )
    assert super_called, (
        "LlamaWebGPUModel._prefill_batch_forward was not called for a model "
        "with empty _nope_layers. SmolLM3 must delegate to super() in that case."
    )


@pytest.mark.integration
def test_smollm3_decode_forward_rope_and_nope(wgpu_device):
    """2-layer SmolLM3 decode step exercises both RoPE (layer 0) and NoPE (layer 1).

    Layer 1 is configured as NoPE so _attn_block routes to _attn_block_nope,
    which skips RoPE and stores unrotated Q/K in the KV cache. The forward
    must complete without error and return a valid greedy token.
    """
    import wgpu as wgpu_lib
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.models.smollm3 import SmolLM3WebGPUModel
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

    # Layer 1 is NoPE; layer 0 uses standard RoPE.
    cfg   = make_tiny_smollm3_config(num_layers=layers, nope_layers=[1])
    dev   = wgpu_device.wgpu_device
    cache = PipelineCache(dev, SHADERS_DIR)
    model = SmolLM3WebGPUModel(cfg, wgpu_device, cache)

    assert model._nope_layers == frozenset({1}), "Expected NoPE at layer 1"

    rw = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST
    rng = np.random.default_rng(3)

    def f16(*shape):
        return WebGPUBuffer.from_numpy(
            dev, (rng.standard_normal(shape) * 0.01).astype(np.float16), usage=rw)

    model.weights["model.embed_tokens.weight"] = f16(vocab, H)
    model.weights["model.norm.weight"]          = f16(H)

    for i in range(layers):
        p = f"model.layers.{i}"
        model.weights[f"{p}.input_layernorm.weight"]          = f16(H)
        model.weights[f"{p}.post_attention_layernorm.weight"] = f16(H)
        model.weights[f"{p}.self_attn.q_proj.weight"]         = f16(q_dim, H)
        model.weights[f"{p}.self_attn.k_proj.weight"]         = f16(kv_dim, H)
        model.weights[f"{p}.self_attn.v_proj.weight"]         = f16(kv_dim, H)
        model.weights[f"{p}.self_attn.o_proj.weight"]         = f16(H, q_dim)
        model.weights[f"{p}.mlp.gate_proj.weight"]            = f16(inter, H)
        model.weights[f"{p}.mlp.up_proj.weight"]              = f16(inter, H)
        model.weights[f"{p}.mlp.down_proj.weight"]            = f16(H, inter)

    kv_bytes = num_blocks * block_size * kv_h * hd * 2
    for _ in range(layers):
        model.kv_pool.append((
            WebGPUBuffer.empty(dev, kv_bytes, usage=rw),
            WebGPUBuffer.empty(dev, kv_bytes, usage=rw),
        ))

    model._batch_matmul_supported = False

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
