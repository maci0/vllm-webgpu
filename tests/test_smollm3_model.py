"""Tests for SmolLM3WebGPUModel (Llama with NoPE layers)."""
import pytest
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
