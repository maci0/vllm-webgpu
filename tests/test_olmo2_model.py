"""Tests for Olmo2WebGPUModel (post-norm architecture)."""
import pytest
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
