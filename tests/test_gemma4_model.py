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
