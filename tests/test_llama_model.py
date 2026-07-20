import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


def make_tiny_llama_config():
    cfg = MagicMock()
    cfg.hidden_size = 64
    cfg.num_hidden_layers = 2
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.intermediate_size = 128
    cfg.vocab_size = 32
    cfg.max_position_embeddings = 128
    cfg.rope_theta = 10000.0
    cfg.architectures = ["LlamaForCausalLM"]
    # Explicit head_dim so MagicMock.head_dim doesn't auto-create a MagicMock object.
    cfg.head_dim = 64 // 4  # 16
    return cfg


def test_llama_model_instantiates(wgpu_device):
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.llama import LlamaWebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    cfg = make_tiny_llama_config()
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = LlamaWebGPUModel(cfg, wgpu_device, cache)
    assert model.weights == {}
    assert model.kv_pool == []


def test_llama_layer_count(wgpu_device):
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.llama import LlamaWebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    cfg = make_tiny_llama_config()
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = LlamaWebGPUModel(cfg, wgpu_device, cache)
    assert model.num_layers == 2
    assert model.num_q_heads == 4
    assert model.num_kv_heads == 2
    assert model.head_dim == 16   # hidden_size // num_q_heads


def test_llama_rope_theta_from_top_level(wgpu_device):
    """rope_theta at the top level config is read correctly."""
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.llama import LlamaWebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    cfg = make_tiny_llama_config()
    cfg.rope_theta = 500000.0
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = LlamaWebGPUModel(cfg, wgpu_device, cache)
    assert model.rope_theta == 500000.0


def test_llama_rope_theta_fallback_from_rope_scaling(wgpu_device):
    """rope_theta stored inside rope_scaling (Phi-4, SmolLM3 pattern) is picked up."""
    from unittest.mock import MagicMock, PropertyMock
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.llama import LlamaWebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    cfg = make_tiny_llama_config()
    # Simulate SmolLM3/Phi-4: rope_theta not at top level, stored in rope_scaling.
    del cfg.rope_theta
    cfg.rope_scaling = {"rope_theta": 5000000.0, "rope_type": "default"}
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = LlamaWebGPUModel(cfg, wgpu_device, cache)
    assert model.rope_theta == 5000000.0, (
        f"Expected rope_theta=5000000.0 from rope_scaling, got {model.rope_theta}"
    )
