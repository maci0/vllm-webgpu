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
