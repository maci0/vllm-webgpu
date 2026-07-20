"""Tests for PhiWebGPUModel (Phi-4 / Phi-4-mini)."""
import pytest
from unittest.mock import MagicMock


def make_tiny_phi_config():
    """Tiny Phi-4-like config: fused QKV and gate_up projections."""
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
    cfg.architectures = ["Phi3ForCausalLM"]
    return cfg


def test_arch_map_includes_phi():
    """Phi3ForCausalLM is registered in ARCH_MAP."""
    from vllm_webgpu.v1.model_runner import ARCH_MAP

    assert "Phi3ForCausalLM" in ARCH_MAP, (
        f"Phi3ForCausalLM missing from ARCH_MAP. Keys: {sorted(ARCH_MAP)}"
    )
    assert ARCH_MAP["Phi3ForCausalLM"] == "phi"


def test_phi_model_instantiates(wgpu_device):
    """PhiWebGPUModel instantiates with correct attributes."""
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.phi import PhiWebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    cfg = make_tiny_phi_config()
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = PhiWebGPUModel(cfg, wgpu_device, cache)

    assert model.num_layers == 2
    assert model.num_q_heads == 4
    assert model.num_kv_heads == 2
    assert model.head_dim == 16
    assert model.hidden_size == 64
    assert model.intermediate_size == 128
    assert model.weights == {}
    assert model.kv_pool == []


def test_phi_inherits_llama(wgpu_device):
    """PhiWebGPUModel is a LlamaWebGPUModel subclass."""
    from vllm_webgpu.models.phi import PhiWebGPUModel
    from vllm_webgpu.models.llama import LlamaWebGPUModel

    assert issubclass(PhiWebGPUModel, LlamaWebGPUModel)


def test_phi_split_row_major(wgpu_device):
    """_split_row_major splits a fused f16 weight buffer into row slices.

    This tests the GPU copy logic without loading real safetensors.
    """
    import numpy as np
    import wgpu as wgpu_lib
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.models.phi import PhiWebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    cfg = make_tiny_phi_config()
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = PhiWebGPUModel(cfg, wgpu_device, cache)

    dev = wgpu_device.wgpu_device
    H  = 64
    Q  = 4 * 16  # 64
    KV = 2 * 16  # 32
    # Fused qkv: shape [Q + 2*KV, H] = [128, 64] f16
    fused_arr = np.arange((Q + 2*KV) * H, dtype=np.float16).reshape(Q + 2*KV, H)
    usage = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST
    fused_buf = WebGPUBuffer.from_numpy(dev, fused_arr, usage=usage)
    fused_buf.dtype = "f16"
    fused_buf.shape = (Q + 2*KV, H)

    model.weights["model.layers.0.self_attn.qkv_proj.weight"] = fused_buf

    model._split_row_major(
        dev, "model.layers.0.self_attn.qkv_proj.weight",
        [("model.layers.0.self_attn.q_proj.weight", Q),
         ("model.layers.0.self_attn.k_proj.weight", KV),
         ("model.layers.0.self_attn.v_proj.weight", KV)],
        K=H,
    )

    # Original fused key should be removed.
    assert "model.layers.0.self_attn.qkv_proj.weight" not in model.weights
    # Split keys should be present.
    assert "model.layers.0.self_attn.q_proj.weight" in model.weights
    assert "model.layers.0.self_attn.k_proj.weight" in model.weights
    assert "model.layers.0.self_attn.v_proj.weight" in model.weights

    q_buf = model.weights["model.layers.0.self_attn.q_proj.weight"]
    k_buf = model.weights["model.layers.0.self_attn.k_proj.weight"]
    v_buf = model.weights["model.layers.0.self_attn.v_proj.weight"]

    # Byte sizes should match the row counts.
    assert q_buf.nbytes == Q  * H * 2
    assert k_buf.nbytes == KV * H * 2
    assert v_buf.nbytes == KV * H * 2
