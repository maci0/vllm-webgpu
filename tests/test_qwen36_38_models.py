"""Qwen3.8-27B and Qwen3.6-35B-A3B compatibility checks."""

import numpy as np
import pytest
from pathlib import Path
from types import SimpleNamespace
import json


_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "architecture,expected_family",
    [
        ("Qwen3_5ForConditionalGeneration", "qwen35"),
        ("Qwen3_5MoeForConditionalGeneration", "qwen35"),
    ],
)
def test_qwen_architecture_routes_to_qwen35(architecture, expected_family):
    from vllm_webgpu.v1.model_runner import ARCH_MAP

    assert ARCH_MAP[architecture] == expected_family


@pytest.mark.parametrize(
    "fixture,expected_layers,expected_v_heads,expected_experts,expected_top_k",
    [
        ("qwen38_dense_config.json", 64, 48, 0, 0),
        ("qwen36_moe_config.json", 40, 32, 256, 8),
    ],
)
def test_qwen_official_config_constructs(
    tmp_path, wgpu_device, fixture, expected_layers, expected_v_heads,
    expected_experts, expected_top_k,
):
    """Instantiate from the published Qwen checkpoint configs without weights."""
    from vllm.transformers_utils.config import get_config

    from vllm_webgpu.models.qwen35 import Qwen35WebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR
    from vllm_webgpu.webgpu.pipeline import PipelineCache

    (tmp_path / "config.json").write_bytes((_FIXTURES / fixture).read_bytes())
    config = get_config(str(tmp_path), trust_remote_code=False)
    model = Qwen35WebGPUModel(
        config, wgpu_device,
        PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR),
    )
    assert model.num_layers == expected_layers
    assert model._lin_v_heads == expected_v_heads
    assert model._num_experts == expected_experts
    assert model._top_k == expected_top_k
    assert model.rope_theta == 10_000_000
    if fixture == "qwen38_dense_config.json":
        assert model._PREFILL_TOKEN_CHUNK == 1


def test_qwen38_dense_weight_prefixes_and_gate_split(tmp_path, wgpu_device):
    """The multimodal checkpoint uses raw keys that differ from model keys."""
    from safetensors.numpy import save_file

    from vllm_webgpu.models.qwen35 import Qwen35WebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR
    from vllm_webgpu.webgpu.pipeline import PipelineCache

    cfg = SimpleNamespace(
        hidden_size=128, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=256, vocab_size=64,
        max_position_embeddings=128, head_dim=32, layer_types=[
            "linear_attention", "full_attention",
        ],
        linear_num_key_heads=1, linear_num_value_heads=1,
        linear_key_head_dim=4, linear_value_head_dim=4,
        linear_conv_kernel_dim=2, num_experts=0,
        num_experts_per_tok=0, num_local_experts=0,
        attn_output_gate=True, partial_rotary_factor=1.0,
        rope_parameters={"rope_theta": 10_000_000, "mrope_interleaved": True},
        output_gate_type="swish",
    )
    model = Qwen35WebGPUModel(
        cfg, wgpu_device,
        PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR),
    )
    tensors = {
        "model.language_model.layers.0.linear_attn.A_log": np.array([1.5], dtype=np.float32),
        "model.language_model.layers.0.linear_attn.dt_bias": np.array([0.25], dtype=np.float32),
        "model.language_model.layers.1.self_attn.q_proj.weight": np.arange(
            2 * 128 * 128, dtype=np.float16,
        ).reshape(256, 128),
        "model.visual.dummy.weight": np.ones((2, 2), dtype=np.float16),
        "mtp.fc.weight": np.ones((2, 2), dtype=np.float16),
    }
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, str(tmp_path / shard))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(t.nbytes for t in tensors.values())},
        "weight_map": {k: shard for k in tensors},
    }))
    model.load_weights(str(tmp_path))

    q_key = "model.layers.1.self_attn.q_proj.weight"
    gate_key = "model.layers.1.self_attn.q_gate_proj.weight"
    assert model.weights[q_key].shape == (128, 128)
    assert model.weights[gate_key].shape == (128, 128)
    assert model.weights["model.layers.0.linear_attn.A_log"].dtype == "f32"
    assert model.weights["model.layers.0.linear_attn.dt_bias"].dtype == "f32"
    assert not any(k.startswith(("model.visual.", "mtp.")) for k in model.weights)


def test_qwen38_packed_int4_gate_split_skips_vision_shard(tmp_path, wgpu_device):
    """Load an HF cache symlink, skipping the missing vision shard."""
    from safetensors.numpy import save_file

    from vllm_webgpu.models.qwen35 import Qwen35WebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR
    from vllm_webgpu.webgpu.pipeline import PipelineCache

    cfg = SimpleNamespace(
        hidden_size=64, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, intermediate_size=128, vocab_size=64,
        max_position_embeddings=128, head_dim=32, layer_types=["full_attention"],
        linear_num_key_heads=1, linear_num_value_heads=1,
        linear_key_head_dim=4, linear_value_head_dim=4,
        linear_conv_kernel_dim=2, num_experts=0, num_experts_per_tok=0,
        num_local_experts=0, attn_output_gate=True,
        partial_rotary_factor=1.0, rope_parameters={"rope_theta": 10_000_000},
        output_gate_type="swish",
    )
    model = Qwen35WebGPUModel(
        cfg, wgpu_device,
        PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR),
    )
    base = "model.language_model.layers.0.self_attn.q_proj"
    packed = np.repeat(np.arange(128, dtype=np.int32)[:, None], 8, axis=1)
    scales = np.repeat(np.arange(128, dtype=np.float16)[:, None], 2, axis=1)
    tensors = {
        f"{base}.weight_packed": packed,
        f"{base}.weight_scale": scales,
    }
    repo = tmp_path / "models--Qwen--fixture"
    snapshot = repo / "snapshots" / "revision"
    blobs = repo / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    save_file(tensors, str(blobs / "weight-blob"))
    (snapshot / "model.safetensors").symlink_to(blobs / "weight-blob")
    (snapshot / "config.json").write_text(json.dumps({
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {"group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4, "type": "int", "strategy": "group",
                    "group_size": 32, "symmetric": True,
                },
            }},
        },
    }))
    weight_map = {k: "model.safetensors" for k in tensors}
    weight_map["model.visual.dummy.weight"] = "model-visual.safetensors"
    (snapshot / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": weight_map,
    }))

    model.load_weights(str(snapshot))

    q_key = "model.layers.0.self_attn.q_proj.weight"
    gate_key = "model.layers.0.self_attn.q_gate_proj.weight"
    expected_q = np.concatenate((packed[:32], packed[64:96]))
    expected_gate = np.concatenate((packed[32:64], packed[96:128]))
    np.testing.assert_array_equal(
        model.weights[q_key].to_numpy().view(np.int32).reshape(64, 8), expected_q)
    np.testing.assert_array_equal(
        model.weights[gate_key].to_numpy().view(np.int32).reshape(64, 8), expected_gate)
    assert model.weights[f"{q_key}.scales"].shape == (2, 64)
    assert model.weights[f"{gate_key}.scales"].shape == (2, 64)
    assert model._uq_for_key(q_key) == model._uq_for_key(gate_key) == 3


def test_qwen36_moe_fused_expert_weights(tmp_path, wgpu_device):
    """The official MoE index stores expert projections as fused tensors."""
    from safetensors.numpy import save_file

    from vllm_webgpu.models.qwen35 import Qwen35WebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR
    from vllm_webgpu.webgpu.pipeline import PipelineCache

    cfg = SimpleNamespace(
        hidden_size=64, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, vocab_size=64, max_position_embeddings=128,
        head_dim=32, layer_types=["full_attention"],
        linear_num_key_heads=1, linear_num_value_heads=1,
        linear_key_head_dim=4, linear_value_head_dim=4,
        linear_conv_kernel_dim=2, num_experts=2,
        num_experts_per_tok=2, num_local_experts=0,
        moe_intermediate_size=16, shared_expert_intermediate_size=16,
        attn_output_gate=False, partial_rotary_factor=1.0,
        rope_parameters={"rope_theta": 10_000_000},
        output_gate_type="swish",
    )
    model = Qwen35WebGPUModel(
        cfg, wgpu_device,
        PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR),
    )
    assert model.intermediate_size == 16
    tensors = {
        "model.language_model.layers.0.mlp.experts.gate_up_proj": np.ones((2, 32, 64), dtype=np.float16),
        "model.language_model.layers.0.mlp.experts.down_proj": np.ones((2, 64, 16), dtype=np.float16),
        "model.language_model.layers.0.mlp.gate.weight": np.ones((2, 64), dtype=np.float16),
        "mtp.fc.weight": np.ones((2, 2), dtype=np.float16),
    }
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, str(tmp_path / shard))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(t.nbytes for t in tensors.values())},
        "weight_map": {k: shard for k in tensors},
    }))
    model.load_weights(str(tmp_path))
    for expert in range(2):
        prefix = f"model.layers.0.mlp.experts.{expert}"
        assert model.weights[f"{prefix}.gate_proj.weight"].shape == (16, 64)
        assert model.weights[f"{prefix}.up_proj.weight"].shape == (16, 64)
        assert model.weights[f"{prefix}.down_proj.weight"].shape == (64, 16)
    assert "model.layers.0.mlp.experts.gate_up_proj" not in model.weights
    assert "model.language_model.layers.0.mlp.experts.gate_up_proj" not in model.weights
    assert "model.language_model.layers.0.mlp.experts.down_proj" not in model.weights
    assert "mtp.fc.weight" not in model.weights
