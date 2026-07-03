"""Tests for Qwen3.5-9B hybrid model (GDN linear attention + full attention)."""
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock


def make_qwen35_config(num_layers=4, vocab_size=32):
    """Tiny Qwen3.5 config for unit tests.

    Uses actual Qwen3.5-9B head dimensions so the GDN path works without
    patching architecture constants.
    """
    cfg = MagicMock()
    cfg.hidden_size = 4096
    cfg.num_hidden_layers = num_layers
    cfg.num_attention_heads = 16
    cfg.num_key_value_heads = 4
    cfg.intermediate_size = 12288
    cfg.vocab_size = vocab_size
    cfg.max_position_embeddings = 128
    cfg.rope_theta = 1_000_000.0
    cfg.head_dim = 256
    # 3 linear-attention layers followed by 1 full-attention layer
    cfg.layer_types = (
        ["linear_attention"] * (num_layers - 1) + ["full_attention"]
    )
    return cfg


def _make_gdn_weights_torch(hidden: int = 4096) -> dict:
    """Random GDN weight dict as torch BF16 CPU tensors (matching _extract_lin_weights output)."""
    g = torch.Generator()
    g.manual_seed(42)

    def r(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.bfloat16) * 0.01

    return {
        "in_proj_qkv": r(8192, hidden),
        "in_proj_z":   r(4096, hidden),
        "in_proj_a":   r(32, hidden),
        "in_proj_b":   r(32, hidden),
        "conv1d":      r(8192, 4),        # already squeezed: [dim, kernel]
        "A_log":       torch.full((32,), -1.0, dtype=torch.float32),
        "dt_bias":     torch.zeros(32, dtype=torch.bfloat16),
        "norm_weight": torch.ones(128, dtype=torch.bfloat16),
        "out_proj":    r(hidden, 4096),
    }


def _make_gdn_ssm_state():
    """Zero SSM state tensor."""
    from vllm_webgpu.models.qwen35 import _LIN_V_HEADS, _LIN_K_DIM, _LIN_V_DIM
    return torch.zeros(1, _LIN_V_HEADS, _LIN_K_DIM, _LIN_V_DIM, dtype=torch.float32)


def _make_gdn_conv_state():
    """Zero conv state tensor."""
    from vllm_webgpu.models.qwen35 import _LIN_CONV_DIM, _LIN_CONV_KERNEL
    return torch.zeros(1, _LIN_CONV_DIM, _LIN_CONV_KERNEL - 1, dtype=torch.bfloat16)


def test_qwen35_model_instantiates(wgpu_device):
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.qwen35 import Qwen35WebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    cfg = make_qwen35_config()
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = Qwen35WebGPUModel(cfg, wgpu_device, cache)
    assert model.weights == {}
    assert model.kv_pool == []
    assert model.num_layers == 4
    assert model.hidden_size == 4096


def test_qwen35_layer_type_detection(wgpu_device):
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.qwen35 import Qwen35WebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    cfg = make_qwen35_config(num_layers=4)
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = Qwen35WebGPUModel(cfg, wgpu_device, cache)

    # layers 0,1,2 = linear; layer 3 = full
    assert not model._is_full_attn(0)
    assert not model._is_full_attn(1)
    assert not model._is_full_attn(2)
    assert model._is_full_attn(3)


def test_qwen35_layer_type_fallback(wgpu_device):
    """When layer_types is absent, fall back to interval rule (every 4th layer)."""
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.qwen35 import Qwen35WebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    cfg = make_qwen35_config(num_layers=8)
    cfg.layer_types = None  # force fallback
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = Qwen35WebGPUModel(cfg, wgpu_device, cache)

    # With interval=4: indices 3, 7 are full attention
    for i in range(8):
        expected = (i + 1) % 4 == 0
        assert model._is_full_attn(i) == expected, f"Layer {i}: expected full={expected}"


def test_gdn_decode_shape_and_stability(wgpu_device):
    """GDN decode output has correct shape and no NaN/Inf."""
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.qwen35 import Qwen35WebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    cfg = make_qwen35_config()
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = Qwen35WebGPUModel(cfg, wgpu_device, cache)

    hidden = cfg.hidden_size
    model._lin_cpu[0] = _make_gdn_weights_torch(hidden)
    model._ssm_states[0] = _make_gdn_ssm_state()
    model._conv_states[0] = _make_gdn_conv_state()

    g = torch.Generator()
    g.manual_seed(7)
    x_bf16 = torch.randn(hidden, generator=g, dtype=torch.bfloat16) * 0.1

    out = model._gdn_decode(0, x_bf16)

    assert out.shape == (hidden,)
    assert out.dtype == torch.bfloat16
    assert not torch.any(torch.isnan(out)), "GDN output contains NaN"
    assert not torch.any(torch.isinf(out)), "GDN output contains Inf"


def test_gdn_decode_conv_state_update(wgpu_device):
    """GDN decode mutates the conv state and SSM state in-place."""
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.qwen35 import Qwen35WebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    cfg = make_qwen35_config()
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = Qwen35WebGPUModel(cfg, wgpu_device, cache)

    hidden = cfg.hidden_size
    model._lin_cpu[0] = _make_gdn_weights_torch(hidden)
    model._ssm_states[0] = _make_gdn_ssm_state()
    model._conv_states[0] = _make_gdn_conv_state()

    x_bf16 = torch.ones(hidden, dtype=torch.bfloat16) * 0.1
    model._gdn_decode(0, x_bf16)

    # conv_state is mutated in-place by causal_conv1d_update_torch
    assert model._conv_states[0].abs().sum().item() > 0, "Conv state not updated"
    # SSM state is mutated in-place by fused_sigmoid_gating_delta_rule_update_cpu
    assert model._ssm_states[0].abs().sum().item() > 0, "SSM state not updated"


def test_gdn_decode_sequential_tokens(wgpu_device):
    """Running GDN decode twice with different inputs yields different outputs."""
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.qwen35 import Qwen35WebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    cfg = make_qwen35_config()
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = Qwen35WebGPUModel(cfg, wgpu_device, cache)

    hidden = cfg.hidden_size
    model._lin_cpu[0] = _make_gdn_weights_torch(hidden)
    model._ssm_states[0] = _make_gdn_ssm_state()
    model._conv_states[0] = _make_gdn_conv_state()

    g = torch.Generator()
    g.manual_seed(99)
    x1 = torch.randn(hidden, generator=g, dtype=torch.bfloat16) * 0.1
    x2 = torch.randn(hidden, generator=g, dtype=torch.bfloat16) * 0.1

    out1 = model._gdn_decode(0, x1)
    out2 = model._gdn_decode(0, x2)

    # After state update, outputs should differ
    assert not torch.allclose(out1, out2), "GDN outputs should differ across sequential calls"


def test_mlx_detect_format():
    """detect_weight_format returns 'mlx_int4' for the MLX model directory."""
    import os
    from vllm_webgpu.quant.gguf_loader import detect_weight_format

    model_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-4bit"
        "/snapshots/8b2b98c00a6b4d291155e4890773ca8f769aee53"
    )
    if not os.path.isdir(model_dir):
        pytest.skip("Qwen3.5-9B MLX model not present on disk")

    fmt = detect_weight_format(model_dir)
    assert fmt == "mlx_int4", f"Expected mlx_int4, got {fmt!r}"


def test_mlx_dequant_correctness():
    """_dequant_mlx_int4 correctly unpacks nibbles and applies affine transform."""
    from vllm_webgpu.quant.gguf_loader import _dequant_mlx_int4

    # All-zero weights + bias of 3.0 -> every output = 3.0
    w = np.zeros((2, 2), dtype=np.uint32)    # 2 rows, 16 input cols
    s = np.ones((2, 1), dtype=np.float32)
    b = np.full((2, 1), 3.0, dtype=np.float32)
    result = _dequant_mlx_int4(w, s, b, group_size=16)
    assert result.shape == (2, 16)
    assert np.allclose(result, 3.0), f"Expected all 3.0, got {result}"

    # Nibble ordering: 0x12345678 should unpack to [8,7,6,5,4,3,2,1]
    w2 = np.array([[0x12345678]], dtype=np.uint32)
    s2 = np.ones((1, 1), dtype=np.float32)
    b2 = np.zeros((1, 1), dtype=np.float32)
    result2 = _dequant_mlx_int4(w2, s2, b2, group_size=8)
    assert list(result2[0].astype(int)) == [8, 7, 6, 5, 4, 3, 2, 1], (
        f"Wrong nibble order: {list(result2[0])}"
    )


def test_arch_map_includes_qwen35():
    """ARCH_MAP in model_runner maps Qwen3_5ForConditionalGeneration to qwen35."""
    from vllm_webgpu.v1.model_runner import ARCH_MAP
    assert "Qwen3_5ForConditionalGeneration" in ARCH_MAP
    assert ARCH_MAP["Qwen3_5ForConditionalGeneration"] == "qwen35"
