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
    # MoE fields — must be integers for comparison in __init__
    cfg.num_experts = 0
    cfg.num_experts_per_tok = 0
    cfg.moe_intermediate_size = 0
    cfg.shared_expert_intermediate_size = 12288
    # Linear attention (GDN) architecture fields — must be integers
    cfg.linear_num_key_heads = 16
    cfg.linear_num_value_heads = 16
    cfg.linear_key_head_dim = 128
    cfg.linear_value_head_dim = 128
    cfg.linear_conv_kernel_dim = 4
    cfg.full_attention_interval = 4
    cfg.attn_output_gate = False
    cfg.partial_rotary_factor = None
    cfg.mrope_interleaved = False
    # 3 linear-attention layers followed by 1 full-attention layer
    cfg.layer_types = (
        ["linear_attention"] * (num_layers - 1) + ["full_attention"]
    )
    return cfg


def _make_gdn_gpu_weights(wgpu_device, hidden: int = 4096) -> dict:
    """Upload random GDN weights as GPU f16 buffers."""
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    import wgpu as wgpu_lib

    dev = wgpu_device.wgpu_device
    rw = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST
    rng = np.random.default_rng(42)

    def r_f16(*shape) -> "WebGPUBuffer":
        arr = (rng.standard_normal(shape) * 0.01).astype(np.float16)
        return WebGPUBuffer.from_numpy(dev, np.ascontiguousarray(arr), usage=rw)

    from vllm_webgpu.models.qwen35 import _LIN_V_HEADS, _LIN_K_DIM, _LIN_V_DIM, _LIN_CONV_DIM

    weights = {
        "model.layers.0.linear_attn.in_proj_qkv.weight": r_f16(8192, hidden),
        "model.layers.0.linear_attn.in_proj_z.weight":   r_f16(4096, hidden),
        "model.layers.0.linear_attn.in_proj_a.weight":   r_f16(16, hidden),
        "model.layers.0.linear_attn.in_proj_b.weight":   r_f16(16, hidden),
        "model.layers.0.linear_attn.conv1d.weight":
            WebGPUBuffer.from_numpy(dev,
                np.ascontiguousarray(
                    (rng.standard_normal((_LIN_CONV_DIM, 4)) * 0.01).astype(np.float16)),
                usage=rw),
        "model.layers.0.linear_attn.A_log":
            WebGPUBuffer.from_numpy(dev,
                np.full(16, -1.0, dtype=np.float16), usage=rw),
        "model.layers.0.linear_attn.dt_bias":
            WebGPUBuffer.from_numpy(dev,
                np.zeros(16, dtype=np.float16), usage=rw),
        "model.layers.0.linear_attn.norm.weight":
            WebGPUBuffer.from_numpy(dev,
                np.ones(_LIN_V_DIM, dtype=np.float16), usage=rw),
        "model.layers.0.linear_attn.out_proj.weight": r_f16(hidden, 4096),
        # Weights needed by _gdn_layer_gpu for the FFN and norm dispatches
        "model.layers.0.post_attention_layernorm.weight":
            WebGPUBuffer.from_numpy(dev, np.ones(hidden, dtype=np.float16), usage=rw),
        "model.layers.0.mlp.gate_proj.weight": r_f16(hidden, hidden),
        "model.layers.0.mlp.up_proj.weight":   r_f16(hidden, hidden),
        "model.layers.0.mlp.down_proj.weight": r_f16(hidden, hidden),
        # Cross-layer norm weight (used when layer_idx < num_layers-1)
        "model.layers.1.input_layernorm.weight":
            WebGPUBuffer.from_numpy(dev, np.ones(hidden, dtype=np.float16), usage=rw),
        "dummy_scales":
            WebGPUBuffer.from_numpy(dev, np.ones(1, dtype=np.float16), usage=rw),
    }
    return weights


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


def _setup_gdn_model(wgpu_device):
    """Create a Qwen35WebGPUModel with minimal GPU weights for GDN testing."""
    import wgpu as wgpu_lib
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.models.qwen35 import (
        Qwen35WebGPUModel, _LIN_V_HEADS, _LIN_K_DIM, _LIN_V_DIM, _LIN_CONV_DIM
    )
    from vllm_webgpu.utils import SHADERS_DIR

    cfg = make_qwen35_config()
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = Qwen35WebGPUModel(cfg, wgpu_device, cache)

    hidden = cfg.hidden_size
    dev = wgpu_device.wgpu_device
    rw = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST
    rng = np.random.default_rng(42)

    def f16(arr): return WebGPUBuffer.from_numpy(dev, np.ascontiguousarray(arr.astype(np.float16)), usage=rw)

    gdn_weights = _make_gdn_gpu_weights(wgpu_device, hidden)
    model.weights.update(gdn_weights)

    # Stub out the norm/FFN weights that _gdn_layer_gpu needs
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    import wgpu as wgpu_lib
    _dev = wgpu_device.wgpu_device
    _rw = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST
    p = "model.layers.0"
    inter = cfg.intermediate_size
    for key, shape in [
        (f"{p}.input_layernorm.weight", (hidden,)),
        (f"{p}.post_attention_layernorm.weight", (hidden,)),
        (f"{p}.mlp.gate_proj.weight", (inter, hidden)),
        (f"{p}.mlp.up_proj.weight", (inter, hidden)),
        (f"{p}.mlp.down_proj.weight", (hidden, inter)),
    ]:
        arr = np.ones(shape, dtype=np.float16) * 0.01
        model.weights[key] = WebGPUBuffer.from_numpy(_dev, np.ascontiguousarray(arr), usage=_rw)

    # Allocate SSM and conv state GPU buffers directly
    model._ssm_gpu = [None] * model.num_layers
    model._conv_gpu = [None] * model.num_layers
    ssm_bytes  = _LIN_V_HEADS * _LIN_K_DIM * _LIN_V_DIM * 4
    conv_bytes = 3 * _LIN_CONV_DIM * 2  # (KERNEL-1) * DIM * f16
    model._ssm_gpu[0]  = WebGPUBuffer.empty(dev, ssm_bytes,  usage=rw)
    model._conv_gpu[0] = WebGPUBuffer.empty(dev, conv_bytes, usage=rw)

    return model, hidden


def test_gdn_decode_shape_and_stability(wgpu_device):
    """GDN GPU kernel output has correct shape and no NaN/Inf."""
    model, hidden = _setup_gdn_model(wgpu_device)
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    import wgpu as wgpu_lib

    rw = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST
    dev = wgpu_device.wgpu_device

    rng = np.random.default_rng(7)
    x_np = (rng.standard_normal(hidden) * 0.1).astype(np.float16)
    x_buf = WebGPUBuffer.from_numpy(dev, x_np, usage=rw)

    # _gdn_layer_gpu(layer_idx, normed_x, x_buf, num_tokens) → (normed_out, raw_out)
    _, raw_out = model._gdn_layer_gpu(0, x_buf, x_buf, 1)
    out = raw_out.to_numpy().view(np.float16).reshape(hidden)

    assert out.shape == (hidden,), f"Expected ({hidden},), got {out.shape}"
    assert not np.any(np.isnan(out)), "GDN output contains NaN"
    assert not np.any(np.isinf(out)), "GDN output contains Inf"


def test_gdn_decode_conv_state_update(wgpu_device):
    """GDN GPU kernel updates conv and SSM state buffers."""
    model, hidden = _setup_gdn_model(wgpu_device)
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    import wgpu as wgpu_lib

    rw = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST
    dev = wgpu_device.wgpu_device

    x_np = np.ones(hidden, dtype=np.float16) * 0.1
    x_buf = WebGPUBuffer.from_numpy(dev, x_np, usage=rw)
    model._gdn_layer_gpu(0, x_buf, x_buf, 1)

    # Conv state should be non-zero after update (causal_conv_step writes to it)
    conv_data = model._conv_gpu[0].to_numpy().view(np.float16)
    assert np.any(conv_data != 0), "Conv state not updated by GPU kernel"

    # SSM state should be non-zero after update (gdn_state_update writes to it)
    ssm_data = model._ssm_gpu[0].to_numpy().view(np.float32)
    assert np.any(ssm_data != 0), "SSM state not updated by GPU kernel"


def test_gdn_decode_sequential_tokens(wgpu_device):
    """Two GDN GPU decode steps with different inputs produce different outputs."""
    model, hidden = _setup_gdn_model(wgpu_device)
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    import wgpu as wgpu_lib

    rw = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST
    dev = wgpu_device.wgpu_device

    rng = np.random.default_rng(99)
    x1 = WebGPUBuffer.from_numpy(dev, (rng.standard_normal(hidden) * 0.1).astype(np.float16), usage=rw)
    x2 = WebGPUBuffer.from_numpy(dev, (rng.standard_normal(hidden) * 0.1).astype(np.float16), usage=rw)

    _, raw1 = model._gdn_layer_gpu(0, x1, x1, 1)
    _, raw2 = model._gdn_layer_gpu(0, x2, x2, 1)
    out1 = raw1.to_numpy().view(np.float16).copy()
    out2 = raw2.to_numpy().view(np.float16).copy()

    assert not np.allclose(out1, out2), "GDN GPU outputs should differ across sequential calls"


def test_mlx_detect_format():
    """detect_weight_format returns 'mlx_int4' for the MLX model directory."""
    import os
    from vllm_webgpu.quant.weight_loader import detect_weight_format

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
    from vllm_webgpu.quant.weight_loader import _dequant_mlx_int4

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
