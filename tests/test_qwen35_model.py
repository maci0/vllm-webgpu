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
    cfg.num_local_experts = 0   # read by MixtralWebGPUModel.__init__ (now in MRO)
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
    cfg.output_gate_type = "silu"
    cfg.partial_rotary_factor = 1.0
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

    from vllm_webgpu.models.qwen35 import _LIN_V_HEADS, _LIN_K_DIM, _LIN_V_DIM

    weights = {
        "model.layers.0.linear_attn.in_proj_qkv.weight": r_f16(8192, hidden),
        "model.layers.0.linear_attn.in_proj_z.weight":   r_f16(4096, hidden),
        "model.layers.0.linear_attn.in_proj_a.weight":   r_f16(16, hidden),
        "model.layers.0.linear_attn.in_proj_b.weight":   r_f16(16, hidden),
        "model.layers.0.linear_attn.conv1d.weight":
            WebGPUBuffer.from_numpy(dev,
                np.ascontiguousarray(
                    (rng.standard_normal((8192, 4)) * 0.01).astype(np.float16)),
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
    """When layer_types is absent, model init raises AssertionError (misconfigured checkpoint)."""
    import pytest
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.qwen35 import Qwen35WebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    cfg = make_qwen35_config(num_layers=8)
    cfg.layer_types = None  # simulate misconfigured checkpoint
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)

    # HF always populates layer_types; if it is missing the model raises at init
    # rather than silently falling through to a wrong per-call ValueError.
    with pytest.raises(AssertionError, match="layer_types"):
        Qwen35WebGPUModel(cfg, wgpu_device, cache)


def _setup_gdn_model(wgpu_device):
    """Create a Qwen35WebGPUModel with minimal GPU weights for GDN testing."""
    import wgpu as wgpu_lib
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.models.qwen35 import (
        Qwen35WebGPUModel, _LIN_V_HEADS, _LIN_K_DIM, _LIN_V_DIM
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
    conv_bytes = 3 * model._lin_conv_dim * 2  # (KERNEL-1) * DIM * f16
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

    # _gdn_layer_gpu(layer_idx, normed_x, x_buf) → (normed_out, raw_out)
    _, raw_out = model._gdn_layer_gpu(0, x_buf, x_buf)
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
    model._gdn_layer_gpu(0, x_buf, x_buf)

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

    _, raw1 = model._gdn_layer_gpu(0, x1, x1)
    _, raw2 = model._gdn_layer_gpu(0, x2, x2)
    out1 = raw1.to_numpy().view(np.float16).copy()
    out2 = raw2.to_numpy().view(np.float16).copy()

    assert not np.allclose(out1, out2), "GDN GPU outputs should differ across sequential calls"


def test_mlx_detect_format():
    """detect_weight_format returns 'safetensors_sharded' for the MLX model directory.

    MLX detection (presence of .biases keys) is deferred to the loader so the
    index JSON is only parsed once (inside load_safetensors_weights_sharded,
    which then dispatches to load_mlx_weights when .biases keys are present).
    """
    import os
    from vllm_webgpu.quant.weight_loader import detect_weight_format

    model_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-4bit"
        "/snapshots/8b2b98c00a6b4d291155e4890773ca8f769aee53"
    )
    if not os.path.isdir(model_dir):
        pytest.skip("Qwen3.5-9B MLX model not present on disk")

    fmt, _index, _resolved = detect_weight_format(model_dir)
    assert fmt == "safetensors_sharded", f"Expected safetensors_sharded, got {fmt!r}"


def test_mlx_dequant_correctness():
    """_dequant_mlx_int4 correctly unpacks nibbles and applies affine transform."""
    from vllm_webgpu.quant.weight_loader import _dequant_mlx_int4

    # All-zero weights + bias of 3.0 -> every output = 3.0
    # group_size derived from shapes: 2 packed cols * 8 // 1 scale group = 16
    w = np.zeros((2, 2), dtype=np.uint32)    # 2 rows, 16 input cols
    s = np.ones((2, 1), dtype=np.float32)
    b = np.full((2, 1), 3.0, dtype=np.float32)
    result = _dequant_mlx_int4(w, s, b)
    assert result.shape == (2, 16)
    assert np.allclose(result, 3.0), f"Expected all 3.0, got {result}"

    # Nibble ordering: 0x12345678 should unpack to [8,7,6,5,4,3,2,1]
    # group_size derived from shapes: 1 packed col * 8 // 1 scale group = 8
    w2 = np.array([[0x12345678]], dtype=np.uint32)
    s2 = np.ones((1, 1), dtype=np.float32)
    b2 = np.zeros((1, 1), dtype=np.float32)
    result2 = _dequant_mlx_int4(w2, s2, b2)
    assert list(result2[0].astype(int)) == [8, 7, 6, 5, 4, 3, 2, 1], (
        f"Wrong nibble order: {list(result2[0])}"
    )


def test_arch_map_includes_qwen35():
    """ARCH_MAP in model_runner maps Qwen3_5ForConditionalGeneration to qwen35."""
    from vllm_webgpu.v1.model_runner import ARCH_MAP
    assert "Qwen3_5ForConditionalGeneration" in ARCH_MAP
    assert ARCH_MAP["Qwen3_5ForConditionalGeneration"] == "qwen35"


def test_arch_map_includes_qwen35_moe():
    """ARCH_MAP also maps the MoE variant Qwen3_5MoeForConditionalGeneration."""
    from vllm_webgpu.v1.model_runner import ARCH_MAP
    assert "Qwen3_5MoeForConditionalGeneration" in ARCH_MAP
    assert ARCH_MAP["Qwen3_5MoeForConditionalGeneration"] == "qwen35"


@pytest.mark.integration
def test_qwen36_moe_forward(wgpu_device):
    """Qwen3.6 MoE path: GPU expert routing via topk_sort, shared expert + K selected experts.

    Uses all-full-attention layers (layer_types override) to isolate the MoE FFN
    from GDN state, keeping the test fast without sacrificing coverage of the
    router → topk_sort → expert dispatch pipeline.

    Qwen35's _forward_moe() manages encoders manually: Phase A (router + topk)
    is submitted and synced before the CPU reads expert indices, so routing is
    correct and expert weights are actually applied.
    """
    import wgpu as wgpu_lib
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.models.qwen35 import Qwen35WebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    hidden      = 128
    inter       = 256          # shared_expert_intermediate_size
    moe_inter   = 64           # moe_intermediate_size (per routed expert)
    n_experts   = 8
    top_k       = 2
    heads       = 4
    kv_heads    = 2
    head_dim    = 32           # hidden // heads
    q_dim       = heads * head_dim    # 128
    kv_dim      = kv_heads * head_dim # 64
    vocab       = 64
    layers      = 1
    block_sz    = 16
    n_blocks    = 8

    class _Cfg:
        hidden_size                      = hidden
        num_hidden_layers                = layers
        num_attention_heads              = heads
        num_key_value_heads              = kv_heads
        intermediate_size                = inter
        vocab_size                       = vocab
        max_position_embeddings          = 128
        rope_theta                       = 1_000_000.0
        head_dim                         = hidden // heads
        # MoE
        num_experts                      = n_experts
        num_experts_per_tok              = top_k
        moe_intermediate_size            = moe_inter
        shared_expert_intermediate_size  = inter
        # GDN — tiny dimensions; never used since all layers are full attention
        linear_num_key_heads             = 1
        linear_key_head_dim              = 4
        linear_num_value_heads           = 1
        linear_value_head_dim            = 4
        linear_conv_kernel_dim           = 2
        full_attention_interval          = 4   # overridden by layer_types
        attn_output_gate                 = False
        partial_rotary_factor            = 1.0
        mrope_interleaved                = False
        # Single full-attention layer — bypasses all GDN machinery
        layer_types                      = ["full_attention"]

    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = Qwen35WebGPUModel(_Cfg(), wgpu_device, cache)

    assert model._is_moe, "_is_moe should be True with num_experts=8, num_experts_per_tok=2"

    dev = wgpu_device.wgpu_device
    rw  = (wgpu_lib.BufferUsage.STORAGE
           | wgpu_lib.BufferUsage.COPY_SRC
           | wgpu_lib.BufferUsage.COPY_DST)
    rng = np.random.default_rng(7)

    def f16(shape):
        arr = (rng.standard_normal(shape) * 0.01).astype(np.float16)
        return WebGPUBuffer.from_numpy(dev, np.ascontiguousarray(arr), usage=rw)

    def ones_f16(shape):
        """Norm weights as ones so _postprocess_weights detects GEMMA_NORM=0."""
        return WebGPUBuffer.from_numpy(dev,
            np.ones(shape, dtype=np.float16), usage=rw)

    # Global weights
    model.weights["model.embed_tokens.weight"] = f16((vocab, hidden))
    model.weights["model.norm.weight"]         = ones_f16((hidden,))

    # Layer 0 weights
    p = "model.layers.0"
    model.weights[f"{p}.input_layernorm.weight"]          = ones_f16((hidden,))
    model.weights[f"{p}.post_attention_layernorm.weight"] = ones_f16((hidden,))

    # Attention projections
    model.weights[f"{p}.self_attn.q_proj.weight"] = f16((q_dim,  hidden))
    model.weights[f"{p}.self_attn.k_proj.weight"] = f16((kv_dim, hidden))
    model.weights[f"{p}.self_attn.v_proj.weight"] = f16((kv_dim, hidden))
    model.weights[f"{p}.self_attn.o_proj.weight"] = f16((hidden, q_dim))
    # Per-head RMSNorm weights (always present in Qwen3/3.5 checkpoints)
    model.weights[f"{p}.self_attn.q_norm.weight"] = ones_f16((head_dim,))
    model.weights[f"{p}.self_attn.k_norm.weight"] = ones_f16((head_dim,))

    # MoE router: gate.weight [num_experts, hidden]
    model.weights[f"{p}.mlp.gate.weight"] = f16((n_experts, hidden))

    # Shared expert (always active, weight 1.0)
    sp = f"{p}.mlp.shared_expert"
    model.weights[f"{sp}.gate_proj.weight"] = f16((inter,  hidden))
    model.weights[f"{sp}.up_proj.weight"]   = f16((inter,  hidden))
    model.weights[f"{sp}.down_proj.weight"] = f16((hidden, inter))

    # All n_experts routed experts (inject all so any top-k selection will succeed)
    for eid in range(n_experts):
        ep = f"{p}.mlp.experts.{eid}"
        model.weights[f"{ep}.gate_proj.weight"] = f16((moe_inter, hidden))
        model.weights[f"{ep}.up_proj.weight"]   = f16((moe_inter, hidden))
        model.weights[f"{ep}.down_proj.weight"] = f16((hidden, moe_inter))

    # _postprocess_weights detects GEMMA_NORM format from input_layernorm mean
    # (ones → mean=1.0 > 0.7 → GEMMA_NORM=0, correct for Qwen3.5).
    model._postprocess_weights()
    # _alloc_lin_states initialises SSM/conv state lists (all None for full-attn layers).
    model._alloc_lin_states()
    model._batch_matmul_supported = False  # set by load_weights() for MoE models

    # KV cache — required by _full_attn_layer
    kv_bytes = n_blocks * block_sz * kv_heads * head_dim * 2  # f16 bytes
    for _ in range(layers):
        model.kv_pool.append((
            WebGPUBuffer.empty(dev, kv_bytes, usage=rw),
            WebGPUBuffer.empty(dev, kv_bytes, usage=rw),
        ))

    class _FakeMeta:
        slot_mapping    = [0]
        block_tables    = [np.zeros(n_blocks, dtype=np.uint32)]
        max_decode_seq_len = 1

    result = model.forward(
        np.array([1], dtype=np.uint32),
        np.array([0], dtype=np.uint32),
        _FakeMeta(),
    )

    # _forward_moe returns [[token_id]] as int32
    assert result.shape == (1, 1), f"Expected shape (1, 1), got {result.shape}"
    token_id = int(result[0, 0])
    assert 0 <= token_id < vocab, (
        f"token_id {token_id} out of range [0, {vocab})"
    )


def test_prefill_chunked_forward_method_exists(wgpu_device):
    """_prefill_chunked_forward exists and forward() routes to it for num_tokens > 1."""
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.qwen35 import Qwen35WebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR
    from unittest.mock import patch, MagicMock

    cfg = make_qwen35_config()
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = Qwen35WebGPUModel(cfg, wgpu_device, cache)
    model._is_moe = False

    assert hasattr(model, "_prefill_chunked_forward"), (
        "_prefill_chunked_forward method missing from Qwen35WebGPUModel")

    sentinel = np.array([[7]], dtype=np.int32)
    am = MagicMock()
    am.block_tables = [[0]]
    am.slot_mapping = [0, 1, 2]
    am.max_decode_seq_len = 3

    with patch.object(model, "_prefill_chunked_forward", return_value=sentinel) as mock_pfc:
        result = model.forward(
            np.array([10, 11, 12], dtype=np.int32),
            np.array([0, 1, 2], dtype=np.int32),
            am,
        )
        mock_pfc.assert_called_once_with(
            pytest.approx(np.array([10, 11, 12], dtype=np.int32)),
            pytest.approx(np.array([0, 1, 2], dtype=np.int32)),
            am, 3,
        )
        assert np.array_equal(result, sentinel)


def test_gdn_forward_dispatch(wgpu_device):
    """forward() with a GDN layer routes through _transformer_layer -> _gdn_layer_gpu.

    Exercises the full forward() path for a non-MoE model with a single
    linear_attention layer: embed -> layer 0 (GDN) -> final norm -> lm_head -> argmax.
    Checks output shape (1, 1) int32 and that the token id is in-range.
    """
    import wgpu as wgpu_lib
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.models.qwen35 import Qwen35WebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    # Single GDN layer; no full-attention layers, so no kv_pool is needed.
    cfg = make_qwen35_config(num_layers=1)
    cfg.layer_types = ["linear_attention"]

    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = Qwen35WebGPUModel(cfg, wgpu_device, cache)

    hidden = cfg.hidden_size
    vocab  = cfg.vocab_size
    inter  = cfg.intermediate_size
    dev = wgpu_device.wgpu_device
    rw = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST
    rng = np.random.default_rng(99)

    def f16(shape):
        arr = (rng.standard_normal(shape) * 0.01).astype(np.float16)
        return WebGPUBuffer.from_numpy(dev, np.ascontiguousarray(arr), usage=rw)

    def ones_f16(shape):
        return WebGPUBuffer.from_numpy(dev, np.ones(shape, dtype=np.float16), usage=rw)

    # Global weights needed by forward()
    model.weights["model.embed_tokens.weight"] = f16((vocab, hidden))
    model.weights["model.norm.weight"]         = ones_f16((hidden,))

    # Layer 0 norm + FFN weights (input_layernorm is read by _run_decode_dispatches)
    p  = "model.layers.0"
    lp = "model.layers.0.linear_attn"
    model.weights[f"{p}.input_layernorm.weight"]          = ones_f16((hidden,))
    model.weights[f"{p}.post_attention_layernorm.weight"] = ones_f16((hidden,))
    model.weights[f"{p}.mlp.gate_proj.weight"] = f16((inter, hidden))
    model.weights[f"{p}.mlp.up_proj.weight"]   = f16((inter, hidden))
    model.weights[f"{p}.mlp.down_proj.weight"] = f16((hidden, inter))

    # GDN linear_attn weights: use config-derived dimensions from model attributes
    # so the buffer sizes match what the shaders expect exactly.
    cd = model._lin_conv_dim   # Q+K+V projection size (e.g. 6144 for k/v_heads=16)
    vd = model._lin_val_dim    # V head total dim (e.g. 2048)
    vh = model._lin_v_heads    # number of V heads (e.g. 16)
    kern = model._lin_conv_kernel  # conv kernel size (e.g. 4)

    model.weights[f"{lp}.in_proj_qkv.weight"] = f16((cd, hidden))
    model.weights[f"{lp}.in_proj_z.weight"]   = f16((vd, hidden))
    model.weights[f"{lp}.in_proj_a.weight"]   = f16((vh, hidden))
    model.weights[f"{lp}.in_proj_b.weight"]   = f16((vh, hidden))
    model.weights[f"{lp}.conv1d.weight"] = WebGPUBuffer.from_numpy(
        dev, np.zeros((cd, kern), dtype=np.float16), usage=rw)
    model.weights[f"{lp}.A_log"] = WebGPUBuffer.from_numpy(
        dev, np.full(vh, -1.0, dtype=np.float16), usage=rw)
    model.weights[f"{lp}.dt_bias"] = WebGPUBuffer.from_numpy(
        dev, np.zeros(vh, dtype=np.float16), usage=rw)
    model.weights[f"{lp}.norm.weight"]     = ones_f16((model._lin_v_dim,))
    model.weights[f"{lp}.out_proj.weight"] = f16((hidden, vd))

    # Allocate SSM and conv recurrent state buffers (zero-initialized by _make_buf).
    model._alloc_lin_states()
    # _postprocess_weights only acts when attn_output_gate=True; safe to call here.
    model._postprocess_weights()
    # Bypass the load_weights() scan check; set to False since MoE flag is irrelevant.
    model._batch_matmul_supported = False

    class _FakeMeta:
        slot_mapping    = [0]
        block_tables    = [np.zeros(8, dtype=np.uint32)]
        max_decode_seq_len = 1

    result = model.forward(
        np.array([1], dtype=np.uint32),
        np.array([0], dtype=np.uint32),
        _FakeMeta(),
    )

    # forward() returns (1, 1) int32 containing the GPU-argmax token id.
    assert result.shape == (1, 1), f"Expected (1, 1), got {result.shape}"
    token_id = int(result[0, 0])
    assert 0 <= token_id < vocab, (
        f"token_id {token_id} out of range [0, {vocab})"
    )
