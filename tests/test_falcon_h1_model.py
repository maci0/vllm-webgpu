"""Tests for FalconH1WebGPUModel (parallel hybrid SSM + attention)."""
import pytest
from unittest.mock import MagicMock


def make_tiny_falcon_h1_config():
    """Tiny FalconH1-like config for unit tests (2 parallel-hybrid layers)."""
    cfg = MagicMock()
    cfg.hidden_size             = 64
    cfg.num_hidden_layers       = 2
    cfg.num_attention_heads     = 4
    cfg.num_key_value_heads     = 2
    cfg.head_dim                = 16    # 64 // 4
    cfg.intermediate_size       = 128   # feed_forward intermediate
    cfg.vocab_size              = 32
    cfg.max_position_embeddings = 64

    # FalconH1 Mamba config (field names match FalconH1Config)
    cfg.mamba_n_heads  = 4     # -> mamba_num_heads
    cfg.mamba_d_head   = 8     # -> mamba_head_dim  (mamba_int = 4*8 = 32)
    cfg.mamba_n_groups = 2     # -> n_groups
    cfg.mamba_d_state  = 4     # -> ssm_state_size
    cfg.mamba_d_conv   = 4     # -> conv_kernel
    cfg.mamba_d_ssm    = 32    # explicit intermediate size (4*8)
    cfg.mamba_expand   = 2.0   # used if mamba_d_ssm is None

    # Bias flags (must be False for current implementation)
    cfg.mlp_bias       = False
    cfg.mamba_proj_bias = False
    cfg.use_bias       = False  # NemotronH compat

    # Activations
    cfg.hidden_act       = "silu"
    cfg.mlp_hidden_act   = "relu2"
    cfg.mamba_hidden_act = "silu"

    # Multipliers (must all be 1.0 for attention/ssm/embedding/key;
    # mlp_multipliers are absorbed into weights at load time)
    cfg.attention_in_multiplier  = 1.0
    cfg.attention_out_multiplier = 1.0
    cfg.ssm_in_multiplier        = 1.0
    cfg.ssm_out_multiplier       = 1.0
    cfg.embedding_multiplier     = 1.0
    cfg.key_multiplier           = 1.0
    cfg.mlp_multipliers          = [1.0, 1.0]

    # RoPE
    cfg.rope_parameters = {"rope_theta": 10000.0}

    # HF config required by MoE guard
    cfg.layers_block_type = ["attention", "attention"]  # overridden in __init__

    return cfg


def test_arch_map_includes_falcon_h1():
    """FalconH1ForCausalLM is registered in ARCH_MAP."""
    from vllm_webgpu.v1.model_runner import ARCH_MAP

    assert "FalconH1ForCausalLM" in ARCH_MAP, (
        f"FalconH1ForCausalLM missing from ARCH_MAP. Keys: {sorted(ARCH_MAP)}"
    )
    assert ARCH_MAP["FalconH1ForCausalLM"] == "falcon_h1"


def test_falcon_h1_model_instantiates(wgpu_device):
    """FalconH1WebGPUModel instantiates with correct attributes."""
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.falcon_h1 import FalconH1WebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    cfg = make_tiny_falcon_h1_config()
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = FalconH1WebGPUModel(cfg, wgpu_device, cache)

    assert model.num_layers == 2
    assert model.hidden_size == 64
    assert model.mamba_num_heads == 4
    assert model.mamba_head_dim == 8
    assert model.n_groups == 2
    assert model.ssm_state_size == 4
    assert model.conv_kernel == 4
    assert model.weights == {}
    assert model.kv_pool == []


def test_falcon_h1_inherits_nemotron_h(wgpu_device):
    """FalconH1WebGPUModel is a NemotronHWebGPUModel subclass."""
    from vllm_webgpu.models.falcon_h1 import FalconH1WebGPUModel
    from vllm_webgpu.models.nemotron_h import NemotronHWebGPUModel

    assert issubclass(FalconH1WebGPUModel, NemotronHWebGPUModel)


def test_falcon_h1_layer_types_all_attention(wgpu_device):
    """FalconH1 sets all layer types to 'attention' for KV spec compatibility."""
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.falcon_h1 import FalconH1WebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    cfg = make_tiny_falcon_h1_config()
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = FalconH1WebGPUModel(cfg, wgpu_device, cache)

    assert all(lt == "attention" for lt in model._layer_types)
    assert len(model._layer_types) == 2


def test_falcon_h1_mamba_states_allocated_for_all_layers(wgpu_device):
    """FalconH1 allocates mamba conv/SSM states for every layer (not just 'mamba' typed).

    _init_mamba_states is not called here because weights aren't loaded.
    The test verifies the override is in place by checking that
    _conv_states and _ssm_states start empty (allocated during load_weights).
    """
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.falcon_h1 import FalconH1WebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    cfg = make_tiny_falcon_h1_config()
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = FalconH1WebGPUModel(cfg, wgpu_device, cache)

    # Before load_weights, conv/SSM states are empty dicts.
    assert model._conv_states == {}
    assert model._ssm_states  == {}


def test_falcon_h1_multiplier_guard(wgpu_device):
    """FalconH1 raises NotImplementedError for non-unit multipliers."""
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.falcon_h1 import FalconH1WebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    cfg = make_tiny_falcon_h1_config()
    cfg.attention_in_multiplier = 0.5  # non-unit

    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    with pytest.raises(NotImplementedError, match="attention_in_multiplier"):
        FalconH1WebGPUModel(cfg, wgpu_device, cache)


def test_falcon_h1_pos_buf_allocated(wgpu_device):
    """FalconH1 includes 'pos' in its _pre scratch buffer dict (for RoPE)."""
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.falcon_h1 import FalconH1WebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    cfg = make_tiny_falcon_h1_config()
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = FalconH1WebGPUModel(cfg, wgpu_device, cache)

    assert "pos" in model._pre, (
        "'pos' buffer not found in FalconH1WebGPUModel._pre; "
        "_init_scratch_buffers override must add it for RoPE dispatches"
    )


def test_falcon_h1_rope_scratch_buffers_allocated(wgpu_device):
    """FalconH1 allocates q_rope and k_rope in _sc (NemotronH parent does not)."""
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.falcon_h1 import FalconH1WebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    cfg = make_tiny_falcon_h1_config()
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = FalconH1WebGPUModel(cfg, wgpu_device, cache)

    assert "q_rope" in model._sc, "'q_rope' buffer missing from FalconH1 _sc"
    assert "k_rope" in model._sc, "'k_rope' buffer missing from FalconH1 _sc"
    q_dim = cfg.num_attention_heads * cfg.head_dim
    k_dim = cfg.num_key_value_heads * cfg.head_dim
    assert model._sc["q_rope"].nbytes == q_dim * 2
    assert model._sc["k_rope"].nbytes == k_dim * 2


def test_falcon_h1_mlp_multipliers_weight_transforms_registered(wgpu_device):
    """Non-unit mlp_multipliers register _weight_transforms for gate/down projections."""
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.falcon_h1 import FalconH1WebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    cfg = make_tiny_falcon_h1_config()
    cfg.mlp_multipliers = [0.5, 0.25]
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = FalconH1WebGPUModel(cfg, wgpu_device, cache)

    assert model._gate_mult == 0.5
    assert model._down_mult == 0.25
    for i in range(cfg.num_hidden_layers):
        gate_key = f"model.layers.{i}.feed_forward.gate_proj.weight"
        down_key = f"model.layers.{i}.feed_forward.down_proj.weight"
        assert gate_key in model._weight_transforms, f"Missing transform for {gate_key}"
        assert down_key in model._weight_transforms, f"Missing transform for {down_key}"
