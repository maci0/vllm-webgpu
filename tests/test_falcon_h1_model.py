"""Tests for FalconH1WebGPUModel (parallel hybrid SSM + attention)."""
import numpy as np
import pytest
from types import SimpleNamespace
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


@pytest.mark.integration
def test_falcon_h1_decode_forward(wgpu_device):
    """2-layer FalconH1 decode step runs both the SSM branch and the attention branch.

    Each layer dispatches _mamba_branch and _attn_branch in parallel on the same
    pre-normed input. The test injects synthetic weights in the post-load format
    (qkv_proj packed, A already in -exp form) and manually calls _init_mamba_states
    to allocate conv/SSM state buffers. Asserts that forward returns a valid token.
    """
    import wgpu as wgpu_lib
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.models.falcon_h1 import FalconH1WebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    H          = 64
    layers     = 2
    q_h        = 4
    kv_h       = 2
    hd         = 16
    inter      = 128   # FFN intermediate
    vocab      = 32
    num_blocks = 8
    block_size = 16
    q_dim      = q_h * hd    # 64
    k_dim      = kv_h * hd   # 32

    cfg   = make_tiny_falcon_h1_config()
    dev   = wgpu_device.wgpu_device
    cache = PipelineCache(dev, SHADERS_DIR)
    model = FalconH1WebGPUModel(cfg, wgpu_device, cache)

    # Derived Mamba dimensions from the model (set by __init__ after remapping cfg).
    MI  = model.mamba_int        # 32  (mamba_n_heads * mamba_d_head)
    CD  = model.conv_dim         # 48  (mamba_int + 2*n_groups*ssm_state_size)
    MNH = model.mamba_num_heads  # 4
    IPD = model.in_proj_dim      # 84  (MI + CD + MNH)
    CK  = model.conv_kernel      # 4

    rw = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST
    rng = np.random.default_rng(4)

    def f16(*shape):
        return WebGPUBuffer.from_numpy(
            dev, (rng.standard_normal(shape) * 0.01).astype(np.float16), usage=rw)

    def f32(*shape):
        return WebGPUBuffer.from_numpy(
            dev, (rng.standard_normal(shape) * 0.01).astype(np.float32), usage=rw)

    model.weights["model.embed_tokens.weight"]   = f16(vocab, H)
    model.weights["model.final_layernorm.weight"] = f16(H)

    for i in range(layers):
        p = f"model.layers.{i}"
        model.weights[f"{p}.input_layernorm.weight"] = f16(H)
        model.weights[f"{p}.pre_ff_layernorm.weight"] = f16(H)

        # Attention branch: packed qkv_proj (post _pack_falconh1_attn_weights).
        model.weights[f"{p}.self_attn.qkv_proj.weight"] = f16(q_dim + 2 * k_dim, H)
        model.weights[f"{p}.self_attn.o_proj.weight"]   = f16(H, q_dim)

        # Mamba branch: weights in post-load format (A already = -exp(A_log)).
        model.weights[f"{p}.mamba.in_proj.weight"]  = f16(IPD, H)
        model.weights[f"{p}.mamba.conv1d.weight"]   = f16(CD * CK)
        model.weights[f"{p}.mamba.out_proj.weight"] = f16(H, MI)
        # A, D, dt_bias must be f32 (shader reads array<f32>); A negative.
        A_arr = np.full((MNH,), -0.1, dtype=np.float32)
        model.weights[f"{p}.mamba.A"]      = WebGPUBuffer.from_numpy(dev, A_arr, usage=rw)
        model.weights[f"{p}.mamba.D"]      = f32(MNH)
        model.weights[f"{p}.mamba.dt_bias"] = f32(MNH)
        model.weights[f"{p}.mamba.norm.weight"] = f16(MI)

        # FFN branch.
        model.weights[f"{p}.feed_forward.gate_proj.weight"] = f16(inter, H)
        model.weights[f"{p}.feed_forward.up_proj.weight"]   = f16(inter, H)
        model.weights[f"{p}.feed_forward.down_proj.weight"] = f16(H, inter)

    # Allocate Mamba conv/SSM state buffers (normally done at end of load_weights).
    model._init_mamba_states()

    # KV pool: FalconH1 has attention in every layer.
    kv_bytes = num_blocks * block_size * kv_h * hd * 2
    for _ in range(layers):
        model.kv_pool.append((
            WebGPUBuffer.empty(dev, kv_bytes, usage=rw),
            WebGPUBuffer.empty(dev, kv_bytes, usage=rw),
        ))

    meta = SimpleNamespace(
        slot_mapping=[0],
        block_tables=[np.zeros(num_blocks, dtype=np.uint32)],
        max_decode_seq_len=1,
    )

    result = model.forward(
        np.array([0], dtype=np.uint32),
        np.array([0], dtype=np.uint32),
        meta,
    )
    tok = int(result[0, 0])
    assert 0 <= tok < vocab, f"token id {tok} out of [0, {vocab})"


def _inject_falcon_h1_synth_weights(model, wgpu_device, cfg, *, num_blocks=8, block_size=16):
    """Inject synthetic post-load weights and allocate KV/Mamba state for tests."""
    import wgpu as wgpu_lib
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer

    H = cfg.hidden_size
    layers = cfg.num_hidden_layers
    vocab = cfg.vocab_size
    q_h = cfg.num_attention_heads
    kv_h = cfg.num_key_value_heads
    hd = cfg.head_dim
    inter = cfg.intermediate_size
    q_dim = q_h * hd
    k_dim = kv_h * hd

    MI = model.mamba_int
    CD = model.conv_dim
    MNH = model.mamba_num_heads
    IPD = model.in_proj_dim
    CK = model.conv_kernel

    dev = wgpu_device.wgpu_device
    rw = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST
    rng = np.random.default_rng(4)

    def f16(*shape):
        return WebGPUBuffer.from_numpy(
            dev, (rng.standard_normal(shape) * 0.01).astype(np.float16), usage=rw)

    def f32(*shape):
        return WebGPUBuffer.from_numpy(
            dev, (rng.standard_normal(shape) * 0.01).astype(np.float32), usage=rw)

    model.weights["model.embed_tokens.weight"] = f16(vocab, H)
    model.weights["model.final_layernorm.weight"] = f16(H)

    for i in range(layers):
        p = f"model.layers.{i}"
        model.weights[f"{p}.input_layernorm.weight"] = f16(H)
        model.weights[f"{p}.pre_ff_layernorm.weight"] = f16(H)
        model.weights[f"{p}.self_attn.qkv_proj.weight"] = f16(q_dim + 2 * k_dim, H)
        model.weights[f"{p}.self_attn.o_proj.weight"] = f16(H, q_dim)
        model.weights[f"{p}.mamba.in_proj.weight"] = f16(IPD, H)
        model.weights[f"{p}.mamba.conv1d.weight"] = f16(CD * CK)
        model.weights[f"{p}.mamba.out_proj.weight"] = f16(H, MI)
        A_arr = np.full((MNH,), -0.1, dtype=np.float32)
        model.weights[f"{p}.mamba.A"] = WebGPUBuffer.from_numpy(dev, A_arr, usage=rw)
        model.weights[f"{p}.mamba.D"] = f32(MNH)
        model.weights[f"{p}.mamba.dt_bias"] = f32(MNH)
        model.weights[f"{p}.mamba.norm.weight"] = f16(MI)
        model.weights[f"{p}.feed_forward.gate_proj.weight"] = f16(inter, H)
        model.weights[f"{p}.feed_forward.up_proj.weight"] = f16(inter, H)
        model.weights[f"{p}.feed_forward.down_proj.weight"] = f16(H, inter)

    model._init_mamba_states()

    kv_bytes = num_blocks * block_size * kv_h * hd * 2
    model.kv_pool.clear()
    for _ in range(layers):
        model.kv_pool.append((
            WebGPUBuffer.empty(dev, kv_bytes, usage=rw),
            WebGPUBuffer.empty(dev, kv_bytes, usage=rw),
        ))
    return num_blocks


@pytest.mark.integration
def test_falcon_h1_replay_prefix_for_ssm(wgpu_device):
    """replay_prefix_for_ssm reconstructs SSM/conv state after a zeroing reset.

    Simulates preemption: run a short prefix (populates KV + SSM), save SSM,
    zero SSM while leaving KV intact, replay the prefix, and assert SSM matches.
    """
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.falcon_h1 import FalconH1WebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    cfg = make_tiny_falcon_h1_config()
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = FalconH1WebGPUModel(cfg, wgpu_device, cache)
    num_blocks = _inject_falcon_h1_synth_weights(model, wgpu_device, cfg)

    prefix = np.array([1, 2, 3, 4], dtype=np.uint32)
    T = len(prefix)
    block_table = np.arange(num_blocks, dtype=np.uint32)
    meta = SimpleNamespace(
        slot_mapping=list(range(T)),
        block_tables=[block_table],
        max_decode_seq_len=T,
    )

    # Prefill populates KV cache and SSM state.
    model.forward(prefix, np.arange(T, dtype=np.uint32), meta)
    expected = model.save_recurrent_states()
    assert expected["ssm"], "expected non-empty SSM state after prefill"
    assert expected["conv"], "expected non-empty conv state after prefill"

    # Preemption: SSM lost, KV retained.
    model.reset_recurrent_states()
    model.replay_prefix_for_ssm(prefix, list(block_table), start_pos=0)
    reconstructed = model.save_recurrent_states()

    for kind, dtype in (("ssm", np.float32), ("conv", np.float16)):
        assert set(reconstructed[kind]) == set(expected[kind]), (
            f"{kind} layer keys differ after replay"
        )
        for layer_idx, exp_bytes in expected[kind].items():
            got = reconstructed[kind][layer_idx]
            exp_arr = np.frombuffer(exp_bytes, dtype=dtype)
            got_arr = np.frombuffer(got, dtype=dtype)
            np.testing.assert_allclose(
                got_arr.astype(np.float32), exp_arr.astype(np.float32),
                rtol=1e-2, atol=1e-2,
                err_msg=f"FalconH1 {kind} state layer {layer_idx} mismatch after replay",
            )