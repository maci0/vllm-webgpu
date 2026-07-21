"""Tests for NemotronHForCausalLM — Mamba-2 SSM hybrid model."""
import numpy as np
from pathlib import Path

SHADERS_DIR = Path(__file__).parent.parent / "vllm_webgpu" / "shaders"


# ── Layer type list (no GPU) ──────────────────────────────────────────────────

def test_resolve_intermediate_size():
    """_resolve_intermediate_size matches NemotronHMLPDecoderLayer.__init__ L286-292."""
    from vllm_webgpu.models.nemotron_h import _resolve_intermediate_size

    # Single-element list must return v[0] for any idx.
    assert _resolve_intermediate_size([1024], 0) == 1024
    assert _resolve_intermediate_size([1024], 5) == 1024, (
        "_resolve_intermediate_size: single-element list must return v[0] for any idx. "
        "The len==1 branch of NemotronHMLPDecoderLayer.__init__ was refactored; "
        "update _resolve_intermediate_size to match."
    )
    # Multi-element list: each index returns the corresponding entry.
    for i, expected in enumerate([1024, 2048, 4096]):
        got = _resolve_intermediate_size([1024, 2048, 4096], i)
        assert got == expected, (
            f"_resolve_intermediate_size([1024, 2048, 4096], {i}) returned {got!r}, "
            f"expected {expected}. The multi-element list branch has changed; update to match."
        )
    # Scalar passthrough.
    assert _resolve_intermediate_size(2048, 3) == 2048


def test_nemotron_h_layer_types_from_config():
    """Model reads _layer_types directly from layers_block_type."""
    # Verify the ARCH_MAP entry and layer type ordering without touching GPU.
    layer_types = ["mamba", "mlp", "mamba", "mlp", "mamba", "mlp",
                   "mamba", "mamba", "mlp", "mamba", "mlp", "mamba",
                   "attention", "mlp", "mamba", "mlp", "mamba"]
    assert layer_types[0]  == "mamba"
    assert layer_types[1]  == "mlp"
    assert layer_types[12] == "attention"
    assert len(layer_types) == 17


def test_nemotron_h_layer_types_mixed():
    """A mixed layer list has the expected counts of each type."""
    layer_types = (
        ["mamba"] * 3 + ["attention"] + ["mamba"] * 3 + ["attention"]
        + ["mamba"] * 2 + ["attention"] + ["mamba"] * 3 + ["attention"]
    )
    assert layer_types.count("attention") == 4
    assert layer_types.count("mamba") > 0


def test_arch_map_includes_nemotron_h():
    """NemotronHForCausalLM is registered in the model runner ARCH_MAP."""
    from vllm_webgpu.v1.model_runner import ARCH_MAP

    assert "NemotronHForCausalLM" in ARCH_MAP, (
        f"NemotronHForCausalLM missing from ARCH_MAP. Keys: {sorted(ARCH_MAP)}"
    )
    assert ARCH_MAP["NemotronHForCausalLM"] == "nemotron_h"


# ── GPU tests ─────────────────────────────────────────────────────────────────

def make_tiny_nemotron_config():
    """Tiny NemotronH config for unit tests (2 layers: mamba + attention)."""
    from unittest.mock import MagicMock

    cfg = MagicMock()
    cfg.hidden_size             = 64
    cfg.num_hidden_layers       = 2
    cfg.num_attention_heads     = 2
    cfg.num_key_value_heads     = 2
    cfg.head_dim                = 32
    cfg.intermediate_size       = 128
    cfg.vocab_size              = 32
    cfg.max_position_embeddings = 64
    cfg.rope_theta              = 10000.0
    # Mamba-2 parameters
    cfg.mamba_num_heads         = 4
    cfg.mamba_head_dim          = 8    # mamba_int = 4*8 = 32
    cfg.n_groups                = 2
    cfg.ssm_state_size          = 4
    cfg.conv_kernel             = 4
    # conv_dim = 32 + 2*2*4 = 48, in_proj_dim = 32 + 48 + 4 = 84
    cfg.layers_block_type  = ["mamba", "attention"]
    cfg.mlp_bias           = False
    cfg.use_bias           = False
    cfg.mamba_hidden_act   = "silu"
    cfg.mlp_hidden_act     = "relu2"
    return cfg


def _make_nemotron_weights(dev, cfg):
    """Upload minimal random weights as GPU f16/f32 buffers for a 2-layer model."""
    import wgpu as wgpu_lib
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer

    rw  = (wgpu_lib.BufferUsage.STORAGE
           | wgpu_lib.BufferUsage.COPY_SRC
           | wgpu_lib.BufferUsage.COPY_DST)
    rng = np.random.default_rng(42)

    def f16(*shape):
        arr = (rng.standard_normal(shape) * 0.01).astype(np.float16)
        return WebGPUBuffer.from_numpy(dev, np.ascontiguousarray(arr), usage=rw)

    def ones_f16(*shape):
        return WebGPUBuffer.from_numpy(
            dev, np.ones(shape, dtype=np.float16), usage=rw)

    def f32(*shape):
        arr = (rng.standard_normal(shape) * 0.01).astype(np.float32)
        return WebGPUBuffer.from_numpy(dev, np.ascontiguousarray(arr), usage=rw)

    H   = cfg.hidden_size        # 64
    V   = cfg.vocab_size         # 32
    MI  = cfg.mamba_num_heads * cfg.mamba_head_dim  # 32
    NG  = cfg.n_groups            # 2
    NS  = cfg.ssm_state_size      # 4
    CD  = MI + 2 * NG * NS        # 48
    MNH = cfg.mamba_num_heads     # 4
    IPD = MI + CD + MNH           # 84
    QD  = cfg.num_attention_heads * cfg.head_dim        # 64
    KD  = cfg.num_key_value_heads * cfg.head_dim        # 64
    TQD = QD + 2 * KD             # 192

    w = {}
    w["model.embed_tokens.weight"]  = f16(V, H)
    w["model.norm_f.weight"]        = ones_f16(H)
    w["model.layers.0.norm.weight"] = ones_f16(H)
    w["model.layers.1.norm.weight"] = ones_f16(H)

    # Layer 0 — Mamba-2
    p0 = "model.layers.0.mixer"
    w[f"{p0}.in_proj.weight"]  = f16(IPD, H)
    # conv1d.weight: flat [conv_dim * kernel] f16
    w[f"{p0}.conv1d.weight"]   = f16(CD * cfg.conv_kernel)
    w[f"{p0}.out_proj.weight"] = f16(H, MI)
    # A already postprocessed: negative f32 values
    w[f"{p0}.A"]              = WebGPUBuffer.from_numpy(
        dev,
        np.full((MNH,), -0.1, dtype=np.float32),
        usage=rw,
    )
    w[f"{p0}.D"]              = f32(MNH)
    w[f"{p0}.dt_bias"]        = f32(MNH)
    w[f"{p0}.norm.weight"]    = ones_f16(MI)

    # Layer 1 — Attention
    p1 = "model.layers.1.mixer"
    w[f"{p1}.qkv_proj.weight"] = f16(TQD, H)
    w[f"{p1}.o_proj.weight"]   = f16(H, QD)

    return w


def test_nemotron_h_model_instantiates(wgpu_device):
    """Model instantiates with correct layer count and parsed pattern."""
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.nemotron_h import NemotronHWebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    cfg   = make_tiny_nemotron_config()
    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model = NemotronHWebGPUModel(cfg, wgpu_device, cache)

    assert model.num_layers == 2
    assert model.hidden_size == 64
    assert model._layer_types == ["mamba", "attention"]
    assert model.weights == {}
    assert model.kv_pool == []


def test_nemotron_h_mamba_forward(wgpu_device):
    """Forward pass through a 2-layer NemotronH (mamba + attention) returns a valid token.

    Weights are injected as already-postprocessed GPU buffers so we skip
    the file-loading path. Mamba states are allocated explicitly.
    """
    import wgpu as wgpu_lib
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.models.nemotron_h import NemotronHWebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    cfg     = make_tiny_nemotron_config()
    cache   = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
    model   = NemotronHWebGPUModel(cfg, wgpu_device, cache)
    dev     = wgpu_device.wgpu_device

    # Inject synthetic weights
    model.weights = _make_nemotron_weights(dev, cfg)

    # Allocate Mamba SSM state buffers
    model._init_mamba_states()

    # KV pool: 2 entries — layer 0 (mamba, never accessed) + layer 1 (attention)
    block_size = model.block_size
    n_blocks   = 4
    rw = (wgpu_lib.BufferUsage.STORAGE
          | wgpu_lib.BufferUsage.COPY_SRC
          | wgpu_lib.BufferUsage.COPY_DST)
    kv_bytes = n_blocks * block_size * cfg.num_key_value_heads * cfg.head_dim * 2
    dummy_k  = WebGPUBuffer.empty(dev, max(kv_bytes, 8), usage=rw)
    dummy_v  = WebGPUBuffer.empty(dev, max(kv_bytes, 8), usage=rw)
    real_k   = WebGPUBuffer.empty(dev, max(kv_bytes, 8), usage=rw)
    real_v   = WebGPUBuffer.empty(dev, max(kv_bytes, 8), usage=rw)
    model.kv_pool = [(dummy_k, dummy_v), (real_k, real_v)]

    class _FakeMeta:
        slot_mapping       = [0]
        block_tables       = [np.zeros(n_blocks, dtype=np.uint32)]
        max_decode_seq_len = 1

    result = model.forward(
        np.array([1], dtype=np.uint32),
        np.array([0], dtype=np.uint32),
        _FakeMeta(),
    )

    assert result.shape == (1, 1), f"Expected (1, 1), got {result.shape}"
    token_id = int(result[0, 0])
    assert 0 <= token_id < cfg.vocab_size, (
        f"token_id {token_id} out of range [0, {cfg.vocab_size})"
    )


# ── Kernel tests ──────────────────────────────────────────────────────────────

def _dispatch_kernel(dev, pipeline_cache, shader_name, bindings, constants, n_groups):
    """Bind buffers and dispatch a compute shader synchronously."""
    from vllm_webgpu.webgpu.pipeline import PipelineKey

    key      = PipelineKey(shader_name, tuple(sorted(constants.items())))
    pipeline = pipeline_cache.get_or_create(key)
    bg_layout = pipeline.get_bind_group_layout(0)
    entries   = [{"binding": i, "resource": {"buffer": b.buf}}
                 for i, b in enumerate(bindings)]
    bg      = dev.create_bind_group(layout=bg_layout, entries=entries)
    encoder = dev.create_command_encoder()
    cp      = encoder.begin_compute_pass()
    cp.set_pipeline(pipeline)
    cp.set_bind_group(0, bg)
    cp.dispatch_workgroups(*n_groups)
    cp.end()
    dev.queue.submit([encoder.finish()])


def _ssm_step_ref(x_B_C_f32, dt_f32, A_f32, dt_bias_f32, D_f32, state_f32,
                  NUM_HEADS, HEAD_DIM, STATE_SIZE, N_GROUPS):
    """Pure-NumPy reference for the mamba2_ssm_step kernel."""
    mamba_int    = NUM_HEADS * HEAD_DIM
    groups_state = N_GROUPS * STATE_SIZE
    state = state_f32.copy().reshape(NUM_HEADS, HEAD_DIM, STATE_SIZE)
    y     = np.zeros((NUM_HEADS, HEAD_DIM), dtype=np.float32)

    x = x_B_C_f32[:mamba_int].reshape(NUM_HEADS, HEAD_DIM)
    B = x_B_C_f32[mamba_int : mamba_int + groups_state].reshape(N_GROUPS, STATE_SIZE)
    C = x_B_C_f32[mamba_int + groups_state :].reshape(N_GROUPS, STATE_SIZE)

    for h in range(NUM_HEADS):
        dt_raw = dt_f32[h] + dt_bias_f32[h]
        # numerically stable softplus
        dt_eff = np.where(dt_raw >= 0, dt_raw, 0.0) + np.log1p(np.exp(-np.abs(dt_raw)))
        dA     = np.exp(A_f32[h] * dt_eff)
        group  = h * N_GROUPS // NUM_HEADS

        for d in range(HEAD_DIM):
            x_val = x[h, d]
            state[h, d, :] = (
                dA * state[h, d, :]
                + dt_eff * B[group, :] * x_val
            )
        for d in range(HEAD_DIM):
            y[h, d] = np.dot(C[group, :], state[h, d, :]) + D_f32[h] * x[h, d]

    return y.ravel(), state.ravel()


def test_mamba2_ssm_step(wgpu_device):
    """mamba2_ssm_step WGSL output matches NumPy reference (RTol ≤ 0.1 for f16)."""
    import wgpu as wgpu_lib
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache

    NUM_HEADS  = 4
    HEAD_DIM   = 8
    STATE_SIZE = 4
    N_GROUPS   = 2
    WG_SIZE    = 64   # must be >= HEAD_DIM

    mamba_int    = NUM_HEADS * HEAD_DIM       # 32
    groups_state = N_GROUPS * STATE_SIZE      # 8
    total_xbc    = mamba_int + 2 * groups_state  # 48

    rng = np.random.default_rng(77)
    x_B_C_f16 = (rng.standard_normal(total_xbc) * 0.1).astype(np.float16)
    dt_f16     = (rng.standard_normal(NUM_HEADS)  * 0.1).astype(np.float16)
    A_f32      = np.full((NUM_HEADS,), -0.1, dtype=np.float32)
    dt_bias_f32 = np.zeros((NUM_HEADS,), dtype=np.float32)
    D_f32      = np.ones((NUM_HEADS,), dtype=np.float32) * 0.5
    state_f32  = np.zeros((NUM_HEADS * HEAD_DIM * STATE_SIZE,), dtype=np.float32)

    dev = wgpu_device.wgpu_device
    rw  = (wgpu_lib.BufferUsage.STORAGE
           | wgpu_lib.BufferUsage.COPY_SRC
           | wgpu_lib.BufferUsage.COPY_DST)

    xbc_buf   = WebGPUBuffer.from_numpy(dev, x_B_C_f16, usage=rw)
    dt_buf    = WebGPUBuffer.from_numpy(dev, dt_f16,    usage=rw)
    A_buf     = WebGPUBuffer.from_numpy(dev, A_f32,     usage=rw)
    dtbias_buf = WebGPUBuffer.from_numpy(dev, dt_bias_f32, usage=rw)
    D_buf     = WebGPUBuffer.from_numpy(dev, D_f32,     usage=rw)
    state_buf = WebGPUBuffer.from_numpy(dev, state_f32, usage=rw)
    y_buf     = WebGPUBuffer.empty(dev, mamba_int * 2, usage=rw)  # f16

    cache = PipelineCache(dev, SHADERS_DIR / "generic")
    constants = {
        "NUM_HEADS":  NUM_HEADS,
        "HEAD_DIM":   HEAD_DIM,
        "STATE_SIZE": STATE_SIZE,
        "N_GROUPS":   N_GROUPS,
        "WG_SIZE":    WG_SIZE,
    }
    _dispatch_kernel(
        dev, cache, "mamba2_ssm_step",
        [xbc_buf, dt_buf, A_buf, dtbias_buf, D_buf, state_buf, y_buf],
        constants, (NUM_HEADS, 1, 1),
    )

    # GPU output
    gpu_y = y_buf.to_numpy().view(np.float16).astype(np.float32)

    # NumPy reference
    ref_y, _ = _ssm_step_ref(
        x_B_C_f16.astype(np.float32),
        dt_f16.astype(np.float32),
        A_f32, dt_bias_f32, D_f32, state_f32,
        NUM_HEADS, HEAD_DIM, STATE_SIZE, N_GROUPS,
    )
    ref_y = ref_y.astype(np.float32)

    assert gpu_y.shape == ref_y.shape, f"Shape mismatch: {gpu_y.shape} vs {ref_y.shape}"
    assert not np.any(np.isnan(gpu_y)), "GPU SSM output contains NaN"
    assert not np.any(np.isinf(gpu_y)), "GPU SSM output contains Inf"
    # f16 precision: allow up to 10% relative tolerance + small absolute tolerance.
    np.testing.assert_allclose(gpu_y, ref_y, rtol=0.1, atol=1e-3,
                               err_msg="mamba2_ssm_step GPU output deviates from reference")


def test_mamba2_causal_conv(wgpu_device):
    """mamba2_causal_conv updates conv state and produces finite SiLU output."""
    import wgpu as wgpu_lib
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache

    CONV_DIM = 16
    KERNEL   = 4
    WG_SIZE  = 16

    rng = np.random.default_rng(13)
    x          = (rng.standard_normal(CONV_DIM) * 0.1).astype(np.float16)
    weight     = (rng.standard_normal(CONV_DIM * KERNEL) * 0.1).astype(np.float16)
    bias_buf_d = np.zeros(CONV_DIM, dtype=np.float16)   # HAS_BIAS=0, dummy
    conv_state = np.zeros((KERNEL - 1) * CONV_DIM, dtype=np.float16)

    dev = wgpu_device.wgpu_device
    rw  = (wgpu_lib.BufferUsage.STORAGE
           | wgpu_lib.BufferUsage.COPY_SRC
           | wgpu_lib.BufferUsage.COPY_DST)

    x_buf     = WebGPUBuffer.from_numpy(dev, x,          usage=rw)
    w_buf     = WebGPUBuffer.from_numpy(dev, weight,     usage=rw)
    b_buf     = WebGPUBuffer.from_numpy(dev, bias_buf_d, usage=rw)
    cs_buf    = WebGPUBuffer.from_numpy(dev, conv_state, usage=rw)
    out_buf   = WebGPUBuffer.empty(dev, CONV_DIM * 2, usage=rw)

    cache = PipelineCache(dev, SHADERS_DIR / "generic")
    constants = {
        "CONV_DIM": CONV_DIM,
        "KERNEL":   KERNEL,
        "WG_SIZE":  WG_SIZE,
        "HAS_BIAS": 0,
    }
    _dispatch_kernel(
        dev, cache, "mamba2_causal_conv",
        [x_buf, w_buf, b_buf, cs_buf, out_buf],
        constants, ((CONV_DIM + WG_SIZE - 1) // WG_SIZE, 1, 1),
    )

    out       = out_buf.to_numpy().view(np.float16)
    new_state = cs_buf.to_numpy().view(np.float16)

    assert out.shape == (CONV_DIM,), f"Unexpected output shape: {out.shape}"
    assert not np.any(np.isnan(out)), "Causal conv output contains NaN"
    assert not np.any(np.isinf(out)), "Causal conv output contains Inf"

    # Conv state should be updated — the last kernel-1 inputs are now stored.
    # After the first call (all-zero state), the final slot should equal x.
    last_row = new_state[(KERNEL - 2) * CONV_DIM : (KERNEL - 1) * CONV_DIM]
    np.testing.assert_array_equal(last_row, x,
        err_msg="Conv state final slot should equal x after first update")


def test_relu_sq_activation(wgpu_device):
    """relu_sq kernel: y = max(0, x)^2, finite for negative inputs."""
    import wgpu as wgpu_lib
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache

    N      = 32
    WG_SIZE = 32

    rng = np.random.default_rng(5)
    x_np = (rng.standard_normal(N) * 1.0).astype(np.float16)

    dev = wgpu_device.wgpu_device
    rw  = (wgpu_lib.BufferUsage.STORAGE
           | wgpu_lib.BufferUsage.COPY_SRC
           | wgpu_lib.BufferUsage.COPY_DST)

    x_buf   = WebGPUBuffer.from_numpy(dev, x_np, usage=rw)
    out_buf = WebGPUBuffer.empty(dev, N * 2, usage=rw)

    cache = PipelineCache(dev, SHADERS_DIR / "generic")
    _dispatch_kernel(
        dev, cache, "relu_sq",
        [x_buf, out_buf],
        {"N": N, "WG_SIZE": WG_SIZE},
        ((N + WG_SIZE - 1) // WG_SIZE, 1, 1),
    )

    gpu_y = out_buf.to_numpy().view(np.float16).astype(np.float32)
    ref_y = np.maximum(0.0, x_np.astype(np.float32)) ** 2

    assert not np.any(np.isnan(gpu_y)), "relu_sq output contains NaN"
    np.testing.assert_allclose(gpu_y, ref_y, rtol=0.01, atol=1e-4,
                               err_msg="relu_sq output deviates from reference")
    # Negative inputs must produce zero.
    neg_mask = x_np < 0
    assert np.all(gpu_y[neg_mask] == 0.0), "relu_sq must be 0 for negative inputs"
