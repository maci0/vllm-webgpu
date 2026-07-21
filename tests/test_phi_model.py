"""Tests for PhiWebGPUModel (Phi-4 / Phi-4-mini)."""
import numpy as np
import pytest
from types import SimpleNamespace
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


def test_phi_split_row_major_gptq(wgpu_device):
    """_split_row_major + _split_gptq_scales correctly split GPTQ int4 weights.

    Sets up a synthetic qkv_proj with dtype=i32 (GPTQ-packed) and matching
    f32 scales, then verifies that _split_fused_phi_weights produces three
    weight buffers (q, k, v) with correct shape, dtype, and nbytes, plus three
    scale buffers with the correct column slices and weight_meta propagated.
    """
    import numpy as np
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.models.phi import PhiWebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    cfg        = make_tiny_phi_config()
    dev        = wgpu_device.wgpu_device
    cache      = PipelineCache(dev, SHADERS_DIR)
    model      = PhiWebGPUModel(cfg, wgpu_device, cache)

    H          = 64          # hidden_size = K (input features)
    Q          = 4 * 16      # q_dim  = 64
    KV         = 2 * 16      # kv_dim = 32
    N_total    = Q + 2 * KV  # 128 output neurons
    group_size = 16
    G          = H // group_size  # 4 quantization groups

    # GPTQ weight: [N, K//8] i32 (transposed by weight_loader for coalesced access)
    qw_arr = np.arange(N_total * (H // 8), dtype=np.int32).reshape(N_total, H // 8)
    qkv_buf = WebGPUBuffer.from_numpy(dev, qw_arr)
    qkv_buf.dtype  = "i32"
    qkv_buf.shape  = qw_arr.shape
    model.weights["model.layers.0.self_attn.qkv_proj.weight"] = qkv_buf

    # GPTQ scales: [G, N_total] f32
    rng       = np.random.default_rng(42)
    sc_arr    = rng.standard_normal((G, N_total)).astype(np.float32)
    sc_buf    = WebGPUBuffer.from_numpy(dev, sc_arr)
    sc_buf.dtype  = "f32"
    sc_buf.shape  = sc_arr.shape
    model.weights["model.layers.0.self_attn.qkv_proj.weight.scales"] = sc_buf

    # weight_meta mirrors what the weight_loader stores for gptq_sym.
    model.weight_meta["model.layers.0.self_attn.qkv_proj"] = {
        "fmt": "gptq_sym",
        "group_size": group_size,
    }

    model._split_fused_phi_weights()

    # Fused key must be gone.
    assert "model.layers.0.self_attn.qkv_proj.weight" not in model.weights
    assert "model.layers.0.self_attn.qkv_proj.weight.scales" not in model.weights

    # All three split weight keys must be present.
    for proj in ("q_proj", "k_proj", "v_proj"):
        wk = f"model.layers.0.self_attn.{proj}.weight"
        sk = f"{wk}.scales"
        assert wk in model.weights,  f"missing weight key {wk!r}"
        assert sk in model.weights,  f"missing scales key {sk!r}"

    q_wbuf = model.weights["model.layers.0.self_attn.q_proj.weight"]
    k_wbuf = model.weights["model.layers.0.self_attn.k_proj.weight"]
    v_wbuf = model.weights["model.layers.0.self_attn.v_proj.weight"]

    # Weight dtype must be i32 on all three.
    assert q_wbuf.dtype == "i32", f"q_proj dtype {q_wbuf.dtype!r}, expected i32"
    assert k_wbuf.dtype == "i32", f"k_proj dtype {k_wbuf.dtype!r}, expected i32"
    assert v_wbuf.dtype == "i32", f"v_proj dtype {v_wbuf.dtype!r}, expected i32"

    # Weight shapes: (n_rows, K//8).
    assert q_wbuf.shape == (Q,  H // 8), f"q_proj shape {q_wbuf.shape}"
    assert k_wbuf.shape == (KV, H // 8), f"k_proj shape {k_wbuf.shape}"
    assert v_wbuf.shape == (KV, H // 8), f"v_proj shape {v_wbuf.shape}"

    # Weight byte counts.
    assert q_wbuf.nbytes == Q  * (H // 8) * 4
    assert k_wbuf.nbytes == KV * (H // 8) * 4
    assert v_wbuf.nbytes == KV * (H // 8) * 4

    # Scales shapes: [G, n_rows_for_proj].
    q_sbuf = model.weights["model.layers.0.self_attn.q_proj.weight.scales"]
    k_sbuf = model.weights["model.layers.0.self_attn.k_proj.weight.scales"]
    v_sbuf = model.weights["model.layers.0.self_attn.v_proj.weight.scales"]

    assert q_sbuf.shape == (G, Q),  f"q scales shape {q_sbuf.shape}"
    assert k_sbuf.shape == (G, KV), f"k scales shape {k_sbuf.shape}"
    assert v_sbuf.shape == (G, KV), f"v scales shape {v_sbuf.shape}"

    # Verify scale values are the correct column slices of the original array.
    q_sc_back = np.frombuffer(q_sbuf.to_numpy(), dtype=np.float32).reshape(G, Q)
    k_sc_back = np.frombuffer(k_sbuf.to_numpy(), dtype=np.float32).reshape(G, KV)
    v_sc_back = np.frombuffer(v_sbuf.to_numpy(), dtype=np.float32).reshape(G, KV)

    np.testing.assert_array_equal(q_sc_back, sc_arr[:, :Q])
    np.testing.assert_array_equal(k_sc_back, sc_arr[:, Q:Q + KV])
    np.testing.assert_array_equal(v_sc_back, sc_arr[:, Q + KV:])

    # weight_meta must be propagated to split keys and removed for fused key.
    assert "model.layers.0.self_attn.qkv_proj" not in model.weight_meta
    for proj in ("q_proj", "k_proj", "v_proj"):
        meta_key = f"model.layers.0.self_attn.{proj}"
        assert meta_key in model.weight_meta, f"missing weight_meta for {meta_key!r}"
        assert model.weight_meta[meta_key] == {"fmt": "gptq_sym", "group_size": group_size}

    # _uq_for_key must return 3 (gptq_sym) for all three split weights.
    for proj in ("q_proj", "k_proj", "v_proj"):
        wk = f"model.layers.0.self_attn.{proj}.weight"
        uq = model._uq_for_key(wk)
        assert uq == 3, f"_uq_for_key({wk!r}) returned {uq}, expected 3"


@pytest.mark.integration
def test_phi_decode_forward(wgpu_device):
    """2-layer Phi decode step returns a valid token id.

    Weights are injected in post-split format (separate q/k/v and gate/up
    projections) to simulate the state after _split_fused_phi_weights runs.
    Exercises the full Llama-inherited decode path with Phi weight keys.
    """
    import wgpu as wgpu_lib
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.models.phi import PhiWebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR

    H          = 64
    layers     = 2
    q_h        = 4
    kv_h       = 2
    hd         = 16
    inter      = 128
    vocab      = 32
    num_blocks = 8
    block_size = 16
    q_dim      = q_h * hd
    kv_dim     = kv_h * hd

    cfg   = make_tiny_phi_config()
    dev   = wgpu_device.wgpu_device
    cache = PipelineCache(dev, SHADERS_DIR)
    model = PhiWebGPUModel(cfg, wgpu_device, cache)

    rw = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST
    rng = np.random.default_rng(2)

    def f16(*shape):
        return WebGPUBuffer.from_numpy(
            dev, (rng.standard_normal(shape) * 0.01).astype(np.float16), usage=rw)

    model.weights["model.embed_tokens.weight"] = f16(vocab, H)
    model.weights["model.norm.weight"]          = f16(H)

    for i in range(layers):
        p = f"model.layers.{i}"
        model.weights[f"{p}.input_layernorm.weight"]          = f16(H)
        model.weights[f"{p}.post_attention_layernorm.weight"] = f16(H)
        # Post-split: separate projections (as produced by _split_fused_phi_weights).
        model.weights[f"{p}.self_attn.q_proj.weight"] = f16(q_dim, H)
        model.weights[f"{p}.self_attn.k_proj.weight"] = f16(kv_dim, H)
        model.weights[f"{p}.self_attn.v_proj.weight"] = f16(kv_dim, H)
        model.weights[f"{p}.self_attn.o_proj.weight"] = f16(H, q_dim)
        model.weights[f"{p}.mlp.gate_proj.weight"]    = f16(inter, H)
        model.weights[f"{p}.mlp.up_proj.weight"]      = f16(inter, H)
        model.weights[f"{p}.mlp.down_proj.weight"]    = f16(H, inter)

    kv_bytes = num_blocks * block_size * kv_h * hd * 2
    for _ in range(layers):
        model.kv_pool.append((
            WebGPUBuffer.empty(dev, kv_bytes, usage=rw),
            WebGPUBuffer.empty(dev, kv_bytes, usage=rw),
        ))

    # Set by load_weights; injecting directly so set explicitly.
    model._batch_matmul_supported = False

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
