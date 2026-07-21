import numpy as np
from pathlib import Path
import struct


_DTYPE_MAP = {np.float16: "F16", np.float32: "F32", np.uint8: "U8", np.int32: "I32"}


def make_fake_safetensors_raw(tmp_path: Path, tensors: list, filename: str = "model.safetensors") -> Path:
    """Write a safetensors file accepting explicit dtype strings (e.g. 'F8_E4M3').

    tensors: list of (name, dtype_str, shape, np_array) tuples.
    """
    import json
    metadata = {}
    offset = 0
    data_parts = []
    for name, dtype_str, shape, arr in tensors:
        data = arr.ravel().view(np.uint8).tobytes()
        metadata[name] = {
            "dtype": dtype_str,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(data)],
        }
        data_parts.append(data)
        offset += len(data)
    header_bytes = json.dumps(metadata).encode("utf-8")
    header_len = struct.pack("<Q", len(header_bytes))
    out = tmp_path / filename
    out.write_bytes(header_len + header_bytes + b"".join(data_parts))
    return out


def make_fake_safetensors(tmp_path: Path, tensors: dict, filename: str = "model.safetensors") -> Path:
    """Write a minimal safetensors file for testing.

    Supports F16, F32, U8, and I32 tensors.
    """
    import json
    metadata = {}
    offset = 0
    data_parts = []
    for name, arr in tensors.items():
        dtype_str = _DTYPE_MAP[arr.dtype.type]
        nbytes = arr.nbytes
        metadata[name] = {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        data_parts.append(arr.tobytes())
        offset += nbytes
    header_bytes = json.dumps(metadata).encode("utf-8")
    header_len = struct.pack("<Q", len(header_bytes))
    out = tmp_path / filename
    out.write_bytes(header_len + header_bytes + b"".join(data_parts))
    return out


def test_detect_format_safetensors(tmp_path):
    from vllm_webgpu.quant.weight_loader import detect_weight_format
    f = tmp_path / "model.safetensors"
    f.write_bytes(b"\x00" * 16)
    fmt, index_path, resolved = detect_weight_format(str(f))
    assert fmt == "safetensors"
    assert index_path is None
    assert resolved == str(f)


def test_detect_format_gguf(tmp_path):
    from vllm_webgpu.quant.weight_loader import detect_weight_format
    f = tmp_path / "model.gguf"
    f.write_bytes(b"GGUF" + b"\x00" * 12)
    fmt, index_path, resolved = detect_weight_format(str(f))
    assert fmt == "gguf"
    assert index_path is None
    assert resolved is None


def test_load_safetensors(wgpu_device, tmp_path):
    from vllm_webgpu.quant.weight_loader import load_safetensors_weights
    tensors = {
        "model.embed_tokens.weight": np.random.randn(32, 64).astype(np.float16),
        "model.layers.0.self_attn.q_proj.weight": np.random.randn(64, 64).astype(np.float16),
    }
    st_path = make_fake_safetensors(tmp_path, tensors)
    weights = load_safetensors_weights(str(st_path), wgpu_device.wgpu_device)
    assert "model.embed_tokens.weight" in weights
    assert weights["model.embed_tokens.weight"].dtype == "f16"
    assert weights["model.embed_tokens.weight"].shape == (32, 64)


def _make_bnb_nf4_tensors(N: int, K: int):
    """Build synthetic BnB NF4 packed codes and absmax for a [N, K] weight."""
    GROUP_K = 64
    # NF4 codes packed as [N//2, K] uint8 (2 codes per byte, flat pairs)
    codes = np.random.randint(0, 16, size=(N // 2, K), dtype=np.uint8)
    lo = np.random.randint(0, 16, size=(N // 2, K), dtype=np.uint8)
    hi = np.random.randint(0, 16, size=(N // 2, K), dtype=np.uint8)
    codes = (lo & 0xF) | ((hi & 0xF) << 4)
    # Absmax: one F32 per block of 64 elements, flat order
    num_blocks = N * K // GROUP_K
    absmax = np.random.rand(num_blocks).astype(np.float32) + 0.1
    return codes, absmax


def test_load_bnb_nf4_old_format(wgpu_device, tmp_path):
    """Old BnB format: weight (U8) + weight_quantized_stats (F32)."""
    from vllm_webgpu.quant.weight_loader import load_safetensors_weights

    N, K = 8, 64
    codes, absmax = _make_bnb_nf4_tensors(N, K)
    base = "model.layers.0.self_attn.q_proj"
    tensors = {
        f"{base}.weight": codes,
        f"{base}.weight_quantized_stats": absmax,
    }
    st_path = make_fake_safetensors(tmp_path, tensors)
    weights = load_safetensors_weights(str(st_path), wgpu_device.wgpu_device)

    w_key = f"{base}.weight"
    s_key = f"{base}.weight.scales"
    assert w_key in weights, f"{w_key} not in weights"
    assert s_key in weights, f"{s_key} not in weights"
    assert weights[w_key].dtype == "u8"
    assert weights[w_key].shape == (N, K // 2)
    assert weights[s_key].dtype == "f32"
    assert weights[s_key].shape == (N, K // 64)

    qmeta = weights.get("__quant_meta__", {})
    assert qmeta.get(base, {}).get("fmt") == "nf4_gpu"
    assert qmeta.get(base, {}).get("group_size") == 64


def test_load_bnb_nf4_new_format(wgpu_device, tmp_path):
    """New BnB format: weight (U8) + weight.absmax (F32)."""
    from vllm_webgpu.quant.weight_loader import load_safetensors_weights

    N, K = 16, 128
    codes, absmax = _make_bnb_nf4_tensors(N, K)
    base = "model.layers.0.self_attn.q_proj"
    tensors = {
        f"{base}.weight": codes,
        f"{base}.weight.absmax": absmax,
    }
    st_path = make_fake_safetensors(tmp_path, tensors)
    weights = load_safetensors_weights(str(st_path), wgpu_device.wgpu_device)

    w_key = f"{base}.weight"
    s_key = f"{base}.weight.scales"
    assert w_key in weights
    assert s_key in weights
    assert weights[w_key].dtype == "u8"
    assert weights[w_key].shape == (N, K // 2)
    assert weights[s_key].dtype == "f32"
    assert weights[s_key].shape == (N, K // 64)

    qmeta = weights.get("__quant_meta__", {})
    assert qmeta.get(base, {}).get("fmt") == "nf4_gpu"


def test_load_bnb_nf4_nibble_layout(wgpu_device, tmp_path):
    """Verify the weight reshape: BnB [N//2, K] → shader [N, K//2], nibbles preserved."""
    from vllm_webgpu.quant.weight_loader import load_safetensors_weights

    N, K = 4, 64
    # Craft known pattern: row r in [N//2, K] has byte value r+1 everywhere.
    codes = np.zeros((N // 2, K), dtype=np.uint8)
    for r in range(N // 2):
        codes[r, :] = r + 1
    absmax = np.ones(N * K // 64, dtype=np.float32)

    base = "model.layers.0.mlp.down_proj"
    tensors = {
        f"{base}.weight": codes,
        f"{base}.weight_quantized_stats": absmax,
    }
    st_path = make_fake_safetensors(tmp_path, tensors)
    weights = load_safetensors_weights(str(st_path), wgpu_device.wgpu_device)

    # Expected shader layout: [N, K//2]
    # BnB row 0: first K//2 bytes → shader row 0; last K//2 bytes → shader row 1
    # BnB row 1: first K//2 bytes → shader row 2; last K//2 bytes → shader row 3
    buf = weights[f"{base}.weight"]
    arr = buf.to_numpy().view(np.uint8).reshape(N, K // 2)
    assert arr.shape == (N, K // 2)
    # Shader row 0 comes from BnB row 0, cols 0..K//2-1 (value=1)
    assert np.all(arr[0] == 1)
    # Shader row 1 comes from BnB row 0, cols K//2..K-1 (value=1)
    assert np.all(arr[1] == 1)
    # Shader row 2 comes from BnB row 1, cols 0..K//2-1 (value=2)
    assert np.all(arr[2] == 2)
    # Shader row 3 comes from BnB row 1, cols K//2..K-1 (value=2)
    assert np.all(arr[3] == 2)


def test_load_fp8_per_tensor_scale(wgpu_device, tmp_path):
    """FP8 with a scalar per-tensor scale is uploaded as fp8_gpu with global_scale."""
    from vllm_webgpu.quant.weight_loader import load_safetensors_weights

    N, K = 8, 16
    base = "model.layers.0.self_attn.q_proj"
    w_fp8 = np.zeros((N, K), dtype=np.uint8)  # all-zero FP8 bytes
    scale = np.array([0.125], dtype=np.float32)

    st_path = make_fake_safetensors_raw(tmp_path, [
        (f"{base}.weight", "F8_E4M3", (N, K), w_fp8),
        (f"{base}.weight_scale", "F32", (1,), scale),
    ])
    weights = load_safetensors_weights(str(st_path), wgpu_device.wgpu_device)

    w_key = f"{base}.weight"
    assert w_key in weights
    assert weights[w_key].dtype == "u8"

    qmeta = weights.get("__quant_meta__", {})
    entry = qmeta.get(base, {})
    assert entry.get("fmt") == "fp8_gpu"
    assert abs(entry.get("global_scale", 0) - 0.125) < 1e-5
    # Per-tensor: no separate scales buffer, no group_size=1
    assert f"{w_key}.scales" not in weights
    assert entry.get("group_size") != 1


def test_load_fp8_per_channel_scale(wgpu_device, tmp_path):
    """FP8 with a per-channel scale tensor uploads a .scales buffer and sets group_size=1."""
    from vllm_webgpu.quant.weight_loader import load_safetensors_weights

    N, K = 8, 16
    base = "model.layers.0.self_attn.q_proj"
    w_fp8 = np.zeros((N, K), dtype=np.uint8)
    # Per-channel: one scale per output row
    scale_per_ch = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    assert scale_per_ch.shape == (N,)

    st_path = make_fake_safetensors_raw(tmp_path, [
        (f"{base}.weight", "F8_E4M3", (N, K), w_fp8),
        (f"{base}.weight_scale", "F32", (N,), scale_per_ch),
    ])
    weights = load_safetensors_weights(str(st_path), wgpu_device.wgpu_device)

    w_key = f"{base}.weight"
    s_key = f"{w_key}.scales"
    assert w_key in weights, f"{w_key} not in weights"
    assert s_key in weights, f"{s_key} not in weights (per-channel scales must be uploaded)"
    assert weights[w_key].dtype == "u8"
    assert weights[s_key].dtype == "f32"
    assert weights[s_key].shape == (N,)

    qmeta = weights.get("__quant_meta__", {})
    entry = qmeta.get(base, {})
    assert entry.get("fmt") == "fp8_gpu"
    assert entry.get("group_size") == 1, "group_size must be 1 for per-channel FP8"
    assert entry.get("global_scale") == 1.0

    # Verify the uploaded scale values match the input (exact for f32)
    uploaded = weights[s_key].to_numpy().view(np.float32)
    np.testing.assert_allclose(uploaded[:N], scale_per_ch, rtol=1e-6, atol=1e-6)


def test_load_int8_per_channel_scale(wgpu_device, tmp_path):
    """Int8 weight with companion weight_scale uploads a .scales buffer and sets group_size=1."""
    from vllm_webgpu.quant.weight_loader import load_safetensors_weights

    N, K = 8, 16
    base = "model.layers.0.self_attn.q_proj"
    # Signed int8 weight values covering the full range
    w_i8 = np.array(
        [i % 256 - 128 for i in range(N * K)], dtype=np.int8
    ).reshape(N, K)
    scale_per_ch = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
                             dtype=np.float32)
    assert scale_per_ch.shape == (N,)

    st_path = make_fake_safetensors_raw(tmp_path, [
        (f"{base}.weight", "I8", (N, K), w_i8.view(np.uint8)),
        (f"{base}.weight_scale", "F32", (N,), scale_per_ch),
    ])
    weights = load_safetensors_weights(str(st_path), wgpu_device.wgpu_device)

    w_key = f"{base}.weight"
    s_key = f"{w_key}.scales"
    assert w_key in weights, f"{w_key} not in weights"
    assert s_key in weights, f"{s_key} not in weights (per-channel int8 scale must be uploaded)"
    assert weights[w_key].dtype == "u8"
    assert weights[s_key].dtype == "f32"
    assert weights[s_key].shape == (N,)

    qmeta = weights.get("__quant_meta__", {})
    entry = qmeta.get(base, {})
    assert entry.get("fmt") == "int8_gpu", f"expected int8_gpu, got {entry.get('fmt')}"
    assert entry.get("group_size") == 1, "group_size must be 1 for per-channel int8"

    # Verify scale values round-trip correctly through f32 (exact)
    uploaded = weights[s_key].to_numpy().view(np.float32)
    np.testing.assert_allclose(uploaded[:N], scale_per_ch, rtol=1e-6, atol=1e-6)


def test_load_int8_no_scale(wgpu_device, tmp_path):
    """Int8 weight without a scale tensor still uploads correctly (no crash)."""
    from vllm_webgpu.quant.weight_loader import load_safetensors_weights

    N, K = 4, 8
    base = "model.layers.0.mlp.down_proj"
    w_i8 = np.zeros((N, K), dtype=np.int8)

    st_path = make_fake_safetensors_raw(tmp_path, [
        (f"{base}.weight", "I8", (N, K), w_i8.view(np.uint8)),
    ])
    weights = load_safetensors_weights(str(st_path), wgpu_device.wgpu_device)

    w_key = f"{base}.weight"
    assert w_key in weights
    assert weights[w_key].dtype == "u8"
    # No companion scale: .scales should not be present, group_size not set
    assert f"{w_key}.scales" not in weights

    qmeta = weights.get("__quant_meta__", {})
    entry = qmeta.get(base, {})
    assert entry.get("fmt") == "int8_gpu"
    assert entry.get("group_size") is None


def _make_gptq_tensors(K: int, N: int, G: int):
    """Build minimal GPTQ int4 tensors. qweight is (K//8, N), scales is (G, N), qzeros is (G, N//8)."""
    qweight = np.random.randint(0, 2**31, size=(K // 8, N), dtype=np.int32)
    scales = np.random.randn(G, N).astype(np.float16)
    qzeros = np.zeros((G, N // 8), dtype=np.int32)
    return qweight, scales, qzeros


def _make_awq_tensors(K: int, N: int, G: int):
    """Build minimal AWQ int4 tensors. qweight is (K, N//8), scales is (G, N), qzeros is (G, N//8).

    qzeros are packed int32 values where every nibble is 8 (zero_point=8), the
    standard symmetric AWQ encoding. All-zero qzeros encode zero_point=0 and are
    routed to CPU dequant; they are not valid test data for the GPU AWQ path.
    """
    qweight = np.random.randint(0, 2**31, size=(K, N // 8), dtype=np.int32)
    scales = np.random.randn(G, N).astype(np.float16)
    # 0x88888888 as signed int32 = -2004318072: each nibble is 0x8 = 8, meaning zero_point=8.
    qzeros = np.full((G, N // 8), fill_value=-2004318072, dtype=np.int32)
    return qweight, scales, qzeros


def test_gptq_with_qzeros_not_misidentified_as_awq(wgpu_device, tmp_path):
    """Asymmetric GPTQ (has qzeros) must not be routed through the AWQ dequant path.

    Before the fix, any qweight file containing .qzeros keys was labelled AWQ.
    GPTQ qweight shape is (K//8, N); AWQ is (K, N//8). The shape check is definitive.

    When correctly identified as GPTQ, the weight is CPU-dequantized via _dequant_gptq
    and uploaded as plain F16 with shape (N, K). The CPU path does not write a
    __quant_meta__ entry, but the weight's presence and shape confirm correct routing.
    If the old AWQ path were taken, _dequant_awq would raise a shape mismatch (its K
    and N calculations produce incompatible dimensions) and the weight would be absent.
    """
    from vllm_webgpu.quant.weight_loader import load_safetensors_weights

    K, N, G = 64, 256, 2   # GPTQ: qweight (8, 256), scales (2, 256), qzeros (2, 32)
    base = "model.layers.0.self_attn.q_proj"
    qweight, scales, qzeros = _make_gptq_tensors(K, N, G)

    st_path = make_fake_safetensors(tmp_path, {
        f"{base}.qweight": qweight,
        f"{base}.scales": scales,
        f"{base}.qzeros": qzeros,
    })
    weights = load_safetensors_weights(str(st_path), wgpu_device.wgpu_device)

    w_key = f"{base}.weight"
    # Weight must be present (AWQ path would throw a shape error and silently drop it).
    assert w_key in weights, (
        f"{w_key} not found — GPTQ with qzeros was likely misrouted through AWQ dequant; "
        f"keys={list(weights.keys())}"
    )
    # Shape after CPU GPTQ dequant is (N, K) = (256, 64). AWQ dequant with these
    # GPTQ shapes raises an exception (incompatible qzeros dims), so the presence
    # and correct shape together confirm the right path was taken.
    assert weights[w_key].shape == (N, K), (
        f"Expected (N={N}, K={K}), got {weights[w_key].shape}"
    )
    assert weights[w_key].dtype == "f16"
    # Must not be tagged as AWQ in quant_meta (GPU AWQ path is shape-gated).
    qmeta = weights.get("__quant_meta__", {})
    entry = qmeta.get(base, {})
    assert entry.get("fmt") not in ("awq_sym", "awq"), (
        f"Weight was incorrectly tagged as AWQ in quant_meta: {entry}"
    )


def test_awq_identified_by_shape(wgpu_device, tmp_path):
    """AWQ format is correctly detected via qweight/scales shape cross-reference."""
    from vllm_webgpu.quant.weight_loader import load_safetensors_weights

    K, N, G = 256, 64, 2   # AWQ: qweight (256, 8), scales (2, 64), qzeros (2, 8)
    base = "model.layers.0.self_attn.q_proj"
    qweight, scales, qzeros = _make_awq_tensors(K, N, G)

    st_path = make_fake_safetensors(tmp_path, {
        f"{base}.qweight": qweight,
        f"{base}.scales": scales,
        f"{base}.qzeros": qzeros,
    })
    weights = load_safetensors_weights(str(st_path), wgpu_device.wgpu_device)

    w_key = f"{base}.weight"
    assert w_key in weights, f"{w_key} not found; keys={list(weights.keys())}"
    qmeta = weights.get("__quant_meta__", {})
    entry = qmeta.get(base, {})
    assert entry.get("fmt") in ("awq_sym", "awq"), (
        f"AWQ model not identified correctly: fmt={entry.get('fmt')!r}"
    )


# --- _detect_mx_quant parity tests ----------------------------------------
# _detect_mx_quant duplicates ModelOptFp8Config._extract_modelopt_quant_algo
# from vllm/model_executor/layers/quantization/modelopt.py because that class
# has top-level CUDA imports that crash on WebGPU. These tests guard against
# drift with vLLM's parsing logic. On each vLLM bump, diff
# _extract_modelopt_quant_algo against the inline block in _detect_mx_quant
# and update if the hf_quant_config.json parsing changes.

def test_detect_mx_quant_mxfp4(tmp_path):
    """MXFP4 quant_algo recognized from hf_quant_config.json."""
    import json
    from vllm_webgpu.quant.weight_loader import _detect_mx_quant
    hf_quant = tmp_path / "hf_quant_config.json"
    hf_quant.write_text(json.dumps({
        "quant_method": "modelopt",
        "quantization": {"quant_algo": "MXFP4"},
    }))
    assert _detect_mx_quant(tmp_path) == "mxfp4"


def test_detect_mx_quant_mxfp8(tmp_path):
    """MXFP8 quant_algo recognized from hf_quant_config.json."""
    import json
    from vllm_webgpu.quant.weight_loader import _detect_mx_quant
    hf_quant = tmp_path / "hf_quant_config.json"
    hf_quant.write_text(json.dumps({
        "quant_method": "modelopt",
        "quantization": {"quant_algo": "MXFP8"},
    }))
    assert _detect_mx_quant(tmp_path) == "mxfp8"


def test_detect_mx_quant_top_level_algo(tmp_path):
    """quant_algo at the top level (no nested quantization key) is recognized."""
    import json
    from vllm_webgpu.quant.weight_loader import _detect_mx_quant
    hf_quant = tmp_path / "hf_quant_config.json"
    hf_quant.write_text(json.dumps({
        "quant_method": "modelopt",
        "quant_algo": "MXFP4",
    }))
    assert _detect_mx_quant(tmp_path) == "mxfp4"


def test_detect_mx_quant_non_dict_quantization(tmp_path):
    """quantization key present but not a dict: quant_algo falls back to None.

    vLLM's _extract_modelopt_quant_algo returns None early in this case.
    Our inline copy sets quant_algo=None and falls through to the config.json
    fallback path. Both produce no MX detection when no config.json is present.
    If this test fails after a vLLM bump, re-audit the inline block.
    """
    import json
    from vllm_webgpu.quant.weight_loader import _detect_mx_quant
    hf_quant = tmp_path / "hf_quant_config.json"
    hf_quant.write_text(json.dumps({
        "quant_method": "modelopt",
        "quantization": "not_a_dict",
    }))
    # Neither mxfp4 nor mxfp8 should be detected.
    assert _detect_mx_quant(tmp_path) == ""


def test_detect_mx_quant_null_quantization(tmp_path):
    """quantization key present with null value: key-presence vs falsy check parity.

    vLLM uses 'if "quantization" in hf_quant_cfg:' (key-presence), so a JSON
    {"quantization": null} enters the quantization branch and returns None for
    quant_algo. Our inline copy must use the same key-presence check so behavior
    is identical. This test catches any regression to cfg.get("quantization") which
    would silently skip the null value and fall through to the top-level quant_algo.
    If this test fails after a vLLM bump, re-audit the inline block against
    ModelOptFp8Config._extract_modelopt_quant_algo in modelopt.py.
    """
    import json
    from vllm_webgpu.quant.weight_loader import _detect_mx_quant
    hf_quant = tmp_path / "hf_quant_config.json"
    # "quantization": null with a top-level quant_algo that would be detected if
    # the null value were skipped (falsy path). The key-presence path enters the
    # quantization branch, gets quant_algo=None from a non-dict, and returns "".
    hf_quant.write_text(json.dumps({
        "quant_method": "modelopt",
        "quantization": None,
        "quant_algo": "MXFP4",
    }))
    # With key-presence check: quantization branch entered, quant_algo=None -> "".
    # With falsy check (bug): quantization branch skipped, top-level quant_algo="MXFP4" -> "mxfp4".
    assert _detect_mx_quant(tmp_path) == ""


# --- ct_pack_int4 (Gemma4 QAT W4A16) tests -----------------------------------

def test_ct_pack_int4_weight_packed(wgpu_device, tmp_path):
    """compressed-tensors W4A16 via .weight_packed (canonical Gemma4 QAT layout).

    Weight: {base}.weight_packed  [N, K//8] I32   (8 nibbles per int32)
    Scale:  {base}.weight_scale   [N, G]    F16   -> transposed to [G, N] on upload
    Expect: weight uploaded as i32 [N, K//8], scales uploaded as f32 [G, N],
            __quant_meta__ records fmt='gptq_sym' with correct group_size.
    """
    from vllm_webgpu.quant.weight_loader import load_safetensors_weights

    N, K, group_size = 32, 64, 32
    G = K // group_size  # 2
    base = "model.layers.0.self_attn.q_proj"

    qw = np.random.randint(0, 2**31 - 1, size=(N, K // 8), dtype=np.int32)
    sc = np.random.randn(N, G).astype(np.float16)

    st_path = make_fake_safetensors(tmp_path, {
        f"{base}.weight_packed": qw,
        f"{base}.weight_scale": sc,
    })

    ct_meta = {"__global__": {"fmt": "gptq_gpu", "group_size": group_size}}
    weights = load_safetensors_weights(
        str(st_path), wgpu_device.wgpu_device,
        ct_meta=ct_meta,
    )

    w_key = f"{base}.weight"
    sc_key = f"{base}.weight.scales"

    assert w_key in weights, f"{w_key} missing; keys={list(weights.keys())}"
    assert sc_key in weights, f"{sc_key} missing; keys={list(weights.keys())}"

    assert weights[w_key].dtype == "i32", f"weight dtype={weights[w_key].dtype}"
    assert weights[w_key].shape == (N, K // 8), (
        f"weight shape={weights[w_key].shape}, expected ({N}, {K // 8})"
    )

    # Scale must be transposed from [N, G] to [G, N] for the gptq_sym shader.
    assert weights[sc_key].shape == (G, N), (
        f"scale shape={weights[sc_key].shape}, expected ({G}, {N}); "
        "ct_pack_int4 must transpose [N,G] to [G,N] before upload"
    )
    assert weights[sc_key].dtype == "f32", f"scale dtype={weights[sc_key].dtype}"

    qmeta = weights.get("__quant_meta__", {})
    entry = qmeta.get(base, {})
    assert entry.get("fmt") == "gptq_sym", (
        f"quant_meta fmt={entry.get('fmt')!r}, expected 'gptq_sym'"
    )
    assert entry.get("group_size") == group_size, (
        f"quant_meta group_size={entry.get('group_size')!r}, expected {group_size}"
    )


def test_ct_pack_int4_weight_key_fallback(wgpu_device, tmp_path):
    """compressed-tensors W4A16 via .weight I32 (fallback key naming convention).

    Some checkpoints store the packed int4 weight as {base}.weight (I32) rather
    than {base}.weight_packed. The loader must fall back to .weight when no
    .weight_packed key is present.
    """
    from vllm_webgpu.quant.weight_loader import load_safetensors_weights

    N, K, group_size = 16, 32, 16
    G = K // group_size  # 2
    base = "model.layers.0.mlp.gate_proj"

    qw = np.random.randint(0, 2**31 - 1, size=(N, K // 8), dtype=np.int32)
    sc = np.random.randn(N, G).astype(np.float16)

    st_path = make_fake_safetensors(tmp_path, {
        f"{base}.weight": qw,
        f"{base}.weight_scale": sc,
    })

    ct_meta = {"__global__": {"fmt": "gptq_gpu", "group_size": group_size}}
    weights = load_safetensors_weights(
        str(st_path), wgpu_device.wgpu_device,
        ct_meta=ct_meta,
    )

    w_key = f"{base}.weight"
    sc_key = f"{base}.weight.scales"

    assert w_key in weights, f"{w_key} missing; keys={list(weights.keys())}"
    assert sc_key in weights, f"{sc_key} missing; keys={list(weights.keys())}"
    assert weights[sc_key].shape == (G, N), (
        f"scale shape={weights[sc_key].shape}, expected ({G}, {N})"
    )
    qmeta = weights.get("__quant_meta__", {})
    assert qmeta.get(base, {}).get("fmt") == "gptq_sym"


def test_ct_pack_int4_weight_packed_wins_over_weight(wgpu_device, tmp_path):
    """When both .weight_packed and .weight are present, .weight_packed wins.

    The two-pass selection in the ct_pack_int4 loader always prefers .weight_packed
    regardless of header iteration order.
    """
    from vllm_webgpu.quant.weight_loader import load_safetensors_weights

    N, K, group_size = 16, 32, 16
    G = K // group_size
    base = "model.layers.0.mlp.up_proj"

    qw_packed = np.random.randint(0, 2**31 - 1, size=(N, K // 8), dtype=np.int32)
    qw_weight = np.random.randint(0, 2**31 - 1, size=(N, K // 8), dtype=np.int32)
    sc = np.random.randn(N, G).astype(np.float16)

    st_path = make_fake_safetensors(tmp_path, {
        f"{base}.weight": qw_weight,
        f"{base}.weight_packed": qw_packed,
        f"{base}.weight_scale": sc,
    })

    ct_meta = {"__global__": {"fmt": "gptq_gpu", "group_size": group_size}}
    weights = load_safetensors_weights(
        str(st_path), wgpu_device.wgpu_device,
        ct_meta=ct_meta,
    )

    w_key = f"{base}.weight"
    assert w_key in weights, f"{w_key} missing; keys={list(weights.keys())}"
    # Verify the uploaded data matches qw_packed (not qw_weight).
    buf = weights[w_key].buf
    raw = wgpu_device.wgpu_device.queue.read_buffer(buf)
    uploaded = np.frombuffer(raw, dtype=np.int32).reshape(N, K // 8)
    assert np.array_equal(uploaded, qw_packed), (
        ".weight_packed data was not preferred over .weight data"
    )
