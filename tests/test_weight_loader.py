import numpy as np
import pytest
from pathlib import Path
import tempfile
import struct


_DTYPE_MAP = {np.float16: "F16", np.float32: "F32", np.uint8: "U8", np.int32: "I32"}


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
    assert detect_weight_format(str(f)) == "safetensors"


def test_detect_format_gguf(tmp_path):
    from vllm_webgpu.quant.weight_loader import detect_weight_format
    f = tmp_path / "model.gguf"
    f.write_bytes(b"GGUF" + b"\x00" * 12)
    assert detect_weight_format(str(f)) == "gguf"


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
    assert weights[s_key].dtype == "f16"
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
    assert weights[s_key].dtype == "f16"
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
