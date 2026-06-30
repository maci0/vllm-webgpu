import numpy as np
import pytest
from pathlib import Path
import tempfile
import struct


def make_fake_safetensors(tmp_path: Path, tensors: dict) -> Path:
    """Write a minimal safetensors file for testing."""
    import json
    metadata = {}
    offset = 0
    data_parts = []
    for name, arr in tensors.items():
        dtype_map = {np.float16: "F16", np.float32: "F32"}
        dtype_str = dtype_map[arr.dtype.type]
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
    out = tmp_path / "model.safetensors"
    out.write_bytes(header_len + header_bytes + b"".join(data_parts))
    return out


def test_detect_format_safetensors(tmp_path):
    from vllm_webgpu.quant.gguf_loader import detect_weight_format
    f = tmp_path / "model.safetensors"
    f.write_bytes(b"\x00" * 16)
    assert detect_weight_format(str(f)) == "safetensors"


def test_detect_format_gguf(tmp_path):
    from vllm_webgpu.quant.gguf_loader import detect_weight_format
    f = tmp_path / "model.gguf"
    f.write_bytes(b"GGUF" + b"\x00" * 12)
    assert detect_weight_format(str(f)) == "gguf"


def test_load_safetensors(wgpu_device, tmp_path):
    from vllm_webgpu.quant.gguf_loader import load_safetensors_weights
    tensors = {
        "model.embed_tokens.weight": np.random.randn(32, 64).astype(np.float16),
        "model.layers.0.self_attn.q_proj.weight": np.random.randn(64, 64).astype(np.float16),
    }
    st_path = make_fake_safetensors(tmp_path, tensors)
    weights = load_safetensors_weights(str(st_path), wgpu_device.wgpu_device)
    assert "model.embed_tokens.weight" in weights
    assert weights["model.embed_tokens.weight"].dtype == "f16"
    assert weights["model.embed_tokens.weight"].shape == (32, 64)
