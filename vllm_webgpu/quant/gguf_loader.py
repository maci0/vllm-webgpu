from __future__ import annotations
import logging
import struct
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_GGUF_MAGIC = b"GGUF"


def detect_weight_format(path: str) -> str:
    p = Path(path)
    if p.suffix == ".gguf":
        return "gguf"
    if p.suffix in {".safetensors", ".bin"}:
        return "safetensors"
    # Try magic bytes
    with open(p, "rb") as f:
        magic = f.read(4)
    if magic == _GGUF_MAGIC:
        return "gguf"
    return "safetensors"


def load_safetensors_weights(path: str, wgpu_device) -> dict:
    """Load safetensors weights, cast bf16->f16, upload to GPU."""
    import json
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer

    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_raw = f.read(header_len)
        data_start = 8 + header_len
        header = json.loads(header_raw)
        f.seek(data_start)
        raw_data = f.read()

    weights: dict = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype_str = meta["dtype"]
        shape = tuple(meta["shape"])
        start, end = meta["data_offsets"]
        raw = raw_data[start:end]

        if dtype_str == "F16":
            arr = np.frombuffer(raw, dtype=np.float16).reshape(shape)
        elif dtype_str == "BF16":
            # Cast bf16 -> f32 -> f16 via view trick.
            # BF16 max magnitude is ~3.4e38; F16 max is 65504.
            # Clip to [-65504, 65504] before casting to avoid silent ±inf in weights.
            u16 = np.frombuffer(raw, dtype=np.uint16)
            f32 = (u16.astype(np.uint32) << 16).view(np.float32)
            f32 = np.clip(f32, -65504.0, 65504.0)
            arr = f32.reshape(shape).astype(np.float16)
        elif dtype_str == "F32":
            arr = np.frombuffer(raw, dtype=np.float32).reshape(shape).astype(np.float16)
        else:
            logger.warning("Unsupported dtype %s for tensor %s, skipping", dtype_str, name)
            continue

        weights[name] = WebGPUBuffer.from_numpy(wgpu_device, np.ascontiguousarray(arr))

    logger.info("Loaded %d tensors from %s", len(weights), path)
    return weights


def load_gguf_weights(path: str, wgpu_device) -> dict:
    """Load GGUF Q4_K_M weights. Raw quantized blocks uploaded as u8."""
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    import wgpu as wgpu_lib

    try:
        import gguf
    except ImportError as e:
        raise ImportError("Install the 'gguf' package to load GGUF files.") from e

    reader = gguf.GGUFReader(path)
    weights: dict = {}

    for tensor in reader.tensors:
        name = tensor.name
        data = tensor.data   # numpy array of raw bytes
        # Upload raw quantized blocks as u8; matmul_quant.wgsl handles dequant
        arr = np.frombuffer(data, dtype=np.uint8)
        # Include COPY_SRC so to_numpy() works on weight buffers (e.g. for debugging).
        weights[name] = WebGPUBuffer.from_numpy(
            wgpu_device,
            arr,
            usage=wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_DST | wgpu_lib.BufferUsage.COPY_SRC,
        )

    logger.info("Loaded %d GGUF tensors from %s", len(weights), path)
    return weights
