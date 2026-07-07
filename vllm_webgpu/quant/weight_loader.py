from __future__ import annotations
import logging
import struct
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_GGUF_MAGIC = b"GGUF"


def _is_mlx_quantized_dir(p: Path) -> bool:
    """Return True if directory contains MLX affine int4 weights (has .biases keys)."""
    import json
    index_path = p / "model.safetensors.index.json"
    try:
        with open(index_path) as f:
            index = json.load(f)
        return any(k.endswith(".biases") for k in index.get("weight_map", {}))
    except Exception:
        return False


def detect_weight_format(path: str) -> str:
    p = Path(path)
    if p.is_dir():
        if (p / "model.safetensors.index.json").exists():
            if _is_mlx_quantized_dir(p):
                return "mlx_int4"
            return "safetensors_sharded"
        if (p / "model.safetensors").exists():
            return "safetensors"
        # Fall through to magic-byte check if single file found
        return "safetensors"
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


def load_safetensors_weights_sharded(model_dir: str, wgpu_device) -> dict:
    """Load multi-shard safetensors from a directory with model.safetensors.index.json."""
    import json
    index_path = Path(model_dir) / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    shard_files = sorted(set(index["weight_map"].values()))
    weights: dict = {}
    # IMPORTANT: keep shard_weights as a local variable (not inline with update()).
    # Inlining as weights.update(load_safetensors_weights(...)) causes Python's GC
    # to drop the temporary dict before wgpu finishes using the mapped GPU buffers,
    # resulting in zeroed buffer contents. The local variable keeps the dict alive.
    # Detect multimodal models with nested language model prefix.
    # Gemma3 multimodal: keys start with "language_model." → strip prefix
    # Qwen3.5 multimodal: keys start with "model.language_model." → remap to "model."
    weight_map = index.get("weight_map", {})
    is_gemma_mm  = any(k.startswith("language_model.") for k in weight_map)
    is_qwen35_mm = any(k.startswith("model.language_model.") for k in weight_map)
    is_multimodal = is_gemma_mm or is_qwen35_mm
    if is_multimodal:
        style = "Gemma3" if is_gemma_mm else "Qwen3.5"
        logger.info("Multimodal model detected (%s style); remapping language_model prefix", style)

    # Detect compressed-tensors quantization format before loading shards.
    # The I8 and F8_E4M3 dtypes are already handled per-shard inside load_safetensors_weights,
    # but we apply comprehensive quant_meta here for any layers not caught by dtype detection.
    ct_meta = detect_compressed_tensors_fmt(Path(model_dir) / "config.json")
    if ct_meta:
        logger.info("compressed-tensors format detected: %s", ct_meta.get("__global__", {}))

    for shard in shard_files:
        shard_path = str(Path(model_dir) / shard)
        logger.info("Loading shard %s", shard)
        shard_weights = load_safetensors_weights(shard_path, wgpu_device)

        # Commit all pending write_buffer operations by submitting a dummy command encoder.
        # queue.write_buffer() is only committed before the NEXT queue.submit(), not by
        # on_submitted_work_done_sync() alone. Without this, 24GB of accumulated writes
        # may be committed simultaneously at the first real submit, causing Metal to
        # silently drop some writes (embedding buffer shows zeros after readback).
        wgpu_device.queue.submit([wgpu_device.create_command_encoder().finish()])
        wgpu_device.queue.on_submitted_work_done_sync()

        weights.update(shard_weights)

    if is_multimodal:
        # Add remapped keys WITHOUT removing originals — freeing non-LM GPU buffers
        # causes Metal memory corruption on adjacent embeddings.
        remapped = {}
        for k, v in weights.items():
            if is_gemma_mm and k.startswith("language_model."):
                # "language_model.model.layers.0.X" → "model.layers.0.X"
                remapped[k[len("language_model."):]] = v
            elif is_qwen35_mm and k.startswith("model.language_model."):
                # "model.language_model.layers.0.X" → "model.layers.0.X"
                remapped["model." + k[len("model.language_model."):]] = v
        weights.update(remapped)
        logger.info("Added %d remapped language_model keys", len(remapped))

    # Apply compressed-tensors quant_meta to any weight layers not already tagged
    # (I8 dtype handling in _upload_plain already sets fmt="int8_gpu" for those layers).
    if ct_meta and "__global__" in ct_meta:
        global_ct = ct_meta["__global__"]
        qmeta = weights.setdefault("__quant_meta__", {})
        applied = 0
        for wkey in list(weights.keys()):
            if wkey.endswith(".weight") and not wkey.startswith("__"):
                base = wkey[:-len(".weight")]
                if base not in qmeta:
                    entry: dict = {"fmt": global_ct["fmt"]}
                    if global_ct.get("group_size") is not None:
                        entry["group_size"] = global_ct["group_size"]
                    qmeta[base] = entry
                    applied += 1
        if applied:
            logger.info("compressed-tensors: applied quant_meta to %d weight layers", applied)

    logger.info("Loaded %d tensors from %d shards in %s", len(weights), len(shard_files), model_dir)
    return weights


def _fp8_e4m3_to_f32(data: np.ndarray) -> np.ndarray:
    """Convert FP8 E4M3 (NVidia/OCP format, exponent bias=7) to float32.

    Format: bit7=sign, bits[6:3]=exponent(bias=7), bits[2:0]=mantissa/8
    Normal: (-1)^s * 2^(exp-7) * (1 + mant/8)  for exp in 1..14
    Denorm: (-1)^s * 2^-6 * (mant/8)             for exp == 0
    NaN:    exp == 15 (no Inf in E4M3)
    """
    # Build 256-entry lookup table once
    LUT = np.zeros(256, dtype=np.float32)
    for i in range(256):
        sign = -1.0 if (i >> 7) else 1.0
        exp = (i >> 3) & 0xF
        mant = i & 0x7
        if exp == 0:
            LUT[i] = sign * (2.0 ** -6) * (mant / 8.0)
        elif exp == 15:
            LUT[i] = float('nan')
        else:
            LUT[i] = sign * (2.0 ** (exp - 7)) * (1.0 + mant / 8.0)
    return LUT[data.ravel().view(np.uint8)].reshape(data.shape)


# NV FP4 E2M1 value table (index = 4-bit code, value = float)
_FP4_LUT = np.array([
    0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=np.float32)


def _dequant_nvfp4(weight_packed: np.ndarray, weight_scale_fp8: np.ndarray,
                   weight_global_scale: float) -> np.ndarray:
    """Dequantize NVFP4 (NVidia FP4) weights to float16.

    NVFP4 packs two FP4 E2M1 values per U8 byte. Each block of 16 weight
    values along K shares one FP8 E4M3 scale. A global F32 scale is also applied.

    weight_packed:    (N, K//2) U8    — two FP4 per byte, lower nibble first
    weight_scale_fp8: (N, K//16) F8_E4M3 — one scale per 16 K-elements
    weight_global_scale: F32 scalar

    Output: (N, K) F16 ready for the matmul_quant shader.
    """
    N, Kh = weight_packed.shape
    K = Kh * 2

    # Unpack 2 FP4 nibbles per byte → (N, K) uint8 FP4 codes
    lo = (weight_packed & 0xF).astype(np.uint8)          # lower nibble (even k)
    hi = ((weight_packed >> 4) & 0xF).astype(np.uint8)   # upper nibble (odd k)
    fp4 = np.empty((N, K), dtype=np.uint8)
    fp4[:, 0::2] = lo
    fp4[:, 1::2] = hi

    # FP4 → F32 via lookup table
    w_f32 = _FP4_LUT[fp4]  # (N, K)

    # FP8 scales → F32: each covers K // num_scale_blocks K values
    scale_f32 = _fp8_e4m3_to_f32(weight_scale_fp8)      # (N, num_blocks)
    num_blocks = weight_scale_fp8.shape[1]
    block_size = K // num_blocks if num_blocks > 0 else K
    scale_exp = np.repeat(scale_f32, block_size, axis=1)  # (N, K)

    w = w_f32 * scale_exp * weight_global_scale  # (N, K)
    return np.ascontiguousarray(np.clip(w, -65504.0, 65504.0).astype(np.float16))


def _dequant_fp8(weight_fp8: np.ndarray, scale: "np.ndarray | float") -> np.ndarray:
    """Dequantize plain FP8 E4M3 weights to float16.

    weight_fp8: (N, K) stored as uint8 bytes (E4M3 encoding)
    scale: scalar F32 or (N, 1) per-channel F32 scale tensor
    Output: (N, K) F16 for the matmul_quant shader.
    """
    w_f32 = _fp8_e4m3_to_f32(weight_fp8)
    result = w_f32 * np.asarray(scale, dtype=np.float32)
    return np.ascontiguousarray(np.clip(result, -65504.0, 65504.0).astype(np.float16))


def _dequant_awq(qweight: np.ndarray, scales: np.ndarray, qzeros: np.ndarray) -> np.ndarray:
    """Dequantize AWQ int4 weights to float16.

    AWQ packs 8 int4 weights per int32 along the output (N) dimension,
    using nibble order [0,4,1,5,2,6,3,7] within each int32. Output is
    the original weight matrix (N, K) = (out_features, in_features) in F16.

    Args:
        qweight: (K, N//8) int32  — packed input dim × output dim
        scales:  (G, N)   float16 — per-group per-output-channel scales
        qzeros:  (G, N//8) int32  — packed zero-points (same nibble order)
    """
    K, N8 = qweight.shape
    N = N8 * 8
    G = scales.shape[0]
    group_size = K // G

    # AWQ nibble reorder: position i in int32 holds nibble at bit offset
    # [0, 16, 4, 20, 8, 24, 12, 28] = [0,4,1,5,2,6,3,7] * 4
    nibble_shifts = np.array([0, 16, 4, 20, 8, 24, 12, 28], dtype=np.int32)

    qw = qweight.astype(np.int32)            # (K, N//8)
    qz = qzeros.astype(np.int32)             # (G, N//8)
    sc = scales.astype(np.float32)           # (G, N)

    # Unpack 8 nibbles per int32 → (K, N) uint8
    w_int4 = np.empty((K, N), dtype=np.uint8)
    z_int4 = np.empty((G, N), dtype=np.uint8)
    for j in range(8):
        shift = nibble_shifts[j]
        w_int4[:, j::8] = (qw >> shift) & 0xF
        z_int4[:, j::8] = (qz >> shift) & 0xF

    # Expand scales/zeros to (K, N) shape
    sc_exp = sc[np.arange(K) // group_size]   # (K, N)
    z_exp  = z_int4[np.arange(K) // group_size].astype(np.float32)  # (K, N)

    # Dequantize: weight(K, N) then transpose to (N, K) for our shader
    w_f32 = sc_exp * (w_int4.astype(np.float32) - z_exp)  # (K, N)
    return np.ascontiguousarray(w_f32.T.astype(np.float16))  # (N, K)


def _dequant_gptq(qweight: np.ndarray, scales: np.ndarray, qzeros: np.ndarray,
                  g_idx: "np.ndarray | None" = None) -> np.ndarray:
    """Dequantize GPTQ int4 weights to float16.

    GPTQ packs 8 int4 weights per int32 along the input (K) dimension,
    using standard nibble order [0,1,2,3,4,5,6,7]. Output is (N, K) F16.

    Args:
        qweight: (K//8, N) int32
        scales:  (G, N)    float16
        qzeros:  (G, N//8) int32
        g_idx:   (K,) int32 optional group index per input dim (desc_act)
    """
    K8, N = qweight.shape
    K = K8 * 8
    G = scales.shape[0]
    group_size = K // G

    qw = qweight.astype(np.int32)  # (K//8, N)
    qz = qzeros.astype(np.int32)   # (G, N//8)
    sc = scales.astype(np.float32)  # (G, N)

    # Unpack 8 nibbles per int32 along K → (K, N)
    w_int4 = np.empty((K, N), dtype=np.uint8)
    for bit in range(8):
        w_int4[bit::8] = (qw >> (bit * 4)) & 0xF

    # Unpack zeros: (G, N//8) → (G, N)
    z_int4 = np.empty((G, N), dtype=np.uint8)
    for bit in range(8):
        z_int4[:, bit::8] = (qz >> (bit * 4)) & 0xF

    # Group index: which group each input dim belongs to
    if g_idx is not None:
        groups = g_idx.astype(np.int32)
    else:
        groups = np.arange(K, dtype=np.int32) // group_size

    sc_exp = sc[groups]             # (K, N)
    z_exp  = z_int4[groups].astype(np.float32)  # (K, N)

    w_f32 = sc_exp * (w_int4.astype(np.float32) - z_exp)  # (K, N)
    return np.ascontiguousarray(w_f32.T.astype(np.float16))  # (N, K)


def _is_awq_format(header: dict) -> bool:
    return any(k.endswith(".qweight") and "qzeros" in "\n".join(header) for k in header)


def _is_gptq_format(header: dict) -> bool:
    return any(k.endswith(".qweight") for k in header)


def _detect_mx_quant(model_dir: Path) -> str:
    """Detect MXFP4 or MXFP8 from config files in the model directory.

    Checks hf_quant_config.json (Nvidia/ModelOpt format) first, then
    config.json quantization_config.quant_type. Returns 'mxfp4', 'mxfp8', or ''.
    """
    import json
    hf_quant = model_dir / "hf_quant_config.json"
    if hf_quant.exists():
        try:
            with open(hf_quant) as f:
                cfg = json.load(f)
            algo = cfg.get("quant_algo", "")
            if "MXFP4" in algo:
                return "mxfp4"
            if "MXFP8" in algo:
                return "mxfp8"
        except Exception:
            pass
    config_json = model_dir / "config.json"
    if config_json.exists():
        try:
            with open(config_json) as f:
                cfg = json.load(f)
            qt = cfg.get("quantization_config", {}).get("quant_type", "")
            if qt.lower() == "mxfp4":
                return "mxfp4"
            if qt.lower() == "mxfp8":
                return "mxfp8"
        except Exception:
            pass
    return ""


def detect_compressed_tensors_fmt(config_path: "str | Path") -> dict:
    """Read config.json and return compressed-tensors quantization metadata.

    compressed-tensors models embed a quantization_config with config_groups that
    describes the actual format. Returns a dict with key '__global__' mapped to
    {use_quant, fmt, group_size} when detected, otherwise empty dict.

    Routing:
      8-bit int  + channel        -> USE_QUANT=7, fmt='int8_gpu'
      8-bit float + tensor/channel -> USE_QUANT=5, fmt='fp8_gpu'
      4-bit int  + group          -> USE_QUANT=3, fmt='gptq_gpu'
      other                       -> empty dict with a logged warning
    """
    import json
    p = Path(config_path)
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            config = json.load(f)
    except Exception:
        return {}
    quant_cfg = config.get("quantization_config", {})
    config_groups = quant_cfg.get("config_groups")
    if not config_groups:
        return {}
    first_group = next(iter(config_groups.values()), {})
    weights_desc = first_group.get("weights", {})
    if not weights_desc:
        return {}
    num_bits = int(weights_desc.get("num_bits", 8))
    wtype = str(weights_desc.get("type", "int")).lower()
    strategy = str(weights_desc.get("strategy", "channel")).lower()
    if num_bits == 8 and wtype == "int" and strategy == "channel":
        return {"__global__": {"use_quant": 7, "fmt": "int8_gpu", "group_size": None}}
    if num_bits == 8 and wtype in ("float", "fp8") and strategy in ("tensor", "channel"):
        return {"__global__": {"use_quant": 5, "fmt": "fp8_gpu", "group_size": None}}
    if num_bits == 4 and wtype == "int" and strategy == "group":
        group_size = int(weights_desc.get("group_size", 128))
        return {"__global__": {"use_quant": 3, "fmt": "gptq_gpu", "group_size": group_size}}
    logger.warning(
        "compressed-tensors: unsupported format (num_bits=%d, type=%s, strategy=%s), "
        "no quant_meta applied", num_bits, wtype, strategy)
    return {}


def load_safetensors_weights(path: str, wgpu_device) -> dict:
    """Load safetensors weights and upload to GPU as F16.

    Handles the following quantization formats (all dequantized on CPU):
    - BF16/F16/F32: direct cast to F16
    - AWQ int4: nibble-packed (K, N//8) with F16 scales and zero-points
    - GPTQ int4: nibble-packed (K//8, N) with F16 scales
    - FP8 E4M3: weight as F8_E4M3 + F32 per-tensor scale
    - NVFP4: weight_packed (U8 = 2 FP4/byte) + F8_E4M3 block scale + F32 global scale

    Uses queue.write_buffer (not mapped_at_creation) for reliable uploads.
    """
    import json
    import wgpu as wgpu_lib
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer

    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_raw = f.read(header_len)
        data_start = 8 + header_len
        header = json.loads(header_raw)
        f.seek(data_start)
        raw_data = f.read()

    usage = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST

    # Detect quantization format from header
    has_qweight   = any(k.endswith(".qweight")      for k in header)
    has_wp        = any(k.endswith(".weight_packed") for k in header)   # standard NVFP4
    # DiffusionGemma NVFP4: *.weight is U8 AND *.weight_scale is F8_E4M3 (ModelOpt format)
    has_diffusion_nvfp4 = any(
        header[k].get("dtype") == "U8" and k.endswith(".weight")
        and k[:-len(".weight")] + ".weight_scale" in header
        and header.get(k[:-len(".weight")] + ".weight_scale", {}).get("dtype") == "F8_E4M3"
        for k in header if k != "__metadata__"
    )
    has_fp8_weight = any(
        header[k].get("dtype") == "F8_E4M3" and k.endswith(".weight")
        for k in header if k != "__metadata__"
    )
    # MXFP4/MXFP8: *.weight U8 + *.weight_scale U8 (exponent bytes, not F8_E4M3 like diffusion_nvfp4)
    has_mx_u8_pair = any(
        header[k].get("dtype") == "U8" and k.endswith(".weight")
        and k[:-len(".weight")] + ".weight_scale" in header
        and header.get(k[:-len(".weight")] + ".weight_scale", {}).get("dtype") == "U8"
        for k in header if k != "__metadata__"
    )
    # BnB NF4: companion keys {base}.weight_quantized_stats (older BnB) or
    # {base}.weight.absmax (newer bitsandbytes >= 0.41) alongside U8 weights.
    has_bnb_nf4 = (
        any(k.endswith(".weight_quantized_stats") for k in header)
        or any("quant_state.bitsandbytes__nf4" in k for k in header if k != "__metadata__")
        or any(
            k.endswith(".weight.absmax")
            and header.get(k[:-len(".absmax")], {}).get("dtype") == "U8"
            for k in header if k != "__metadata__"
        )
    )

    if has_qweight:
        fmt = "awq" if any(k.endswith(".qzeros") for k in header) else "gptq"
    elif has_wp:
        fmt = "nvfp4"
    elif has_diffusion_nvfp4:
        fmt = "diffusion_nvfp4"  # ModelOpt NVFP4: .weight U8 + .weight_scale F8 + .weight_scale_2 F32
    elif has_fp8_weight:
        fmt = "fp8"
    elif has_bnb_nf4:
        fmt = "bnb_nf4"
    elif has_mx_u8_pair:
        # MXFP4 or MXFP8: U8 weight + U8 exponent scale. Distinguish via config files.
        _mx = _detect_mx_quant(Path(path).parent)
        fmt = _mx if _mx in ("mxfp4", "mxfp8") else "plain"
    else:
        fmt = "plain"

    if fmt != "plain":
        logger.info("Detected %s quantization in %s", fmt.upper(), path)

    def _load_raw(name: str) -> np.ndarray:
        meta = header[name]
        dtype_str = meta["dtype"]
        start, end = meta["data_offsets"]
        raw = raw_data[start:end]
        shape = tuple(meta["shape"])
        if dtype_str == "I32":
            return np.frombuffer(raw, dtype=np.int32).reshape(shape)
        if dtype_str == "U8":
            return np.frombuffer(raw, dtype=np.uint8).reshape(shape)
        if dtype_str == "F16":
            return np.frombuffer(raw, dtype=np.float16).reshape(shape)
        if dtype_str == "BF16":
            u16 = np.frombuffer(raw, dtype=np.uint16)
            f32 = (u16.astype(np.uint32) << 16).view(np.float32)
            return f32.reshape(shape)
        if dtype_str == "F32":
            return np.frombuffer(raw, dtype=np.float32).reshape(shape)
        if dtype_str == "F8_E4M3":
            # Load as uint8 bytes; pass to _fp8_e4m3_to_f32 later
            return np.frombuffer(raw, dtype=np.uint8).reshape(shape)
        raise ValueError(f"Unsupported dtype {dtype_str} for {name}")

    def _pad4(data: bytes) -> bytes:
        """Pad to 4-byte boundary — WebGPU write_buffer requires 4-byte-aligned size."""
        r = len(data) % 4
        return data if r == 0 else data + b"\x00" * (4 - r)

    # Track pending write_buffer bytes to flush periodically.
    # Metal silently drops write_buffer operations when the pending write queue
    # exceeds the GPU staging buffer capacity (~1-2GB). For large single-file
    # models (e.g. Gemma4-12B at 22GB), we must flush periodically.
    _pending_bytes: list = [0]
    _FLUSH_THRESHOLD = 512 * 1024 * 1024  # flush every 512MB of pending writes

    def _maybe_flush() -> None:
        if _pending_bytes[0] >= _FLUSH_THRESHOLD:
            wgpu_device.queue.submit([wgpu_device.create_command_encoder().finish()])
            wgpu_device.queue.on_submitted_work_done_sync()
            _pending_bytes[0] = 0

    def _upload(arr: np.ndarray, name: str, weights: dict) -> None:
        arr = np.ascontiguousarray(arr.astype(np.float16))
        data = _pad4(arr.tobytes())
        buf = wgpu_device.create_buffer(size=len(data), usage=usage)
        wgpu_device.queue.write_buffer(buf, 0, data)
        _pending_bytes[0] += len(data)
        _maybe_flush()
        weights[name] = WebGPUBuffer(buf=buf, device=wgpu_device,
                                     shape=tuple(arr.shape), dtype="f16")

    def _upload_u8(arr: np.ndarray, name: str, weights: dict) -> None:
        """Upload uint8 raw bytes to GPU (packed 4/u32 as shader binding).

        Used for FP8 E4M3 and NVFP4 packed weights. The shader reads via
        rd_byte_at() which unpacks individual bytes from the u32 array.
        """
        arr_flat = np.ascontiguousarray(arr.ravel().view(np.uint8))
        # Pad to multiple of 4 bytes so u32 reinterpretation is clean.
        r = len(arr_flat) % 4
        if r:
            arr_flat = np.concatenate([arr_flat, np.zeros(4 - r, dtype=np.uint8)])
        data = _pad4(arr_flat.tobytes())
        buf = wgpu_device.create_buffer(size=len(data), usage=usage)
        wgpu_device.queue.write_buffer(buf, 0, data)
        _pending_bytes[0] += len(data)
        _maybe_flush()
        weights[name] = WebGPUBuffer(buf=buf, device=wgpu_device,
                                     shape=tuple(arr.shape), dtype="u8")

    def _upload_int32(arr: np.ndarray, name: str, weights: dict) -> None:
        """Upload an INT32 array (quantized weights) directly to GPU without conversion."""
        arr = np.ascontiguousarray(arr.astype(np.int32))
        data = _pad4(arr.tobytes())
        buf = wgpu_device.create_buffer(size=len(data), usage=usage)
        wgpu_device.queue.write_buffer(buf, 0, data)
        _pending_bytes[0] += len(data)
        _maybe_flush()
        weights[name] = WebGPUBuffer(buf=buf, device=wgpu_device,
                                     shape=tuple(arr.shape), dtype="i32")

    def _upload_f16(arr: np.ndarray, name: str, weights: dict) -> None:
        """Upload an F16 array (scales/norms) directly to GPU."""
        arr = np.ascontiguousarray(arr.astype(np.float16))
        data = _pad4(arr.tobytes())
        buf = wgpu_device.create_buffer(size=len(data), usage=usage)
        wgpu_device.queue.write_buffer(buf, 0, data)
        _pending_bytes[0] += len(data)
        _maybe_flush()
        weights[name] = WebGPUBuffer(buf=buf, device=wgpu_device,
                                     shape=tuple(arr.shape), dtype="f16")

    weights: dict = {}

    # ── Helper: upload a single tensor from the header (plain dtypes) ──────────
    def _upload_plain(name: str, weights: dict) -> bool:
        meta = header.get(name)
        if meta is None:
            return False
        dtype_str = meta["dtype"]
        shape = tuple(meta["shape"])
        start, end = meta["data_offsets"]
        raw = raw_data[start:end]
        if dtype_str == "F16":
            arr = np.frombuffer(raw, dtype=np.float16).reshape(shape)
        elif dtype_str == "BF16":
            u16 = np.frombuffer(raw, dtype=np.uint16)
            f32 = (u16.astype(np.uint32) << 16).view(np.float32)
            arr = np.clip(f32, -65504.0, 65504.0).reshape(shape).astype(np.float16)
        elif dtype_str == "F32":
            arr = np.frombuffer(raw, dtype=np.float32).reshape(shape).astype(np.float16)
        elif dtype_str == "I8":
            # Int8 per-channel weight (BnB int8 / compressed-tensors int8).
            # Upload raw bytes; shader does sign extension via int8_to_f32().
            # dtype="u8" so _uq_weight() detects it via fmt="int8_gpu".
            arr_u8 = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
            r = len(arr_u8.ravel()) % 4
            arr_pad = np.concatenate([arr_u8.ravel(), np.zeros(4 - r if r else 0, dtype=np.uint8)])
            data_u8 = _pad4(arr_pad.tobytes())
            buf = wgpu_device.create_buffer(size=len(data_u8), usage=usage)
            wgpu_device.queue.write_buffer(buf, 0, data_u8)
            _pending_bytes[0] += len(data_u8)
            _maybe_flush()
            weights[name] = WebGPUBuffer(buf=buf, device=wgpu_device,
                                         shape=shape, dtype="u8")
            # Record int8 format in quant_meta for _uq() detection.
            qmeta = weights.setdefault("__quant_meta__", {})
            base_key = name[:-7] if name.endswith(".weight") else name
            qmeta.setdefault(base_key, {})["fmt"] = "int8_gpu"
            # Load companion per-channel weight scale if present.
            # compressed-tensors int8 (strategy=channel) stores a (N,) F32 scale at
            # {base}.weight_scale. Without it the USE_QUANT=7 shader reads scales[row]
            # from an uninitialized or wrong buffer, producing ~127x magnitude error.
            for sc_key in (f"{base_key}.weight_scale", f"{base_key}.scale"):
                if sc_key in header:
                    try:
                        sc_meta = header[sc_key]
                        sc_dtype = sc_meta["dtype"]
                        sc_start, sc_end = sc_meta["data_offsets"]
                        sc_raw = raw_data[sc_start:sc_end]
                        if sc_dtype == "F32":
                            sc_arr = np.frombuffer(sc_raw, dtype=np.float32).ravel()
                        elif sc_dtype == "F16":
                            sc_arr = np.frombuffer(sc_raw, dtype=np.float16).ravel()
                        elif sc_dtype == "BF16":
                            sc_u16 = np.frombuffer(sc_raw, dtype=np.uint16)
                            sc_arr = ((sc_u16.astype(np.uint32) << 16).view(np.float32)).ravel()
                        else:
                            logger.warning("Int8 scale %s has unsupported dtype %s",
                                           sc_key, sc_dtype)
                            break
                        _upload_f16(sc_arr.astype(np.float16), f"{name}.scales", weights)
                        qmeta[base_key]["group_size"] = 1
                        logger.debug("Int8 per-channel: %s scale n=%d", base_key, sc_arr.size)
                    except Exception as exc:
                        logger.warning("Int8 scale load failed for %s: %s", base_key, exc)
                    break
            return True
        else:
            return False  # not a plain dtype
        arr = np.ascontiguousarray(arr)
        data = _pad4(arr.tobytes())
        buf = wgpu_device.create_buffer(size=len(data), usage=usage)
        wgpu_device.queue.write_buffer(buf, 0, data)
        _pending_bytes[0] += len(data)
        _maybe_flush()
        weights[name] = WebGPUBuffer(buf=buf, device=wgpu_device,
                                     shape=tuple(arr.shape), dtype="f16")
        return True

    if fmt in ("awq", "gptq"):
        # Collect quantized bases
        quant_bases = sorted(set(
            k[:-len(".qweight")] for k in header if k.endswith(".qweight")
        ))
        quant_set = set()
        for base in quant_bases:
            for suf in (".qweight", ".scales", ".qzeros", ".g_idx"):
                if f"{base}{suf}" in header:
                    quant_set.add(f"{base}{suf}")

        for name in header:
            if name == "__metadata__" or name in quant_set:
                continue
            if not _upload_plain(name, weights):
                dt = header[name].get("dtype", "?")
                if dt not in ("F8_E4M3", "U8", "I32"):
                    logger.warning("Skipping %s (dtype=%s)", name, dt)

        for base in quant_bases:
            try:
                qw = _load_raw(f"{base}.qweight")
                sc = _load_raw(f"{base}.scales")
                qz_key = f"{base}.qzeros"
                qz = _load_raw(qz_key) if qz_key in header else None
                g_idx = _load_raw(f"{base}.g_idx") if f"{base}.g_idx" in header else None

                # GPU dequant path: upload raw quantized data directly.
                # GPTQ qweight [K//8, N] is transposed to [N, K//8] so all 256
                # threads in split-K read consecutive INT32s (coalesced access).
                group_size = int(sc.shape[0]) and (qw.shape[1] if fmt == "gptq" else qw.shape[0]) // sc.shape[0] if sc.ndim == 2 else 128
                if fmt == "gptq" and qz is None and g_idx is None:
                    # GPU GPTQ: transpose qweight [K//8, N] → [N, K//8] for coalesced access.
                    K8, N_ = qw.shape
                    group_size = (K8 * 8) // sc.shape[0] if sc.ndim == 2 else 128
                    qw_t = np.ascontiguousarray(qw.T)  # [N, K//8]
                    sc_gn = sc.astype(np.float16)      # [G, N] f16
                    _upload_int32(qw_t, f"{base}.weight", weights)
                    _upload_f16(sc_gn, f"{base}.weight.scales", weights)
                    weights["__quant_meta__"] = weights.get("__quant_meta__", {})
                    weights["__quant_meta__"][base] = {"fmt": "gptq_sym", "group_size": group_size}
                    logger.debug("GPU GPTQ: %s (K=%d, N=%d, G=%d)", base, K8*8, N_, sc.shape[0])
                elif fmt == "awq" and qz is not None and g_idx is None:
                    # GPU AWQ: check if qzeros decode to symmetric (all zero_point=8).
                    # AWQ qzeros use same [0,4,1,5,2,6,3,7] nibble ordering.
                    # All-zero qzeros → all zero_points = 0 → NOT symmetric.
                    # qzeros with all nibbles = 8 → zero_point = 8 (symmetric).
                    # We check for all-zero qzeros which means zero_point=0 (also
                    # supported: just use -0 for zero instead of -8).
                    K_, N8_ = qw.shape  # qw is [K, N//8]
                    N_ = N8_ * 8
                    G_ = sc.shape[0] if sc.ndim == 2 else K_ // 128
                    group_size = K_ // G_ if G_ > 0 else 128
                    # GPU AWQ: store [K, N//8] INT32 directly (no transpose needed
                    # since AWQ access pattern is already per-k, per-output-group)
                    sc_gn = sc.astype(np.float16) if sc.ndim == 2 else sc  # [G, N]
                    _upload_int32(qw, f"{base}.weight", weights)    # [K, N//8]
                    _upload_f16(sc_gn, f"{base}.weight.scales", weights)  # [G, N]
                    weights["__quant_meta__"] = weights.get("__quant_meta__", {})
                    weights["__quant_meta__"][base] = {"fmt": "awq_sym", "group_size": group_size}
                    logger.debug("GPU AWQ: %s (K=%d, N=%d, G=%d)", base, K_, N_, G_)
                else:
                    # Fall back to CPU dequantization.
                    if fmt == "awq" and qz is not None:
                        w_f16 = _dequant_awq(qw, sc, qz)
                    else:
                        w_f16 = _dequant_gptq(
                            qw, sc, qz if qz is not None else np.zeros_like(sc), g_idx)
                    _upload(w_f16, f"{base}.weight", weights)
            except Exception as exc:
                logger.warning("Failed to process %s: %s", base, exc)

    elif fmt == "nvfp4":
        # NVFP4: weight_packed (U8) + weight_scale (F8_E4M3) + weight_global_scale (F32)
        nvfp4_bases = sorted(set(
            k[:-len(".weight_packed")] for k in header if k.endswith(".weight_packed")
        ))
        nvfp4_set = set()
        for base in nvfp4_bases:
            for suf in (".weight_packed", ".weight_scale", ".weight_global_scale",
                        ".input_global_scale"):
                if f"{base}{suf}" in header:
                    nvfp4_set.add(f"{base}{suf}")

        for name in header:
            if name == "__metadata__" or name in nvfp4_set:
                continue
            if not _upload_plain(name, weights):
                dt = header[name].get("dtype", "?")
                if dt not in ("F8_E4M3", "U8", "I32", "F32"):
                    logger.warning("Skipping %s (dtype=%s)", name, dt)

        for base in nvfp4_bases:
            try:
                wp = _load_raw(f"{base}.weight_packed")    # (N, K//2) U8
                ws = _load_raw(f"{base}.weight_scale")     # (N, K//16) F8_E4M3 as uint8
                wgs_key = f"{base}.weight_global_scale"
                wgs = float(_load_raw(wgs_key).ravel()[0]) if wgs_key in header else 1.0
                N_, K2_ = wp.shape
                K_ = K2_ * 2
                # GPU NVFP4: upload raw weight_packed + F16-converted block scales.
                # The shader uses GLOBAL_SCALE as an override constant and
                # reads F8_E4M3 scales via the standard f16 scales binding.
                ws_f16 = np.ascontiguousarray(_fp8_e4m3_to_f32(ws).astype(np.float16))
                _upload_u8(wp, f"{base}.weight", weights)
                _upload_f16(ws_f16, f"{base}.weight.scales", weights)
                weights["__quant_meta__"] = weights.get("__quant_meta__", {})
                weights["__quant_meta__"][base] = {
                    "fmt": "nvfp4_gpu", "global_scale": wgs,
                    "group_size": K_ // (ws_f16.shape[1] if ws_f16.ndim == 2 else 1)}
                logger.debug("GPU NVFP4: %s (N=%d, K=%d, wgs=%.4f)", base, N_, K_, wgs)
            except Exception as exc:
                logger.warning("Failed to process NVFP4 %s: %s", base, exc)

    elif fmt == "diffusion_nvfp4":
        # DiffusionGemma ModelOpt NVFP4: *.weight (U8) + *.weight_scale (F8_E4M3) + *.weight_scale_2 (F32)
        # Used for quantized expert weights. Non-expert weights (BF16) uploaded normally.
        dnvfp4_bases = sorted(set(
            k[:-len(".weight")] for k in header
            if k.endswith(".weight") and header[k].get("dtype") == "U8"
            and k[:-len(".weight")] + ".weight_scale" in header
        ))
        dnvfp4_set = set()
        for base in dnvfp4_bases:
            for suf in (".weight", ".weight_scale", ".weight_scale_2", ".input_scale"):
                if f"{base}{suf}" in header:
                    dnvfp4_set.add(f"{base}{suf}")

        for name in header:
            if name == "__metadata__" or name in dnvfp4_set:
                continue
            if not _upload_plain(name, weights):
                dt = header[name].get("dtype", "?")
                if dt not in ("F8_E4M3", "U8", "I32", "F32"):
                    logger.warning("Skipping %s (dtype=%s)", name, dt)

        for base in dnvfp4_bases:
            try:
                wp = _load_raw(f"{base}.weight")        # (N, K//2) U8
                ws = _load_raw(f"{base}.weight_scale")  # (N, K//group_size) F8_E4M3 as uint8
                wgs_key = f"{base}.weight_scale_2"
                wgs = float(_load_raw(wgs_key).ravel()[0]) if wgs_key in header else 1.0
                N_, K2_ = wp.shape
                K_ = K2_ * 2
                ws_f16 = np.ascontiguousarray(_fp8_e4m3_to_f32(ws).astype(np.float16))
                _upload_u8(wp, f"{base}.weight", weights)
                _upload_f16(ws_f16, f"{base}.weight.scales", weights)
                weights["__quant_meta__"] = weights.get("__quant_meta__", {})
                weights["__quant_meta__"][base] = {
                    "fmt": "nvfp4_gpu", "global_scale": wgs,
                    "group_size": K_ // (ws_f16.shape[1] if ws_f16.ndim == 2 else 1)}
            except Exception as exc:
                logger.warning("Failed to process diffusion NVFP4 %s: %s", base, exc)

    elif fmt == "fp8":
        # Plain FP8 E4M3: weight stored as F8_E4M3, scale as F32 in *.weight_scale
        fp8_names = {
            k for k in header
            if k != "__metadata__"
            and k.endswith(".weight")
            and header[k].get("dtype") == "F8_E4M3"
        }
        fp8_scale_names = {
            k[:-len(".weight")] + ".weight_scale"
            for k in fp8_names
        }
        fp8_set = fp8_names | fp8_scale_names

        for name in header:
            if name == "__metadata__" or name in fp8_set:
                continue
            if not _upload_plain(name, weights):
                dt = header[name].get("dtype", "?")
                if dt not in ("F8_E4M3", "U8", "I32"):
                    logger.warning("Skipping %s (dtype=%s)", name, dt)

        for wname in fp8_names:
            base = wname[:-len(".weight")]
            try:
                w_fp8 = _load_raw(wname)   # uint8 array (F8_E4M3 bytes), shape (N, K)
                scale_key = f"{base}.weight_scale"
                # GPU FP8: upload raw F8 bytes; shader decodes inline.
                _upload_u8(w_fp8, wname, weights)
                weights.setdefault("__quant_meta__", {})
                if scale_key in header:
                    scale_arr = _load_raw(scale_key)
                    if scale_arr.ndim == 0 or scale_arr.size == 1:
                        # Per-tensor scale: scalar or single-element.
                        scale_val = float(scale_arr.ravel()[0])
                        weights["__quant_meta__"][base] = {
                            "fmt": "fp8_gpu", "global_scale": scale_val}
                        logger.debug("GPU FP8 (per-tensor): %s scale=%.6f", base, scale_val)
                    else:
                        # Per-channel scale: shape (N,) — one float per output channel.
                        # Upload as F16 scales buffer; shader reads scales[row] when GROUP_K=1.
                        scale_f16 = np.ascontiguousarray(
                            scale_arr.ravel().astype(np.float16))
                        _upload_f16(scale_f16, wname + ".scales", weights)
                        weights["__quant_meta__"][base] = {
                            "fmt": "fp8_gpu", "global_scale": 1.0, "group_size": 1}
                        logger.debug("GPU FP8 (per-channel): %s n_scales=%d",
                                     base, scale_f16.size)
                else:
                    weights["__quant_meta__"][base] = {
                        "fmt": "fp8_gpu", "global_scale": 1.0}
                    logger.debug("GPU FP8: %s (no scale key)", base)
            except Exception as exc:
                logger.warning("Failed to process FP8 %s: %s", base, exc)

    elif fmt == "mxfp4":
        # MXFP4 (microscaling FP4): *.weight [N, K//2] U8 packed FP4 + *.weight_scale [N, K//32] U8 exponents.
        # Scales are u8 exponents (not F8_E4M3): scale_f16 = 2^(u8 - 127).
        # Reuses the NVFP4 GPU shader path (USE_QUANT=6) with GROUP_K=32 instead of 16.
        mxfp4_bases = sorted(set(
            k[:-len(".weight")] for k in header
            if k.endswith(".weight")
            and header[k].get("dtype") == "U8"
            and k[:-len(".weight")] + ".weight_scale" in header
            and header.get(k[:-len(".weight")] + ".weight_scale", {}).get("dtype") == "U8"
        ))
        mx4_set: set = set()
        for base in mxfp4_bases:
            mx4_set.add(f"{base}.weight")
            mx4_set.add(f"{base}.weight_scale")

        for name in header:
            if name == "__metadata__" or name in mx4_set:
                continue
            if not _upload_plain(name, weights):
                dt = header[name].get("dtype", "?")
                if dt not in ("U8", "I32", "F32"):
                    logger.warning("Skipping %s (dtype=%s)", name, dt)

        for base in mxfp4_bases:
            try:
                wp    = _load_raw(f"{base}.weight")        # (N, K//2) U8 packed FP4
                ws_u8 = _load_raw(f"{base}.weight_scale")  # (N, K//32) U8 exponents
                # Convert U8 exponents to F16: scale = 2^(u8 - 127)
                ws_f16 = np.ascontiguousarray(
                    (np.float32(2.0) ** (ws_u8.astype(np.float32) - 127.0)).astype(np.float16))
                N_, K2_ = wp.shape
                K_ = K2_ * 2
                _upload_u8(wp, f"{base}.weight", weights)
                _upload_f16(ws_f16, f"{base}.weight.scales", weights)
                weights["__quant_meta__"] = weights.get("__quant_meta__", {})
                weights["__quant_meta__"][base] = {
                    "fmt": "nvfp4_gpu", "global_scale": 1.0, "group_size": 32}
                logger.debug("GPU MXFP4: %s (N=%d, K=%d)", base, N_, K_)
            except Exception as exc:
                logger.warning("Failed to process MXFP4 %s: %s", base, exc)

    elif fmt == "mxfp8":
        # MXFP8 (microscaling FP8): *.weight [N, K] U8 FP8-E4M3 + *.weight_scale [N, K//32] U8 exponents.
        # Scales are u8 exponents: scale = 2^(u8 - 127), one per block of 32 K-elements.
        # CPU dequant: avoids shader changes for per-block FP8.
        # TODO: USE_QUANT=9 for GPU MXFP8 per-block decode
        mxfp8_bases = sorted(set(
            k[:-len(".weight")] for k in header
            if k.endswith(".weight")
            and header[k].get("dtype") == "U8"
            and k[:-len(".weight")] + ".weight_scale" in header
            and header.get(k[:-len(".weight")] + ".weight_scale", {}).get("dtype") == "U8"
        ))
        mx8_set: set = set()
        for base in mxfp8_bases:
            mx8_set.add(f"{base}.weight")
            mx8_set.add(f"{base}.weight_scale")

        for name in header:
            if name == "__metadata__" or name in mx8_set:
                continue
            if not _upload_plain(name, weights):
                dt = header[name].get("dtype", "?")
                if dt not in ("U8", "I32", "F32"):
                    logger.warning("Skipping %s (dtype=%s)", name, dt)

        for base in mxfp8_bases:
            try:
                w_u8  = _load_raw(f"{base}.weight")        # (N, K) U8 FP8 E4M3 bytes
                ws_u8 = _load_raw(f"{base}.weight_scale")  # (N, K//32) U8 exponents
                N_, K_ = w_u8.shape
                # Convert U8 exponents to F32 block scales: scale = 2^(u8 - 127)
                block_scale = (np.float32(2.0) ** (ws_u8.astype(np.float32) - 127.0))
                n_blocks = ws_u8.shape[1] if ws_u8.ndim == 2 else 1
                block_size = K_ // n_blocks if n_blocks > 0 else K_
                # Expand block scales to (N, K) for element-wise multiply
                block_scale_exp = np.repeat(block_scale, block_size, axis=1)
                # FP8 E4M3 → F32, scale, clip, cast to F16
                w_f32 = _fp8_e4m3_to_f32(w_u8)
                w_f16 = np.ascontiguousarray(
                    np.clip(w_f32 * block_scale_exp, -65504.0, 65504.0).astype(np.float16))
                _upload(w_f16, f"{base}.weight", weights)
                logger.debug("CPU MXFP8: %s (N=%d, K=%d, blocks=%d)", base, N_, K_, n_blocks)
            except Exception as exc:
                logger.warning("Failed to process MXFP8 %s: %s", base, exc)

    elif fmt == "bnb_nf4":
        # BitsAndBytes NF4: weight [N//2, K] U8 (2 NF4 codes per byte, flattened row-pairs)
        # + absmax [N*K//64] F32 (one per block of 64 elements in flat row-major order).
        #
        # BnB packs flat weight elements in pairs: byte i holds codes for elements 2i and
        # 2i+1.  Reshaped to [N//2, K], row r covers weight rows 2r and 2r+1:
        #   - columns [0, K//2): codes for weight[2r, 0..K-1] (lo=even k, hi=odd k)
        #   - columns [K//2, K): codes for weight[2r+1, 0..K-1]
        # The shader expects [N, K//2] — reshape [N//2, K] → [N//2, 2, K//2] → [N, K//2].
        # Absmax is in flat block order (block b covers weight[b//G, b%G*64..(b%G+1)*64]
        # when K%64==0), so reshape [N*K//64] → [N, K//64] aligns with scales[n, blk].
        #
        # Companion key patterns supported:
        #   old BnB: {base}.weight_quantized_stats  (F32/F16 absmax tensor)
        #   new BnB: {base}.weight.absmax            (F32 absmax tensor)

        # Collect BnB bases and the set of companion keys to skip in the plain pass.
        bnb_bases: set = set()
        bnb_set: set = set()

        for k in header:
            if k == "__metadata__":
                continue
            if k.endswith(".weight_quantized_stats"):
                base = k[:-len(".weight_quantized_stats")]
                bnb_bases.add(base)
                bnb_set.add(k)
                wk = f"{base}.weight"
                if wk in header:
                    bnb_set.add(wk)
            elif k.endswith(".weight.absmax"):
                w_key = k[:-len(".absmax")]          # "{base}.weight"
                if header.get(w_key, {}).get("dtype") == "U8":
                    base = w_key[:-len(".weight")]
                    bnb_bases.add(base)
                    bnb_set.add(k)
                    bnb_set.add(w_key)
            elif "quant_state.bitsandbytes__nf4" in k:
                w_idx = k.find(".weight.quant_state")
                if w_idx >= 0:
                    base = k[:w_idx]
                    bnb_bases.add(base)
                    bnb_set.add(k)
                    wk = f"{base}.weight"
                    if wk in header:
                        bnb_set.add(wk)
                    absmax_k = f"{base}.weight.absmax"
                    if absmax_k in header:
                        bnb_set.add(absmax_k)

        # Upload all non-BnB tensors normally.
        for name in header:
            if name == "__metadata__" or name in bnb_set:
                continue
            if not _upload_plain(name, weights):
                dt = header[name].get("dtype", "?")
                if dt not in ("U8", "I32", "F32"):
                    logger.warning("Skipping %s (dtype=%s)", name, dt)

        _BNB_GROUP_K = 64

        for base in sorted(bnb_bases):
            try:
                w_key = f"{base}.weight"
                if w_key not in header or header[w_key].get("dtype") != "U8":
                    logger.warning("BnB NF4: missing or non-U8 weight for %s, skipping", base)
                    continue

                bnb_codes = _load_raw(w_key)     # [N//2, K] uint8

                if bnb_codes.ndim != 2:
                    logger.warning(
                        "BnB NF4: expected 2D weight, got shape %s for %s — skipping",
                        bnb_codes.shape, base)
                    continue

                N_half, K = bnb_codes.shape
                N = N_half * 2
                K_half = K // 2

                # Find absmax (try each companion key pattern in priority order).
                absmax_arr = None
                for abs_key in (
                    f"{base}.weight_quantized_stats",
                    f"{base}.weight.absmax",
                ):
                    if abs_key in header:
                        absmax_arr = _load_raw(abs_key).astype(np.float32).ravel()
                        break

                if absmax_arr is None:
                    logger.warning("BnB NF4: no absmax found for %s, skipping", base)
                    continue

                expected_blocks = N * K // _BNB_GROUP_K
                if absmax_arr.size != expected_blocks:
                    logger.warning(
                        "BnB NF4: absmax size %d != expected %d (N=%d, K=%d) for %s — skipping",
                        absmax_arr.size, expected_blocks, N, K, base)
                    continue

                # Reshape: [N//2, K] → [N//2, 2, K//2] → [N, K//2]
                # BnB row r: first K//2 bytes → shader row 2r, last K//2 bytes → shader row 2r+1.
                shader_codes = np.ascontiguousarray(
                    bnb_codes.reshape(N_half, 2, K_half).reshape(N, K_half))

                # Reshape absmax: [N*K//64] → [N, K//64] (flat block order matches row-major).
                absmax_2d = np.ascontiguousarray(
                    absmax_arr.reshape(N, K // _BNB_GROUP_K).astype(np.float16))

                _upload_u8(shader_codes, f"{base}.weight", weights)
                _upload_f16(absmax_2d, f"{base}.weight.scales", weights)
                weights.setdefault("__quant_meta__", {})[base] = {
                    "fmt": "nf4_gpu",
                    "group_size": _BNB_GROUP_K,
                }
                logger.debug("GPU NF4: %s (N=%d, K=%d, blocks=%d)", base, N, K, expected_blocks)

            except Exception as exc:
                logger.warning("Failed to process BnB NF4 %s: %s", base, exc)

    else:
        # Plain BF16/F16/F32
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if not _upload_plain(name, weights):
                logger.warning("Unsupported dtype %s for %s, skipping",
                                meta.get("dtype", "?"), name)

    # Multimodal remapping for single-file models (same patterns as sharded loader).
    # Gemma4 unified: model.language_model.X → model.X
    # Gemma3 multimodal (rare single-file): language_model.X → X
    keys = list(weights.keys())
    if any(k.startswith("model.language_model.") for k in keys):
        remapped = {}
        for k, v in weights.items():
            if k.startswith("model.language_model."):
                remapped["model." + k[len("model.language_model."):]] = v
        weights.update(remapped)
        logger.info("Remapped %d model.language_model.* keys", len(remapped))
    elif any(k.startswith("language_model.") for k in keys):
        remapped = {}
        for k, v in weights.items():
            if k.startswith("language_model."):
                remapped[k[len("language_model."):]] = v
        weights.update(remapped)
        logger.info("Remapped %d language_model.* keys", len(remapped))

    # Commit all pending write_buffer calls before returning.
    wgpu_device.queue.submit([wgpu_device.create_command_encoder().finish()])
    wgpu_device.queue.on_submitted_work_done_sync()
    logger.info("Loaded %d tensors from %s", len(weights), path)
    return weights


def _bf16_raw_to_f32(raw: bytes, shape: tuple) -> "np.ndarray":
    """Convert raw BF16 bytes to float32 numpy array."""
    u16 = np.frombuffer(raw, dtype=np.uint16)
    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
    return f32.reshape(shape)


def _dequant_mlx_int4(
    weight_u32: "np.ndarray",
    scales_f32: "np.ndarray",
    biases_f32: "np.ndarray",
    group_size: int = 64,
) -> "np.ndarray":
    """Dequantize MLX affine int4 weights to float32.

    weight_u32: [out_rows, in_cols/8] — 8 packed uint4 nibbles per uint32
    scales_f32: [out_rows, in_cols/group_size]
    biases_f32: [out_rows, in_cols/group_size]
    Returns float32 [out_rows, in_cols].
    """
    out_rows, packed_cols = weight_u32.shape
    in_cols = packed_cols * 8
    w = weight_u32.astype(np.uint32)
    nibbles = np.stack([(w >> (4 * i)) & 0xF for i in range(8)], axis=-1).reshape(out_rows, in_cols).astype(np.float32)
    n_groups = in_cols // group_size
    scales_bc = np.repeat(scales_f32.reshape(out_rows, n_groups), group_size, axis=1)
    biases_bc = np.repeat(biases_f32.reshape(out_rows, n_groups), group_size, axis=1)
    return scales_bc * nibbles + biases_bc


def _mlx_strip_prefix(key: str) -> str:
    """Strip 'language_model.' wrapper from MLX weight key."""
    prefix = "language_model."
    if key.startswith(prefix):
        return key[len(prefix):]
    return key


def load_mlx_weights(model_dir: str, wgpu_device) -> dict:
    """Load MLX affine int4 safetensors weights, dequantize to f16, upload to GPU."""
    import json
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer

    p = Path(model_dir)
    index_path = p / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)

    group_size = 64
    config_path = p / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        qs = (cfg.get("quantization", {}).get("group_size")
              or cfg.get("quantization_config", {}).get("group_size"))
        if qs:
            group_size = int(qs)

    shard_to_keys: dict = {}
    for key, shard_file in index["weight_map"].items():
        shard_to_keys.setdefault(shard_file, []).append(key)

    raw_tensors: dict = {}
    for shard_file in sorted(shard_to_keys.keys()):
        shard_path = str(p / shard_file)
        logger.info("Loading MLX shard %s", shard_file)
        with open(shard_path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header_raw = f.read(header_len)
            data_start = 8 + header_len
            header = json.loads(header_raw)
            f.seek(data_start)
            raw_data = f.read()
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            raw_tensors[name] = (meta["dtype"], tuple(meta["shape"]), raw_data[start:end])

    weights: dict = {}
    all_keys = set(raw_tensors.keys())
    processed: set = set()

    for key in sorted(all_keys):
        if key in processed:
            continue
        dtype_str, shape, raw = raw_tensors[key]

        if key.endswith(".weight") and dtype_str == "U32":
            base = key[:-len(".weight")]
            scales_key = base + ".scales"
            biases_key = base + ".biases"
            if scales_key in all_keys and biases_key in all_keys:
                _, s_shape, s_raw = raw_tensors[scales_key]
                _, b_shape, b_raw = raw_tensors[biases_key]
                processed.update({key, scales_key, biases_key})
                w_u32 = np.frombuffer(raw, dtype=np.uint32).reshape(shape)
                scales_f32 = _bf16_raw_to_f32(s_raw, s_shape)
                biases_f32 = _bf16_raw_to_f32(b_raw, b_shape)
                dequant = _dequant_mlx_int4(w_u32, scales_f32, biases_f32, group_size)
                arr = np.clip(dequant, -65504.0, 65504.0).astype(np.float16)
                local_key = _mlx_strip_prefix(base) + ".weight"
                weights[local_key] = WebGPUBuffer.from_numpy(wgpu_device, np.ascontiguousarray(arr))
                continue

        if key in processed:
            continue
        processed.add(key)

        if key.endswith(".scales") or key.endswith(".biases"):
            base_w = key.rsplit(".", 1)[0] + ".weight"
            if base_w in all_keys and raw_tensors[base_w][0] == "U32":
                continue

        if dtype_str == "BF16":
            f32 = _bf16_raw_to_f32(raw, shape)
            arr = np.clip(f32, -65504.0, 65504.0).astype(np.float16)
        elif dtype_str == "F32":
            arr = np.frombuffer(raw, dtype=np.float32).reshape(shape).astype(np.float16)
        elif dtype_str == "F16":
            arr = np.frombuffer(raw, dtype=np.float16).reshape(shape)
        else:
            logger.warning("Unsupported dtype %s for tensor %s, skipping", dtype_str, key)
            continue

        local_key = _mlx_strip_prefix(key)
        weights[local_key] = WebGPUBuffer.from_numpy(wgpu_device, np.ascontiguousarray(arr))

    logger.info("Loaded %d tensors from MLX int4 dir %s", len(weights), model_dir)
    return weights
