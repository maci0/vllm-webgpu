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

    if has_qweight:
        fmt = "awq" if any(k.endswith(".qzeros") for k in header) else "gptq"
    elif has_wp:
        fmt = "nvfp4"
    elif has_diffusion_nvfp4:
        fmt = "diffusion_nvfp4"  # ModelOpt NVFP4: .weight U8 + .weight_scale F8 + .weight_scale_2 F32
    elif has_fp8_weight:
        fmt = "fp8"
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

    def _upload(arr: np.ndarray, name: str, weights: dict) -> None:
        arr = np.ascontiguousarray(arr.astype(np.float16))
        data = _pad4(arr.tobytes())
        buf = wgpu_device.create_buffer(size=len(data), usage=usage)
        wgpu_device.queue.write_buffer(buf, 0, data)
        weights[name] = WebGPUBuffer(buf=buf, device=wgpu_device,
                                     shape=tuple(arr.shape), dtype="f16")

    def _upload_int32(arr: np.ndarray, name: str, weights: dict) -> None:
        """Upload an INT32 array (quantized weights) directly to GPU without conversion."""
        arr = np.ascontiguousarray(arr.astype(np.int32))
        data = _pad4(arr.tobytes())
        buf = wgpu_device.create_buffer(size=len(data), usage=usage)
        wgpu_device.queue.write_buffer(buf, 0, data)
        weights[name] = WebGPUBuffer(buf=buf, device=wgpu_device,
                                     shape=tuple(arr.shape), dtype="i32")

    def _upload_f16(arr: np.ndarray, name: str, weights: dict) -> None:
        """Upload an F16 array (scales/norms) directly to GPU."""
        arr = np.ascontiguousarray(arr.astype(np.float16))
        data = _pad4(arr.tobytes())
        buf = wgpu_device.create_buffer(size=len(data), usage=usage)
        wgpu_device.queue.write_buffer(buf, 0, data)
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
        else:
            return False  # not a plain dtype
        arr = np.ascontiguousarray(arr)
        data = _pad4(arr.tobytes())
        buf = wgpu_device.create_buffer(size=len(data), usage=usage)
        wgpu_device.queue.write_buffer(buf, 0, data)
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
                w_f16 = _dequant_nvfp4(wp, ws, wgs)
                _upload(w_f16, f"{base}.weight", weights)
            except Exception as exc:
                logger.warning("Failed to dequantize NVFP4 %s: %s", base, exc)

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
                w_f16 = _dequant_nvfp4(wp, ws, wgs)
                _upload(w_f16, f"{base}.weight", weights)
            except Exception as exc:
                logger.warning("Failed to dequantize DiffusionGemma NVFP4 %s: %s", base, exc)

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
                w_fp8 = _load_raw(wname)   # uint8 array (F8_E4M3 bytes)
                scale_key = f"{base}.weight_scale"
                if scale_key in header:
                    scale = _load_raw(scale_key).astype(np.float32)
                else:
                    scale = np.float32(1.0)
                w_f16 = _dequant_fp8(w_fp8, scale)
                _upload(w_f16, wname, weights)
            except Exception as exc:
                logger.warning("Failed to dequantize FP8 %s: %s", base, exc)

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


_GGUF_TO_HF_LLAMA = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "model.norm.weight",
}

_GGUF_BLK_MAP = {
    "attn_q.weight":             "self_attn.q_proj.weight",
    "attn_k.weight":             "self_attn.k_proj.weight",
    "attn_v.weight":             "self_attn.v_proj.weight",
    "attn_output.weight":        "self_attn.o_proj.weight",
    "attn_norm.weight":          "input_layernorm.weight",
    "ffn_gate.weight":           "mlp.gate_proj.weight",
    "ffn_up.weight":             "mlp.up_proj.weight",
    "ffn_down.weight":           "mlp.down_proj.weight",
    # Llama/Qwen: ffn_norm is the single pre-FFN norm (same key as post-attention in their schema)
    "ffn_norm.weight":           "post_attention_layernorm.weight",
    "attn_q_norm.weight":        "self_attn.q_norm.weight",
    "attn_k_norm.weight":        "self_attn.k_norm.weight",
    # Gemma4 has 4 distinct norms per layer (HF: Gemma3DecoderLayer):
    #   input_layernorm          = attn_norm       (pre-attention)
    #   post_attention_layernorm = post_attention_norm (post-attn residual)
    #   pre_feedforward_layernorm = ffn_norm       (pre-FFN) — overrides Llama mapping below
    #   post_feedforward_layernorm = post_ffw_norm (post-FFN residual)
    "post_attention_norm.weight":  "post_attention_layernorm.weight",
    "post_ffw_norm.weight":        "post_feedforward_layernorm.weight",
    "layer_output_scale.weight":   "self_attn.layer_scale",
}

# Gemma4 overrides: ffn_norm maps to pre_feedforward_layernorm, not post_attention_layernorm.
# Detect by architecture and remap at load time (see _gguf_to_hf_name).
_GGUF_BLK_MAP_GEMMA4 = dict(_GGUF_BLK_MAP)
_GGUF_BLK_MAP_GEMMA4["ffn_norm.weight"] = "pre_feedforward_layernorm.weight"


def _gguf_to_hf_name(gguf_name: str, blk_map: dict | None = None) -> str | None:
    """Map a GGUF tensor name to its HuggingFace equivalent. Returns None to skip."""
    if blk_map is None:
        blk_map = _GGUF_BLK_MAP
    if gguf_name in _GGUF_TO_HF_LLAMA:
        return _GGUF_TO_HF_LLAMA[gguf_name]
    # blk.{i}.{suffix} pattern
    if gguf_name.startswith("blk."):
        parts = gguf_name.split(".", 2)  # ["blk", "i", "suffix"]
        if len(parts) == 3:
            layer_idx = parts[1]
            suffix = parts[2]
            hf_suffix = blk_map.get(suffix)
            if hf_suffix:
                return f"model.layers.{layer_idx}.{hf_suffix}"
    # Unknown tensor (e.g. rope_freqs, vision layers): skip
    return None


def load_gguf_weights(path: str, wgpu_device, arch_prefix: str = "") -> dict:
    """Load GGUF weights with HF-style key remapping. Raw quantized blocks uploaded as u8."""
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    import wgpu as wgpu_lib

    try:
        import gguf
    except ImportError as e:
        raise ImportError("Install the 'gguf' package to load GGUF files.") from e

    reader = gguf.GGUFReader(path)

    # Detect arch from GGUF metadata if not provided
    if not arch_prefix:
        arch_bytes = reader.fields.get("general.architecture")
        if arch_bytes:
            raw = arch_bytes.parts[-1].tolist()
            arch_prefix = bytes(raw).decode() if isinstance(raw, list) else str(raw)

    blk_map = _GGUF_BLK_MAP_GEMMA4 if arch_prefix in ("gemma3", "gemma4") else _GGUF_BLK_MAP

    from gguf import GGMLQuantizationType

    from gguf import dequantize as gguf_dequantize

    # Tensors decoded to f16 immediately (small or required by non-matmul shaders).
    # The embedding table and LM-head weight must be f16 because embedding_lookup.wgsl
    # reads them as vec4<f16> and matmul_quant's f16 path is used for the LM-head GEMV.
    # All other quantized weights stay as raw bytes and are decoded on the GPU.
    _EAGER_F16_NAMES = {"token_embd.weight", "output.weight"}  # GGUF raw names

    weights: dict = {}
    # Track tensor types for use_quant detection.
    weight_quant_type: dict[str, int] = {}
    usage = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_DST | wgpu_lib.BufferUsage.COPY_SRC
    skipped = 0

    for tensor in reader.tensors:
        hf_name = _gguf_to_hf_name(tensor.name, blk_map)
        if hf_name is None:
            skipped += 1
            continue

        tt = int(tensor.tensor_type)
        # Determine whether to decode now (f16) or keep raw bytes for GPU decoding.
        force_eager = tensor.name in _EAGER_F16_NAMES

        if tt == int(GGMLQuantizationType.F32):
            arr = np.ascontiguousarray(
                tensor.data.view(np.float32).astype(np.float16))
        elif tt == int(GGMLQuantizationType.F16):
            arr = np.ascontiguousarray(tensor.data.view(np.float16))
        elif tt == int(GGMLQuantizationType.BF16):
            u16 = tensor.data.view(np.uint16)
            f32 = (u16.astype(np.uint32) << 16).view(np.float32)
            arr = np.ascontiguousarray(
                np.clip(f32, -65504.0, 65504.0).astype(np.float16))
        elif force_eager or tt == int(GGMLQuantizationType.Q6_K):
            # Eagerly dequantize: embedding/lm-head (f16 required by shaders)
            # and Q6_K (different block format; Q4_K-only GPU decoder can't handle it).
            f32 = gguf_dequantize(tensor.data, tensor.tensor_type)
            arr = np.ascontiguousarray(
                np.clip(f32, -65504.0, 65504.0).astype(np.float16))
            tt = int(GGMLQuantizationType.F16)  # treat as f16 for quant_types
        else:
            # Quantized weight (Q4_K, Q6_K, Q8_0, etc.): upload raw bytes.
            # The WGSL matmul shader decodes on the fly using the GGUF block format.
            arr = np.ascontiguousarray(tensor.data)

        weights[hf_name] = WebGPUBuffer.from_numpy(wgpu_device, arr, usage=usage)
        weight_quant_type[hf_name] = tt

    # Attach quant metadata so models can detect quantization format per tensor.
    weights["__quant_types__"] = weight_quant_type  # type: ignore[assignment]

    logger.info("Loaded %d GGUF tensors (%d skipped) from %s", len(weights), skipped, path)
    return weights


def _vocab_from_tokens(reader) -> int | None:
    """Extract vocab size from GGUF token list metadata, or None if unavailable."""
    try:
        field = reader.fields.get("tokenizer.ggml.tokens")
        if field is not None:
            # The field contains a list of token strings; count them.
            return len(field.data)
    except Exception:
        pass
    # Fall back to the embedding table outermost dimension.
    try:
        for t in reader.tensors:
            if t.name == "token_embd.weight":
                # GGUF shape: [hidden, vocab] → outermost dim = vocab
                return int(t.shape[-1]) if len(t.shape) >= 2 else None
    except Exception:
        pass
    return None


def gguf_read_config(path: str) -> dict:
    """Read model configuration from GGUF metadata. Returns a dict compatible with run_inference."""
    try:
        import gguf
    except ImportError as e:
        raise ImportError("Install the 'gguf' package.") from e

    reader = gguf.GGUFReader(path)

    def _get(key):
        if key not in reader.fields:
            return None
        v = reader.fields[key].parts[-1].tolist()
        if isinstance(v, list) and len(v) == 1:
            return v[0]
        return v

    def _get_str(key):
        v = _get(key)
        if isinstance(v, list):
            return bytes(v).decode("utf-8", errors="replace")
        return v

    # Detect architecture prefix
    arch_bytes = _get("general.architecture")
    arch_prefix = bytes(arch_bytes).decode() if isinstance(arch_bytes, list) else str(arch_bytes)
    p = arch_prefix  # e.g. "llama", "gemma4", "qwen3"

    cfg = {
        "architectures":     [_get_str("general.name") or "LlamaForCausalLM"],
        "hidden_size":       _get(f"{p}.embedding_length") or 4096,
        "num_hidden_layers": _get(f"{p}.block_count") or 32,
        "num_attention_heads": _get(f"{p}.attention.head_count") or 32,
        "num_key_value_heads": _get(f"{p}.attention.head_count_kv") or 8,
        "intermediate_size": _get(f"{p}.feed_forward_length") or 14336,
        "vocab_size":        _get(f"{p}.vocab_size") or _vocab_from_tokens(reader) or 151936,
        "rope_theta":        _get(f"{p}.rope.freq_base") or 10000.0,
        "head_dim":          _get(f"{p}.attention.key_length"),
        "max_position_embeddings": _get(f"{p}.context_length") or 8192,
        "tie_word_embeddings": False,
        # softcap for Gemma
        "final_logit_softcapping": _get(f"{p}.final_logit_softcapping"),
        "_gguf_arch_prefix": arch_prefix,
    }

    # If head_dim not explicit, derive from hidden / heads
    if cfg["head_dim"] is None:
        cfg["head_dim"] = cfg["hidden_size"] // cfg["num_attention_heads"]

    # Map GGUF arch name to HF architecture class
    _arch_map = {
        "llama": "LlamaForCausalLM",
        "qwen3": "Qwen3ForCausalLM",
        "qwen2": "Qwen2ForCausalLM",
        "gemma3": "Gemma3ForCausalLM",
        "gemma4": "Gemma3ForCausalLM",  # map to our Gemma4 model
    }
    cfg["architectures"] = [_arch_map.get(arch_prefix, "LlamaForCausalLM")]

    # For Gemma4: detect per-layer attention params from tensor shapes.
    # Local layers use head_dim=256, 8 KV heads; global layers (idx%6==5)
    # use head_dim=512, 1 KV head, and have no separate v_proj (V=K).
    if arch_prefix in ("gemma3", "gemma4"):
        tensor_shapes = {t.name: t.shape for t in reader.tensors}
        n_layers = cfg["num_hidden_layers"]
        layer_params = []
        for i in range(n_layers):
            k_norm_key = f"blk.{i}.attn_k_norm.weight"
            q_proj_key = f"blk.{i}.attn_q.weight"
            has_v = f"blk.{i}.attn_v.weight" in tensor_shapes
            # GGUF stores weights as [in_dim, out_dim] (innermost first / column-major).
            # For a weight [hidden, q_dim]: shape[0]=hidden (input), shape[-1]=q_dim (output).
            # Use shape[-1] for output dimensions and shape[0] for the k_norm (scalar head_dim).
            if k_norm_key in tensor_shapes:
                hd = int(tensor_shapes[k_norm_key][-1])  # k_norm shape is [head_dim]
            elif q_proj_key in tensor_shapes:
                # q_proj shape: [hidden, q_dim] → q_dim = num_q_heads * head_dim
                q_dim_gguf = int(tensor_shapes[q_proj_key][-1])  # outermost = output dim
                hd = q_dim_gguf // cfg["num_attention_heads"]
            else:
                hd = cfg["head_dim"]
            # k_proj shape: [hidden, kv_dim] → kv_dim = num_kv_heads * head_dim (outermost)
            k_proj_key = f"blk.{i}.attn_k.weight"
            if k_proj_key in tensor_shapes:
                kv_dim = int(tensor_shapes[k_proj_key][-1])
                num_kv_heads = kv_dim // hd
            else:
                num_kv_heads = cfg["num_key_value_heads"]
            q_dim = cfg["num_attention_heads"] * hd
            layer_params.append({
                "head_dim": hd,
                "num_q_heads": cfg["num_attention_heads"],
                "num_kv_heads": num_kv_heads,
                "q_dim": q_dim,
                "kv_dim": num_kv_heads * hd,
                "has_v_proj": has_v,
            })
        cfg["_layer_attention_params"] = layer_params
        logger.info("Gemma4: detected %d layers, %d local (hd=256,kv=8) + %d global (hd=512,kv=1)",
                    n_layers,
                    sum(1 for lp in layer_params if lp["head_dim"] == 256),
                    sum(1 for lp in layer_params if lp["head_dim"] == 512))

    logger.info("GGUF config: arch=%s hid=%d layers=%d heads=%d/%d head_dim=%d inter=%d",
                arch_prefix, cfg["hidden_size"], cfg["num_hidden_layers"],
                cfg["num_attention_heads"], cfg["num_key_value_heads"],
                cfg["head_dim"], cfg["intermediate_size"])
    return cfg


def _bf16_raw_to_f32(raw: bytes, shape: tuple) -> np.ndarray:
    """Convert raw BF16 bytes to float32 numpy array."""
    u16 = np.frombuffer(raw, dtype=np.uint16)
    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
    return f32.reshape(shape)


def _dequant_mlx_int4(
    weight_u32: np.ndarray,
    scales_f32: np.ndarray,
    biases_f32: np.ndarray,
    group_size: int = 64,
) -> np.ndarray:
    """Dequantize MLX affine int4 weights to float32.

    weight_u32: [out_rows, in_cols/8]  -- 8 packed uint4 nibbles per uint32
    scales_f32: [out_rows, in_cols/group_size]
    biases_f32: [out_rows, in_cols/group_size]
    Returns float32 [out_rows, in_cols].
    """
    out_rows, packed_cols = weight_u32.shape
    in_cols = packed_cols * 8

    w = weight_u32.astype(np.uint32)
    nibbles = np.stack([
        (w >> (4 * i)) & 0xF for i in range(8)
    ], axis=-1).reshape(out_rows, in_cols).astype(np.float32)

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
