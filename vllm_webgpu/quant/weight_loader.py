from __future__ import annotations
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

# AWQ nibble reorder: position i in int32 holds nibble at bit offset
# [0, 16, 4, 20, 8, 24, 12, 28] = [0,4,1,5,2,6,3,7] * 4
# Matches vllm.model_executor.layers.quantization.auto_awq._REVERSE_AWQ_PACK_ORDER * 4.
_AWQ_NIBBLE_SHIFTS: np.ndarray = np.array([0, 16, 4, 20, 8, 24, 12, 28], dtype=np.int32)
# GPTQ nibble unpack: each int32 holds 8 nibbles at bit offsets [0, 4, 8, ..., 28].
_GPTQ_NIBBLE_SHIFTS: np.ndarray = np.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=np.int32)
_F16_MAX: float = np.finfo(np.float16).max

# When set, GDN projection weights with BF16 dtype are uploaded in their native
# bf16 bit pattern (packed u16 pairs → u32) under key + "__bf16", in addition to
# the standard f16 version. The matmul_quant shader decodes them with
# bitcast<f32>(word << 16u), preserving the full 8-bit bf16 exponent range.
_GDN_BF16 = os.environ.get("GDN_BF16", "0") == "1"


def _is_gdn_weight_key(key: str) -> bool:
    """True for GDN linear-attention projection weights that benefit from bf16 storage."""
    return "linear_attn" in key and any(
        p in key for p in ("in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z",
                           "out_proj", "conv1d")
    )

logger = logging.getLogger(__name__)

_GGUF_MAGIC = b"GGUF"

# Flush every 512 MB of pending write_buffer calls. Metal silently drops
# write_buffer operations when the GPU staging buffer queue is saturated
# (~1-2 GB). Periodic flushes prevent this for large single-file models.
_FLUSH_THRESHOLD = 512 * 1024 * 1024


from transformers.utils import SAFE_WEIGHTS_INDEX_NAME as _SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME


def _is_mlx_quantized_dir(p: Path) -> bool:
    """Return True if directory contains MLX affine int4 weights (has .biases keys)."""
    index_path = p / _SAFE_WEIGHTS_INDEX_NAME
    try:
        with open(index_path) as f:
            index = json.load(f)
        return any(k.endswith(".biases") for k in index.get("weight_map", {}))
    except Exception:
        return False


_UNSUPPORTED_QUANT_TYPES = frozenset({"aqlm", "hqq", "quip#", "quip"})


def _collect_mx_bases(header: dict) -> list:
    """Return sorted base names for MX-format weight pairs (*.weight + *.weight_scale, both U8)."""
    return sorted(
        base
        for k in header
        if k.endswith(".weight") and header[k].get("dtype") == "U8"
        and (base := k.removesuffix(".weight"))
        and header.get(base + ".weight_scale", {}).get("dtype") == "U8"
    )


def _load_quant_cfg(config_path: Path) -> dict:
    """Return the quantization config dict for a model.

    Uses compressed_tensors.get_quantization_config, which handles nested
    locations (text_config, compression_config) and multimodal variants.
    Returns {} on any failure.
    """
    try:
        from compressed_tensors import get_quantization_config as _get_ct_config
        return _get_ct_config(str(config_path)) or {}
    except Exception:
        return {}


def _check_unsupported_quant(model_dir: Path) -> None:
    """Raise ValueError if config.json names an unsupported quantization scheme.

    Detects AQLM, HQQ, and QuIP# by reading quant_type/quant_method from
    config.json. Uses compressed_tensors.get_quantization_config() so that
    quantization config nested under text_config.quantization_config (multimodal
    models like Gemma3/Qwen3.5-MM) and compression_config are both covered.
    These formats cannot be loaded as safetensors by this plugin; raise early
    with a clear message rather than silently loading wrong data.
    """
    config_json = model_dir / "config.json"
    if not config_json.exists():
        return
    qcfg = _load_quant_cfg(config_json)
    qt = (qcfg.get("quant_type") or qcfg.get("quant_method") or "").lower().strip()
    if qt in _UNSUPPORTED_QUANT_TYPES:
        raise ValueError(
            f"Quantization format {qt!r} is not supported by vllm-webgpu. "
            f"Supported formats: safetensors (fp16/bf16), GPTQ, AWQ, FP8, INT8, NF4, MLX-int4. "
            f"For {qt!r}, use a dedicated plugin or dequantize the model first."
        )


def _apply_multimodal_remap(weights: dict) -> int:
    """Remap language_model key prefixes for multimodal checkpoints.

    Gemma3 multimodal: 'language_model.X' -> 'X'
    Qwen3.5 multimodal: 'model.language_model.X' -> 'model.X'

    Adds remapped keys without removing originals (freeing non-LM GPU buffers
    causes Metal memory corruption on adjacent embeddings).
    Returns the number of keys added.
    """
    remapped = {}
    for k, v in weights.items():
        if k.startswith("model.language_model."):
            remapped["model." + k[len("model.language_model."):]] = v
        elif k.startswith("language_model."):
            remapped[k[len("language_model."):]] = v
    weights.update(remapped)
    return len(remapped)


def detect_weight_format(path: str) -> str:
    p = Path(path)
    if p.is_dir():
        _check_unsupported_quant(p)
        if (p / _SAFE_WEIGHTS_INDEX_NAME).exists():
            if _is_mlx_quantized_dir(p):
                return "mlx_int4"
            return "safetensors_sharded"
        if (p / SAFE_WEIGHTS_NAME).exists():
            return "safetensors"
        # No known safetensors manifest found in directory; default.
        return "safetensors"
    if p.suffix == ".gguf":
        return "gguf"
    if p.suffix == ".safetensors":
        return "safetensors"
    if p.suffix == ".bin":
        raise ValueError(
            f"Legacy .bin (PyTorch pickle) format not supported; convert to safetensors first: {path}"
        )
    # Try magic bytes
    with open(p, "rb") as f:
        magic = f.read(4)
    if magic == _GGUF_MAGIC:
        return "gguf"
    return "safetensors"


def load_safetensors_weights_sharded(
    model_dir: str,
    wgpu_device,
    f32_keys: "frozenset[str] | None" = None,
) -> dict:
    """Load multi-shard safetensors from a directory with model.safetensors.index.json."""
    index_path = Path(model_dir) / _SAFE_WEIGHTS_INDEX_NAME
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
        shard_weights = load_safetensors_weights(
            shard_path, wgpu_device, ct_meta=ct_meta, f32_keys=f32_keys,
            skip_remap=True)

        # Commit all pending write_buffer operations by submitting a dummy command encoder.
        # queue.write_buffer() is only committed before the NEXT queue.submit(), not by
        # on_submitted_work_done_sync() alone. Without this, 24GB of accumulated writes
        # may be committed simultaneously at the first real submit, causing Metal to
        # silently drop some writes (embedding buffer shows zeros after readback).
        wgpu_device.queue.submit([wgpu_device.create_command_encoder().finish()])
        wgpu_device.queue.on_submitted_work_done_sync()

        weights.update(shard_weights)

    if is_multimodal:
        n_remapped = _apply_multimodal_remap(weights)
        logger.info("Added %d remapped language_model keys", n_remapped)

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
    """Convert FP8 E4M3 (OCP float8_e4m3fn) to float32.

    Only bytes 0x7F and 0xFF are NaN; exp=15 with mant<7 (0x78-0x7E, 0xF8-0xFE)
    are valid normals (256-448 and their negatives). Use torch's native conversion
    to get the correct result for all 256 byte values.
    """
    import torch
    return (
        torch.from_numpy(np.ascontiguousarray(data))
        .view(torch.float8_e4m3fn)
        .to(torch.float32)
        .numpy()
    )


def _awq_qzeros_symmetric(qzeros: np.ndarray) -> bool:
    """Return True only when every AWQ qzero nibble is exactly 8 (symmetric midpoint).

    AWQ packs zero-points with nibble order [0,4,1,5,2,6,3,7], i.e. bit
    offsets [0,16,4,20,8,24,12,28].  Symmetric checkpoints produced by
    standard AWQ have all zero-points equal to 8 (midpoint of uint4).
    All-zero qzeros encode zero_point=0, which the GPU shader cannot handle
    correctly because the shader hardcodes `nibble - 8` and never receives
    qzeros. Those checkpoints must fall through to CPU dequantisation.
    Any other value means the checkpoint uses per-group asymmetric zeros
    and must also fall back to CPU dequantisation.
    """
    all_nibbles = (np.asarray(qzeros, dtype=np.int32)[..., np.newaxis] >> _AWQ_NIBBLE_SHIFTS) & 0xF
    return bool(np.all(all_nibbles == 8))


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

    qw = qweight.astype(np.int32)            # (K, N//8)
    qz = qzeros.astype(np.int32)             # (G, N//8)
    sc = scales.astype(np.float32)           # (G, N)

    # Unpack 8 nibbles per int32 → (K, N) uint8
    w_int4 = ((qw[:, :, np.newaxis] >> _AWQ_NIBBLE_SHIFTS) & 0xF).reshape(K, N).astype(np.uint8)
    z_int4 = ((qz[:, :, np.newaxis] >> _AWQ_NIBBLE_SHIFTS) & 0xF).reshape(G, N).astype(np.uint8)

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

    # Unpack 8 nibbles per int32 along K → (K, N) and zeros (G, N//8) → (G, N)
    shifts = _GPTQ_NIBBLE_SHIFTS
    w_int4 = ((qw[:, np.newaxis, :] >> shifts[:, np.newaxis]) & 0xF).reshape(K, N).astype(np.uint8)
    z_int4 = ((qz[:, :, np.newaxis] >> shifts) & 0xF).reshape(G, N).astype(np.uint8)

    # Group index: which group each input dim belongs to
    if g_idx is not None:
        groups = g_idx.astype(np.int32)
    else:
        groups = np.arange(K, dtype=np.int32) // group_size

    sc_exp = sc[groups]             # (K, N)
    z_exp  = z_int4[groups].astype(np.float32)  # (K, N)

    w_f32 = sc_exp * (w_int4.astype(np.float32) - z_exp)  # (K, N)
    return np.ascontiguousarray(w_f32.T.astype(np.float16))  # (N, K)


def _detect_mx_quant(model_dir: Path) -> str:
    """Detect MXFP4 or MXFP8 from config files in the model directory.

    Checks hf_quant_config.json (Nvidia/ModelOpt format) first, then
    config.json quantization_config.quant_type. Returns 'mxfp4', 'mxfp8', or ''.
    """
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
        qt = _load_quant_cfg(config_json).get("quant_type", "").lower()
        if qt == "mxfp4":
            return "mxfp4"
        if qt == "mxfp8":
            return "mxfp8"
    return ""


def detect_compressed_tensors_fmt(config_path: "str | Path") -> dict:
    """Read config.json and return compressed-tensors quantization metadata.

    compressed-tensors models embed a quantization_config with config_groups that
    describes the actual format. Returns a dict with key '__global__' mapped to
    {fmt, group_size} when detected, otherwise empty dict.

    Routing:
      8-bit int  + channel        -> fmt='int8_gpu'
      8-bit float + tensor/channel -> fmt='fp8_gpu'
      4-bit int  + group          -> fmt='gptq_gpu'
      other                       -> empty dict with a logged warning
    """
    p = Path(config_path)
    if not p.exists():
        return {}
    quant_cfg = _load_quant_cfg(p)
    if not quant_cfg.get("config_groups"):
        return {}
    try:
        from compressed_tensors import QuantizationConfig
        from compressed_tensors.quantization import QuantizationType, QuantizationStrategy
        cfg = QuantizationConfig.model_validate(quant_cfg)
        w_args = next((s.weights for s in cfg.config_groups.values() if s.weights), None)
    except Exception:
        return {}
    if w_args is None:
        return {}
    if w_args.num_bits == 8 and w_args.type == QuantizationType.INT and w_args.strategy == QuantizationStrategy.CHANNEL:
        return {"__global__": {"fmt": "int8_gpu", "group_size": None}}
    if w_args.num_bits == 8 and w_args.type == QuantizationType.FLOAT and w_args.strategy in (QuantizationStrategy.TENSOR, QuantizationStrategy.CHANNEL):
        return {"__global__": {"fmt": "fp8_gpu", "group_size": None}}
    if w_args.num_bits == 4 and w_args.type == QuantizationType.INT and w_args.strategy == QuantizationStrategy.GROUP:
        return {"__global__": {"fmt": "gptq_gpu", "group_size": w_args.group_size or 128}}
    logger.warning(
        "compressed-tensors: unsupported format (num_bits=%d, type=%s, strategy=%s), "
        "no quant_meta applied", w_args.num_bits, w_args.type, w_args.strategy)
    return {}


def load_safetensors_weights(
    path: str,
    wgpu_device,
    ct_meta: dict | None = None,
    f32_keys: "frozenset[str] | None" = None,
    skip_remap: bool = False,
) -> dict:
    """Load safetensors weights and upload to GPU as F16.

    Handles the following quantization formats (all dequantized on CPU):
    - BF16/F16/F32: direct cast to F16
    - AWQ int4: nibble-packed (K, N//8) with F16 scales and zero-points
    - GPTQ int4: nibble-packed (K//8, N) with F16 scales
    - FP8 E4M3: weight as F8_E4M3 + F32 per-tensor scale
    - NVFP4: weight_packed (U8 = 2 FP4/byte) + F8_E4M3 block scale + F32 global scale

    Uses queue.write_buffer (not mapped_at_creation) for reliable uploads.

    Args:
        ct_meta: Pre-computed compressed-tensors metadata from detect_compressed_tensors_fmt().
                 When None, config.json is read from the parent directory of path.
                 Pass this from load_safetensors_weights_sharded to avoid re-parsing per shard.
        f32_keys: Optional set of weight key names that must be uploaded as float32 instead
                  of the default float16. Intended for scalar/vector parameters (e.g. Mamba D
                  and dt_bias) where the checkpoint stores F32 and the shader reads f32 — the
                  default F16 downcast would silently lose 13 mantissa bits.
    """
    import safetensors.torch as sft
    import torch
    import wgpu as wgpu_lib
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer

    with sft.safe_open(path, framework="pt") as sf:
        # Build header for format detection from safetensors metadata.
        # get_slice() reads only the file header bytes — no tensor data is loaded yet.
        header = {}
        for _n in sf.keys():
            _t = sf.get_slice(_n)
            header[_n] = {
                "dtype": str(_t.get_dtype()).upper(),
                "shape": list(_t.get_shape()),
            }

        usage = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST

        # Detect compressed-tensors config from the model directory (needed for
        # pack-quantized INT4 format where weight dtype alone is insufficient).
        # Use the caller-supplied ct_meta when available to avoid re-parsing per shard.
        if ct_meta is None:
            ct_meta = detect_compressed_tensors_fmt(Path(path).parent / "config.json")

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
        # compressed-tensors pack-quantized INT4: .weight I32 + .weight_scale F16/BF16/F32
        # (e.g. google/gemma-4-12B-it-qat-w4a16-ct). The quantization_config in config.json
        # is read by detect_compressed_tensors_fmt() and stored in ct_meta; the weight tensors
        # themselves use different key names than standard GPTQ (.weight not .qweight, and
        # .weight_scale not .scales), so they need a dedicated loading path.
        has_ct_pack_int4 = (
            bool(ct_meta and ct_meta.get("__global__", {}).get("fmt") == "gptq_gpu")
            and any(
                header[k].get("dtype") == "I32"
                and (k.endswith(".weight") or k.endswith(".weight_packed"))
                and (
                    (k.endswith(".weight") and k[:-len(".weight")] + ".weight_scale" in header)
                    or (k.endswith(".weight_packed") and k[:-len(".weight_packed")] + ".weight_scale" in header)
                )
                for k in header if k != "__metadata__"
            )
        )

        if has_qweight:
            # Distinguish AWQ from GPTQ by qweight shape, not qzeros presence.
            # Both formats may have qzeros. The packing axis differs:
            #   GPTQ: qweight = (K//8, N), scales = (G, N)  → scales.shape[-1] == qweight.shape[-1]
            #   AWQ:  qweight = (K, N//8), scales = (G, N)  → scales.shape[-1] == qweight.shape[-1] * 8
            # Cross-referencing scales is definitive. If scales are absent, fall back
            # to the shape ratio: AWQ has shape[0] > shape[1]; GPTQ has shape[0] < shape[1].
            fmt = "gptq"
            for _qw_key in header:
                if _qw_key == "__metadata__" or not _qw_key.endswith(".qweight"):
                    continue
                _qw_shape = tuple(header[_qw_key]["shape"])
                _base = _qw_key[:-len(".qweight")]
                _sc_key = f"{_base}.scales"
                if _sc_key in header:
                    _sc_shape = tuple(header[_sc_key]["shape"])
                    if _sc_shape and _qw_shape and _sc_shape[-1] == _qw_shape[-1] * 8:
                        fmt = "awq"
                else:
                    # AWQ: shape[0] > shape[1] (K > N//8); GPTQ: shape[0] < shape[1] (K//8 < N).
                    if len(_qw_shape) == 2 and _qw_shape[0] > _qw_shape[1]:
                        fmt = "awq"
                break
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
        elif has_ct_pack_int4:
            fmt = "ct_pack_int4"
        else:
            fmt = "plain"

        if fmt != "plain":
            logger.info("Detected %s quantization in %s", fmt.upper(), path)

        def _load_raw(name: str) -> np.ndarray:
            """Load a tensor from the open safetensors file as numpy.

            Returns float32 for BF16 tensors (bit-shifted from uint16), uint8 for
            F8_E4M3 (raw bytes for the FP8 LUT decoder), and the native numpy dtype
            for all other formats (F16, F32, I32, U8, I8).
            """
            dtype_str = header[name]["dtype"]
            t = sf.get_tensor(name)        # torch.Tensor on CPU
            if dtype_str == "BF16":
                return t.to(torch.float32).numpy()
            if "F8_" in dtype_str:
                # FP8 variants (F8_E4M3, F8_E4M3FN, …): return raw uint8 bytes for
                # the LUT-based decoder in _fp8_e4m3_to_f32.
                return t.view(torch.uint8).numpy()
            return t.numpy()

        def _pad4(data: bytes) -> bytes:
            """Pad to 4-byte boundary — WebGPU write_buffer requires 4-byte-aligned size."""
            return data + b"\x00" * (-len(data) % 4)

        # Track pending write_buffer bytes to flush periodically.
        # Metal silently drops write_buffer operations when the pending write queue
        # exceeds the GPU staging buffer capacity (~1-2GB). For large single-file
        # models (e.g. Gemma4-12B at 22GB), we must flush periodically.
        _pending_bytes = 0

        def _maybe_flush() -> None:
            nonlocal _pending_bytes
            if _pending_bytes >= _FLUSH_THRESHOLD:
                wgpu_device.queue.submit([wgpu_device.create_command_encoder().finish()])
                wgpu_device.queue.on_submitted_work_done_sync()
                _pending_bytes = 0

        def _upload_u8(arr: np.ndarray, name: str, weights: dict) -> None:
            """Upload uint8 raw bytes to GPU (packed 4/u32 as shader binding).

            Used for FP8 E4M3 and NVFP4 packed weights. The shader reads via
            rd_byte_at() which unpacks individual bytes from the u32 array.
            """
            nonlocal _pending_bytes
            arr_flat = np.ascontiguousarray(arr.ravel().view(np.uint8))
            # Pad to multiple of 4 bytes so u32 reinterpretation is clean.
            data = _pad4(arr_flat.tobytes())
            buf = wgpu_device.create_buffer(size=len(data), usage=usage)
            wgpu_device.queue.write_buffer(buf, 0, data)
            _pending_bytes += len(data)
            _maybe_flush()
            weights[name] = WebGPUBuffer(buf=buf, device=wgpu_device,
                                         shape=tuple(arr.shape), dtype="u8")

        def _upload_int32(arr: np.ndarray, name: str, weights: dict) -> None:
            """Upload an INT32 array (quantized weights) directly to GPU without conversion."""
            nonlocal _pending_bytes
            arr = np.ascontiguousarray(arr.astype(np.int32))
            data = _pad4(arr.tobytes())
            buf = wgpu_device.create_buffer(size=len(data), usage=usage)
            wgpu_device.queue.write_buffer(buf, 0, data)
            _pending_bytes += len(data)
            _maybe_flush()
            weights[name] = WebGPUBuffer(buf=buf, device=wgpu_device,
                                         shape=tuple(arr.shape), dtype="i32")

        def _upload_f16(arr: np.ndarray, name: str, weights: dict) -> None:
            """Upload an F16 array (scales/norms) directly to GPU."""
            nonlocal _pending_bytes
            arr = np.ascontiguousarray(arr.astype(np.float16))
            data = _pad4(arr.tobytes())
            buf = wgpu_device.create_buffer(size=len(data), usage=usage)
            wgpu_device.queue.write_buffer(buf, 0, data)
            _pending_bytes += len(data)
            _maybe_flush()
            weights[name] = WebGPUBuffer(buf=buf, device=wgpu_device,
                                         shape=tuple(arr.shape), dtype="f16")

        weights: dict = {}

        # Keys that _upload_plain consumed as I8 companion scales. These are
        # re-encountered in the outer iteration but should not be re-uploaded.
        _i8_companion_skip: set = set()

        # ── Helper: upload a single tensor from the header (plain dtypes) ──────────
        def _upload_plain(name: str, weights: dict) -> bool:  # noqa: E501
            nonlocal _pending_bytes
            meta = header.get(name)
            if meta is None:
                return False
            dtype_str = meta["dtype"]
            shape = tuple(meta["shape"])

            # Keep native F32 precision for explicitly requested keys (e.g. Mamba D
            # and dt_bias). The default path casts every F32/BF16 checkpoint value to
            # F16, silently discarding 13 mantissa bits. Shaders that declare these
            # bindings as array<f32> need the full-precision values.
            if f32_keys and name in f32_keys and dtype_str in ("F32", "BF16", "F16"):
                if dtype_str == "BF16":
                    arr_f32 = np.ascontiguousarray(
                        sf.get_tensor(name).to(torch.float32).numpy())
                else:
                    arr_f32 = np.ascontiguousarray(
                        sf.get_tensor(name).numpy().astype(np.float32))
                data = _pad4(arr_f32.tobytes())
                buf = wgpu_device.create_buffer(size=len(data), usage=usage)
                wgpu_device.queue.write_buffer(buf, 0, data)
                _pending_bytes += len(data)
                _maybe_flush()
                weights[name] = WebGPUBuffer(
                    buf=buf, device=wgpu_device,
                    shape=tuple(arr_f32.shape), dtype="f32")
                return True

            if dtype_str == "F16":
                arr = sf.get_tensor(name).numpy()    # torch.float16 → np.float16
            elif dtype_str == "BF16":
                t_bf16 = sf.get_tensor(name)
                if _GDN_BF16 and _is_gdn_weight_key(name):
                    # Preserve bf16 bit pattern: pack u16 pairs into u32 (same storage
                    # cost as f16 pairs). The shader decodes via bitcast<f32>(w << 16u),
                    # recovering the full 8-bit bf16 exponent — avoids f16 range loss.
                    u16 = t_bf16.view(torch.int16).numpy().view(np.uint16)
                    u16_flat = np.ascontiguousarray(u16.ravel())
                    if u16_flat.size % 2 != 0:
                        u16_flat = np.concatenate([u16_flat, np.zeros(1, dtype=np.uint16)])
                    arr_u32 = u16_flat.view(np.uint32)
                    data_u32 = _pad4(arr_u32.tobytes())
                    buf_bf16 = wgpu_device.create_buffer(size=len(data_u32), usage=usage)
                    wgpu_device.queue.write_buffer(buf_bf16, 0, data_u32)
                    _pending_bytes += len(data_u32)
                    _maybe_flush()
                    weights[name + "__bf16"] = WebGPUBuffer(buf=buf_bf16, device=wgpu_device,
                                                            shape=tuple(shape), dtype="u32")
                f32 = t_bf16.to(torch.float32).numpy()
                arr = np.clip(f32, -_F16_MAX, _F16_MAX).astype(np.float16)
            elif dtype_str == "F32":
                arr = np.clip(sf.get_tensor(name).numpy(), -_F16_MAX, _F16_MAX).astype(np.float16)
            elif dtype_str == "I8":
                # Int8 per-channel weight (BnB int8 / compressed-tensors int8).
                # Upload raw bytes; shader does sign extension via int8_to_f32().
                # dtype="u8" so _uq_weight() detects it via fmt="int8_gpu".
                arr_u8 = sf.get_tensor(name).numpy().view(np.uint8).reshape(shape)
                _upload_u8(arr_u8, name, weights)
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
                            sc_dtype = header[sc_key]["dtype"]
                            sc_t = sf.get_tensor(sc_key)
                            if sc_dtype in ("F32", "F16"):
                                sc_arr = sc_t.numpy().ravel()
                            elif sc_dtype == "BF16":
                                sc_arr = sc_t.to(torch.float32).numpy().ravel()
                            else:
                                logger.warning("Int8 scale %s has unsupported dtype %s",
                                               sc_key, sc_dtype)
                                break
                            _upload_f16(sc_arr, f"{name}.scales", weights)
                            qmeta[base_key]["group_size"] = 1
                            logger.debug("Int8 per-channel: %s scale n=%d", base_key, sc_arr.size)
                        except Exception as exc:
                            logger.warning("Int8 scale load failed for %s: %s", base_key, exc)
                        _i8_companion_skip.add(sc_key)
                        break
                return True
            else:
                return False  # not a plain dtype
            arr = np.ascontiguousarray(arr)
            data = _pad4(arr.tobytes())
            buf = wgpu_device.create_buffer(size=len(data), usage=usage)
            wgpu_device.queue.write_buffer(buf, 0, data)
            _pending_bytes += len(data)
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
                if name == "__metadata__" or name in quant_set or name in _i8_companion_skip:
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
                    if fmt == "gptq" and qz is None and g_idx is None:
                        # GPU GPTQ: transpose qweight [K//8, N] → [N, K//8] for coalesced access.
                        K8, N_ = qw.shape
                        group_size = (K8 * 8) // sc.shape[0] if sc.ndim == 2 else 128
                        qw_t = np.ascontiguousarray(qw.T)  # [N, K//8]
                        sc_gn = sc.astype(np.float16)      # [G, N] f16
                        _upload_int32(qw_t, f"{base}.weight", weights)
                        _upload_f16(sc_gn, f"{base}.weight.scales", weights)
                        weights.setdefault("__quant_meta__", {})[base] = {"fmt": "gptq_sym", "group_size": group_size}
                        logger.debug("GPU GPTQ: %s (K=%d, N=%d, G=%d)", base, K8*8, N_, sc.shape[0])
                    elif (fmt == "awq" and qz is not None and g_idx is None
                          and _awq_qzeros_symmetric(qz)):
                        # GPU AWQ: qzeros verified symmetric (every nibble is 0 or 8).
                        # Asymmetric checkpoints fall through to the CPU dequant branch.
                        K_, N8_ = qw.shape  # qw is [K, N//8]
                        N_ = N8_ * 8
                        G_ = sc.shape[0] if sc.ndim == 2 else K_ // 128
                        group_size = K_ // G_ if G_ > 0 else 128
                        # GPU AWQ: store [K, N//8] INT32 directly (no transpose needed
                        # since AWQ access pattern is already per-k, per-output-group)
                        sc_gn = sc.astype(np.float16) if sc.ndim == 2 else sc  # [G, N]
                        _upload_int32(qw, f"{base}.weight", weights)    # [K, N//8]
                        _upload_f16(sc_gn, f"{base}.weight.scales", weights)  # [G, N]
                        weights.setdefault("__quant_meta__", {})[base] = {"fmt": "awq_sym", "group_size": group_size}
                        logger.debug("GPU AWQ: %s (K=%d, N=%d, G=%d)", base, K_, N_, G_)
                    else:
                        # Fall back to CPU dequantization.
                        if fmt == "awq" and qz is not None:
                            w_f16 = _dequant_awq(qw, sc, qz)
                        elif fmt == "awq" and qz is None:
                            # AWQ without qzeros: assume symmetric (all zero-points = 8).
                            # Symmetric AWQ uses zero_point=8 (uint4 midpoint), so every nibble
                            # is 8, encoded as 0x88888888 per int32 word.
                            # qzeros shape is (G, N//8) where G=sc.shape[0], N//8=qw.shape[1].
                            qz_sym = np.full((sc.shape[0], qw.shape[1]), fill_value=0x88888888, dtype=np.int32)
                            w_f16 = _dequant_awq(qw, sc, qz_sym)
                        else:
                            w_f16 = _dequant_gptq(
                                qw, sc,
                                qz if qz is not None else np.full((sc.shape[0], qw.shape[1] // 8), 0x88888888, dtype=np.int32),
                                g_idx)
                        _upload_f16(w_f16, f"{base}.weight", weights)
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
                if name == "__metadata__" or name in nvfp4_set or name in _i8_companion_skip:
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
                    weights.setdefault("__quant_meta__", {})[base] = {
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
                if name == "__metadata__" or name in dnvfp4_set or name in _i8_companion_skip:
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
                    weights.setdefault("__quant_meta__", {})[base] = {
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
                if name == "__metadata__" or name in fp8_set or name in _i8_companion_skip:
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
            mxfp4_bases = _collect_mx_bases(header)
            mx4_set: set = set()
            for base in mxfp4_bases:
                mx4_set.add(f"{base}.weight")
                mx4_set.add(f"{base}.weight_scale")

            for name in header:
                if name == "__metadata__" or name in mx4_set or name in _i8_companion_skip:
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
                    weights.setdefault("__quant_meta__", {})[base] = {
                        "fmt": "nvfp4_gpu", "global_scale": 1.0, "group_size": 32}
                    logger.debug("GPU MXFP4: %s (N=%d, K=%d)", base, N_, K_)
                except Exception as exc:
                    logger.warning("Failed to process MXFP4 %s: %s", base, exc)

        elif fmt == "mxfp8":
            # MXFP8 (microscaling FP8): *.weight [N, K] U8 FP8-E4M3 + *.weight_scale [N, K//32] U8 exponents.
            # Scales are u8 exponents: scale = 2^(u8 - 127), one per block of 32 K-elements.
            # CPU dequant: avoids shader changes for per-block FP8.
            # TODO: USE_QUANT=9 for GPU MXFP8 per-block decode
            mxfp8_bases = _collect_mx_bases(header)
            mx8_set: set = set()
            for base in mxfp8_bases:
                mx8_set.add(f"{base}.weight")
                mx8_set.add(f"{base}.weight_scale")

            for name in header:
                if name == "__metadata__" or name in mx8_set or name in _i8_companion_skip:
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
                        np.clip(w_f32 * block_scale_exp, -_F16_MAX, _F16_MAX).astype(np.float16))
                    _upload_f16(w_f16, f"{base}.weight", weights)
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
                if name == "__metadata__" or name in bnb_set or name in _i8_companion_skip:
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

        elif fmt == "ct_pack_int4":
            # compressed-tensors pack-quantized INT4 (W4A16).
            # Weight: {base}.weight [N, K//8] I32 (8 nibbles/u32, already in [N,K//8] layout)
            # Scale:  {base}.weight_scale [N, G] F16/BF16/F32 → transpose to [G, N] for shader
            # group_size comes from the quantization_config parsed in ct_meta.
            _ct_group_size = ct_meta["__global__"]["group_size"]

            # Collect (base, weight_key) pairs for all I32 packed weight tensors.
            # Canonical compressed-tensors saves as .weight_packed; some checkpoints use .weight.
            _ct_base_map = {}  # base -> weight_key
            for _k in header:
                if _k == "__metadata__":
                    continue
                if header[_k].get("dtype") != "I32":
                    continue
                if _k.endswith(".weight_packed"):
                    _base = _k[:-len(".weight_packed")]
                    if _base + ".weight_scale" in header:
                        _ct_base_map.setdefault(_base, _k)
                elif _k.endswith(".weight"):
                    _base = _k[:-len(".weight")]
                    if _base + ".weight_scale" in header:
                        _ct_base_map.setdefault(_base, _k)
            ct_bases = sorted(_ct_base_map)
            ct_reserved = set()
            for _b in ct_bases:
                ct_reserved.add(_ct_base_map[_b])
                ct_reserved.add(f"{_b}.weight_scale")

            for name in header:
                if name == "__metadata__" or name in ct_reserved or name in _i8_companion_skip:
                    continue
                if not _upload_plain(name, weights):
                    logger.warning("Skipping %s (dtype=%s)", name, header[name].get("dtype", "?"))

            weights.setdefault("__quant_meta__", {})
            for base in ct_bases:
                try:
                    weight_key = _ct_base_map[base]
                    qw = _load_raw(weight_key)                 # [N, K//8] I32
                    sc_raw = _load_raw(f"{base}.weight_scale") # [N, G] F16/BF16/F32

                    sc = sc_raw.astype(np.float16)

                    # Transpose scale [N, G] → [G, N] to match shader expectation.
                    if sc.ndim == 2:
                        sc = np.ascontiguousarray(sc.T)

                    _upload_int32(qw, f"{base}.weight", weights)
                    _upload_f16(sc, f"{base}.weight.scales", weights)
                    weights["__quant_meta__"][base] = {
                        "fmt": "gptq_sym",
                        "group_size": _ct_group_size,
                    }
                    logger.debug(
                        "CT pack-int4: %s key=%s (N=%d, K=%d, G=%d, group_size=%d)",
                        base, weight_key, qw.shape[0], qw.shape[1] * 8,
                        sc.shape[0], _ct_group_size)
                except Exception as exc:
                    logger.warning("Failed to process CT pack-int4 %s: %s", base, exc)

        else:
            # Plain BF16/F16/F32
            for name, meta in header.items():
                if name == "__metadata__" or name in _i8_companion_skip:
                    continue
                if not _upload_plain(name, weights):
                    logger.warning("Unsupported dtype %s for %s, skipping",
                                    meta.get("dtype", "?"), name)

        # Multimodal remapping for single-file models (same patterns as sharded loader).
        # Gemma4 unified: model.language_model.X → model.X
        # Gemma3 multimodal (rare single-file): language_model.X → X
        # Skipped when called from load_safetensors_weights_sharded, which applies
        # the combined remap once after all shards are merged.
        if not skip_remap:
            keys = list(weights.keys())
            if any(k.startswith("model.language_model.") or k.startswith("language_model.") for k in keys):
                n_remapped = _apply_multimodal_remap(weights)
                if n_remapped:
                    logger.info("Remapped %d language_model.* keys", n_remapped)

        # Commit all pending write_buffer calls before returning.
        wgpu_device.queue.submit([wgpu_device.create_command_encoder().finish()])
        wgpu_device.queue.on_submitted_work_done_sync()
        logger.info("Loaded %d tensors from %s", len(weights), path)
        return weights


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

    mlx.core.dequantize() covers the same operation, but mlx is not required
    at inference time (MLX-format checkpoints can be loaded without mlx installed,
    as long as the weights are dequantized to f16 before upload). The numpy path
    keeps mlx optional and avoids the Metal-device init that mlx triggers on import.
    """
    out_rows, packed_cols = weight_u32.shape
    in_cols = packed_cols * 8
    w = weight_u32.astype(np.uint32)
    shifts = np.arange(8, dtype=np.uint32) * 4
    nibbles = ((w[:, :, np.newaxis] >> shifts) & 0xF).reshape(out_rows, in_cols).astype(np.float32)
    n_groups = in_cols // group_size
    scales_bc = np.repeat(scales_f32.reshape(out_rows, n_groups), group_size, axis=1)
    biases_bc = np.repeat(biases_f32.reshape(out_rows, n_groups), group_size, axis=1)
    return scales_bc * nibbles + biases_bc


def load_mlx_weights(model_dir: str, wgpu_device) -> dict:
    """Load MLX affine int4 safetensors weights, dequantize to f16, upload to GPU."""
    import torch as _torch
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer

    p = Path(model_dir)
    index_path = p / _SAFE_WEIGHTS_INDEX_NAME
    with open(index_path) as f:
        index = json.load(f)

    group_size = 64
    config_path = p / "config.json"
    if config_path.exists():
        # Try compressed_tensors-aware loader for quantization_config.group_size first,
        # then fall back to the raw 'quantization' key used by some MLX formats.
        qs = _load_quant_cfg(config_path).get("group_size")
        if not qs:
            with open(config_path) as f:
                cfg_raw = json.load(f)
            qs = cfg_raw.get("quantization", {}).get("group_size")
        if qs:
            group_size = int(qs)

    weight_map: dict = index["weight_map"]

    # Pass 1: build key -> shard_path index without loading any tensor data.
    key_to_shard: dict[str, str] = {k: str(p / v) for k, v in weight_map.items()}
    all_keys: set[str] = set(key_to_shard)

    import safetensors.torch as _sft

    # Pass 2: process tensors shard-by-shard, opening each shard at most once per group.
    # Quantized triplets (weight + scales + biases) are loaded together from their
    # respective shards; non-quantized tensors are streamed one shard at a time.
    weights: dict = {}
    processed: set = set()

    # First handle quantized triplets: find all .weight keys that form a quant group.
    quant_bases: list[str] = []
    for key in sorted(all_keys):
        if key.endswith(".weight"):
            base = key[:-len(".weight")]
            if base + ".scales" in all_keys and base + ".biases" in all_keys:
                quant_bases.append(base)

    # Group quantized triplets by the shard that holds the .weight key.
    # Opening a shard once per base avoids re-opening the same file for .scales and .biases.
    shard_to_quant_bases: dict[str, list[str]] = defaultdict(list)
    for base in quant_bases:
        shard_to_quant_bases[key_to_shard[base + ".weight"]].append(base)

    for shard_path, bases in shard_to_quant_bases.items():
        with _sft.safe_open(shard_path, framework="pt") as sf_w:
            for base in bases:
                wk = base + ".weight"
                sk = base + ".scales"
                bk = base + ".biases"
                processed.update({wk, sk, bk})

                t = sf_w.get_tensor(wk)
                if t.dtype != _torch.uint32:
                    # Not actually an int4 weight; upload as plain float.
                    processed.discard(sk)
                    processed.discard(bk)
                    arr = t.to(_torch.float16).numpy() if t.dtype == _torch.float16 else np.clip(t.to(_torch.float32).numpy(), -_F16_MAX, _F16_MAX).astype(np.float16)
                    local_key = wk.removeprefix("language_model.")
                    weights[local_key] = WebGPUBuffer.from_numpy(wgpu_device, np.ascontiguousarray(arr))
                    continue

                # Load scales and biases from their respective shards (may differ from weight shard).
                with _sft.safe_open(key_to_shard[sk], framework="pt") as sf_s:
                    s_t = sf_s.get_tensor(sk)
                with _sft.safe_open(key_to_shard[bk], framework="pt") as sf_b:
                    b_t = sf_b.get_tensor(bk)
                w_u32 = t.numpy()
                scales_f32 = s_t.to(_torch.float32).numpy()
                biases_f32 = b_t.to(_torch.float32).numpy()
                dequant = _dequant_mlx_int4(w_u32, scales_f32, biases_f32, group_size)
                arr = np.clip(dequant, -_F16_MAX, _F16_MAX).astype(np.float16)
                local_key = base.removeprefix("language_model.") + ".weight"
                weights[local_key] = WebGPUBuffer.from_numpy(wgpu_device, np.ascontiguousarray(arr))

    # Stream non-quantized tensors shard-by-shard.
    shard_files = sorted(set(weight_map.values()))
    for shard_file in shard_files:
        shard_path = str(p / shard_file)
        logger.info("Loading MLX shard %s", shard_file)
        with _sft.safe_open(shard_path, framework="pt") as sf:
            for key in sf.keys():
                if key in processed:
                    continue
                t = sf.get_tensor(key)
                if t.dtype == _torch.bfloat16:
                    arr = np.clip(t.to(_torch.float32).numpy(), -_F16_MAX, _F16_MAX).astype(np.float16)
                elif t.dtype == _torch.float32:
                    arr = np.clip(t.numpy(), -_F16_MAX, _F16_MAX).astype(np.float16)
                elif t.dtype == _torch.float16:
                    arr = t.numpy()
                else:
                    logger.warning("Unsupported dtype %s for tensor %s, skipping", t.dtype, key)
                    continue
                local_key = key.removeprefix("language_model.")
                weights[local_key] = WebGPUBuffer.from_numpy(wgpu_device, np.ascontiguousarray(arr))

    _apply_multimodal_remap(weights)
    logger.info("Loaded %d tensors from MLX int4 dir %s", len(weights), model_dir)
    return weights
