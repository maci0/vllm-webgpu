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
    # Detect if this is a multimodal model with language_model.* prefix by checking the index.
    is_multimodal = any(
        k.startswith("language_model.")
        for k in index.get("weight_map", {})
    )
    if is_multimodal:
        logger.info("Multimodal model detected; remapping language_model.* prefix")

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
        # Add remapped language_model.* keys WITHOUT removing the originals.
        # Removing non-LM buffers (e.g. vision encoder) causes Metal GPU memory
        # corruption: freeing GPU buffers that neighbored the embed buffer zeros it out.
        # Keeping all buffers alive avoids this Metal driver quirk at the cost of
        # ~1-2GB extra VRAM for the vision encoder weights (harmless, they're unused).
        remapped = {}
        for k, v in weights.items():
            if k.startswith("language_model."):
                remapped[k[len("language_model."):]] = v
        weights.update(remapped)
        logger.info("Added %d remapped language_model.* keys", len(remapped))

    logger.info("Loaded %d tensors from %d shards in %s", len(weights), len(shard_files), model_dir)
    return weights


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
    """Load safetensors weights, cast bf16->f16, upload to GPU.

    Handles plain BF16/F16/F32, AWQ int4, and GPTQ int4 safetensors.
    Quantized weights are dequantized on CPU and uploaded as F16 to the GPU.

    Uses queue.write_buffer (not mapped_at_creation) to upload data.
    mapped_at_creation corrupts large buffers (>1GB) when many GPU buffers
    coexist in the same process — the Metal driver appears to lose the
    written data. queue.write_buffer is reliable at any size.
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

    # Check for quantized format
    quant_bases = sorted(set(
        k[:-len(".qweight")]
        for k in header
        if k.endswith(".qweight") and k != "__metadata__"
    ))
    is_quantized = len(quant_bases) > 0
    if is_quantized:
        fmt = "awq" if any(k.endswith(".qzeros") for k in header) else "gptq"
        logger.info("Detected %s quantization (%d layers)", fmt.upper(), len(quant_bases))

    usage = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST

    def _load_raw(name: str) -> np.ndarray:
        """Load one tensor as a numpy array (CPU only, no GPU upload)."""
        meta = header[name]
        dtype_str = meta["dtype"]
        start, end = meta["data_offsets"]
        raw = raw_data[start:end]
        shape = tuple(meta["shape"])
        if dtype_str == "I32":
            return np.frombuffer(raw, dtype=np.int32).reshape(shape)
        if dtype_str == "F16":
            return np.frombuffer(raw, dtype=np.float16).reshape(shape)
        if dtype_str == "BF16":
            u16 = np.frombuffer(raw, dtype=np.uint16)
            f32 = (u16.astype(np.uint32) << 16).view(np.float32)
            return f32.reshape(shape)
        if dtype_str == "F32":
            return np.frombuffer(raw, dtype=np.float32).reshape(shape)
        raise ValueError(f"Unsupported dtype {dtype_str} for {name}")

    def _upload(arr: np.ndarray, name: str, weights: dict) -> None:
        """Upload F16 array to GPU and record in weights dict."""
        arr = np.ascontiguousarray(arr.astype(np.float16))
        buf = wgpu_device.create_buffer(size=arr.nbytes, usage=usage)
        wgpu_device.queue.write_buffer(buf, 0, arr.tobytes())
        weights[name] = WebGPUBuffer(buf=buf, device=wgpu_device,
                                     shape=tuple(arr.shape), dtype="f16")

    weights: dict = {}

    if is_quantized:
        # Build set of keys that are part of quantized layers
        quant_set = set()
        for base in quant_bases:
            for suffix in (".qweight", ".scales", ".qzeros", ".g_idx", ".bias"):
                if f"{base}{suffix}" in header:
                    quant_set.add(f"{base}{suffix}")

        # Process all tensors
        for name, meta in header.items():
            if name == "__metadata__" or name in quant_set:
                continue
            dtype_str = meta["dtype"]
            if dtype_str not in ("F16", "BF16", "F32"):
                logger.warning("Unsupported dtype %s for %s, skipping", dtype_str, name)
                continue
            _upload(_load_raw(name), name, weights)

        # Dequantize quantized layers and upload as .weight
        for base in quant_bases:
            try:
                qw    = _load_raw(f"{base}.qweight")
                sc    = _load_raw(f"{base}.scales")
                qz_key = f"{base}.qzeros"
                qz    = _load_raw(qz_key) if qz_key in header else None
                g_idx_key = f"{base}.g_idx"
                g_idx = _load_raw(g_idx_key) if g_idx_key in header else None

                if fmt == "awq" and qz is not None:
                    w_f16 = _dequant_awq(qw, sc, qz)
                else:
                    w_f16 = _dequant_gptq(qw, sc, qz if qz is not None else np.zeros_like(sc),
                                          g_idx)

                _upload(w_f16, f"{base}.weight", weights)
            except Exception as exc:
                logger.warning("Failed to dequantize %s: %s", base, exc)
    else:
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
                u16 = np.frombuffer(raw, dtype=np.uint16)
                f32 = (u16.astype(np.uint32) << 16).view(np.float32)
                f32 = np.clip(f32, -65504.0, 65504.0)
                arr = f32.reshape(shape).astype(np.float16)
            elif dtype_str == "F32":
                arr = np.frombuffer(raw, dtype=np.float32).reshape(shape).astype(np.float16)
            else:
                logger.warning("Unsupported dtype %s for tensor %s, skipping", dtype_str, name)
                continue

            arr = np.ascontiguousarray(arr)
            buf = wgpu_device.create_buffer(size=arr.nbytes, usage=usage)
            wgpu_device.queue.write_buffer(buf, 0, arr.tobytes())
            weights[name] = WebGPUBuffer(buf=buf, device=wgpu_device,
                                         shape=tuple(arr.shape), dtype="f16")

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
