from __future__ import annotations
import logging
import struct
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_GGUF_MAGIC = b"GGUF"


def detect_weight_format(path: str) -> str:
    p = Path(path)
    if p.is_dir():
        if (p / "model.safetensors.index.json").exists():
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
    for shard in shard_files:
        shard_path = str(Path(model_dir) / shard)
        logger.info("Loading shard %s", shard)
        shard_weights = load_safetensors_weights(shard_path, wgpu_device)
        weights.update(shard_weights)
    logger.info("Loaded %d tensors from %d shards in %s", len(weights), len(shard_files), model_dir)
    return weights


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
    "ffn_norm.weight":           "post_attention_layernorm.weight",
    "attn_q_norm.weight":        "self_attn.q_norm.weight",
    "attn_k_norm.weight":        "self_attn.k_norm.weight",
    # Gemma4-specific
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "post_ffw_norm.weight":      "post_feedforward_layernorm.weight",
    "layer_output_scale.weight": "self_attn.layer_scale",
}


def _gguf_to_hf_name(gguf_name: str) -> str | None:
    """Map a GGUF tensor name to its HuggingFace equivalent. Returns None to skip."""
    if gguf_name in _GGUF_TO_HF_LLAMA:
        return _GGUF_TO_HF_LLAMA[gguf_name]
    # blk.{i}.{suffix} pattern
    if gguf_name.startswith("blk."):
        parts = gguf_name.split(".", 2)  # ["blk", "i", "suffix"]
        if len(parts) == 3:
            layer_idx = parts[1]
            suffix = parts[2]
            hf_suffix = _GGUF_BLK_MAP.get(suffix)
            if hf_suffix:
                return f"model.layers.{layer_idx}.{hf_suffix}"
    # Unknown tensor (e.g. rope_freqs, vision layers): skip
    return None


def load_gguf_weights(path: str, wgpu_device) -> dict:
    """Load GGUF weights with HF-style key remapping. Raw quantized blocks uploaded as u8."""
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    import wgpu as wgpu_lib

    try:
        import gguf
    except ImportError as e:
        raise ImportError("Install the 'gguf' package to load GGUF files.") from e

    reader = gguf.GGUFReader(path)
    weights: dict = {}
    usage = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_DST | wgpu_lib.BufferUsage.COPY_SRC
    skipped = 0

    for tensor in reader.tensors:
        hf_name = _gguf_to_hf_name(tensor.name)
        if hf_name is None:
            skipped += 1
            continue
        data = tensor.data
        arr = np.frombuffer(data, dtype=np.uint8)
        weights[hf_name] = WebGPUBuffer.from_numpy(wgpu_device, arr, usage=usage)

    logger.info("Loaded %d GGUF tensors (%d skipped) from %s", len(weights), skipped, path)
    return weights


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
        "vocab_size":        151936,
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

    logger.info("GGUF config: arch=%s hid=%d layers=%d heads=%d/%d head_dim=%d inter=%d",
                arch_prefix, cfg["hidden_size"], cfg["num_hidden_layers"],
                cfg["num_attention_heads"], cfg["num_key_value_heads"],
                cfg["head_dim"], cfg["intermediate_size"])
    return cfg
