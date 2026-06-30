"""
Integration smoke test: plugin registration -> platform -> worker -> one decode step.
Requires a real WebGPU adapter and a tiny Llama-shaped safetensors checkpoint.
Skips if no GPU or no checkpoint available.
"""
import json
import logging
import struct
from pathlib import Path

import numpy as np
import pytest

logger = logging.getLogger(__name__)


def _write_tiny_safetensors(tmp_dir: Path, hidden: int, layers: int, vocab: int, heads: int, kv_heads: int, inter: int) -> Path:
    """Write a tiny Llama-shaped safetensors checkpoint for smoke testing."""
    tensors = {}

    def r(*shape): return np.random.randn(*shape).astype(np.float16)

    tensors["model.embed_tokens.weight"] = r(vocab, hidden)
    tensors["model.norm.weight"] = r(hidden)
    for i in range(layers):
        p = f"model.layers.{i}"
        tensors[f"{p}.input_layernorm.weight"] = r(hidden)
        tensors[f"{p}.post_attention_layernorm.weight"] = r(hidden)
        tensors[f"{p}.self_attn.q_proj.weight"] = r(heads * (hidden // heads), hidden)
        tensors[f"{p}.self_attn.k_proj.weight"] = r(kv_heads * (hidden // heads), hidden)
        tensors[f"{p}.self_attn.v_proj.weight"] = r(kv_heads * (hidden // heads), hidden)
        tensors[f"{p}.self_attn.o_proj.weight"] = r(hidden, heads * (hidden // heads))
        tensors[f"{p}.mlp.gate_proj.weight"] = r(inter, hidden)
        tensors[f"{p}.mlp.up_proj.weight"] = r(inter, hidden)
        tensors[f"{p}.mlp.down_proj.weight"] = r(hidden, inter)

    metadata = {}
    offset = 0
    data_parts = []
    for name, arr in tensors.items():
        metadata[name] = {"dtype": "F16", "shape": list(arr.shape),
                          "data_offsets": [offset, offset + arr.nbytes]}
        data_parts.append(arr.tobytes())
        offset += arr.nbytes

    header_bytes = json.dumps(metadata).encode("utf-8")
    out = tmp_dir / "model.safetensors"
    out.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(data_parts))
    return out


def _write_tiny_hf_config(tmp_dir: Path, hidden, layers, vocab, heads, kv_heads, inter) -> Path:
    cfg = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "intermediate_size": inter,
        "vocab_size": vocab,
        "max_position_embeddings": 128,
        "rope_theta": 10000.0,
        "model_type": "llama",
    }
    out = tmp_dir / "config.json"
    out.write_text(json.dumps(cfg))
    return out


@pytest.mark.integration
def test_full_pipeline_smoke(wgpu_device, tmp_path):
    """
    Smoke test: build model, load tiny weights, run one decode step, get a token.
    Does NOT go through vLLM's full engine -- tests only the plugin's own stack.
    """
    hidden, layers, vocab, heads, kv_heads, inter = 64, 2, 128, 4, 2, 128
    head_dim = hidden // heads
    block_size = 16
    num_blocks = 8

    weight_path = _write_tiny_safetensors(tmp_path, hidden, layers, vocab, heads, kv_heads, inter)
    _write_tiny_hf_config(tmp_path, hidden, layers, vocab, heads, kv_heads, inter)

    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.models.llama import LlamaWebGPUModel
    from vllm_webgpu.utils import SHADERS_DIR
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    import wgpu as wgpu_lib

    cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)

    class _FakeConfig:
        hidden_size = hidden
        num_hidden_layers = layers
        num_attention_heads = heads
        num_key_value_heads = kv_heads
        intermediate_size = inter
        vocab_size = vocab
        max_position_embeddings = 128
        rope_theta = 10000.0
        architectures = ["LlamaForCausalLM"]

    model = LlamaWebGPUModel(_FakeConfig(), wgpu_device, cache)
    model.load_weights(str(weight_path))
    assert len(model.weights) > 0

    # Allocate KV cache
    dev = wgpu_device.wgpu_device
    rw = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST
    for _ in range(layers):
        k = WebGPUBuffer.empty(dev, num_blocks * block_size * kv_heads * head_dim * 2, usage=rw)
        v = WebGPUBuffer.empty(dev, num_blocks * block_size * kv_heads * head_dim * 2, usage=rw)
        model.kv_pool.append((k, v))

    # Build fake attn_metadata
    class _FakeMeta:
        slot_mapping = [0]
        block_tables = [np.zeros(num_blocks, dtype=np.uint32)]
        max_decode_seq_len = 1

    input_ids = np.array([42], dtype=np.uint32)
    positions = np.array([0], dtype=np.uint32)

    logits = model.forward(input_ids, positions, _FakeMeta())

    assert logits.shape == (1, vocab), f"Expected (1, {vocab}), got {logits.shape}"
    assert np.isfinite(logits).all(), "Logits contain NaN or Inf"
    token_id = int(logits.argmax(axis=-1)[0])
    assert 0 <= token_id < vocab
    logger.info("Smoke test passed: predicted token_id=%d", token_id)
