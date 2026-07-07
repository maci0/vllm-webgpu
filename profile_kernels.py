#!/usr/bin/env python3
"""Profile GPU kernel timing per transformer layer.

Usage:
    source .venv/bin/activate
    python3 profile_kernels.py [--model MODEL_PATH] [--tokens N]
"""
import argparse
import sys
import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3-4B")
parser.add_argument("--prompt", default="The capital of France is Paris and the Eiffel Tower")
parser.add_argument("--decode-steps", type=int, default=5,
                    help="Number of decode steps to profile (averaged)")
parser.add_argument("--warmup-steps", type=int, default=2,
                    help="Warmup steps before profiling (not counted)")
args = parser.parse_args()

# ── Setup device ──────────────────────────────────────────────────────────────
import wgpu
from vllm_webgpu.webgpu.device import WebGPUDevice
from vllm_webgpu.webgpu.pipeline import PipelineCache
from vllm_webgpu.utils import SHADERS_DIR
from vllm_webgpu.quant.weight_loader import detect_weight_format
from vllm_webgpu.models.base import BaseWebGPUModel

wgpu_dev = WebGPUDevice.initialize()
pipeline_cache = PipelineCache(wgpu_dev.wgpu_device, SHADERS_DIR)

# ── Load model ────────────────────────────────────────────────────────────────
model_path = BaseWebGPUModel._resolve_model_path(args.model)
fmt = detect_weight_format(model_path)
print(f"Model: {model_path}")
print(f"Format: {fmt}")

if fmt == "gguf":
    raise RuntimeError("GGUF profiling not supported in this build (use safetensors models).")
else:
    from transformers import AutoConfig
    hf_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    arch = (getattr(hf_cfg, 'architectures', None) or ['LlamaForCausalLM'])[0]

print(f"Architecture: {arch}")

ARCH_MAP = {
    "LlamaForCausalLM": "llama",
    "MistralForCausalLM": "llama",
    "Qwen2ForCausalLM": "llama",
    "Qwen3ForCausalLM": "llama",
    "Gemma3ForCausalLM": "gemma4",
}
family = ARCH_MAP.get(arch, "llama")

if family == "llama":
    from vllm_webgpu.models.llama import LlamaWebGPUModel
    model = LlamaWebGPUModel(hf_cfg, wgpu_dev, pipeline_cache)
elif family == "gemma4":
    from vllm_webgpu.models.gemma4 import Gemma4WebGPUModel
    model = Gemma4WebGPUModel(hf_cfg, wgpu_dev, pipeline_cache)
else:
    raise ValueError(f"Unknown family {family}")

print("Loading weights...")
t0 = time.perf_counter()
model.load_weights(model_path)
print(f"Weights loaded in {time.perf_counter()-t0:.1f}s")

# ── Tokenize prompt ──────────────────────────────────────────────────────────
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tok_ids = tok.encode(args.prompt)
    print(f"Prompt: {len(tok_ids)} tokens")
except Exception as e:
    print(f"Tokenizer unavailable ({e}), using synthetic 32-token prompt")
    tok_ids = list(range(1, 33))

# ── Setup fake KV pool ────────────────────────────────────────────────────────
import wgpu as wgpu_lib
from vllm_webgpu.webgpu.buffer import WebGPUBuffer

rw = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST
dev = wgpu_dev.wgpu_device
block_size = 16
num_blocks = 512  # enough for profiling
num_kv_heads = hf_cfg.num_key_value_heads
head_dim = getattr(hf_cfg, "head_dim", hf_cfg.hidden_size // hf_cfg.num_attention_heads)
num_layers = hf_cfg.num_hidden_layers

kv_block_bytes = num_blocks * block_size * num_kv_heads * head_dim * 2  # f16
model.kv_pool = [
    (WebGPUBuffer.empty(dev, kv_block_bytes, usage=rw),
     WebGPUBuffer.empty(dev, kv_block_bytes, usage=rw))
    for _ in range(num_layers)
]

# ── Run prefill ───────────────────────────────────────────────────────────────
# Allocate enough blocks for prompt + warmup + profiling steps
total_toks = len(tok_ids) + args.warmup_steps + args.decode_steps + 4
blk_ids = list(range((total_toks + block_size - 1) // block_size))
bt = np.array(blk_ids, dtype=np.uint32)

print("Running prefill...")
t0 = time.perf_counter()
for i, tok in enumerate(tok_ids):
    slot = blk_ids[i // block_size] * block_size + (i % block_size)

    class _PM:
        _s = slot; _b = bt; _c = i + 1
        slot_mapping = [_s]; block_tables = [_b]; max_decode_seq_len = _c

    logits = model.forward(np.array([tok], dtype=np.uint32), np.array([i], dtype=np.uint32), _PM())

_has_gpu_argmax = hasattr(model, 'logit_readback')

def _top1(logits_out):
    """Extract the greedy token from either GPU-argmax (int32 [1,1]) or float logits."""
    if _has_gpu_argmax:
        return int(logits_out[0, 0])
    return int(np.argmax(logits_out[0]))

decode_tok = _top1(logits)
pos = len(tok_ids)
print(f"Prefill done in {(time.perf_counter()-t0)*1000:.1f}ms, first decode token: {decode_tok}")

# ── Decode warmup + production timing ─────────────────────────────────────────
print(f"Warming up ({args.warmup_steps} steps)...")
prod_times = []
for step in range(args.warmup_steps + 5):  # 5 extra for production timing
    slot = blk_ids[pos // block_size] * block_size + (pos % block_size)

    class _DM:
        _s = [slot]; _b = [bt]; _c = pos + 1
        slot_mapping = _s; block_tables = _b; max_decode_seq_len = _c

    t0 = time.perf_counter()
    logits = model.forward(np.array([decode_tok], dtype=np.uint32), np.array([pos], dtype=np.uint32), _DM())
    elapsed_ms = (time.perf_counter() - t0) * 1000
    decode_tok = _top1(logits)
    pos += 1
    if step >= args.warmup_steps:
        prod_times.append(elapsed_ms)

prod_avg_ms = sum(prod_times) / len(prod_times)
print(f"Production throughput: {prod_avg_ms:.1f} ms/tok = {1000/prod_avg_ms:.1f} tok/s")

# ── Profiled decode steps ──────────────────────────────────────────────────────
print(f"Profiling {args.decode_steps} decode steps...")
model.profiling = True
model.profile_reset()

decode_times = []
for step in range(args.decode_steps):
    slot = blk_ids[pos // block_size] * block_size + (pos % block_size)

    class _DM2:
        _s = [slot]; _b = [bt]; _c = pos + 1
        slot_mapping = _s; block_tables = _b; max_decode_seq_len = _c

    t0 = time.perf_counter()
    logits = model.forward(np.array([decode_tok], dtype=np.uint32), np.array([pos], dtype=np.uint32), _DM2())
    decode_tok = _top1(logits)
    decode_times.append((time.perf_counter() - t0) * 1000.0)
    pos += 1

model.profiling = False
avg_step_ms = sum(decode_times) / len(decode_times)
print(f"\nAverage decode step: {avg_step_ms:.1f} ms  ({1000/avg_step_ms:.1f} tok/s)")
print()
print(model.profile_report())
print()

# ── Per-component breakdown ────────────────────────────────────────────────────
stats = model._prof_stats
if stats:
    total = sum(sum(v)/len(v) for v in stats.values())

    # Aggregate by category (strip L00_ prefix)
    from collections import defaultdict
    cat_totals: dict = defaultdict(float)
    for label, times in stats.items():
        cat = label.split("_", 1)[1] if "_" in label else label  # "L00_attn" → "attn"
        cat_totals[cat] += sum(times) / len(times)

    print(f"Total GPU time: {total:.2f} ms")
    print(f"Python overhead: {avg_step_ms - total:.2f} ms")
    print(f"Each layer avg: {total/num_layers:.3f} ms")
    print(f"\nBreakdown by category (summed across {num_layers} layers):")
    for cat, ms in sorted(cat_totals.items(), key=lambda x: -x[1]):
        pct = 100 * ms / total
        per_layer = ms / num_layers
        print(f"  {cat:<12s} {ms:7.2f} ms total  {per_layer:.3f} ms/layer  {pct:.1f}%")

    print(f"\nBottleneck analysis:")
    hid = hf_cfg.hidden_size
    inter_sz = hf_cfg.intermediate_size
    # Estimate weight bytes per layer
    q_dim2 = num_kv_heads * head_dim * (hf_cfg.num_attention_heads // num_kv_heads)  # total Q dim
    attn_w = 2 * (hid * q_dim2 + hid * num_kv_heads * head_dim * 2 + hid * hid)  # qkvo in f16 bytes
    ffn_w = 2 * (hid * inter_sz * 3)  # gate, up, down
    total_w_mb = (attn_w + ffn_w) * num_layers / 1e6
    bw_util_gb_s = total_w_mb * 1000 / total  # GB/s
    print(f"  Weight data moved: {total_w_mb:.0f} MB")
    print(f"  Effective BW: {bw_util_gb_s:.0f} GB/s  (M3 Peak: ~200-400 GB/s)")
    print(f"  BW utilization: {bw_util_gb_s/300*100:.1f}%")
