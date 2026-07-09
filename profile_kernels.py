#!/usr/bin/env python3
"""Profile GPU kernel timing per transformer layer.

Usage:
    source .venv/bin/activate
    python3 profile_kernels.py [--model MODEL_PATH] [--tokens N]
"""
import argparse
import time
from collections import defaultdict
from types import SimpleNamespace
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
from vllm_webgpu.webgpu.device import WebGPUDevice
from vllm_webgpu.webgpu.pipeline import PipelineCache
from vllm_webgpu.utils import SHADERS_DIR
from vllm.transformers_utils.repo_utils import get_model_path

wgpu_dev = WebGPUDevice.initialize()
pipeline_cache = PipelineCache(wgpu_dev.wgpu_device, SHADERS_DIR)

# ── Load model ────────────────────────────────────────────────────────────────
model_path = str(get_model_path(args.model))
print(f"Model: {model_path}")

from transformers import AutoConfig
hf_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
arch = (getattr(hf_cfg, 'architectures', None) or ['LlamaForCausalLM'])[0]
num_layers = hf_cfg.num_hidden_layers
head_dim = getattr(hf_cfg, 'head_dim', hf_cfg.hidden_size // hf_cfg.num_attention_heads)

print(f"Architecture: {arch}")

from vllm_webgpu.config import get_config
from vllm_webgpu.v1.model_runner import _build_model
from vllm_webgpu.v1.cache_policy import allocate_kv_from_hf_config, get_num_kv_heads
num_kv_heads = get_num_kv_heads(hf_cfg)
model = _build_model(arch, hf_cfg, wgpu_dev, pipeline_cache)

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
dev = wgpu_dev.wgpu_device
block_size = get_config().block_size
num_blocks = 512  # enough for profiling

allocate_kv_from_hf_config(dev, model, hf_cfg, num_blocks=num_blocks, block_size=block_size)

# ── Run prefill ───────────────────────────────────────────────────────────────
# Allocate enough blocks for prompt + warmup + profiling steps
total_toks = len(tok_ids) + args.warmup_steps + args.decode_steps * 2
bt_blocks = (total_toks + block_size - 1) // block_size
bt = np.arange(bt_blocks, dtype=np.uint32)

print("Running prefill...")
t0 = time.perf_counter()
slots = list(range(len(tok_ids)))
_pm = SimpleNamespace(slot_mapping=slots, block_tables=[bt], max_decode_seq_len=len(tok_ids))
logits = model.forward(np.array(tok_ids, dtype=np.uint32), np.arange(len(tok_ids), dtype=np.uint32), _pm)

_has_gpu_argmax = getattr(model, 'logit_returns_token_id', False)

decode_tok = int(logits[0, 0]) if _has_gpu_argmax else int(np.argmax(logits[-1]))
pos = len(tok_ids)
print(f"Prefill done in {(time.perf_counter()-t0)*1000:.1f}ms, first decode token: {decode_tok}")

# ── Decode warmup + production timing ─────────────────────────────────────────
print(f"Warming up ({args.warmup_steps} steps)...")
prod_times = []
for step in range(args.warmup_steps + args.decode_steps):  # decode_steps extra for production timing
    slot = pos

    _dm = SimpleNamespace(slot_mapping=[slot], block_tables=[bt], max_decode_seq_len=pos + 1)
    t0 = time.perf_counter()
    logits = model.forward(np.array([decode_tok], dtype=np.uint32), np.array([pos], dtype=np.uint32), _dm)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    decode_tok = int(logits[0, 0]) if _has_gpu_argmax else int(np.argmax(logits[-1]))
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
    slot = pos

    _dm2 = SimpleNamespace(slot_mapping=[slot], block_tables=[bt], max_decode_seq_len=pos + 1)
    t0 = time.perf_counter()
    logits = model.forward(np.array([decode_tok], dtype=np.uint32), np.array([pos], dtype=np.uint32), _dm2)
    decode_tok = int(logits[0, 0]) if _has_gpu_argmax else int(np.argmax(logits[-1]))
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

    # Aggregate by category (strip L00_ prefix; labels without a suffix, e.g.
    # "L00", are grouped under "layer" so that per_layer divides by the actual
    # number of labels in that category, not the total layer count).
    cat_totals: dict = defaultdict(float)
    cat_layer_count: dict = defaultdict(int)
    for label, times in stats.items():
        cat = label.split("_", 1)[1] if "_" in label else "layer"  # "L00_attn" → "attn"
        cat_totals[cat] += sum(times) / len(times)
        cat_layer_count[cat] += 1

    print(f"Total GPU time: {total:.2f} ms")
    print(f"Python overhead: {avg_step_ms - total:.2f} ms")
    print(f"Each layer avg: {total/num_layers:.3f} ms")
    print(f"\nBreakdown by category (summed across {num_layers} layers):")
    for cat, ms in sorted(cat_totals.items(), key=lambda x: -x[1]):
        pct = 100 * ms / total
        count = cat_layer_count[cat]
        per_layer = ms / count
        print(f"  {cat:<12s} {ms:7.2f} ms total  {per_layer:.3f} ms/layer  {pct:.1f}%")

    print(f"\nBottleneck analysis:")
    hid = hf_cfg.hidden_size
    if hasattr(hf_cfg, 'num_attention_heads'):
        raw_inter_sz = hf_cfg.intermediate_size
        inter_sz = max(raw_inter_sz) if isinstance(raw_inter_sz, list) else raw_inter_sz
        q_dim2 = hf_cfg.num_attention_heads * head_dim  # total Q projection dim
        attn_w = 2 * (hid * q_dim2 + hid * num_kv_heads * head_dim * 2 + q_dim2 * hid)  # qkvo in f16 bytes
        # FFN weight bytes: 3 matrices (gate, up, down) for SwiGLU models.
        # NemotronH '-' layers use only 2 matrices (up, down; relu^2, no gate).
        # MoE multiplies by top_k (activated experts per token).
        ffn_matrices = 3
        ffn_w = 2 * (hid * inter_sz * ffn_matrices)
        if getattr(model, '_is_moe', False):
            top_k = getattr(model, '_top_k', 1)
            ffn_w *= top_k

        # For hybrid architectures (e.g. NemotronH), layer types differ per layer.
        # Use per-layer type weights rather than applying (attn_w + ffn_w) uniformly.
        layer_types = getattr(model, '_layer_types', None)
        if layer_types is None:
            layer_types = getattr(hf_cfg, 'layers_block_type', None)

        if layer_types is not None and len(layer_types) == num_layers:
            # Mamba-2 SSM layer weight bytes (f16 unless noted):
            #   in_proj:    hid * in_proj_dim * 2
            #   conv1d:     conv_dim * conv_kernel * 2
            #   A_log (f32): mamba_num_heads * 4
            #   D (f32):     mamba_num_heads * 4
            #   dt_bias (f32): mamba_num_heads * 4
            #   out_proj:   mamba_int * hid * 2
            mnh  = getattr(hf_cfg, 'mamba_num_heads', 0)
            mhd  = getattr(hf_cfg, 'mamba_head_dim', 0)
            ng   = getattr(hf_cfg, 'n_groups', 0)
            ss   = getattr(hf_cfg, 'ssm_state_size', 0)
            ck   = getattr(hf_cfg, 'conv_kernel', 0)
            mi   = mnh * mhd                           # mamba_int
            cd   = mi + 2 * ng * ss                   # conv_dim
            ipd  = mi + cd + mnh                      # in_proj_dim
            ssm_w = (
                2 * hid * ipd       # in_proj (f16)
                + 2 * cd * ck       # conv1d  (f16)
                + 4 * mnh * 3       # A_log + D + dt_bias (f32 each)
                + 2 * mi * hid      # out_proj (f16)
            ) if (mnh and mhd) else 0

            raw_inter = hf_cfg.intermediate_size
            inter_list = raw_inter if isinstance(raw_inter, list) else None

            total_w_bytes = 0
            _mlp_idx = 0
            for idx, lt in enumerate(layer_types):
                if lt == 'attention':
                    total_w_bytes += attn_w
                elif lt in ('mlp', 'ffn'):
                    layer_inter = inter_list[_mlp_idx] if inter_list else inter_sz
                    _mlp_idx += 1
                    # NemotronH '-' (mlp) layers: up_proj + down_proj only (relu^2, no gate_proj).
                    layer_ffn_matrices = 2 if (lt == 'mlp' and 'NemotronH' in arch) else ffn_matrices
                    total_w_bytes += 2 * (hid * layer_inter * layer_ffn_matrices)
                elif lt == 'mamba':
                    total_w_bytes += ssm_w
                # 'moe' and unknown types are skipped (no reliable generic formula)
        else:
            # Uniform architecture: every layer has both attention and FFN.
            total_w_bytes = (attn_w + ffn_w) * num_layers

        total_w_mb = total_w_bytes / 1e6
        bw_util_gb_s = total_w_mb / total  # 1 MB/ms = 1 GB/s
        print(f"  Weight data moved: {total_w_mb:.0f} MB")
        print(f"  Effective BW: {bw_util_gb_s:.0f} GB/s  (M3 Peak: ~200-400 GB/s)")
        print(f"  BW utilization: {bw_util_gb_s/300*100:.1f}%")
    else:
        print("  (Bottleneck analysis not available for this architecture)")
