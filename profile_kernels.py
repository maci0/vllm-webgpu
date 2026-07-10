#!/usr/bin/env python3
"""Profile GPU kernel timing per transformer layer.

Usage:
    source .venv/bin/activate
    python3 profile_kernels.py [--model MODEL_PATH] [--decode-steps N] [--warmup-steps N]
"""
import argparse
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--prompt", default="The capital of France is Paris and the Eiffel Tower")
    parser.add_argument("--decode-steps", type=int, default=5,
                        help="Number of decode steps to profile (averaged)")
    parser.add_argument("--warmup-steps", type=int, default=2,
                        help="Warmup steps before profiling (not counted)")
    args = parser.parse_args()

    # ── Setup device ──────────────────────────────────────────────────────────────
    from huggingface_hub import snapshot_download
    from vllm_webgpu.webgpu.device import WebGPUDevice
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.utils import SHADERS_DIR
    from vllm_webgpu.config import get_config

    wgpu_dev = WebGPUDevice.initialize(get_config().power_preference)
    pipeline_cache = PipelineCache(wgpu_dev.wgpu_device, SHADERS_DIR)

    # ── Load model ────────────────────────────────────────────────────────────────
    model_path = args.model if Path(args.model).is_dir() else snapshot_download(args.model)
    print(f"Model: {model_path}")

    from transformers import AutoConfig, AutoTokenizer
    hf_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    arch = (getattr(hf_cfg, 'architectures', None) or ['LlamaForCausalLM'])[0]
    num_layers = hf_cfg.num_hidden_layers
    print(f"Architecture: {arch}")

    from vllm_webgpu.v1.model_runner import _build_model
    from vllm_webgpu.scripts.kv_utils import allocate_kv_from_hf_config
    import vllm_webgpu.envs as _envs
    block_size = _envs.VLLM_WEBGPU_BLOCK_SIZE
    model = _build_model(arch, hf_cfg, wgpu_dev, pipeline_cache, block_size=block_size)

    print("Loading weights...")
    t0 = time.perf_counter()
    model.load_weights(model_path)
    print(f"Weights loaded in {time.perf_counter()-t0:.1f}s")

    # ── Tokenize prompt ──────────────────────────────────────────────────────────
    try:
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tok_ids = tok.encode(args.prompt)
        print(f"Prompt: {len(tok_ids)} tokens")
    except Exception as e:
        print(f"Tokenizer unavailable ({e}), using synthetic 32-token prompt")
        tok_ids = list(range(1, 33))

    # ── Setup fake KV pool ────────────────────────────────────────────────────────
    dev = wgpu_dev.wgpu_device

    # Compute block count before allocating so the pool covers every block ID in bt.
    total_toks = len(tok_ids) + args.warmup_steps + args.decode_steps * 2
    bt_blocks = (total_toks + block_size - 1) // block_size
    num_blocks = max(512, bt_blocks)

    allocate_kv_from_hf_config(dev, model, hf_cfg, num_blocks=num_blocks, block_size=block_size)

    model.warmup()

    # ── Run prefill ───────────────────────────────────────────────────────────────
    bt = np.arange(bt_blocks, dtype=np.uint32)

    print("Running prefill...")
    t0 = time.perf_counter()
    slots = list(range(len(tok_ids)))
    _pm = SimpleNamespace(slot_mapping=slots, block_tables=[bt], max_decode_seq_len=len(tok_ids))
    logits = model.forward(np.array(tok_ids, dtype=np.uint32), np.arange(len(tok_ids), dtype=np.uint32), _pm)

    _has_gpu_argmax = model.logit_returns_token_id

    def _next_tok(lg):
        return int(lg[0, 0]) if _has_gpu_argmax else int(np.argmax(lg[-1]))

    decode_tok = _next_tok(logits)
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
        decode_tok = _next_tok(logits)
        pos += 1
        if step >= args.warmup_steps:
            prod_times.append(elapsed_ms)

    if not prod_times:
        print("No production steps measured")
    else:
        prod_avg_ms = np.mean(prod_times)
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
        decode_tok = _next_tok(logits)
        decode_times.append((time.perf_counter() - t0) * 1000.0)
        pos += 1

    model.profiling = False
    avg_step_ms = np.mean(decode_times) if decode_times else 0.0
    if not decode_times:
        print("\nNo profiled decode steps measured")
    else:
        print(f"\nAverage decode step: {avg_step_ms:.1f} ms  ({1000/avg_step_ms:.1f} tok/s)")
    print()
    print(model.profile_report())
    print()

    # ── Per-component breakdown ────────────────────────────────────────────────────
    stats = model.get_prof_stats()
    if stats:
        total = sum(np.mean(v) for v in stats.values())

        print(f"Total GPU time: {total:.2f} ms")
        print(f"Python overhead: {avg_step_ms - total:.2f} ms")
        print(f"Each layer avg: {total/num_layers:.3f} ms")

        print(f"\nBottleneck analysis:")
        # Sum actual compressed buffer sizes from loaded weights. This is correct for
        # all quantization formats (F16, GPTQ INT4, FP8, NF4) because WebGPUBuffer.nbytes
        # returns buf.size, which reflects the real on-device allocation.
        # Exclude metadata keys (__*) and scale tensors (.scales) that are not weight
        # data streamed through the shader per step.
        total_w_bytes = sum(
            v.nbytes for k, v in model.weights.items()
            if not k.startswith('__') and not k.endswith('.scales')
        )
        total_w_mb = total_w_bytes / 1e6
        bw_util_gb_s = total_w_mb / total  # 1 MB/ms = 1 GB/s
        print(f"  Weight data moved: {total_w_mb:.0f} MB  ({total_w_mb/num_layers:.1f} MB/layer avg)")
        print(f"  Effective BW: {bw_util_gb_s:.0f} GB/s  (M3 Peak: ~200-400 GB/s)")
        print(f"  BW utilization: {bw_util_gb_s/300*100:.1f}%")


if __name__ == '__main__':
    main()
