#!/usr/bin/env python3
"""Profile GPU kernel timing per transformer layer.

Usage:
    source .venv/bin/activate
    python3 profile_kernels.py [--model MODEL_PATH] [--decode-steps N] [--warmup-steps N]
"""
import argparse
import math
import os
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
    model_path = args.model if os.path.isdir(args.model) else snapshot_download(args.model)
    print(f"Model: {model_path}")

    from transformers import AutoConfig, AutoTokenizer
    hf_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    arch = (getattr(hf_cfg, 'architectures', None) or ['LlamaForCausalLM'])[0]
    num_layers = hf_cfg.num_hidden_layers
    print(f"Architecture: {arch}")

    from vllm_webgpu.v1.model_runner import _build_model
    from vllm_webgpu.scripts.kv_utils import allocate_kv_from_hf_config
    from vllm_webgpu.scripts.kv_utils import get_kv_dims_from_hf_config
    from vllm_webgpu.v1.cache_policy import KV_ATTN_TYPES, get_layer_types
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
    bt_blocks = math.ceil(total_toks / block_size)
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
        hid = hf_cfg.hidden_size
        if hasattr(hf_cfg, 'num_attention_heads'):
            num_kv_heads, head_dim = get_kv_dims_from_hf_config(hf_cfg)
            raw_inter_sz = hf_cfg.intermediate_size
            inter_sz = max(raw_inter_sz) if isinstance(raw_inter_sz, list) else raw_inter_sz
            q_dim2 = hf_cfg.num_attention_heads * head_dim  # total Q projection dim
            attn_w = 4 * hid * (q_dim2 + num_kv_heads * head_dim)  # Q+K+V+O projections in f16 bytes
            # Per-head QK-norm weights (Qwen3, Llama3.2): q_norm [num_q_heads*head_dim] f16
            # + k_norm [num_kv_heads*head_dim] f16. Add if the checkpoint carries q_norm weights.
            if any('q_norm' in k for k in model.weights):
                attn_w += (q_dim2 + num_kv_heads * head_dim) * 2  # f16 = 2 bytes

            # For hybrid architectures (e.g. NemotronH, Gemma4), layer types differ per layer.
            # Use per-layer type weights rather than applying (attn_w + ffn_w) uniformly.
            layer_types = get_layer_types(model, hf_cfg)

            if layer_types is not None and len(layer_types) == num_layers:
                # Mamba-2 SSM layer weight bytes (f16 unless noted):
                #   in_proj:    hid * in_proj_dim * 2
                #   conv1d:     conv_dim * conv_kernel * 2
                #   A_log (f32): mamba_num_heads * 4
                #   D (f32):     mamba_num_heads * 4
                #   dt_bias (f32): mamba_num_heads * 4
                #   out_proj:   mamba_int * hid * 2
                mnh  = getattr(model, 'mamba_num_heads', getattr(hf_cfg, 'mamba_num_heads', 0))
                mhd  = getattr(model, 'mamba_head_dim', getattr(hf_cfg, 'mamba_head_dim', 0))
                ng   = getattr(model, 'n_groups',        getattr(hf_cfg, 'n_groups', 0))
                ss   = getattr(model, 'ssm_state_size',  getattr(hf_cfg, 'ssm_state_size', 0))
                ck   = getattr(model, 'conv_kernel',     getattr(hf_cfg, 'conv_kernel', 0))
                mi   = getattr(model, 'mamba_int',  mnh * mhd)                # mamba_int
                cd   = getattr(model, 'conv_dim',   mi + 2 * ng * ss)         # conv_dim
                ipd  = getattr(model, 'in_proj_dim', mi + cd + mnh)           # in_proj_dim
                ssm_w = (
                    2 * hid * ipd       # in_proj (f16)
                    + 2 * cd * ck       # conv1d  (f16)
                    + 4 * mnh * 3       # A_log + D + dt_bias (f32 each)
                    + 2 * mi * hid      # out_proj (f16)
                    + 2 * mi            # norm.weight (f16, shape [mamba_int])
                ) if (mnh and mhd) else 0

                # Use per-layer intermediate sizes from the loaded model when available,
                # falling back to the raw hf_config list or scalar.
                layer_int_size = getattr(model, '_layer_int_size', None)
                if layer_int_size is not None and len(layer_int_size) != num_layers:
                    layer_int_size = None
                inter_list = raw_inter_sz if isinstance(raw_inter_sz, list) else None

                ffn_matrices = 3
                total_w_bytes = 0
                _mlp_idx = 0
                for idx, lt in enumerate(layer_types):
                    if lt in KV_ATTN_TYPES:
                        total_w_bytes += attn_w
                    elif lt in ('mlp', 'ffn'):
                        if layer_int_size is not None:
                            layer_inter = layer_int_size[idx]
                        elif inter_list:
                            layer_inter = inter_list[_mlp_idx]
                            _mlp_idx += 1
                        else:
                            layer_inter = inter_sz
                        # NemotronH '-' (mlp) layers: up_proj + down_proj only (relu^2, no gate_proj).
                        layer_ffn_matrices = 2 if (lt == 'mlp' and 'NemotronH' in arch) else ffn_matrices
                        total_w_bytes += 2 * (hid * layer_inter * layer_ffn_matrices)
                    elif lt == 'mamba':
                        total_w_bytes += ssm_w
                    # 'moe' and unknown types are skipped (no reliable generic formula)
            else:
                # Uniform architecture: every layer has both attention and FFN.
                # FFN weight bytes: 3 matrices (gate, up, down) for SwiGLU models.
                # MoE multiplies by top_k (activated experts per token).
                ffn_matrices = 3
                ffn_w = 2 * (hid * inter_sz * ffn_matrices)
                if getattr(model, '_is_moe', False):
                    top_k = getattr(model, '_top_k', 1)
                    ffn_w *= top_k
                # attn_w already includes the qk_norm correction applied above for
                # models that carry q_norm weights (Qwen3, Llama3.2).
                total_w_bytes = (attn_w + ffn_w) * num_layers

            total_w_mb = total_w_bytes / 1e6
            bw_util_gb_s = total_w_mb / total  # 1 MB/ms = 1 GB/s
            print(f"  Weight data moved: {total_w_mb:.0f} MB")
            print(f"  Effective BW: {bw_util_gb_s:.0f} GB/s  (M3 Peak: ~200-400 GB/s)")
            print(f"  BW utilization: {bw_util_gb_s/300*100:.1f}%")
        else:
            print("  (Bottleneck analysis not available for this architecture)")


if __name__ == '__main__':
    main()
