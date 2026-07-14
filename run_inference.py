"""Standalone inference script for vllm-webgpu.

The run() function requires transformers and wgpu. When invoked from __main__,
huggingface_hub.snapshot_download is used to resolve a repo ID to a local path.
"""
import time
from types import SimpleNamespace

import numpy as np


def _pick_token(logits_2d, greedy: bool, temperature: float, top_p: float) -> int:
    """Sample or greedily decode the next token from a (1, vocab_or_1) logits row."""
    from vllm_webgpu.utils import sample_token
    row = logits_2d[0]
    if greedy and logits_2d.shape[-1] == 1:
        return int(row[0])
    if greedy:
        return int(row.argmax())
    return sample_token(row, temperature=temperature, top_p=top_p)


def run(model_dir: str, prompt: str, max_tokens: int = 64, temperature: float = 0.0, top_p: float = 0.9):
    print(f"\nLoading model from: {model_dir}")

    # Use vLLM's config loader so Mistral-format repos (params.json) are handled correctly.
    from transformers import AutoTokenizer
    from vllm.transformers_utils.config import get_config as _vllm_get_config
    cfg = _vllm_get_config(model_dir, trust_remote_code=True)

    arch = (cfg.architectures or ["LlamaForCausalLM"])[0]
    print(f"Architecture: {arch}")

    # AutoTokenizer handles chat templates, special tokens, and all tokenizer variants.
    print("\nLoading tokenizer...")
    from vllm_webgpu.scripts import apply_chat_template_or_encode
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    input_ids_list = apply_chat_template_or_encode(tok, prompt)
    _eos_raw = getattr(cfg, 'eos_token_id', None)
    _eos_raw = _eos_raw if _eos_raw is not None else tok.eos_token_id
    eos_ids = set(_eos_raw if isinstance(_eos_raw, list) else [_eos_raw]) - {None}
    print(f"Input tokens: {len(input_ids_list)}")
    print(f"Prompt (after template): {repr(tok.decode(input_ids_list)[:120])}")

    # GPU device
    print("\nInitializing WebGPU device...")
    from vllm_webgpu.webgpu.device import WebGPUDevice
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.utils import GREEDY_TEMP
    from vllm_webgpu.config import get_config

    device = WebGPUDevice.initialize(get_config().power_preference)
    print(f"  Adapter: f16={device.supports_f16}")
    pipeline_cache = PipelineCache(device.wgpu_device)

    # Build model
    print("\nBuilding model...")
    from vllm_webgpu.v1.model_runner import _build_model, ARCH_MAP
    import vllm_webgpu.envs as _envs
    block_size = _envs.VLLM_WEBGPU_BLOCK_SIZE
    family = ARCH_MAP.get(arch)
    model = _build_model(arch, family, cfg, device, pipeline_cache, block_size)
    print(f"  hidden={model.hidden_size}, layers={model.num_layers}, "
          f"heads={getattr(model, 'num_q_heads', '?')}, "
          f"kv_heads={getattr(model, 'num_kv_heads', '?')}")

    # Load weights
    print("\nLoading weights (this may take a while)...")
    t0 = time.perf_counter()
    model.load_weights(model_dir)
    print(f"  Loaded {len(model.weights)} tensors in {time.perf_counter() - t0:.1f}s")

    # KV cache
    from vllm_webgpu.scripts.kv_utils import allocate_kv_from_hf_config
    from vllm.utils.math_utils import cdiv
    # num_blocks is capped at 4096 (the KV pool ceiling for this script).
    max_ctx = getattr(cfg, "max_position_embeddings", 8192)
    num_blocks = min(cdiv(max_ctx, block_size) + 4, 4096)

    allocate_kv_from_hf_config(device.wgpu_device, model, cfg, num_blocks=num_blocks, block_size=block_size)

    model.warmup()
    if hasattr(model, 'reset_recurrent_states'):
        model.reset_recurrent_states()

    # Prefill
    print(f"\nRunning prefill ({len(input_ids_list)} tokens)...")
    T = len(input_ids_list)
    needed_blocks = min(cdiv(len(input_ids_list) + max_tokens, block_size), num_blocks)
    block_table = np.arange(needed_blocks, dtype=np.uint32)
    slots = list(range(T))

    _greedy = temperature < GREEDY_TEMP
    model._greedy_decode = _greedy

    batch_meta = SimpleNamespace(slot_mapping=slots, block_tables=[block_table], max_decode_seq_len=T)
    logits = model.forward(
        np.array(input_ids_list, dtype=np.uint32),
        np.arange(T, dtype=np.uint32),
        batch_meta,
    )

    # DiffusionGemma always returns full (num_tokens, vocab) float32 logits regardless of
    # _greedy_decode, so check shape before trusting logits[-1, 0] as a token ID.
    last_token = _pick_token(logits[-1:], _greedy, temperature, top_p)
    if logits.shape[-1] != 1:
        _best = int(np.argmax(logits[-1]))
        print(f"  Last prefill logit: argmax={_best}, value={float(logits[-1][_best]):.2f}, "
              f"std={float(logits[-1].std()):.2f}")
    else:
        print(f"  Last prefill logit: argmax={last_token}")

    # Decode
    print(f"\nDecoding (max {max_tokens} tokens)...")

    generated = []
    t_start = time.perf_counter()

    for step in range(max_tokens):
        if eos_ids and last_token in eos_ids:
            print(f"  [EOS at step {step}]")
            break
        generated.append(last_token)

        slot = len(input_ids_list) + step
        if slot // block_size >= needed_blocks:
            print(f"  [KV cache full at step {step}]")
            break
        meta   = SimpleNamespace(slot_mapping=[slot], block_tables=[block_table], max_decode_seq_len=len(input_ids_list) + step + 1)
        logits = model.forward(
            np.array([last_token], dtype=np.uint32),
            np.array([slot], dtype=np.uint32),
            meta,
        )

        # DiffusionGemma always returns full (1, vocab) float32 logits regardless of
        # _greedy_decode, so check shape before trusting logits[0, 0] as a token ID.
        last_token = _pick_token(logits, _greedy, temperature, top_p)

        if (step + 1) % 5 == 0:
            print(f"  [{step+1} tokens]: {repr(tok.decode(generated)[-60:])}", flush=True)

    n_tok       = len(generated)
    tok_per_sec = n_tok / max(time.perf_counter() - t_start, 0.001)
    output_text = tok.decode(generated, skip_special_tokens=True)

    print(f"\n{'='*60}")
    print(f"Prompt: {repr(prompt[:80])}")
    print(f"Output: {output_text}")
    print(f"{'='*60}")
    print(f"Generated {n_tok} tokens at {tok_per_sec:.1f} tok/s")
    return output_text


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       required=True)
    parser.add_argument("--prompt",      default="What is 2+2?")
    parser.add_argument("--max_tokens",  type=int,   default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p",       type=float, default=0.9)
    parser.add_argument("--gdn-bf16",    action="store_true",
                        help="Experimental: bf16 GDN weights for Qwen3.5 (safetensors BF16 only)")
    args = parser.parse_args()

    if args.gdn_bf16:
        os.environ["GDN_BF16"] = "1"

    from vllm_webgpu.scripts import resolve_model_path
    model_path = resolve_model_path(args.model)
    run(model_path, args.prompt, args.max_tokens, args.temperature, args.top_p)
