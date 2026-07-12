"""Standalone inference script for vllm-webgpu.

The run() function requires transformers and wgpu. When invoked from __main__,
huggingface_hub.snapshot_download is used to resolve a repo ID to a local path.
"""
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# Numerical stability threshold for temperature comparisons, matching
# vllm.v1.sample.sampler._SAMPLING_EPS. Defined locally to avoid a module-level
# import of vllm_webgpu.utils (and transitively vllm internals) in a script that
# may be imported as a library without the full vLLM stack present.
_GREEDY_TEMP: float = 1e-5


def run(model_dir: str, prompt: str, max_tokens: int = 64, temperature: float = 0.0, top_p: float = 0.9):
    print(f"\nLoading model from: {model_dir}")

    if Path(model_dir).suffix == ".gguf":
        raise ValueError(
            "GGUF format is not supported by this plugin. "
            "Use the vllm-gguf plugin instead."
        )

    # Use vLLM's config loader so Mistral-format repos (params.json) are handled correctly.
    from transformers import AutoTokenizer
    from vllm.transformers_utils.config import get_config as _vllm_get_config
    cfg = _vllm_get_config(model_dir, trust_remote_code=True)

    arch = (cfg.architectures or ["LlamaForCausalLM"])[0]
    print(f"Architecture: {arch}")
    from vllm_webgpu.scripts.kv_utils import _make_convertor
    # For multimodal wrapper configs cfg.num_hidden_layers is the outer wrapper's
    # count; the convertor reads from hf_text_config and returns the correct value.
    _num_layers = _make_convertor(cfg).get_num_hidden_layers()
    print(f"  hidden={cfg.hidden_size}, layers={_num_layers}, "
          f"heads={cfg.num_attention_heads}, kv_heads={getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)}")

    # AutoTokenizer handles chat templates, special tokens, and all tokenizer variants.
    print("\nLoading tokenizer...")
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    messages = [{"role": "user", "content": prompt}]
    try:
        input_ids_list = tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True)
    except Exception as _e:
        print(f"  Warning: apply_chat_template failed ({_e}), falling back to tok.encode")
        input_ids_list = tok.encode(prompt)
    _eos_raw = getattr(cfg, 'eos_token_id', None)
    _eos_raw = _eos_raw if _eos_raw is not None else tok.eos_token_id
    eos_ids = set(_eos_raw if isinstance(_eos_raw, list) else [_eos_raw]) - {None}
    print(f"Input tokens: {len(input_ids_list)}")
    print(f"Prompt (after template): {repr(tok.decode(input_ids_list)[:120])}")

    # GPU device
    print("\nInitializing WebGPU device...")
    from vllm_webgpu.webgpu.device import WebGPUDevice
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.utils import SHADERS_DIR, sample_token
    from vllm_webgpu.config import get_config
    from vllm.utils.math_utils import cdiv

    device = WebGPUDevice.initialize(get_config().power_preference)
    print(f"  Adapter: f16={device.supports_f16}")
    pipeline_cache = PipelineCache(device.wgpu_device, SHADERS_DIR)

    # Build model
    print("\nBuilding model...")
    from vllm_webgpu.v1.model_runner import _build_model, ARCH_MAP
    import vllm_webgpu.envs as _envs
    block_size = _envs.VLLM_WEBGPU_BLOCK_SIZE
    family = ARCH_MAP.get(arch)
    model = _build_model(arch, family, cfg, device, pipeline_cache, block_size)

    # Load weights
    print("\nLoading weights (this may take a while)...")
    t0 = time.perf_counter()
    model.load_weights(model_dir)
    print(f"  Loaded {len(model.weights)} tensors in {time.perf_counter() - t0:.1f}s")

    # KV cache
    from vllm_webgpu.scripts.kv_utils import allocate_kv_from_hf_config
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
    block_table = np.arange(num_blocks, dtype=np.uint32)
    slots = list(range(T))

    model._greedy_decode = (temperature < _GREEDY_TEMP)

    batch_meta = SimpleNamespace(slot_mapping=slots, block_tables=[block_table], max_decode_seq_len=T)
    logits = model.forward(
        np.array(input_ids_list, dtype=np.uint32),
        np.arange(T, dtype=np.uint32),
        batch_meta,
    )

    if temperature < _GREEDY_TEMP:
        last_token = int(logits[-1, 0])
        print(f"  Last prefill logit: argmax={last_token}")
    else:
        _best = int(np.argmax(logits[-1]))
        last_token = sample_token(logits[-1], temperature=temperature, top_p=top_p)
        print(f"  Last prefill logit: argmax={_best}, value={float(logits[-1][_best]):.2f}, "
              f"std={float(logits[-1].std()):.2f}")

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
        bi   = slot // block_size
        if bi >= num_blocks:
            print(f"  [KV cache full at step {step}]")
            break
        meta   = SimpleNamespace(slot_mapping=[slot], block_tables=[block_table], max_decode_seq_len=len(input_ids_list) + step + 1)
        logits = model.forward(
            np.array([last_token], dtype=np.uint32),
            np.array([slot], dtype=np.uint32),
            meta,
        )

        if temperature < _GREEDY_TEMP:
            last_token = int(logits[0, 0])
        else:
            # _greedy_decode=False: forward() already returned full (1, vocab) logits.
            # No logit_readback() call needed.
            last_token = sample_token(logits[0], temperature=temperature, top_p=top_p)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       required=True)
    parser.add_argument("--prompt",      default="What is 2+2?")
    parser.add_argument("--max_tokens",  type=int,   default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p",       type=float, default=0.9)
    parser.add_argument("--gdn_bf16",    action="store_true",
                        help="Experimental: bf16 GDN weights for Qwen3.5 (safetensors BF16 only)")
    args = parser.parse_args()

    if args.gdn_bf16:
        import os
        os.environ["GDN_BF16"] = "1"

    from huggingface_hub import snapshot_download
    model_path = args.model if Path(args.model).is_dir() else snapshot_download(args.model)
    run(model_path, args.prompt, args.max_tokens, args.temperature, args.top_p)
