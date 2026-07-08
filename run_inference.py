"""Standalone inference script for vllm-webgpu. No vLLM required."""
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))



def run(model_dir: str, prompt: str, max_tokens: int = 64, temperature: float = 0.0, top_p: float = 0.9):
    print(f"\nLoading model from: {model_dir}")

    if Path(model_dir).suffix == ".gguf":
        raise ValueError(
            "GGUF format is not supported by this plugin. "
            "Use the vllm-gguf plugin instead."
        )

    # AutoConfig handles text_config merging for multimodal models automatically.
    from transformers import AutoConfig, AutoTokenizer
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

    arch = (cfg.architectures or ["LlamaForCausalLM"])[0]
    print(f"Architecture: {arch}")
    print(f"  hidden={cfg.hidden_size}, layers={cfg.num_hidden_layers}, "
          f"heads={cfg.num_attention_heads}, kv_heads={cfg.num_key_value_heads}")

    # AutoTokenizer handles chat templates, special tokens, and all tokenizer variants.
    print("\nLoading tokenizer...")
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    messages = [{"role": "user", "content": prompt}]
    try:
        result = tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True)
        # transformers may return a BatchEncoding (dict subclass) or a plain list.
        input_ids_list = result["input_ids"] if isinstance(result, dict) else result
    except Exception:
        input_ids_list = tok.encode(prompt)
    eos_id = tok.eos_token_id
    print(f"Input tokens: {len(input_ids_list)}")
    print(f"Prompt (after template): {repr(tok.decode(input_ids_list)[:120])}")

    # GPU device
    print("\nInitializing WebGPU device...")
    from vllm_webgpu.webgpu.device import WebGPUDevice
    from vllm_webgpu.webgpu.pipeline import PipelineCache
    from vllm_webgpu.utils import SHADERS_DIR

    device = WebGPUDevice.initialize("high-performance")
    print(f"  Adapter: f16={device.supports_f16}")
    pipeline_cache = PipelineCache(device.wgpu_device, SHADERS_DIR)

    # Build model
    print("\nBuilding model...")
    from vllm_webgpu.v1.model_runner import ARCH_MAP, _build_model
    if arch not in ARCH_MAP:
        raise NotImplementedError(
            f"Architecture {arch!r} not supported. Supported: {sorted(ARCH_MAP)}")
    model = _build_model(arch, cfg, device, pipeline_cache)

    # Load weights
    print("\nLoading weights (this may take a while)...")
    t0 = time.perf_counter()
    model.load_weights(model_dir)
    print(f"  Loaded {len(model.weights)} tensors in {time.perf_counter() - t0:.1f}s")

    # KV cache
    from vllm_webgpu.config import get_config
    from vllm_webgpu.v1.cache_policy import allocate_kv_pool_hybrid, allocate_kv_pool_per_layer

    block_size = get_config().block_size
    max_ctx = min(getattr(cfg, "max_position_embeddings", 8192), 65535)
    num_blocks = min(math.ceil(max_ctx / block_size) + 4, 4096)

    layer_params = getattr(model, "_lp", None)
    if layer_params:
        print(f"\nAllocating per-layer KV cache ({cfg.num_hidden_layers} layers, mixed dims)")
        allocate_kv_pool_per_layer(device.wgpu_device, model, num_blocks=num_blocks, block_size=block_size, layer_params=layer_params)
    else:
        kv_h = cfg.num_key_value_heads
        hd   = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        allocate_kv_pool_hybrid(
            device.wgpu_device,
            model,
            num_blocks=num_blocks,
            num_layers=cfg.num_hidden_layers,
            block_size=block_size,
            num_kv_heads=kv_h,
            head_dim=hd,
            layer_types=getattr(model, "_layer_types", None),
        )

    # Prefill
    print(f"\nRunning prefill ({len(input_ids_list)} tokens)...")
    T = len(input_ids_list)
    block_table = np.zeros(num_blocks, dtype=np.uint32)
    slots = []
    for i in range(T):
        bi = i // block_size
        block_table[bi] = bi
        slots.append(bi * block_size + (i % block_size))

    batch_meta = SimpleNamespace(slot_mapping=slots, block_tables=[block_table.copy()], max_decode_seq_len=T)
    logits = model.forward(
        np.array(input_ids_list, dtype=np.uint32),
        np.arange(T, dtype=np.uint32),
        batch_meta,
    )

    has_gpu_argmax = getattr(model, "logit_returns_token_id", False)
    if has_gpu_argmax:
        top1 = int(logits[0, 0])
        _real = model.logit_readback()
        print(f"  Last prefill logit: argmax={top1}, value={float(_real[0][top1]):.2f}, "
              f"std={float(_real[0].std()):.2f}")
    else:
        top1 = int(np.argmax(logits[0]))
        print(f"  Last prefill logit: argmax={top1}, value={float(logits[0][top1]):.2f}, "
              f"std={float(logits[0].std()):.2f}")

    # Decode
    print(f"\nDecoding (max {max_tokens} tokens)...")

    generated = []
    t_start = time.perf_counter()
    last_token = int(logits[0, 0]) if has_gpu_argmax else int(np.argmax(logits[0]))

    for step in range(max_tokens):
        if last_token == eos_id:
            print(f"  [EOS at step {step}]")
            break
        generated.append(last_token)

        slot = len(input_ids_list) + step
        bi   = slot // block_size
        if bi >= num_blocks:
            print(f"  [KV cache full at step {step}]")
            break
        block_table[bi] = bi

        meta   = SimpleNamespace(slot_mapping=[slot], block_tables=[block_table.copy()], max_decode_seq_len=len(input_ids_list) + step + 1)
        logits = model.forward(
            np.array([last_token], dtype=np.uint32),
            np.array([slot], dtype=np.uint32),
            meta,
        )

        if temperature == 0.0:
            last_token = int(logits[0, 0]) if has_gpu_argmax else int(np.argmax(logits[0]))
        else:
            from vllm_webgpu.utils import sample_token
            full = model.logit_readback() if has_gpu_argmax else logits
            last_token = sample_token(full[0], temperature=temperature, top_p=top_p)

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
        os.environ["GDN_BF16"] = "1"

    from vllm.transformers_utils.repo_utils import get_model_path
    model_path = str(get_model_path(args.model))
    run(model_path, args.prompt, args.max_tokens, args.temperature, args.top_p)
