"""Standalone inference script for vllm-webgpu. No vLLM required.

Usage:
    python run_inference.py --model <path_or_hf_id> --prompt "Hello" --max_tokens 50
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent))


def load_tokenizer_from_gguf(gguf_path: str):
    """Extract tokenizer from GGUF metadata and create a tokenizers.Tokenizer."""
    import gguf as gguf_lib
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    import json, tempfile, os

    reader = gguf_lib.GGUFReader(gguf_path)

    def get_str_list(key):
        if key not in reader.fields:
            return []
        parts = reader.fields[key].parts
        # Each part is a bytes array; decode to string
        result = []
        for p in parts:
            try:
                result.append(bytes(p.tolist()).decode("utf-8"))
            except Exception:
                pass
        return result

    def get_int(key, default=None):
        if key not in reader.fields:
            return default
        v = reader.fields[key].parts[-1].tolist()
        return v[0] if isinstance(v, list) else v

    tok_model = get_str_list("tokenizer.ggml.model")
    vocab_list = get_str_list("tokenizer.ggml.tokens")
    merges_list = get_str_list("tokenizer.ggml.merges")
    bos_id = get_int("tokenizer.ggml.bos_token_id", 1)
    eos_id = get_int("tokenizer.ggml.eos_token_id", 2)
    chat_tmpl = None
    if "tokenizer.chat_template" in reader.fields:
        try:
            parts = reader.fields["tokenizer.chat_template"].parts
            chat_tmpl = "".join(bytes(p.tolist()).decode("utf-8") for p in parts)
        except Exception:
            pass

    if not vocab_list:
        return None, eos_id, bos_id, chat_tmpl

    # Build vocab dict from list
    vocab = {tok: idx for idx, tok in enumerate(vocab_list)}
    merges_pairs = [tuple(m.split(" ", 1)) for m in merges_list if " " in m]

    # Write to temp dir and load
    with tempfile.TemporaryDirectory() as tmp:
        tok_data = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [],
            "normalizer": None,
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
            "post_processor": None,
            "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
            "model": {
                "type": "BPE",
                "dropout": None,
                "unk_token": None,
                "continuing_subword_prefix": None,
                "end_of_word_suffix": None,
                "fuse_unk": False,
                "byte_fallback": False,
                "vocab": vocab,
                "merges": merges_pairs,
            },
        }
        tok_path = os.path.join(tmp, "tokenizer.json")
        with open(tok_path, "w") as f:
            json.dump(tok_data, f)
        try:
            tok = Tokenizer.from_file(tok_path)
            return tok, eos_id, bos_id, chat_tmpl
        except Exception as e:
            print(f"  GGUF tokenizer build failed: {e}")
            return None, eos_id, bos_id, chat_tmpl


def load_tokenizer(model_dir: str):
    """Load HuggingFace tokenizer from a model directory."""
    from tokenizers import Tokenizer
    tok_path = Path(model_dir) / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(f"tokenizer.json not found in {model_dir}")
    tok = Tokenizer.from_file(str(tok_path))

    # Load tokenizer config for special tokens
    cfg_path = Path(model_dir) / "tokenizer_config.json"
    eos_id = None
    bos_id = None
    chat_template = None
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        eos_token = cfg.get("eos_token")
        bos_token = cfg.get("bos_token")
        chat_template = cfg.get("chat_template")
        if eos_token:
            enc = tok.encode(eos_token, add_special_tokens=False)
            if enc.ids:
                eos_id = enc.ids[0]
        if bos_token:
            enc = tok.encode(bos_token, add_special_tokens=False)
            if enc.ids:
                bos_id = enc.ids[0]

    return tok, eos_id, bos_id, chat_template


def build_prompt(text: str, chat_template: str | None, model_dir: str) -> str:
    """Wrap text in chat template if available."""
    if chat_template is None:
        return text

    # Try applying the Jinja template via transformers if available
    try:
        from jinja2 import Template
        messages = [{"role": "user", "content": text}]
        tmpl = Template(chat_template)
        return tmpl.render(
            messages=messages,
            add_generation_prompt=True,
            bos_token="",
            eos_token="",
        )
    except ImportError:
        pass

    # Fallback: look for a chat_template.jinja file
    jinja_path = Path(model_dir) / "chat_template.jinja"
    if jinja_path.exists():
        try:
            from jinja2 import Template
            tmpl = Template(jinja_path.read_text())
            messages = [{"role": "user", "content": text}]
            return tmpl.render(messages=messages, add_generation_prompt=True)
        except ImportError:
            pass

    return text


def resolve_model_dir(model_id: str) -> str:
    """Resolve HuggingFace model ID or local path to actual directory."""
    p = Path(model_id)
    if p.exists():
        return str(p)

    # Try HuggingFace cache
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    safe_id = model_id.replace("/", "--")
    model_cache = hf_cache / f"models--{safe_id}"
    if model_cache.exists():
        snapshots = list((model_cache / "snapshots").iterdir())
        if snapshots:
            return str(sorted(snapshots)[-1])

    raise FileNotFoundError(
        f"Model not found: {model_id}. "
        "Pass a local directory or a model that is already cached in ~/.cache/huggingface/hub"
    )


class FakeAttnMetadata:
    """Minimal attn_metadata compatible with LlamaWebGPUModel._transformer_layer."""
    def __init__(self, ctx_len: int, block_size: int = 16):
        num_blocks = math.ceil(ctx_len / block_size) + 1
        self.slot_mapping = [0]  # decode: one token, slot 0
        self.block_tables = [np.zeros(num_blocks, dtype=np.uint32)]
        self.max_decode_seq_len = ctx_len


def run(model_dir: str, prompt: str, max_tokens: int = 64, temperature: float = 0.0):
    print(f"\nLoading model from: {model_dir}")

    # Read config — either from JSON or GGUF metadata
    gguf_path = None
    model_path_obj = Path(model_dir)
    if model_path_obj.suffix == ".gguf":
        gguf_path = str(model_path_obj)
        model_dir_for_tok = str(model_path_obj.parent)
    else:
        # Check if model_dir itself is a gguf file
        model_dir_for_tok = model_dir

    if gguf_path:
        from vllm_webgpu.quant.gguf_loader import gguf_read_config
        config = gguf_read_config(gguf_path)
    else:
        cfg_path = Path(model_dir) / "config.json"
        if not cfg_path.exists():
            # Maybe the dir itself has a single GGUF
            gguf_files = list(Path(model_dir).glob("*.gguf"))
            if gguf_files:
                gguf_path = str(gguf_files[0])
                from vllm_webgpu.quant.gguf_loader import gguf_read_config
                config = gguf_read_config(gguf_path)
            else:
                raise FileNotFoundError(f"No config.json or .gguf in {model_dir}")
        else:
            with open(cfg_path) as f:
                config = json.load(f)

    arch = config.get("architectures", ["LlamaForCausalLM"])[0]
    print(f"Architecture: {arch}")
    print(f"  hidden_size={config['hidden_size']}, layers={config['num_hidden_layers']}, "
          f"heads={config['num_attention_heads']}, kv_heads={config['num_key_value_heads']}")
    if gguf_path:
        print(f"  GGUF: {Path(gguf_path).name}")

    # Tokenizer — for GGUF, try embedded tokenizer first, then companion dir
    print("\nLoading tokenizer...")
    tok, eos_id, bos_id, chat_template = None, None, None, None
    if gguf_path:
        tok, eos_id, bos_id, chat_template = load_tokenizer_from_gguf(gguf_path)
        if tok is None:
            print("  GGUF tokenizer failed, trying companion dir...")
    if tok is None:
        tok_dir = model_dir_for_tok if gguf_path else model_dir
        tok, eos_id, bos_id, chat_template = load_tokenizer(tok_dir)
    full_prompt = build_prompt(prompt, chat_template, model_dir_for_tok if gguf_path else model_dir)
    print(f"Prompt (after template): {repr(full_prompt[:120])}")

    enc = tok.encode(full_prompt)
    input_ids_list = enc.ids
    if bos_id is not None and (not input_ids_list or input_ids_list[0] != bos_id):
        input_ids_list = [bos_id] + input_ids_list
    print(f"Input tokens: {len(input_ids_list)}")

    # Build config object
    class ModelConfig:
        pass
    cfg = ModelConfig()
    cfg.num_hidden_layers       = config["num_hidden_layers"]
    cfg.num_attention_heads     = config["num_attention_heads"]
    cfg.num_key_value_heads     = config["num_key_value_heads"]
    cfg.hidden_size             = config["hidden_size"]
    cfg.intermediate_size       = config["intermediate_size"]
    cfg.vocab_size              = config["vocab_size"]
    cfg.max_position_embeddings = config.get("max_position_embeddings", 4096)
    cfg.rope_theta              = config.get("rope_theta", 10000.0)
    # Some models (Gemma4, Qwen3 variants) specify head_dim explicitly
    cfg.head_dim                = config.get("head_dim", cfg.hidden_size // cfg.num_attention_heads)
    cfg.architectures           = config["architectures"]

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
    from vllm_webgpu.models.llama import LlamaWebGPUModel
    from vllm_webgpu.models.gemma4 import Gemma4WebGPUModel
    ARCH_MAP = {
        "LlamaForCausalLM": LlamaWebGPUModel,
        "MistralForCausalLM": LlamaWebGPUModel,
        "Qwen2ForCausalLM": LlamaWebGPUModel,
        "Qwen3ForCausalLM": LlamaWebGPUModel,
        "Gemma3ForCausalLM": Gemma4WebGPUModel,
        "Gemma4ForCausalLM": Gemma4WebGPUModel,
    }
    ModelClass = ARCH_MAP.get(arch)
    if ModelClass is None:
        raise NotImplementedError(f"Architecture {arch!r} not supported. Supported: {sorted(ARCH_MAP)}")

    model = ModelClass(cfg, device, pipeline_cache)

    # Load weights
    print("\nLoading weights (this may take a while)...")
    t0 = time.perf_counter()
    model.load_weights(gguf_path if gguf_path else model_dir)
    elapsed = time.perf_counter() - t0
    print(f"  Loaded {len(model.weights)} tensors in {elapsed:.1f}s")

    # Allocate KV cache (small for decode test)
    import wgpu as wgpu_lib
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.config import get_config

    block_size = get_config().block_size
    max_ctx = min(cfg.max_position_embeddings, 2048)
    num_blocks = math.ceil(max_ctx / block_size) + 4
    kv_heads = cfg.num_key_value_heads
    head_dim = cfg.head_dim
    rw = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST

    print(f"\nAllocating KV cache: {num_blocks} blocks × {block_size} × {kv_heads} heads × {head_dim} dim")
    for layer in range(cfg.num_hidden_layers):
        k = WebGPUBuffer.empty(device.wgpu_device, num_blocks * block_size * kv_heads * head_dim * 2, usage=rw)
        v = WebGPUBuffer.empty(device.wgpu_device, num_blocks * block_size * kv_heads * head_dim * 2, usage=rw)
        model.kv_pool.append((k, v))

    # --- Prefill: run the prompt tokens one by one (decode-only MVP) ---
    # We simulate prefill by running each token as a decode step.
    # This is slow but correct for the MVP single-token decode architecture.
    print(f"\nRunning prefill ({len(input_ids_list)} tokens)...")
    ctx_len = 0
    slot_idx = 0

    class Meta:
        def __init__(self, slot, blk_table, ctx):
            self.slot_mapping = [slot]
            self.block_tables = [blk_table]
            self.max_decode_seq_len = ctx

    block_table = np.zeros(num_blocks, dtype=np.uint32)
    for i, tid in enumerate(input_ids_list):
        slot = i
        block_idx = slot // block_size
        block_table[block_idx] = block_idx
        # ctx_len = i+1: token i attends to positions 0..i (itself + all prior).
        # kv_cache_store writes at slot i first, then attn_score reads positions 0..i.
        meta = Meta(slot, block_table.copy(), i + 1)
        input_arr = np.array([tid], dtype=np.uint32)
        positions_arr = np.array([i], dtype=np.uint32)
        logits = model.forward(input_arr, positions_arr, meta)
        ctx_len = i + 1
        if (i + 1) % 10 == 0 or i == len(input_ids_list) - 1:
            print(f"  prefill {i+1}/{len(input_ids_list)}", end="\r", flush=True)
    print()

    # Sanity: check logit quality before decode
    top1 = int(np.argmax(logits[0]))
    top1_val = float(logits[0][top1])
    print(f"  Last prefill logit: argmax={top1}, value={top1_val:.2f}, std={logits[0].std():.2f}")

    # Decode loop
    print(f"\nDecoding (max {max_tokens} tokens)...")
    generated = []
    t_start = time.perf_counter()
    last_token = np.argmax(logits[0]).item()

    for step in range(max_tokens):
        if last_token == eos_id:
            print(f"  [EOS at step {step}]")
            break
        generated.append(last_token)

        slot = len(input_ids_list) + step
        block_idx = slot // block_size
        if block_idx >= num_blocks:
            print(f"  [KV cache full at step {step}]")
            break
        block_table[block_idx] = block_idx
        ctx_len_now = len(input_ids_list) + step + 1

        meta = Meta(slot, block_table.copy(), ctx_len_now)
        input_arr = np.array([last_token], dtype=np.uint32)
        positions_arr = np.array([slot], dtype=np.uint32)

        logits = model.forward(input_arr, positions_arr, meta)

        if temperature == 0.0:
            last_token = int(np.argmax(logits[0]))
        else:
            scaled = logits[0].astype(np.float64) / temperature
            scaled -= scaled.max()
            probs = np.exp(scaled)
            probs /= probs.sum()
            last_token = int(np.random.choice(len(probs), p=probs))

        if (step + 1) % 5 == 0:
            partial = tok.decode(generated)
            print(f"  [{step+1} tokens]: {repr(partial[-60:])}", flush=True)

    t_end = time.perf_counter()
    n_tok = len(generated)
    tok_per_sec = n_tok / max(t_end - t_start, 0.001)

    output_text = tok.decode(generated)
    print(f"\n{'='*60}")
    print(f"Prompt: {repr(prompt[:80])}")
    print(f"Output: {output_text}")
    print(f"{'='*60}")
    print(f"Generated {n_tok} tokens at {tok_per_sec:.1f} tok/s")
    return output_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model path or HuggingFace ID")
    parser.add_argument("--prompt", default="What is 2+2?", help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0, help="0=greedy")
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model)
    run(model_dir, args.prompt, args.max_tokens, args.temperature)
