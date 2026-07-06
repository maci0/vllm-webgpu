# vllm-webgpu

A [vLLM](https://github.com/vllm-project/vllm) out-of-tree platform plugin that runs LLM inference on WebGPU via [wgpu-py](https://github.com/pygfx/wgpu-py). Compute kernels are written in WGSL and run on Metal (Apple Silicon), Vulkan, or DX12.

## What it does

- Registers as a vLLM platform plugin — the full vLLM scheduler, block allocator, and engine remain in use.
- Replaces only the compute layer with custom WGSL kernels.
- Loads models from HuggingFace safetensors (f16/bf16) or GGUF (Q4\_K, Q6\_K, f32).
- Runs on any WebGPU-capable GPU via wgpu-py: Apple M-series (Metal), NVIDIA/AMD (Vulkan/DX12).

## Supported models

See [MODELS.md](MODELS.md) for the full matrix including quantization formats and known limitations.

| Architecture | Example models | Quant formats |
|---|---|---|
| LlamaForCausalLM / Qwen3ForCausalLM | Llama 3, Qwen3-0.6B–72B | f16, Q4_K, GPTQ, AWQ, FP8, NVFP4 |
| Qwen2ForCausalLM / MistralForCausalLM | Qwen2.5, Mistral-7B | f16, Q4_K, GPTQ, AWQ, FP8, NVFP4 |
| Qwen3_5ForConditionalGeneration | Qwen3.5-9B, Qwen3.6-27B | f16, Q4_K (attn/FFN); f16 only (GDN layers) |
| Qwen3_5MoeForConditionalGeneration | Qwen3.6-35B-A3B | same as Qwen3.5; MoE routing on GPU |
| Gemma3/4ForCausalLM | Gemma3-1B–27B, Gemma4-12B | f16, Q4_K |
| DiffusionGemmaForBlockDiffusion | DiffusionGemma | f16, Q4_K |

**Throughput** (Apple M3, single-sequence decode, no CPU↔GPU transfers):

| Model | tok/s |
|---|---|
| Qwen3-4B (f16) | ~18 tok/s |
| Qwen3-0.6B (f16) | ~60 tok/s |

## Install

```bash
# Create isolated environment
uv venv .venv
source .venv/bin/activate

# Install plugin + dependencies
uv pip install -e .

# Optional: for MLX format (Qwen3.5)
uv pip install vllm>=0.24.0
```

Python 3.12+ required. vLLM 0.24.0 tested.

## Standalone inference

```bash
# Qwen3-4B (HuggingFace safetensors)
python run_inference.py --model Qwen/Qwen3-4B --prompt "The capital of France is"

# Gemma4-12B (GGUF)
python run_inference.py \
  --model ~/.cache/huggingface/hub/models--unsloth--gemma-4-12b-it-GGUF/.../gemma-4-12b-it-Q4_K_M.gguf \
  --prompt "2+2="

# Qwen3.5-9B (MLX 4-bit)
python run_inference.py \
  --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-4bit/... \
  --prompt "Hello"
```

## vLLM engine integration

The plugin auto-registers via the entry point. Use vLLM as normal:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-4B",
    max_model_len=512,
    enforce_eager=True,
)
outputs = llm.generate("The capital of France is", SamplingParams(max_tokens=50))
print(outputs[0].outputs[0].text)
```

vLLM selects the WebGPU platform automatically when the plugin is installed and a WebGPU adapter is available.

## Kernel profiling

```bash
python profile_kernels.py --model Qwen/Qwen3-4B --decode-steps 10 --warmup-steps 3
```

Outputs per-layer timing and effective memory bandwidth. Set `model.profiling = True` in code for programmatic access.

## Architecture

```
vLLM engine  (scheduler, block allocator, request lifecycle)
     │
     └─ WebGPU platform plugin
           │
           ├─ WebGPUWorker  (WorkerBase)
           │      └─ WebGPUModelRunner
           │            ├─ load_model()       — selects model class by architecture
           │            ├─ execute_model()     — runs forward() per decode step
           │            └─ get_kv_cache_spec() — reports per-layer KV dims
           │
           └─ Model classes
                 ├─ LlamaWebGPUModel        — Llama/Qwen2/Qwen3
                 ├─ Gemma4WebGPUModel       — Gemma3/4 (heterogeneous attention, f32 residual)
                 ├─ Qwen35WebGPUModel       — Qwen3.5/3.6 (hybrid GDN + standard attention)
                 └─ DiffusionGemmaWebGPUModel — DiffusionGemma (MoE, block diffusion)
```

### Forward pass (single encoder, one GPU submit per token)

All dispatches for an entire 36-layer forward pass are recorded into one `CommandEncoder` and submitted in a single `queue.submit()`. This eliminates per-layer submit overhead (~2.3ms/submit on Metal), which was the dominant bottleneck.

### Paged KV cache

Block size 16 tokens. Each layer has its own KV buffer pair. Gemma4-12B uses heterogeneous block sizes (local layers 8 kv-heads×256-dim, global layers 1 kv-head×512-dim). KV blocks are managed by vLLM's block allocator.

### Quantization

`matmul_quant.wgsl` decodes all quantized formats entirely on GPU (no CPU dequant in the forward path):

| USE_QUANT | Format | Notes |
|---|---|---|
| 0 | f16 | Two f16 per u32, standard HF format |
| 2 | GGUF Q4_K | 144 bytes/256 weights, GPU block decoder |
| 3 | GPTQ int4 | Symmetric, group_size=128, transposed to [N, K/8] |
| 4 | AWQ int4 | [K, N/8] nibble order, zero_point=8 |
| 5 | FP8 E4M3 | Raw u8 bytes, GLOBAL_SCALE constant |
| 6 | NVFP4 | [N, K/2] packed FP4 + [N, K/16] f16 scales |

Q6_K, F32, and MLX affine-int4 are dequantized to f16 at load time.

## WGSL kernels

See [KERNELS.md](KERNELS.md) for the full reference with dispatch shapes, overrides, and composability notes.

**Core kernels (all models):**
`matmul_quant`, `rms_norm`, `fused_per_head_norm_rope`, `kv_cache_store_both`, `attn_score`, `softmax`, `attn_output`, `embedding_lookup`, `add`, `add_rms_norm`, `argmax_f16`

**Fused kernels (dispatch count reduction):**
`fused_qkv` (Q+K+V projections), `fused_qk_norm_rope` (Q+K norm+rope), `fused_gate_act` (gate+up+SiLU/GELU), `kv_cache_store_both` (K+V cache write), `add_rms_norm` / `add_f32_rms_norm` (residual add + next layer norm)

**Decode dispatch count per layer** (f16, fused QKV path): **11 dispatches** (vs 16 before fusions)

## Weight loading

| Source | Format detected by |
|---|---|
| HuggingFace model ID or local dir | `model.safetensors.index.json` present |
| Single-shard `.safetensors` | file extension |
| `.gguf` file | magic bytes `GGUF` |
| MLX directory | `.biases` keys in safetensors index |

Model paths are resolved through the HF cache (`~/.cache/huggingface/hub/`) automatically.

## Adding a new architecture

1. Create `vllm_webgpu/models/mymodel.py` extending `BaseWebGPUModel`.
2. Implement `forward(input_ids, positions, attn_metadata) → np.ndarray`.
3. Add the HF architecture string to `ARCH_MAP` in `vllm_webgpu/v1/model_runner.py`.
4. Add it to `run_inference.py`'s `ARCH_MAP` if standalone inference is needed.

## Tests

```bash
pytest tests/ -q
```

68 tests covering: kernel correctness (softmax sum-to-one, RMSNorm scale invariance, RoPE norm preservation, matmul linearity, attention pipeline), quantization round-trips, model instantiation, vLLM platform integration, Qwen3.5/Gemma4 layer dispatch.

## Limitations

- Single-sequence decode only (no request batching).
- Prefill is token-by-token (the model runner loops over prompt tokens).
- ctx_len limited to 65535 (WebGPU dispatch dimension limit).
- Gemma models: f16 residual precision gap vs bfloat16; WebGPU has no bfloat16 support.
- Qwen3.5 GDN layers: quantization not supported (always f16).
- Block table capped at 512 blocks per sequence at init time.

See [MODELS.md](MODELS.md) for per-model details.
