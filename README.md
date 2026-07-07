# vllm-webgpu

A [vLLM](https://github.com/vllm-project/vllm) out-of-tree platform plugin that runs LLM inference on WebGPU via [wgpu-py](https://github.com/pygfx/wgpu-py). Compute kernels are written in WGSL and run on Metal (Apple Silicon), Vulkan, or DX12.

## What it does

- Registers as a vLLM platform plugin — the full vLLM scheduler, block allocator, and engine remain in use.
- Replaces only the compute layer with custom WGSL kernels.
- Loads models from HuggingFace safetensors (f16/bf16) and MLX format.
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
| Qwen3-4B (f16) | ~18-19 tok/s |
| Qwen3-0.6B (f16) | ~60 tok/s |
| Qwen3.5-9B (MLX 4-bit) | ~11 tok/s |

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

# Gemma4-12B (safetensors)
python run_inference.py \
  --model google/gemma-4-12b-it \
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
| 3 | GPTQ int4 | Symmetric, group_size=128, transposed to [N, K/8] |
| 4 | AWQ int4 | [K, N/8] nibble order, zero_point=8 |
| 5 | FP8 E4M3 | Raw u8 bytes, GLOBAL_SCALE constant |
| 6 | NVFP4 | [N, K/2] packed FP4 + [N, K/16] f16 scales |
| 7 | Int8 per-channel | Raw I8 bytes; shader sign-extends to f32 |
| 8 | BnB NF4 | 4-bit normal-float, GROUP_K=64 absmax block |

MLX affine-int4 and other non-native formats are dequantized to f16 at load time.

## WGSL kernels

See [KERNELS.md](KERNELS.md) for the full reference with dispatch shapes, overrides, and composability notes.

**Decode path** (f16 weights, fused QKV, per-head norms — e.g. Qwen3-4B): **11 dispatches/layer** (down from 16 before fusion work).

| # | Shader | Operation |
|---|--------|-----------|
| 1 | `fused_qkv` | Q+K+V projections in one dispatch |
| 2 | `fused_qk_norm_rope` | Q+K per-head RMSNorm + RoPE |
| 3 | `kv_cache_store_both` | K+V paged cache write |
| 4 | `attn_score` | QK dot-products vs paged K cache |
| 5 | `softmax` | Online 2-pass softmax |
| 6 | `attn_output` | Weighted V sum from paged V cache |
| 7 | `matmul_quant` | Output projection (o_proj) |
| 8 | `add_rms_norm` | Post-attn residual add + FFN pre-norm |
| 9 | `fused_gate_act` | Gate+up GEMV with inline SiLU/GELU |
| 10 | `matmul_quant` | Down projection |
| 11 | `add_rms_norm` | Post-FFN residual add + next layer pre-norm |

Fallback paths (quantized weights, no per-head norms): 14-16 dispatches/layer.

**Prefill path:** `matmul_quant_mr4` (tiled GEMM, M×4 output tile per workgroup). Layers are chunked across separate command encoders (4 layers each) to stay under the Metal command-buffer timeout. T is not bounded.

**Long-context fallback:** `flash_attn_decode` replaces the three-pass `attn_score + softmax + attn_output` when `ctx_len > 65535` (the WebGPU dispatch-dimension limit for the three-pass approach). The fused shader loops over all KV positions internally and has no per-axis limit.

**Kernels by category:**
- Core: `matmul_quant`, `matmul_quant_mr4`, `rms_norm`, `rms_norm_f32in`, `add_rms_norm`, `add_f32_rms_norm`, `embedding_lookup`, `add`, `argmax_f16`
- Attention: `attn_score`, `softmax`, `attn_output`, `flash_attn_decode`, `kv_cache_store_both`, `kv_cache_store`
- Fused: `fused_qkv`, `fused_qk_norm_rope`, `fused_per_head_norm_rope`, `fused_gate_act`, `fused_gate_up`
- Sampling: `gumbel_sample`, `topk256`, `topk_sort`
- Gemma-specific: `logit_softcap`, `per_head_rms_norm_no_weight`, `ple_gelu_mul`, `ple_skip_scale_add`, `ple_stage1_fuse`
- Qwen3.5 GDN: `causal_conv_step`, `gdn_state_update`, `linear_attn_norm_gate`, `sigmoid_gate`

## Weight loading

| Source | Format detected by |
|---|---|
| HuggingFace model ID or local dir | `model.safetensors.index.json` present |
| Single-shard `.safetensors` | file extension |
| MLX directory | `.biases` keys in safetensors index |

Model paths are resolved through the HF cache (`~/.cache/huggingface/hub/`) automatically. All format detection and upload logic lives in `vllm_webgpu/quant/weight_loader.py`.

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
- Batch prefill: up to T unlimited tokens per forward call (chunked encoder submission, no GPU timeout).
- Standard 3-pass attention supports ctx_len up to 65535 tokens. `flash_attn_decode` activates automatically for ctx_len > 65535 (loops internally, no per-axis dispatch limit).
- Gemma models: f16 residual precision gap vs bfloat16; WebGPU has no bfloat16 support.
- Qwen3.5 GDN layers: quantization not supported (always f16).
- Block table capped at 512 blocks per sequence at init time.

See [MODELS.md](MODELS.md) for per-model details.
