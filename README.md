# vllm-webgpu

A [vLLM](https://github.com/vllm-project/vllm) out-of-tree platform plugin that runs LLM inference on WebGPU via [wgpu-py](https://github.com/pygfx/wgpu-py). Compute kernels are written in WGSL and run on Metal (Apple Silicon), Vulkan, or DX12.

## What it does

- Registers as a vLLM platform plugin — the full vLLM scheduler, block allocator, and engine remain in use.
- Replaces only the compute layer with custom WGSL kernels.
- Loads models from HuggingFace safetensors (f16/bf16) or GGUF (Q4\_K, Q6\_K, f32).
- Runs on any WebGPU-capable GPU via wgpu-py: Apple M-series (Metal), NVIDIA/AMD (Vulkan/DX12).

## Supported models

| Model family | Format | Notes |
|---|---|---|
| Llama 3.x | safetensors (f16) | LlamaForCausalLM |
| Qwen 2.5 / 3.x | safetensors (f16) | Qwen2ForCausalLM, Qwen3ForCausalLM |
| Gemma 4-12B | GGUF Q4\_K\_M | Heterogeneous local+global attention; Q4\_K decoded on GPU |
| Qwen 3.5-9B | MLX affine int4 | Hybrid GDN linear attention + standard attention |

Tested: Qwen3-4B at **8.2 tok/s**, Qwen3-8B at ~4 tok/s on Apple M4 Pro.

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
                 ├─ LlamaWebGPUModel   — Llama/Qwen2/Qwen3
                 ├─ Gemma4WebGPUModel  — Gemma4 (heterogeneous attention)
                 └─ Qwen35WebGPUModel  — Qwen3.5 (hybrid GDN + standard)
```

### Forward pass (single encoder, one GPU submit per token)

All dispatches for an entire 36-layer forward pass are recorded into one `CommandEncoder` and submitted in a single `queue.submit()`. This eliminates per-layer submit overhead (~2.3ms/submit on Metal), which was the dominant bottleneck.

### Paged KV cache

Block size 16 tokens. Each layer has its own KV buffer pair. Gemma4-12B uses heterogeneous block sizes (local layers 8 kv-heads×256-dim, global layers 1 kv-head×512-dim). KV blocks are managed by vLLM's block allocator.

### Quantization

| Format | WGSL path | Notes |
|---|---|---|
| f16 safetensors | USE\_QUANT=0: two f16 per u32 | Standard HF format |
| GGUF Q4\_K | USE\_QUANT=2: GPU Q4\_K block decoder | 144 bytes/256 weights, deinterleaved nibbles |
| GGUF Q6\_K | Eager dequant at load time | → f16 |
| GGUF F32 | Eager cast at load time | → f16 |
| MLX affine int4 | Dequant at load time | uint32 nibbles + bf16 scales/biases → f16 |

The Q4\_K GPU decoder reads the raw GGUF block format (d, dmin, 12 scale bytes, 128 nibble bytes) and dequantizes entirely in the shader. No CPU decode.

## WGSL kernels

| Shader | Purpose |
|---|---|
| `matmul_quant.wgsl` | GEMV (decode, M=1): f16 or Q4\_K |
| `matmul_quant_mr4.wgsl` | Batched GEMM (prefill): 256-thread K-reduction |
| `attn_score.wgsl` | QK dot-product (paged, GQA) |
| `attn_output.wgsl` | Weighted V sum (paged, GQA) |
| `softmax.wgsl` | Online 2-pass (Milakov-Divanov) |
| `rms_norm.wgsl` | RMSNorm with register-tile |
| `fused_per_head_norm_rope.wgsl` | Per-head RMSNorm + RoPE fused |
| `kv_cache_store.wgsl` | Paged KV write (vec2<f16>) |
| `embedding_lookup.wgsl` | Token embedding (vec4<f16>) |
| `gelu_mul.wgsl` | SwiGLU (vec4<f16>) |
| `add.wgsl` | Residual add (vec4<f16>) |
| `causal_conv_step.wgsl` | Single-step causal depthwise conv (Qwen3.5) |
| `gdn_state_update.wgsl` | GDN delta-rule SSM update (Qwen3.5) |
| `linear_attn_norm_gate.wgsl` | Per-head RMSNorm + sigmoid gate (Qwen3.5) |
| `logit_softcap.wgsl` | Gemma logit softcap: tanh(x/cap)×cap |
| `per_head_rms_norm_no_weight.wgsl` | Per-head RMSNorm, no learnable weight (Gemma4 V) |

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

- Single-sequence decode only (no batching).
- Prefill runs token-by-token (KV cache populated sequentially).
- ctx\_len limited to 65535 (WebGPU dispatch limit per axis).
- Gemma4-12B generates plausible tokens but thinking-mode output quality depends on Q4\_K approximation accuracy.
- Qwen3.5 GDN uses a simplified delta rule (b-projection term omitted).
