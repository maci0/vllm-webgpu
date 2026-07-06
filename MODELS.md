# Model Support

## Architecture map

| HuggingFace architecture | Backend | Model class |
|---|---|---|
| `LlamaForCausalLM` | `llama` | `LlamaWebGPUModel` |
| `MistralForCausalLM` | `llama` | `LlamaWebGPUModel` |
| `Qwen2ForCausalLM` | `llama` | `LlamaWebGPUModel` |
| `Qwen3ForCausalLM` | `llama` | `LlamaWebGPUModel` |
| `Gemma3ForCausalLM` | `gemma4` | `Gemma4WebGPUModel` |
| `Gemma3ForConditionalGeneration` | `gemma4` | `Gemma4WebGPUModel` |
| `Gemma4ForCausalLM` | `gemma4` | `Gemma4WebGPUModel` |
| `Gemma4UnifiedForConditionalGeneration` | `gemma4` | `Gemma4WebGPUModel` |
| `Qwen3_5ForConditionalGeneration` | `qwen35` | `Qwen35WebGPUModel` |
| `Qwen3_5MoeForConditionalGeneration` | `qwen35` | `Qwen35WebGPUModel` |
| `DiffusionGemmaForBlockDiffusion` | `diffusion_gemma` | `DiffusionGemmaWebGPUModel` |

## Quantization support

Quantization is handled inside `matmul_quant.wgsl` via the `USE_QUANT` override. The model's `_uq()` function inspects weight dtype and metadata to select the right value.

| USE_QUANT | Format | vLLM name | Detection | Group size |
|-----------|--------|-----------|-----------|------------|
| 0 | f16 (plain) | — | default | — |
| 1 | Simple Q4 + external scales | — | `.scales` key present | per-tensor |
| 2 | GGUF Q4_K | (outside registry) | `__quant_types__[key] == 12` | 256 |
| 3 | GPTQ int4 | `gptq`, `gptq_marlin` | `dtype == "i32"`, `fmt != "awq_sym"` | 128 (configurable) |
| 4 | AWQ int4 | `awq`, `awq_marlin` | `dtype == "i32"`, `fmt == "awq_sym"` | 128 (configurable) |
| 5 | FP8 E4M3 | `fp8`, `modelopt` | `dtype == "u8"`, `fmt == "fp8_gpu"` | global scale |
| 6 | NVFP4 | `modelopt_fp4` | `dtype == "u8"`, `fmt == "nvfp4_gpu"` | 16 |
| 7 | Int8 per-channel | `bitsandbytes` int8, `compressed-tensors` int8 | `dtype == "u8"`, `fmt == "int8_gpu"` | per-row |
| 8 | NF4 (Normal Float 4) | `bitsandbytes` nf4 | `dtype == "u8"`, `fmt == "nf4_gpu"` | 64 |

**Not yet supported** (WebGPU-feasible but loader/detection work needed):

| Format | vLLM name | Blocker |
|--------|-----------|---------|
| MXFP8 | `mxfp8`, `modelopt_mxfp8` | u8 exponent scales need load-time → f16 conversion |
| MXFP4 | `mxfp4` | same as MXFP8; block_size=32 vs NVFP4's 16 |
| NF4 double-quant | `bitsandbytes` (advanced) | nested absmax: scales of scales |
| compressed-tensors int8/fp8 | `compressed-tensors` | sub-format detection from `config_groups` JSON |
| torchao int4/int8 | `torchao` | checkpoint-specific format |

**Not feasible for WebGPU** (CUDA-specific memory layouts or missing hardware support):

| Format | Reason |
|--------|--------|
| Marlin (GPTQ/AWQ Marlin) | Tensor-core tile layout requires CUDA |
| ExllamaV2 | CUDA-specific |
| bfloat16 weights | WebGPU has no bf16 compute |
| DeepSeek V4 FP8 | Architecture-specific mixed-precision MoE |

### LlamaWebGPUModel

Full support for USE_QUANT 0–6. Applies to all projections: Q, K, V, o_proj, gate, up, down. LM head is always USE_QUANT=0 with SPLIT_K=0 (vocab_size > 65535 exceeds the per-axis dispatch limit).

The fused QKV path (`fused_qkv.wgsl`) and `fused_gate_act.wgsl` require f16 weights (USE_QUANT=0). Quantized weights fall back to three separate `matmul_quant` calls for QKV, and separate gate/up matmuls followed by `gelu_mul`.

Q/K norms (`q_norm.weight`, `k_norm.weight`) that are shape `(head_dim,)` rather than `(num_heads * head_dim,)` are tiled to the expected shape at load time (Qwen3 uses shared norms across heads).

### Gemma4WebGPUModel

USE_QUANT 0, 1, 2 only. Q6_K is eagerly dequantized to f16 at load time; GPTQ/AWQ/FP8/NVFP4 are not detected.

Uses an f32 residual stream to avoid saturation from large `output_norm` weights (up to ~600). Adds and norms operate in f32 via `add_f32.wgsl`, `rms_norm_f32in.wgsl`, `add_f32_rms_norm.wgsl`.

Gemma3 vs Gemma4 distinction:
- `_apply_v_norm = True` only when "Gemma4" appears in the architecture name. Gemma3 skips per-head V normalization.
- Layer type detection (local vs global attention) reads `layer_types` from config, or infers from `layer_idx % 6 == 5`.
- Global attention layers (every 6th): HEAD_DIM=512, 1 KV head, V=K (no separate v_proj weight).

`layer_output_scale` (≈0.053) cached at load time for all layers — avoids one blocking GPU→CPU readback per layer per token.

### Qwen35WebGPUModel

Full-attention layers: USE_QUANT 0–6 (same `_uq_weight()` logic as LlamaWebGPUModel).

GDN linear-attention layers: always USE_QUANT=0. Projections `in_proj_qkv`, `in_proj_a`, `in_proj_b`, `in_proj_z`, `out_proj`, `conv1d` are dispatched as f16 matmul_quant regardless of the loaded weight format.

GDN recurrent state (SSM matrix + conv history) is stored in persistent GPU buffers and updated in-place each decode step. Call `reset_recurrent_states()` at the start of each new sequence.

Qwen3.6-27B: same architecture as Qwen3.5, maps here via `Qwen3_5ForConditionalGeneration`.

Qwen3.6-35B-A3B (MoE): maps here via `Qwen3_5MoeForConditionalGeneration`. Dense GDN and full-attention layers work; MoE FFN routing dispatches `topk_sort.wgsl` on GPU (256 experts, top-8 selection). Expert FFN matmuls are dispatched sequentially per selected expert.

Interleaved RoPE (`mrope_interleaved=True` in config) is supported via `INTERLEAVED=1` override in `fused_per_head_norm_rope` / `fused_qk_norm_rope`. Partial RoPE (`partial_rotary_factor`) is supported via `ROTARY_DIM`.

### DiffusionGemmaWebGPUModel

Extends `Gemma4WebGPUModel`. Architecture differences:
- Weight key prefix: `model.decoder.layers.N.*` instead of `model.layers.N.*`.
- `forward()` returns full f32 logits (no GPU argmax; diffusion generation needs the full distribution).
- MoE router: `topk_sort.wgsl` on GPU for 128 experts, top-8 active. Reads back 2×8 scalars (64 bytes) per MoE layer for Python-side dispatch. Expert FFN buffers (moe_ping/moe_pong) not pre-allocated.

USE_QUANT 0, 1, 2, 3 only. NVFP4 (USE_QUANT=6) not detected despite NVFP4 being the primary expert weight format for DiffusionGemma NVFP4 checkpoints.

---

## Known limitations (all models)

| Limitation | Detail |
|---|---|
| Single-sequence only | `forward()` accepts one sequence at a time; no request batching |
| Token-by-token prefill | The model runner loops over prompt tokens; no batch prefill GEMM |
| ctx_len ≤ 65535 | WebGPU dispatch dimension limit; long contexts require attention splitting |
| hidden/inter must be divisible by 4 | vec4 shader requirement |
| head_dim must be even | f16 GEMV packing requirement |
| Block table cap | 512 blocks per sequence, set at init time |
| No bfloat16 | WebGPU lacks bfloat16; Gemma models trained in bf16 see reduced output quality |
| Grammar/structured output | `sample_tokens()` returns the cached greedy token unchanged |

---

## Fused dispatch summary

The table below shows which fused kernels apply per model. "f16 only" means the fusion requires all-f16 weights; quantized weights fall back to the unfused path.

| Fusion | `llama` | `gemma4` | `qwen35` full-attn | `qwen35` GDN |
|--------|---------|---------|-------------------|-------------|
| `fused_qkv` (Q+K+V → 1) | f16 only | no | no | no |
| `fused_qk_norm_rope` (Q+K → 1) | f16+qnorm | no (sep. q_buf/k_buf) | no | no |
| `fused_gate_act` (gate+up+act → 1) | f16 only | f16 only | f16 only | always |
| `kv_cache_store_both` (K+V → 1) | always | always | always | no (no KV cache) |
| `add_rms_norm` (add+norm → 1) | always | — | always | always |
| `add_f32_rms_norm` (add+norm f32 → 1) | — | always | — | — |
| Cross-layer norm fusion | always | always | always | always |

"always" = applies regardless of weight format. "no" = not implemented for this model/path.
