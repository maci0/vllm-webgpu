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

**Loader-only (no shader changes needed, CPU conversion at load time):**

| Format | vLLM name | Status |
|--------|-----------|--------|
| MXFP8 | `mxfp8`, `modelopt_mxfp8` | ✓ Implemented — u8 exponent scales dequanted to f16 at load time, plain f16 weights uploaded |
| MXFP4 | `mxfp4` | ✓ Implemented — u8 exponent scales → f16 via 2^(e-127), routes to USE_QUANT=6 (GROUP_K=32) |
| compressed-tensors | `compressed-tensors` | ✓ Implemented — config_groups JSON parsed, routes to USE_QUANT 3/5/7 by sub-format |
| NF4 double-quant | `bitsandbytes` (advanced) | Planned — nested absmax not yet decoded |
| torchao int4/int8 | `torchao` | Planned — checkpoint-specific format |

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

**Verified models:**

| Model | Quantization | Status |
|-------|-------------|--------|
| Qwen3.5-0.8B | f16 (safetensors) | Working. Correct predictions for single-token inputs. Longer sequences (chat template adds 16+ prefill tokens) show f16 precision degradation from accumulated bfloat16→float16 rounding across 24 layers. |
| Qwen3.5-9B | MLX 4-bit | Working correctly. Verified factual answers (arithmetic, factual recall). |

Full-attention layers: USE_QUANT 0–6 (same `_uq_weight()` logic as LlamaWebGPUModel).

GDN linear-attention layers: always USE_QUANT=0. Projections `in_proj_qkv`, `in_proj_a`, `in_proj_b`, `in_proj_z`, `out_proj`, `conv1d` are dispatched as f16 matmul_quant regardless of the loaded weight format.

GDN recurrent state (SSM matrix + conv history) is stored in persistent GPU buffers and updated in-place each decode step. Call `reset_recurrent_states()` at the start of each new sequence.

**GDN quality note:** Single-token predictions match HuggingFace reference output. Multi-token quality (generation beyond the prompt) degrades relative to HuggingFace due to a known mismatch in GDN computation: this runtime processes each token sequentially with a step-by-step SSM update, while HuggingFace's training and reference inference uses chunked parallel GDN evaluation. The sequential step-by-step path accumulates small numerical differences across layers that compound over longer output sequences.

Qwen3.6-27B: same architecture as Qwen3.5, maps here via `Qwen3_5ForConditionalGeneration`.

Qwen3.6-35B-A3B (MoE): maps here via `Qwen3_5MoeForConditionalGeneration`. Dense GDN and full-attention layers work; MoE FFN routing dispatches `topk_sort.wgsl` on GPU (256 experts, top-8 selection). Expert FFN matmuls are dispatched sequentially per selected expert.

Interleaved RoPE (`mrope_interleaved=True` in config) is supported via `INTERLEAVED=1` override in `fused_per_head_norm_rope` / `fused_qk_norm_rope`. Partial RoPE (`partial_rotary_factor`) is supported via `ROTARY_DIM`.

**Formula bugs fixed during bring-up (7 total):**

1. `attn_output_gate` not applied at all: `q_proj.weight` has shape `[2*q_dim, hidden]` — first half Q, second half gate. The gate was silently ignored; `o_proj` received untrained input.
2. `attn_output_gate` wrong weight split: code used a first-half/second-half split across all heads, but HuggingFace interleaves per head: rows `[h*2*hd : h*2*hd+hd]` are Q and `[h*2*hd+hd : (h+1)*2*hd]` are gate for each head `h`.
3. `attn_output_gate` wrong activation: gate was passed through SiLU (`gelu_mul.wgsl`) but HF uses `torch.sigmoid(gate)`. Fixed with `sigmoid_gate.wgsl`.
4. `GEMMA_NORM=0` for all Qwen3.5 RMSNorm dispatches: `Qwen3_5RMSNorm` uses `(1+weight)` scaling. Without `GEMMA_NORM=1`, all logits are approximately 2× off.
5. GDN `linear_attn_norm_gate` gate formula: shader used `sigmoid(z)` but `Qwen3_5RMSNormGated` applies `F.silu(z) = z * sigmoid(z)`.
6. GDN `causal_conv_step` SiLU: HF applies `F.silu` after the depthwise conv1d. This was incorrectly removed (then restored after layer-by-layer verification).
7. GDN `gdn_state_update` indexing: `A_log` and `dt_bias` were indexed by V-head (`vh`) but the buffer is sized for K-heads (`num_k_heads`). When `num_v_heads > num_k_heads` (Qwen3.5-9B: 32 vs 16), this caused out-of-bounds reads for `vh >= num_k_heads`.

GEMMA_NORM auto-detection: safetensors format stores layernorm weights as deviations from 1 (mean ≈ 0.24, use `GEMMA_NORM=1`). MLX format pre-absorbs the +1 into the weight (mean ≈ 1.03, use `GEMMA_NORM=0`). Detection: if `mean(|weight|) > 0.7` → absolute format, else → deviation format.

### DiffusionGemmaWebGPUModel

Extends `Gemma4WebGPUModel`. Architecture differences:
- Weight key prefix: `model.decoder.layers.N.*` instead of `model.layers.N.*`.
- `forward()` returns full f32 logits (no GPU argmax; diffusion generation needs the full distribution).
- MoE router: `topk_sort.wgsl` on GPU for 128 experts, top-8 active. Reads back 2×8 scalars (64 bytes) per MoE layer for Python-side dispatch. Expert FFN buffers (moe_ping/moe_pong) not pre-allocated.

USE_QUANT 0, 1, 2, 3 only. NVFP4 (USE_QUANT=6) not detected despite NVFP4 being the primary expert weight format for DiffusionGemma NVFP4 checkpoints.

---

## Long-context support

Standard 3-pass attention (attn_score + softmax + attn_output) dispatches one workgroup per (query head, context position). WebGPU's per-axis dispatch limit caps this at ctx_len = 65535.

For ctx_len > 65535, the model runner automatically falls back to `flash_attn_decode.wgsl`, which loops over all KV positions inside each workgroup. This removes the per-axis limit but reduces parallelism to num_q_heads workgroups rather than num_q_heads × ctx_len.

KV cache allocation uses `min(max_position_embeddings, 65535)` as the slot count. Models with max_position_embeddings > 65535 (e.g. Llama-3.1 at 131072) are capped at 65535 slots; sequences beyond that length are not supported.

---

## Known limitations (all models)

| Limitation | Detail |
|---|---|
| Single-sequence only | `forward()` accepts one sequence at a time; no request batching |
| Token-by-token prefill | The model runner loops over prompt tokens; no batch prefill GEMM |
| ctx_len > 65535 | Standard 3-pass attention is capped at 65535 by the WebGPU dispatch limit; flash_attn_decode is used as an automatic fallback beyond that point |
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
| `sigmoid_gate` (attn output gate) | no | no | always | no |
| `kv_cache_store_both` (K+V → 1) | always | always | always | no (no KV cache) |
| `add_rms_norm` (add+norm → 1) | always | — | always | always |
| `add_f32_rms_norm` (add+norm f32 → 1) | — | always | — | — |
| `rms_norm_add_f32_rms_norm` (double-norm + f32 add → 1) | — | always | — | — |
| Cross-layer norm fusion | always | always | always | always |

"always" = applies regardless of weight format. "no" = not implemented for this model/path.

**Qwen3.5-specific formulas that differ from Llama:**

| Component | Llama formula | Qwen3.5 formula |
|-----------|--------------|-----------------|
| RMSNorm weight scale | `w` (standard) | `1 + w` (GEMMA_NORM=1) |
| Attention output gate | none | `sigmoid(gate_proj(h)) * attn_out` before `o_proj` |
| Q/gate weight split | first half Q, second half gate | interleaved per head: rows `[h*2*hd : h*2*hd+hd]` = Q, next `hd` rows = gate |
| GDN layer norms | n/a | `Qwen3_5RMSNorm`: `(1+w)` scaling; `Qwen3_5RMSNormGated`: `silu(z) * norm(h)` |
| GDN state update | n/a | `A^dt * state + outer(b, v)`, A indexed by K-head, A_log/dt_bias indexed by V-head |
