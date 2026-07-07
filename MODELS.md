# Model Support

## Architecture map

| HuggingFace architecture | Backend | Model class |
|---|---|---|
| `LlamaForCausalLM` | `llama` | `LlamaWebGPUModel` |
| `MistralForCausalLM` | `mixtral` | `MixtralWebGPUModel` |
| `MixtralForCausalLM` | `mixtral` | `MixtralWebGPUModel` |
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
| 3 | GPTQ int4 | `gptq`, `gptq_marlin` | `dtype == "i32"`, `fmt != "awq_sym"` | 128 (configurable) |
| 4 | AWQ int4 | `awq`, `awq_marlin` | `dtype == "i32"`, `fmt == "awq_sym"` | 128 (configurable) |
| 5 | FP8 E4M3 | `fp8`, `modelopt` | `dtype == "u8"`, `fmt == "fp8_gpu"` | global scale |
| 6 | NVFP4 | `modelopt_fp4` | `dtype == "u8"`, `fmt == "nvfp4_gpu"` | 16 |
| 7 | Int8 per-channel | `bitsandbytes` int8, `compressed-tensors` int8 | `dtype == "u8"`, `fmt == "int8_gpu"` | per-row |
| 8 | NF4 (Normal Float 4) | `bitsandbytes` nf4 | `dtype == "u8"`, `fmt == "nf4_gpu"` | 64 |

**Loader-only (no shader changes needed, CPU conversion at load time):**

| Format | vLLM name | Status |
|--------|-----------|--------|
| MXFP8 | `mxfp8`, `modelopt_mxfp8` | ✓ Implemented — u8 exponent scales → f16 at load time, runs as USE_QUANT=0 at runtime |
| MXFP4 | `mxfp4` | ✓ Implemented — u8 exponent scales → f16 at load time, runs as USE_QUANT=0 at runtime |
| compressed-tensors | `compressed-tensors` | ✓ Implemented — config_groups JSON parsed, routes to USE_QUANT 3/5/7 by sub-format |
| NF4 | `bitsandbytes` | ✓ Implemented — [N//2, K] packed codes + absmax scales → USE_QUANT=8 |
| torchao int4/int8 | `torchao` | Planned — checkpoint-specific format |

**Not feasible for WebGPU** (CUDA-specific memory layouts or missing hardware support):

| Format | Reason |
|--------|--------|
| Marlin (GPTQ/AWQ Marlin) | Tensor-core tile layout requires CUDA |
| ExllamaV2 | CUDA-specific |
| bfloat16 weights | WebGPU has no bf16 compute |
| DeepSeek V4 FP8 | Architecture-specific mixed-precision MoE |

### LlamaWebGPUModel

Full support for USE_QUANT 0–8. Applies to all projections: Q, K, V, o_proj, gate, up, down. LM head is always USE_QUANT=0 with SPLIT_K=0 (vocab_size > 65535 exceeds the per-axis dispatch limit).

The fused QKV path (`fused_qkv.wgsl`) and `fused_gate_act.wgsl` require f16 weights (USE_QUANT=0). Quantized weights fall back to three separate `matmul_quant` calls for QKV, and separate gate/up matmuls followed by `gelu_mul`.

Q/K norms (`q_norm.weight`, `k_norm.weight`) that are shape `(head_dim,)` rather than `(num_heads * head_dim,)` are tiled to the expected shape at load time (Qwen3 uses shared norms across heads).

### MixtralWebGPUModel

Extends `LlamaWebGPUModel` for Mistral (dense) and Mixtral (sparse MoE FFN) architectures.

**Sliding Window Attention (SWA):** When `sliding_window` is set in the model config, `_effective_ctx(ctx_len)` caps the context length passed to attention dispatches. For the decode path this means `attn_score MAX_SEQ_LEN`, `softmax SEQ_LEN`, `attn_output CTX_LEN`, and `flash_attn_decode CTX_LEN` are all bounded by the window size. For the batch prefill path, `flash_attn_prefill.wgsl` accepts a `WINDOW_SIZE` override that restricts each query token to attend only to its most recent `WINDOW_SIZE` tokens.

**MoE FFN (Mixtral):** When `num_local_experts > 0` and `num_experts_per_tok > 0`, the standard gate/up/down FFN is replaced by sparse expert dispatch using the Phase A/B pattern:

- Phase A: router matmul (`block_sparse_moe.gate.weight`) + `topk_sort` dispatched into the current command encoder, then flushed and CPU-synced so expert indices can be read back.
- Phase B: a new encoder is created; the top-K selected experts are dispatched sequentially. Each expert runs gate (`w1`) + up (`w3`) + SiLU via `fused_gate_act` (f16) or separate `matmul_quant` + `gelu_mul` (quantized), then down (`w2`) into a per-expert scratch buffer. `moe_accumulate` adds `weight[k] * expert_tmp` into the shared `expert_out` accumulator.

There is no shared expert in Mixtral (unlike Qwen3.6-35B-A3B). The accumulation buffer is zero-initialized via `write_buffer` before Phase B begins.

Weight naming: `model.layers.{i}.block_sparse_moe.{gate|experts.{j}.w1|experts.{j}.w3|experts.{j}.w2}`.

**Quant support:** USE_QUANT 0-8 via the same `_uq()` closure and `_uq_for_key()` logic as `LlamaWebGPUModel`. Applies to all projections including router, expert gate/up/down, Q/K/V, and o_proj.

### Gemma4WebGPUModel

USE_QUANT 0–8. All projections (Q, K, V, o_proj, gate, up, down) dispatch `matmul_quant.wgsl` with the same detection logic as `LlamaWebGPUModel._uq()`. The fused QKV path requires f16 weights (USE_QUANT=0); quantized weights fall back to separate Q/K/V matmuls.

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

GDN linear-attention layers: always USE_QUANT=0. Projections `in_proj_qkv`, `in_proj_a`, `in_proj_b`, `in_proj_z`, `out_proj`, `conv1d` are dispatched as f16 matmul_quant regardless of the loaded weight format. A_log and dt_bias SSM parameters are precision-upgraded to f32 at load time to reduce accumulation drift.

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

USE_QUANT 0–8. Scales key and `_quant_extra` overrides fixed in all dispatch sites (Q, K, V, o_proj, shared gate/up, shared down, router). Expert FFN dispatches already used `_scales_buf` and `_quant_extra` correctly.

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
| Quantized prefill | f16 models use `matmul_quant_mr4` batch GEMM for all T prompt tokens; quantized (GPTQ/AWQ/FP8/NF4) models fall back to sequential per-token decode-path processing during prefill |
| ctx_len > 65535 | Standard 3-pass attention is capped at 65535 by the WebGPU dispatch limit; flash_attn_decode is used as an automatic fallback beyond that point |
| hidden/inter must be divisible by 4 | vec4 shader requirement |
| head_dim must be even | f16 GEMV packing requirement |
| Block table cap | 4096 blocks per sequence (65536 tokens at block_size=16), set at init time |
| No native bfloat16 | WebGPU/WGSL has no bf16 type. Weights and activations use f16 (5-bit exponent) instead of bf16 (8-bit exponent, same range as f32). GDN SSM parameters (A_log, dt_bias) are kept as f32 to preserve decay range. An experimental `GDN_BF16=1` env var (or `--gdn_bf16` flag) stores Qwen3.5 GDN projection weights as packed bf16 u16 to further reduce rounding loss. |
| Logprobs | Supported when requested via `SamplingParams(num_logprobs=N)`. Top-N computed on CPU from full logit readback; adds latency proportional to vocab_size per step. Prompt logprobs supported for prefill passes. |
| Grammar/structured output | Raises `NotImplementedError`. Applying token masks from a grammar FSM requires the full logit distribution, which is not available on the WebGPU decode path. |
| Multi-modal inputs | Images, audio, and video are not implemented. Conditional-generation architectures (`Gemma3ForConditionalGeneration`, `Gemma4UnifiedForConditionalGeneration`, `Qwen3_5ForConditionalGeneration`, `Qwen3_5MoeForConditionalGeneration`) are registered for text-only use. Passing mm_inputs raises `NotImplementedError`. Use the `CausalLM` variant instead. |
| Speculative decoding | Not implemented. No draft model support. |
| Multi-sequence batching | One sequence per `forward()` call. True batching requires N separate scratch-buffer sets and per-sequence block-table dispatch; it is an architectural change and not currently planned. |

### WebGPU platform constraints

These limits apply to all models and stem from the WebGPU/WGSL specification:

- **No bfloat16**: WGSL float types are `f16`, `f32`, `f64`. `bf16` requires the `enable dual_source_blending` extension, which is not compute-related; bf16 is simply absent.
- **No GPU-to-GPU atomic floats**: Reduction kernels use shared-memory tree reduction instead of `atomicAdd<f32>`.
- **65535 dispatch limit per axis**: Standard `attn_score` is limited to ctx_len <= 65535. `flash_attn_decode` (fused, single-pass) has no such limit.
- **No persistent threads**: WebGPU forbids infinite loops or occupancy-based kernel tricks; all loops must have statically-bounded iteration counts or dynamic exit conditions that the driver can verify as non-infinite.
- **Metal GPU timeout (~4-8 s per command buffer)**: Batch prefill chunked encoder submission (4 layers per encoder) keeps each submit under the limit.

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
