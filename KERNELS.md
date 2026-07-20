# Kernel Reference

All compute shaders live under `vllm_webgpu/shaders/`. Each shader is parameterized by WGSL `override` constants set at pipeline compile time — different constant combinations produce different pipeline variants, cached in `PipelineCache`.

Shaders are designed to be composable: they express one operation with clean override interfaces so any model can mix and match them. Model files wire them together in dispatch sequences.

## Dispatch count per layer

For f16 weights with per-head Q+K norms (e.g., Qwen3-4B), the optimized decode path uses:

| # | Shader | Notes |
|---|--------|-------|
| 1 | `fused_qkv` | Q+K+V projections into one buffer |
| 2 | `fused_qk_norm_rope` | Q+K per-head norm + RoPE in one dispatch |
| 3 | `kv_cache_store_both` | K+V cache write |
| 4 | `flash_attn_decode` | Fused QK scores + online softmax + V sum |
| 5 | `matmul_quant` | Output projection (o_proj) |
| 6 | `add_rms_norm` | Post-attn residual add + FFN pre-norm (fused) |
| 7 | `fused_gate_act` | Gate+up projection with inline SiLU |
| 8 | `matmul_quant` | Down projection |
| 9 | `add_rms_norm` | Post-FFN residual add + next layer pre-norm (cross-layer fused) |

**9 dispatches/layer** (down from 16 before fusion work).

Fallback paths (quantized weights, no per-head norms): 12-14 dispatches/layer.

---

## generic/ shaders

### matmul_quant.wgsl

GEMV for single-token decode (M=1). Supports 7 quantization formats via `USE_QUANT`.

**Dispatch:** `(N, 1, 1)` split-K (one WG per output row, 256 threads each), or `(ceil(N/256), 1, 1)` row-per-thread when `SPLIT_K=0`.

| Override | Default | Description |
|----------|---------|-------------|
| `K` | 4096 | Input (hidden) dimension |
| `N` | 4096 | Output dimension |
| `USE_QUANT` | 0 | 0=f16, 3=GPTQ, 4=AWQ, 5=FP8, 6=NVFP4, 7=Int8, 8=NF4 |
| `SPLIT_K` | 1 | 1=split-K (coalesced, for large N), 0=row-per-thread (for N>65535) |
| `GROUP_K` | 128 | Quantization group size (GPTQ/AWQ: 128, NF4: 64, USE_QUANT=3/4/8) |
| `GLOBAL_SCALE` | 1.0 | Weight scale constant (FP8/NVFP4, USE_QUANT=5/6) |

**Bindings:** 0=x(f16), 1=weights(u32), 2=scales(f16), 3=output(f16)

GPU dequant formats:
- `USE_QUANT=3` GPTQ: weights `[N, K/8]` int32, scales `[G, N]` f16, zero_point=8
- `USE_QUANT=4` AWQ: weights `[K, N/8]` int32, AWQ nibble reorder `[0,4,1,5,2,6,3,7]`
- `USE_QUANT=5` FP8 E4M3: weights `[N, K]` raw bytes as u32, GLOBAL_SCALE
- `USE_QUANT=6` NVFP4: weights `[N, K/2]` packed FP4, scales `[N, K/16]` f16
- `USE_QUANT=7` Int8: weights `[N, K]` raw i8 bytes, per-row scale
- `USE_QUANT=8` NF4: weights `[N//2, K]` packed 4-bit codes, scales `[N, K//GROUP_K]` f16 absmax

---

### matmul_quant_mr4.wgsl

Tiled GEMM for prefill (M>1). Each WG handles an MR×4 output tile.

**Dispatch:** `(ceil(N/4), ceil(M/MR), 1)` where MR is the row-tile depth.

| Override | Default | Description |
|----------|---------|-------------|
| `K`, `N`, `M` | — | Matrix dimensions |
| `MR` | 4 | Row tile depth per WG |
| `BLOCK_K`, `USE_QUANT` | — | Same as matmul_quant |

---

### fused_qkv.wgsl

Q, K, V projections in one dispatch. Output `qkv[Q_DIM | KV_DIM | KV_DIM]` is used by `fused_qk_norm_rope` and `kv_cache_store_both` (via `V_IN_OFFSET`).

**Dispatch:** `(Q_DIM + 2*KV_DIM, 1, 1)`

| Override | Default | Description |
|----------|---------|-------------|
| `K` | 2048 | Hidden dim |
| `Q_DIM` | 2048 | Q output dim = num_q_heads × head_dim |
| `KV_DIM` | 512 | K (and V) output dim = num_kv_heads × head_dim |

**Bindings:** 0=x(f16), 1=q_w(u32), 2=k_w(u32), 3=v_w(u32), 4=qkv(f16 out)

Only for f16 weights. Falls back to three separate `matmul_quant` calls for quantized weights.

---

### fused_qk_norm_rope.wgsl

Per-head RMSNorm + RoPE for Q and K in a single dispatch. Reads from one input buffer at two offsets. Writes Q and K to separate output buffers.

**Dispatch:** `(NUM_Q_HEADS + NUM_KV_HEADS, num_tokens, 1)`

Workgroups with `wgid.x < NUM_Q_HEADS` process Q; others process K (at element offset `INPUT_OFFSET_K` in the input buffer).

| Override | Default | Description |
|----------|---------|-------------|
| `HEAD_DIM` | 128 | Head dimension |
| `NUM_Q_HEADS` | 32 | Number of query heads |
| `NUM_KV_HEADS` | 8 | Number of key/value heads |
| `ROPE_BASE` | 10000 | RoPE theta |
| `LN_ROPE_BASE` | 9.21 | `log(ROPE_BASE)`, pre-computed on host |
| `HAS_WEIGHT` | 1 | 0=identity norm, 1=per-head learned weight |
| `GEMMA_NORM` | 0 | 1=(1+w) Gemma-style norm, 0=standard w |
| `ROTARY_DIM` | HEAD_DIM | Dimensions to rotate (partial RoPE) |
| `INTERLEAVED` | 0 | 0=pairs (i, half+i), 1=interleaved pairs (2i, 2i+1) |
| `INPUT_OFFSET_K` | 0 | Element offset in input[] for the K section |
| `K_SEPARATE` | 0 | 1: read K from a separate k_input buffer (binding 6); 0: read from input[] at INPUT_OFFSET_K |

**Bindings:** 0=input(f16), 1=q_norm_w(f16), 2=k_norm_w(f16), 3=positions(u32), 4=q_rope_out(f16), 5=k_rope_out(f16), 6=k_input(f16, read only when K_SEPARATE=1; bind any buffer otherwise)

Both weight arrays are always bound; a scalar `select()` chooses the correct value per WG. WebGPU clamps OOB reads to 0, so out-of-range accesses on the unused array are harmless.

---

### fused_per_head_norm_rope.wgsl

Per-head RMSNorm + RoPE for a single projection (Q or K). Used when `fused_qk_norm_rope` is not applicable (quantized weights, no per-head norms, or separate Q/K buffers).

**Dispatch:** `(NUM_HEADS, num_tokens, 1)`

| Override | Default | Description |
|----------|---------|-------------|
| `HEAD_DIM`, `NUM_HEADS` | — | Head dimensions |
| `ROPE_BASE`, `LN_ROPE_BASE` | — | RoPE parameters |
| `HAS_WEIGHT`, `GEMMA_NORM`, `ROTARY_DIM`, `INTERLEAVED` | — | Same as fused_qk_norm_rope |
| `INPUT_OFFSET` | 0 | Element offset into input[] (for reading K from a QKV buffer) |

**Bindings:** 0=input(f16), 1=weight(f16), 2=positions(u32), 3=output(f16)

---

### fused_gate_act.wgsl

Gate+up projection GEMV with inline activation. Eliminates the separate `gelu_mul_fused` pass and the intermediate gate_up scratch buffer.

**Dispatch:** `(N, 1, 1)` — one WG per output row, 256 threads split-K.

| Override | Default | Description |
|----------|---------|-------------|
| `K` | 2560 | Hidden dim |
| `N` | 9728 | Intermediate dim (gate and up both N×K) |
| `GELU` | 0 | 0=SiLU `x·σ(x)` (Llama/Qwen), 1=tanh-GELU (Gemma) |

**Bindings:** 0=x(f16), 1=gate_w(u32), 2=up_w(u32), 3=ffn_act(f16 out)

---

### add_rms_norm.wgsl

Fused residual add + RMSNorm. One dispatch replaces the separate `add` + `rms_norm` sequence. Used twice per layer: post-attention and cross-layer (final add of layer N fused with pre-norm of layer N+1).

**Dispatch:** `(num_tokens, 1, 1)`

| Override | Default | Description |
|----------|---------|-------------|
| `HIDDEN_DIM` | 4096 | Hidden dimension |
| `VALS_PER_THREAD` | 16 | Registers per thread (HIDDEN_DIM/256, max 16); 0=fallback loop |
| `GEMMA_NORM` | 0 | 1=(1+w) Gemma-style norm |

**Bindings:** 0=a(f16), 1=b(f16), 2=weight(f16), 3=residual_out(f16), 4=normed_out(f16)

Computes: `residual_out = a + b`, `normed_out = rms_norm(residual_out, weight)`. Register-tiled: a+b stored in thread-local registers during pass 1, avoiding a second global memory read in pass 2.

---

### add_f32_rms_norm.wgsl

Same fusion for Gemma4's f32 residual stream. Reads f32 a, f16 b, writes f32 residual and f16 normed output.

**Dispatch:** `(num_tokens, 1, 1)`

| Override | Default | Description |
|----------|---------|-------------|
| `HIDDEN_DIM`, `VALS_PER_THREAD`, `GEMMA_NORM` | — | Same as add_rms_norm |
| `SCALE` | 1.0 | Residual contribution scale (layer_output_scale) |

**Bindings:** 0=a(f32), 1=b(f16), 2=weight(f16), 3=residual_out(f32), 4=normed_out(f16)

---

### kv_cache_store_both.wgsl

Stores K and V into the paged KV cache in one dispatch. Saves 1 dispatch vs two separate `kv_cache_store` calls.

**Dispatch:** `(num_tokens, num_kv_heads, 1)`

| Override | Default | Description |
|----------|---------|-------------|
| `BLOCK_SIZE` | 16 | KV cache block size in tokens |
| `NUM_KV_HEADS` | 8 | Number of KV heads |
| `HEAD_DIM` | 128 | Head dimension |
| `V_IN_OFFSET` | 0 | f16 element offset into v_in[] for fused-QKV mode |

**Bindings:** 0=k_in(vec2f16), 1=k_cache(vec2f16 rw), 2=v_in(vec2f16), 3=v_cache(vec2f16 rw), 4=slot_mapping(u32)

`V_IN_OFFSET` allows V to be read from a `[Q|K|V]` fused buffer at offset `q_dim + kv_dim` without an extra copy dispatch.

---

### rms_norm.wgsl

RMSNorm, f16 input and output. Register-tiled for `HIDDEN_DIM ≤ 4096` (avoids a second global read in pass 2). Falls back to two-pass global re-read for larger dims.

**Dispatch:** `(num_tokens, 1, 1)`

| Override | Default | Description |
|----------|---------|-------------|
| `HIDDEN_DIM` | 4096 | Hidden dimension |
| `WG_SIZE` | 256 | Workgroup size |
| `VALS_PER_THREAD` | 16 | Register slots per thread; 0=fallback |
| `GEMMA_NORM` | 0 | 1=(1+w) weight scaling |

**Bindings:** 0=input(f16), 1=weight(f16), 2=output(f16)

---

### rms_norm_f32in.wgsl

RMSNorm reading f32 input, writing f16 output. Used in Gemma4's f32 residual pipeline for the initial per-layer pre-norm.

Same overrides and dispatch as `rms_norm.wgsl`.

**Bindings:** 0=input(f32), 1=weight(f16), 2=output(f16)

---

### rms_norm_add_f32_rms_norm.wgsl

Double-norm fusion for Gemma4 sublayer pairs. Replaces the two-dispatch sequence used at each sublayer boundary with a single pass that keeps the intermediate in thread registers:

1. `rms_norm(delta_in, post_weight)` — f16 in, intermediate stays in registers (no global write)
2. `residual = residual_in + SCALE * intermediate` — f32 residual update, written to residual_out
3. `rms_norm_f32in(residual, pre_weight)` — f32 in, f16 out

Two shared-memory reductions are performed sequentially (sq_sum1 for the first norm, sq_sum2 for the second). The register-tiled path (VALS_PER_THREAD > 0) avoids a second global read of delta_in in phase 2; a fallback re-read path handles HIDDEN_DIM > WG_SIZE × max register slots.

**Dispatch:** `(num_tokens, 1, 1)` — one workgroup per token.

| Override | Default | Description |
|----------|---------|-------------|
| `HIDDEN_DIM` | 4096 | Hidden dimension |
| `WG_SIZE` | 256 | Workgroup size |
| `VALS_PER_THREAD` | 16 | Register slots per thread (HIDDEN_DIM / WG_SIZE); 0=fallback re-read path |
| `GEMMA_NORM` | 1 | 1=(1+w) Gemma-style norm; 0=standard w |
| `SCALE` | 1.0 | Residual contribution scale (layer_output_scale) |

**Bindings:** 0=delta_in(f16), 1=post_weight(f16), 2=residual_in(f32), 3=pre_weight(f16), 4=residual_out(f32), 5=normed_out(f16)

---

### attn_score.wgsl

QK dot-products against the paged K cache. Produces the full attention score matrix `[num_q_heads, ctx_len]`.

**Dispatch:** `(num_q_heads, ctx_len, 1)` — one WG per (query head, context position).

128-thread WG: each thread handles one HEAD_DIM element. 7-step tree reduction. GQA: `kv_head = q_head / (NUM_Q_HEADS / NUM_KV_HEADS)`.

| Override | Default | Description |
|----------|---------|-------------|
| `BLOCK_SIZE` | 16 | KV cache block size |
| `NUM_Q_HEADS`, `NUM_KV_HEADS`, `HEAD_DIM` | — | Attention dimensions |
| `MAX_SEQ_LEN` | 4096 | Output stride for scores_out |
| `Q_TOKEN_OFFSET` | 0 | Element offset into Q[] for batch prefill; for token t use `t * NUM_Q_HEADS * HEAD_DIM` |

**Bindings:** 0=Q(f16), 1=K_cache(f16), 2=block_table(u32), 3=scores_out(f16)

---

### softmax.wgsl

Online 2-pass softmax (Milakov-Divanov). Pass 1 computes per-thread `(local_max, local_sum)` in one scan; pass 2 normalizes. 256-thread WG, handles arbitrary SEQ_LEN via strided loop.

**Dispatch:** `(num_rows, 1, 1)` — one WG per query head.

| Override | Default | Description |
|----------|---------|-------------|
| `SEQ_LEN` | 128 | Context length (number of attention scores per head) |

**Bindings:** 0=input(f16), 1=output(f16)

---

### attn_output.wgsl

Weighted sum of paged V cache using softmaxed scores. Produces `[num_q_heads, head_dim]` output.

**Dispatch:** `(num_q_heads, 1, 1)` — one WG per query head.

128-thread WG: each thread accumulates one HEAD_DIM element across all ctx positions.

| Override | Default | Description |
|----------|---------|-------------|
| `BLOCK_SIZE` | 16 | KV cache block size |
| `NUM_Q_HEADS`, `NUM_KV_HEADS`, `HEAD_DIM`, `CTX_LEN` | — | Attention dimensions |
| `ATTN_TOKEN_OFFSET` | 0 | Output element offset for batch prefill; token t writes at `t * NUM_Q_HEADS * HEAD_DIM` |

**Bindings:** 0=scores(f16), 1=V_cache(f16), 2=block_table(u32), 3=out(f16)

---

### argmax_f16.wgsl

GPU argmax over the f16 logit vector. 256-thread WG, stride loop for arbitrary vocab size, 8-step tree reduction. Returns one u32 token ID.

**Dispatch:** `(1, 1, 1)`

| Override | Default | Description |
|----------|---------|-------------|
| `N` | 151936 | Vocabulary size |

**Bindings:** 0=logits(f16), 1=result(u32)

---

### embedding_lookup.wgsl / embedding_lookup_f32.wgsl

Token embedding lookup. The f32 variant applies `sqrt(HIDDEN_DIM)` scaling (Gemma requirement) and outputs f32 for the Gemma4 f32 residual pipeline.

**Dispatch:** `(num_tokens, 1, 1)`

| Override | Default | Description |
|----------|---------|-------------|
| `HIDDEN_DIM` | 4096 | Embedding dimension |

**Bindings:** 0=table(vec4f16), 1=token_ids(u32), 2=output(vec4f16 or f32)

---

### add.wgsl / add_f32.wgsl

Residual add. `add.wgsl` is f16→f16 (vec4); `add_f32.wgsl` is `f32 + SCALE*f16 → f32` for Gemma4. Used in fallback paths where `add_rms_norm` is not applicable (last layer, quantized FFN).

**Dispatch:** `(ceil(N/4/256), 1, 1)`

| Override | Default | Description |
|----------|---------|-------------|
| `N` | — | Total element count |
| `SCALE` | 1.0 | Multiplier for b (add_f32 only) |

---

### gelu_mul.wgsl

SiLU/GELU activation applied to the gate projection, then multiplied by the up projection. Used in the quantized FFN fallback path (f16 path uses `fused_gate_act`). Reads from separate gate and up buffers. Override `GELU=0` for SiLU, `GELU=1` for tanh-GELU.

---

### kv_cache_store.wgsl

Single-buffer KV cache write (K or V). Used in the quantized fallback path; prefer `kv_cache_store_both` for f16.

---

### fused_qk_norm_rope.wgsl

Per-head RMSNorm + RoPE for Q and K in one dispatch. Replaces two `fused_per_head_norm_rope` calls (one for Q, one for K). Supports two input layouts: Q and K from the same buffer at different offsets (default), or K from a separate buffer via K_SEPARATE=1 (used by models with distinct q_buf/k_buf such as Gemma4 and Qwen3.5).

**Dispatch:** `(NUM_Q_HEADS + NUM_KV_HEADS, num_tokens, 1)` — routes on `wgid.x`.

| Override | Default | Description |
|----------|---------|-------------|
| `HEAD_DIM`, `NUM_Q_HEADS`, `NUM_KV_HEADS` | — | Attention dimensions |
| `ROPE_BASE`, `LN_ROPE_BASE` | — | RoPE parameters |
| `HAS_WEIGHT`, `GEMMA_NORM`, `ROTARY_DIM`, `INTERLEAVED` | — | Same as fused_per_head_norm_rope |
| `INPUT_OFFSET_K` | 0 | Element offset for K section in input[] (= q_dim from fused_qkv); ignored when K_SEPARATE=1 |
| `K_SEPARATE` | 0 | 1: read K from k_input (binding 6); 0: read from input[] at INPUT_OFFSET_K |

**Bindings:** 0=input(f16), 1=q_norm_w(f16), 2=k_norm_w(f16), 3=positions(u32), 4=q_rope_out(f16), 5=k_rope_out(f16), 6=k_input(f16, read only when K_SEPARATE=1)

---

### fused_gate_act.wgsl

Gate+up projection GEMV with inline activation — fuses `fused_gate_up` + `gelu_mul_fused` into one dispatch. Writes activated output directly to `ffn_act`, eliminating the intermediate gate_up buffer.

**Dispatch:** `(N, 1, 1)` — one WG per output row, 256 threads split-K.

| Override | Default | Description |
|----------|---------|-------------|
| `K` | 2560 | Hidden dim |
| `N` | 9728 | Intermediate dim |
| `GELU` | 0 | 0=SiLU `x·σ(x)` (Llama/Qwen), 1=tanh-GELU (Gemma) |

**Bindings:** 0=x(f16), 1=gate_w(u32), 2=up_w(u32), 3=ffn_act(f16 out)

---

### sigmoid_gate.wgsl

Element-wise `sigmoid(gate) * value`. Used for Qwen3.5 `attn_output_gate`: HuggingFace applies `torch.sigmoid(gate)` before multiplying the attention output, not SiLU.

**Dispatch:** `(ceil(N/4/256), 1, 1)` — 256 threads, each handles 4 elements (vec4).

| Override | Default | Description |
|----------|---------|-------------|
| `N` | 2048 | Total element count (must be divisible by 4) |

**Bindings:** 0=gate(vec4 f16), 1=value(vec4 f16), 2=output(vec4 f16)

---

### flash_attn_prefill.wgsl

Prefill causal self-attention (dense Q/K/V, online softmax, causal masking). Replaces T×3 dispatches (attn_score + softmax + attn_output per token) with one dispatch that covers all T prompt tokens. Dense inputs: Q, K, V each shaped [NUM_T, heads, HEAD_DIM] — no paged cache needed.

**Dispatch:** `(NUM_Q_HEADS, NUM_T, 1)` — one workgroup per (query head, query token).

| Override | Default | Description |
|----------|---------|-------------|
| `NUM_Q_HEADS` | 32 | Number of query heads |
| `NUM_KV_HEADS` | 8 | Number of key/value heads (GQA) |
| `HEAD_DIM` | 128 | Per-head dimension |
| `NUM_T` | 64 | Number of prompt tokens |

**Bindings:** 0=Q(f16), 1=K(f16), 2=V(f16), 3=out(f16)

GQA: `kv_head = q_head / (NUM_Q_HEADS / NUM_KV_HEADS)`. Each workgroup loads its query into shared memory, then streams over t_k in [0, t_q] applying online Milakov-Divanov softmax, accumulating V directly into registers. Output written once at the end.

---

### flash_attn_decode.wgsl

Fused QK dot-products + online Milakov-Divanov softmax + V-weighted sum for decode (M=1). Always used for the single-token decode path. The fused shader loops over all KV positions inside the workgroup, so it has no per-axis dispatch limit (the former three-pass approach — attn_score + softmax + attn_output — dispatched one workgroup per (query head, context position) and hit the WebGPU 65535 per-axis limit at long contexts). Using flash_attn_decode unconditionally also saves 2 dispatches per layer per decode step.

**Dispatch:** `(NUM_Q_HEADS, 1, 1)` — one WG per query head.

| Override | Default | Description |
|----------|---------|-------------|
| `BLOCK_SIZE`, `NUM_Q_HEADS`, `NUM_KV_HEADS`, `HEAD_DIM` | — | Attention dimensions |
| `CTX_LEN` | 512 | Context length |

**Bindings:** 0=Q(f16), 1=K_cache(f16), 2=V_cache(f16), 3=block_table(u32), 4=out(f16)

---

### rope.wgsl

RoPE without per-head norm. Used for Llama models that have no `q_norm`/`k_norm` weights.

**Dispatch:** `(num_tokens, num_heads, 1)`

---

### gumbel_sample.wgsl

Temperature sampling via the Gumbel-max trick. Adds Gumbel noise to logits, returns the argmax. Greedy decoding uses `argmax_f16` instead.

---

### topk_sort.wgsl

GPU top-K selection for MoE routing. Parallel load of N expert logits, then sequential insertion sort for K winners in thread 0, then softmax over the K selected logits.

**Dispatch:** `(1, 1, 1)`

| Override | Default | Description |
|----------|---------|-------------|
| `N_EXPERTS` | 128 | Total expert count (≤256) |
| `K` | 8 | Number of experts to select |

---

### topk256.wgsl

Iterative top-K over the full vocabulary with a mask buffer. For nucleus/top-k sampling.

---

## gemma/ shaders

| Shader | Purpose |
|--------|---------|
| `gelu_mul.wgsl` | Tanh-GELU activation (Gemma3/4 FFN, separate gate/up buffers) |
| `logit_softcap.wgsl` | `tanh(x/CAP) * CAP`, CAP=30.0 for Gemma4 |
| `per_head_rms_norm_no_weight.wgsl` | Per-head V normalization before KV cache (Gemma4 only) |

---

## Qwen3.5 GDN kernels (generic/)

These three kernels implement Gated Delta Networks (GDN) linear attention, replacing the standard QKV attention in alternating layers of Qwen3.5/3.6.

### causal_conv_step.wgsl

Single-step causal depthwise conv1d. Updates the conv ring buffer in-place, writes the convolved output.

**Dispatch:** `(ceil(CONV_DIM/WG_SIZE), 1, 1)`

| Override | Default | Description |
|----------|---------|-------------|
| `CONV_DIM` | — | Total conv dimension (QKV combined) |
| `KERNEL` | 4 | Conv kernel size |
| `WG_SIZE` | 256 | Threads per WG |

**Bindings:** 0=x(f16), 1=weight(f16), 2=conv_state(f16 rw), 3=output(f16)

### gdn_state_update.wgsl

Delta-rule SSM state update. Updates the SSM matrix in-place, writes the GDN output `h = (A^dt * state + outer(b, v)) @ I`.

**Dispatch:** `(num_v_heads, 1, 1)`

| Override | Default | Description |
|----------|---------|-------------|
| `K_DIM`, `V_DIM` | — | Key and value dims per head |
| `NUM_K_HEADS`, `NUM_V_HEADS` | — | Number of heads |
| `Q_BASE`, `K_BASE`, `V_BASE` | — | Offsets into the flat QKV conv output buffer |

**Bindings:** 0=qkv_conv(f16), 1=a_buf(f16), 2=b_buf(f16), 3=A_log, 4=dt_bias, 5=ssm_state(f32 rw), 6=gdn_out(f16)

### linear_attn_norm_gate.wgsl

Per-head RMSNorm on the GDN output followed by sigmoid gating: `output = norm(gdn_out) * sigmoid(z)`.

**Dispatch:** `(num_v_heads, 1, 1)`

**Bindings:** 0=gdn_in(f16), 1=norm_weight(f16), 2=z_in(f16), 3=output(f16)

---

## NemotronH Mamba-2 kernels (generic/)

These three kernels implement Mamba-2 selective state space model (SSM) steps for NemotronH's SSM layers. They replace the standard attention + FFN computation in those layers.

### mamba2_ssm_step.wgsl

Single-step Mamba-2 SSM state update. Reads input and the current SSM state, applies the selective scan, writes the updated state and output.

**Dispatch:** `(num_heads, 1, 1)`

**Bindings:** 0=x(f16), 1=ssm_state(f32 rw), 2=A(f32), 3=B(f16), 4=C(f16), 5=dt(f16), 6=dt_bias(f32), 7=D(f32), 8=output(f16)

---

### mamba2_causal_conv.wgsl

Single-step causal depthwise conv1d for NemotronH. Updates the conv ring buffer in-place, writes convolved output. Shared interface with Qwen3.5's `causal_conv_step.wgsl` but parameterized separately.

**Dispatch:** `(ceil(CONV_DIM/WG_SIZE), 1, 1)`

**Bindings:** 0=x(f16), 1=weight(f16), 2=conv_state(f16 rw), 3=output(f16)

---

### mamba2_norm_gate.wgsl

Per-head RMSNorm followed by gated activation on the Mamba-2 output: `output = norm(x) * silu(z)`.

**Dispatch:** `(num_heads, 1, 1)`

**Bindings:** 0=x(f16), 1=norm_weight(f16), 2=z(f16), 3=output(f16)
