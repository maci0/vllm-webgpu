enable f16;

// flash_attn_decode.wgsl — fused attention score + softmax + output for decode (M=1).
//
// Fuses three separate passes (attn_score, softmax, attn_output) into one dispatch.
// Eliminates scores_buf [num_q_heads × ctx_len] and sm_buf [same] — no intermediate
// global writes between QK dot-products and the final V-weighted sum.
//
// Algorithm: online Milakov-Divanov softmax (same as softmax.wgsl) applied one
// KV token at a time inside the workgroup loop. Each token's V contribution is
// accumulated immediately with the current running scale factor.
//
// Dispatch (NUM_Q_HEADS, 1, 1) — one workgroup per query head.
// WG_SIZE threads: each thread is responsible for one HEAD_DIM element of Q and V.
// For HEAD_DIM=128: WG_SIZE=128 (thread t → dimension t).
// For HEAD_DIM>128: threads loop over dimensions (t, t+WG_SIZE, ...).
//
// Parallelism: only NUM_Q_HEADS workgroups run, vs (NUM_Q_HEADS × ctx_len) in the
// three-pass approach. Less parallelism for short ctx, but eliminates two rounds
// of global memory traffic across all ctx_len positions.
//
// Overrides:
//   BLOCK_SIZE    — KV cache block size (tokens per block)
//   NUM_Q_HEADS   — number of query heads
//   NUM_KV_HEADS  — number of key/value heads (GQA)
//   HEAD_DIM      — per-head dimension
//   CTX_LEN       — context length (number of KV tokens)
//   WG_SIZE       — workgroup size (= HEAD_DIM for HEAD_DIM≤256)
//
// Bindings:
//   0: Q          [NUM_Q_HEADS, HEAD_DIM] f16
//   1: K_cache    [num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM] f16
//   2: V_cache    [num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM] f16
//   3: block_table [max_blocks] u32
//   4: out        [NUM_Q_HEADS, HEAD_DIM] f16

override BLOCK_SIZE:   u32 = 16u;
override NUM_Q_HEADS:  u32 = 32u;
override NUM_KV_HEADS: u32 = 8u;
override HEAD_DIM:     u32 = 128u;
override CTX_LEN:      u32 = 512u;
// WG_SIZE is always 128 (matches attn_score.wgsl). Handles HEAD_DIM > 128 by
// having each thread accumulate multiple dimensions (acc0/acc1/acc2/acc3).

@group(0) @binding(0) var<storage, read>       Q            : array<f16>;
@group(0) @binding(1) var<storage, read>       K_cache      : array<f16>;
@group(0) @binding(2) var<storage, read>       V_cache      : array<f16>;
@group(0) @binding(3) var<storage, read>       block_table  : array<u32>;
@group(0) @binding(4) var<storage, read_write> out          : array<f16>;

// sh_q: cached query, sh_dot: QK partial reduction
var<workgroup> sh_q:   array<f32, HEAD_DIM>;
var<workgroup> sh_dot: array<f32, 128>;

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(local_invocation_id) lid:  vec3<u32>,
    @builtin(workgroup_id)        wgid: vec3<u32>,
) {
    let q_head  = wgid.x;
    let kv_head = q_head / (NUM_Q_HEADS / NUM_KV_HEADS);
    let tid     = lid.x;
    let WS      = 128u;  // workgroup size constant (matches @workgroup_size above)
    let scale   = 1.0f / sqrt(f32(HEAD_DIM));

    // ── Phase 1: Load Q into shared memory ──────────────────────────────────
    var d = tid;
    loop {
        if (d >= HEAD_DIM) { break; }
        sh_q[d] = f32(Q[q_head * HEAD_DIM + d]);
        d += WS;
    }
    workgroupBarrier();

    // ── Phase 2: Stream over KV tokens with online softmax + V accumulation ─
    // Each thread maintains its own slice of the V-weighted output accumulator.
    // Thread t handles output dimension t, t+WG_SIZE, ... (for HEAD_DIM > WG_SIZE).
    // Running state: (m = max score seen, d_inv = 1/normalizer, acc[dims]).
    var running_m: f32 = -1e30f;
    var running_d: f32 = 0.0f;

    // Output accumulator: one f32 per dimension this thread handles.
    // For HEAD_DIM=128, WG_SIZE=128: each thread holds exactly one dimension.
    // For HEAD_DIM>128: thread cycles over HEAD_DIM/WG_SIZE dimensions.
    // WGSL does not allow dynamic arrays in registers, so unroll for max 4 dims.
    // HEAD_DIM ≤ WG_SIZE*4 (e.g. HEAD_DIM=512, WG_SIZE=128 → 4 dims/thread).
    var acc0: f32 = 0.0f;
    var acc1: f32 = 0.0f;
    var acc2: f32 = 0.0f;
    var acc3: f32 = 0.0f;

    var ctx = 0u;
    loop {
        if (ctx >= CTX_LEN) { break; }

        // Locate this KV token in the paged cache.
        let block_idx = block_table[ctx / BLOCK_SIZE];
        let block_off = ctx % BLOCK_SIZE;
        let kv_base   = ((block_idx * BLOCK_SIZE + block_off) * NUM_KV_HEADS + kv_head) * HEAD_DIM;

        // ── QK dot product via parallel reduction ──────────────────────────
        var dot: f32 = 0.0f;
        var dk = tid;
        loop {
            if (dk >= HEAD_DIM) { break; }
            dot += sh_q[dk] * f32(K_cache[kv_base + dk]);
            dk += WS;
        }
        sh_dot[tid] = dot;
        workgroupBarrier();

        // 7-step tree reduction for 128 threads.
        var stride = 64u;
        loop {
            if (stride == 0u) { break; }
            if (tid < stride) { sh_dot[tid] += sh_dot[tid + stride]; }
            workgroupBarrier();
            stride /= 2u;
        }

        // Thread 0 holds the full dot product; broadcast via sh_dot[0].
        let score = sh_dot[0] * scale;   // sh_dot[0] already visible to all threads
        // (workgroupBarrier at end of reduction left sh_dot[0] visible)

        // ── Online softmax update (all threads in sync) ─────────────────────
        let m_new   = max(running_m, score);
        let rescale = exp(running_m - m_new);  // re-scale old accumulator
        let s_new   = exp(score - m_new);       // weight for this token

        // Re-scale previous V accumulation.
        acc0 *= rescale; acc1 *= rescale; acc2 *= rescale; acc3 *= rescale;
        running_d = running_d * rescale + s_new;
        running_m = m_new;

        // ── Accumulate V weighted by s_new ─────────────────────────────────
        if (tid < HEAD_DIM) {
            acc0 += s_new * f32(V_cache[kv_base + tid]);
        }
        if (tid + WS < HEAD_DIM) {
            acc1 += s_new * f32(V_cache[kv_base + tid + WS]);
        }
        if (tid + WS * 2u < HEAD_DIM) {
            acc2 += s_new * f32(V_cache[kv_base + tid + WS * 2u]);
        }
        if (tid + WS * 3u < HEAD_DIM) {
            acc3 += s_new * f32(V_cache[kv_base + tid + WS * 3u]);
        }

        ctx += 1u;
    }

    // ── Phase 3: Normalize and write output ─────────────────────────────────
    let inv_d = 1.0f / max(running_d, 1e-10f);

    let out_base = q_head * HEAD_DIM;
    if (tid < HEAD_DIM) {
        out[out_base + tid] = f16(clamp(acc0 * inv_d, -65504.0f, 65504.0f));
    }
    if (tid + WS < HEAD_DIM) {
        out[out_base + tid + WS] = f16(clamp(acc1 * inv_d, -65504.0f, 65504.0f));
    }
    if (tid + WS * 2u < HEAD_DIM) {
        out[out_base + tid + WS * 2u] = f16(clamp(acc2 * inv_d, -65504.0f, 65504.0f));
    }
    if (tid + WS * 3u < HEAD_DIM) {
        out[out_base + tid + WS * 3u] = f16(clamp(acc3 * inv_d, -65504.0f, 65504.0f));
    }
}
