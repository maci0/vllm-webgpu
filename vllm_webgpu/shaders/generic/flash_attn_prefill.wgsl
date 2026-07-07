enable f16;

// flash_attn_prefill.wgsl — fused causal self-attention for prefill (M=T tokens).
//
// Replaces T×3 dispatches (attn_score × T + softmax × T + attn_output × T) with one
// dispatch that covers all T prompt tokens at once. Dense Q/K/V inputs, not paged cache.
//
// Algorithm: online Milakov-Divanov softmax (same as flash_attn_decode.wgsl),
// applied over the causal prefix [0, t_q] for each query token t_q.
//
// Dispatch (NUM_Q_HEADS, NUM_T, 1) — one workgroup per (query head, query token).
// 128 threads: each thread covers one HEAD_DIM element (for HEAD_DIM≤512, up to 4 dims/thread).
// GQA: kv_head = q_head / (NUM_Q_HEADS / NUM_KV_HEADS).
// Causal mask: token t_q attends to t_k in [0, t_q] inclusive.
//
// Buffer layout (dense, token-major):
//   Q[(t_q * NUM_Q_HEADS + q_head) * HEAD_DIM + d]
//   K[(t_k * NUM_KV_HEADS + kv_head) * HEAD_DIM + d]
//   V[(t_k * NUM_KV_HEADS + kv_head) * HEAD_DIM + d]
//   out[(t_q * NUM_Q_HEADS + q_head) * HEAD_DIM + d]
//
// Bindings:
//   0: Q   [NUM_T, NUM_Q_HEADS, HEAD_DIM] f16
//   1: K   [NUM_T, NUM_KV_HEADS, HEAD_DIM] f16
//   2: V   [NUM_T, NUM_KV_HEADS, HEAD_DIM] f16
//   3: out [NUM_T, NUM_Q_HEADS, HEAD_DIM] f16

override NUM_Q_HEADS:  u32 = 32u;
override NUM_KV_HEADS: u32 = 8u;
override HEAD_DIM:     u32 = 128u;
override NUM_T:        u32 = 64u;
override WINDOW_SIZE:  u32 = 0u;  // 0 = full causal; >0 = sliding-window prefix length

@group(0) @binding(0) var<storage, read>       Q   : array<f16>;
@group(0) @binding(1) var<storage, read>       K   : array<f16>;
@group(0) @binding(2) var<storage, read>       V   : array<f16>;
@group(0) @binding(3) var<storage, read_write> out : array<f16>;

// sh_q: query vector for this workgroup cached in shared memory
// sh_dot: partial dot-product for the parallel tree reduction
var<workgroup> sh_q:   array<f32, HEAD_DIM>;
var<workgroup> sh_dot: array<f32, 128>;

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(local_invocation_id) lid:  vec3<u32>,
    @builtin(workgroup_id)        wgid: vec3<u32>,
) {
    let q_head  = wgid.x;
    let t_q     = wgid.y;
    let kv_head = q_head / (NUM_Q_HEADS / NUM_KV_HEADS);
    let tid     = lid.x;
    let WS      = 128u;  // workgroup size constant (matches @workgroup_size above)
    let scale   = 1.0f / sqrt(f32(HEAD_DIM));

    // ── Phase 1: Load Q[t_q, q_head, :] into shared memory ─────────────────
    // All 128 threads collaborate. For HEAD_DIM=128: each thread loads 1 dim.
    // For HEAD_DIM>128: each thread loads HEAD_DIM/128 dims in a strided loop.
    var d = tid;
    loop {
        if (d >= HEAD_DIM) { break; }
        sh_q[d] = f32(Q[(t_q * NUM_Q_HEADS + q_head) * HEAD_DIM + d]);
        d += WS;
    }
    workgroupBarrier();

    // ── Phase 2: Causal attention with online softmax ────────────────────────
    // Running state: m = max score seen, d = normalizer sum.
    // Each thread accumulates its own slice of the V-weighted output.
    // For HEAD_DIM=128, WS=128: one dim per thread (acc0 only).
    // For HEAD_DIM>128: up to HEAD_DIM/WG_SIZE dims per thread (acc0..acc3).
    var running_m: f32 = -1e30f;
    var running_d: f32 = 0.0f;
    var acc0: f32 = 0.0f;
    var acc1: f32 = 0.0f;
    var acc2: f32 = 0.0f;
    var acc3: f32 = 0.0f;

    // Causal: token t_q attends to t_k in [window_start, t_q] inclusive.
    // When WINDOW_SIZE==0 (default), window_start=0 — full causal attention, unchanged.
    // When WINDOW_SIZE>0, only the last WINDOW_SIZE tokens are attended to (SWA).
    let window_start = select(0u, t_q + 1u - WINDOW_SIZE, WINDOW_SIZE > 0u && t_q + 1u > WINDOW_SIZE);
    var t_k = window_start;
    loop {
        if (t_k > t_q) { break; }

        let kv_base = (t_k * NUM_KV_HEADS + kv_head) * HEAD_DIM;

        // ── QK dot product via parallel reduction ────────────────────────────
        var dot: f32 = 0.0f;
        var dk = tid;
        loop {
            if (dk >= HEAD_DIM) { break; }
            dot += sh_q[dk] * f32(K[kv_base + dk]);
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

        // All threads read sh_dot[0] — the last workgroupBarrier in the
        // stride=1 step makes it visible to the whole workgroup.
        let score = sh_dot[0] * scale;

        // ── Online softmax update (all threads in sync, no divergence) ────────
        let m_new   = max(running_m, score);
        let rescale = exp(running_m - m_new);  // re-scale old accumulator
        let s_new   = exp(score - m_new);       // weight for this token

        acc0 *= rescale; acc1 *= rescale; acc2 *= rescale; acc3 *= rescale;
        running_d = running_d * rescale + s_new;
        running_m = m_new;

        // ── Accumulate V weighted by s_new ────────────────────────────────────
        if (tid < HEAD_DIM) {
            acc0 += s_new * f32(V[kv_base + tid]);
        }
        if (tid + WS < HEAD_DIM) {
            acc1 += s_new * f32(V[kv_base + tid + WS]);
        }
        if (tid + WS * 2u < HEAD_DIM) {
            acc2 += s_new * f32(V[kv_base + tid + WS * 2u]);
        }
        if (tid + WS * 3u < HEAD_DIM) {
            acc3 += s_new * f32(V[kv_base + tid + WS * 3u]);
        }

        t_k += 1u;
    }

    // ── Phase 3: Normalize and write output ───────────────────────────────────
    let inv_d    = 1.0f / max(running_d, 1e-10f);
    let out_base = (t_q * NUM_Q_HEADS + q_head) * HEAD_DIM;

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
