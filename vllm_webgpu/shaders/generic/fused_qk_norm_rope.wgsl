enable f16;

// fused_qk_norm_rope.wgsl — per-head RMSNorm + RoPE for Q and K in one dispatch.
//
// Fuses two fused_per_head_norm_rope calls (one for Q, one for K) into a single
// dispatch. Halves dispatch overhead for the norm+rope step.
//
// Dispatch (NUM_Q_HEADS + NUM_KV_HEADS, num_tokens, 1).
// Workgroups with wgid.x < NUM_Q_HEADS process Q; others process K.
//
// Reads from a single input buffer at different offsets:
//   Q section starts at element 0 (default INPUT_OFFSET_K = q_dim).
//   K section starts at element INPUT_OFFSET_K.
// Set INPUT_OFFSET_K = 0 when Q and K are separate buffers (bind the same buf twice).
//
// Overrides match fused_per_head_norm_rope for composability:
//   HEAD_DIM, NUM_Q_HEADS, NUM_KV_HEADS
//   ROPE_BASE, LN_ROPE_BASE (= log(ROPE_BASE), pre-computed on host)
//   HAS_WEIGHT    — 1: apply per-head norm weight, 0: identity weight
//   GEMMA_NORM    — 1: (1+w) Gemma-style, 0: standard w (Llama/Qwen3)
//   ROTARY_DIM    — dimensions to rotate (rest pass through); default HEAD_DIM
//   INTERLEAVED   — 0: pairs (i, half+i), 1: pairs (2i, 2i+1) for mrope
//   INPUT_OFFSET_K — element offset in input[] where K section begins (= q_dim)
//
// Bindings:
//   0: input     [q_dim + kv_dim] f16  (qkv_buf; Q at [0..q_dim), K at [INPUT_OFFSET_K..))
//   1: q_norm_w  [NUM_Q_HEADS * HEAD_DIM] f16
//   2: k_norm_w  [NUM_KV_HEADS * HEAD_DIM] f16
//   3: positions [num_tokens] u32
//   4: q_rope_out [NUM_Q_HEADS * HEAD_DIM] f16  (output)
//   5: k_rope_out [NUM_KV_HEADS * HEAD_DIM] f16 (output)
//   6: k_input — separate K buffer (only when K_SEPARATE=1; bind any buf otherwise)
//   7: inv_freq_buf [HEAD_DIM/2] f32 — precomputed RoPE freqs (only when USE_FREQ_BUF=1)

override HEAD_DIM:       u32 = 128u;
override NUM_Q_HEADS:    u32 = 32u;
override NUM_KV_HEADS:   u32 = 8u;
override ROPE_BASE:      f32 = 10000.0;
override LN_ROPE_BASE:   f32 = 9.210340372;   // log(ROPE_BASE); host sets this
override HAS_WEIGHT:     u32 = 1u;
override GEMMA_NORM:     u32 = 0u;
override ROTARY_DIM:     u32 = HEAD_DIM;
// FREQ_DIM: denominator for the RoPE frequency exponent -2i/FREQ_DIM.
// Defaults to ROTARY_DIM (correct for standard and default rope types).
// Set to HEAD_DIM for "proportional" rope (Gemma4 full-attention layers), where
// Gemma4RotaryEmbedding._compute_inv_freq uses head_size as the denominator
// regardless of partial_rotary_factor, so rotating fewer pairs still uses
// the full head_dim in the exponent.
override FREQ_DIM:       u32 = ROTARY_DIM;
override INTERLEAVED:    u32 = 0u;
override INPUT_OFFSET_K: u32 = 0u;            // f16 element offset for K in input[]
// K_SEPARATE=1: K data is in k_input (binding 6) at element 0 (set INPUT_OFFSET_K=0).
// K_SEPARATE=0: K data is in input (binding 0) at element INPUT_OFFSET_K (default).
// When K_SEPARATE=0, bind any buffer at slot 6 (it will not be read).
override K_SEPARATE: u32 = 0u;
// USE_FREQ_BUF=1: read precomputed inv_freq from binding 7 instead of computing inline.
// Enables YaRN and other scaled RoPE variants via CPU-side frequency precomputation.
override USE_FREQ_BUF: u32 = 0u;
// YARN_MSCALE: YaRN post-rope multiplier applied to both Q and K.
// Set to mscale (0.1 * ln(factor) + 1.0) when USE_FREQ_BUF=1; leave at 1.0 otherwise.
override YARN_MSCALE: f32  = 1.0;

@group(0) @binding(0) var<storage, read>       input        : array<f16>;
@group(0) @binding(1) var<storage, read>       q_norm_w     : array<f16>;
@group(0) @binding(2) var<storage, read>       k_norm_w     : array<f16>;
@group(0) @binding(3) var<storage, read>       positions    : array<u32>;
@group(0) @binding(4) var<storage, read_write> q_rope_out   : array<f16>;
@group(0) @binding(5) var<storage, read_write> k_rope_out   : array<f16>;
// Separate K input buffer. When K_SEPARATE=0, bind any buffer here (not read).
@group(0) @binding(6) var<storage, read>       k_input      : array<f16>;
@group(0) @binding(7) var<storage, read>       inv_freq_buf : array<f32>;

var<workgroup> shared_sq:    array<f32, 64>;
var<workgroup> shared_input: array<f32, HEAD_DIM>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(local_invocation_id) lid:  vec3<u32>,
    @builtin(workgroup_id)        wgid: vec3<u32>,
) {
    let seq_idx = wgid.y;
    let tid     = lid.x;
    let half    = HEAD_DIM / 2u;
    let eps     = 1e-6f;

    // Route: Q heads occupy wgid.x in [0, NUM_Q_HEADS), K in [NUM_Q_HEADS, total).
    let is_k     = wgid.x >= NUM_Q_HEADS;
    let head_idx = select(wgid.x, wgid.x - NUM_Q_HEADS, is_k);
    let n_heads  = select(NUM_Q_HEADS, NUM_KV_HEADS, is_k);

    // Input element base for this head's data.
    let in_offset = select(0u, INPUT_OFFSET_K, is_k);
    let in_base   = in_offset + (seq_idx * n_heads + head_idx) * HEAD_DIM;
    // Output element base (separate Q and K output buffers, each indexed from 0).
    let out_base  = (seq_idx * n_heads + head_idx) * HEAD_DIM;

    // Phase 1: load into shared mem, accumulate sq_sum for RMSNorm.
    // When K_SEPARATE=1, K heads read from k_input (binding 6) instead of input.
    // select() evaluates both branches; both buffers must be bound and in-range.
    var sq_sum: f32 = 0.0;
    var col = tid;
    loop {
        if (col >= HEAD_DIM) { break; }
        let val = select(
            f32(input[in_base + col]),
            f32(k_input[in_base + col]),
            is_k && K_SEPARATE != 0u
        );
        shared_input[col] = val;
        sq_sum += val * val;
        col += 64u;
    }
    shared_sq[tid] = sq_sum;
    workgroupBarrier();

    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) { shared_sq[tid] += shared_sq[tid + stride]; }
        workgroupBarrier();
        stride /= 2u;
    }

    let rms_inv = inverseSqrt(shared_sq[0] / f32(HEAD_DIM) + eps);
    let w_base  = head_idx * HEAD_DIM;
    let pos     = f32(positions[seq_idx]);
    let rot_half = ROTARY_DIM / 2u;

    // Phase 2: norm weight + RoPE, write to the correct output array.
    var i = tid;
    loop {
        if (i >= half) { break; }

        var n1: f32; var n2: f32;
        var out_idx1: u32; var out_idx2: u32;

        if (INTERLEAVED == 0u) {
            n1 = shared_input[i]        * rms_inv;
            n2 = shared_input[half + i] * rms_inv;
            if (HAS_WEIGHT != 0u) {
                // select() on scalars — both arrays are read, only one result used.
                // WebGPU bounds-clamps OOB reads to 0, so selecting the wrong value is safe.
                let w1 = select(f32(q_norm_w[w_base + i]),        f32(k_norm_w[w_base + i]),        is_k);
                let w2 = select(f32(q_norm_w[w_base + half + i]), f32(k_norm_w[w_base + half + i]), is_k);
                let weff1 = select(w1, 1.0 + w1, GEMMA_NORM != 0u);
                let weff2 = select(w2, 1.0 + w2, GEMMA_NORM != 0u);
                n1 *= weff1; n2 *= weff2;
            }
            out_idx1 = out_base + i;
            out_idx2 = out_base + half + i;
        } else {
            n1 = shared_input[i * 2u]      * rms_inv;
            n2 = shared_input[i * 2u + 1u] * rms_inv;
            if (HAS_WEIGHT != 0u) {
                let w1 = select(f32(q_norm_w[w_base + i * 2u]),      f32(k_norm_w[w_base + i * 2u]),      is_k);
                let w2 = select(f32(q_norm_w[w_base + i * 2u + 1u]), f32(k_norm_w[w_base + i * 2u + 1u]), is_k);
                let weff1 = select(w1, 1.0 + w1, GEMMA_NORM != 0u);
                let weff2 = select(w2, 1.0 + w2, GEMMA_NORM != 0u);
                n1 *= weff1; n2 *= weff2;
            }
            out_idx1 = out_base + i * 2u;
            out_idx2 = out_base + i * 2u + 1u;
        }

        if (i < rot_half) {
            var theta_i: f32;
            if (USE_FREQ_BUF == 1u) {
                theta_i = inv_freq_buf[i];
            } else {
                theta_i = exp(-f32(i * 2u) / f32(FREQ_DIM) * LN_ROPE_BASE);
            }
            let angle   = pos * theta_i;
            let cos_v   = YARN_MSCALE * cos(angle);
            let sin_v   = YARN_MSCALE * sin(angle);
            let r1 = f16(n1 * cos_v - n2 * sin_v);
            let r2 = f16(n2 * cos_v + n1 * sin_v);
            if (!is_k) { q_rope_out[out_idx1] = r1; q_rope_out[out_idx2] = r2; }
            else       { k_rope_out[out_idx1] = r1; k_rope_out[out_idx2] = r2; }
        } else {
            if (!is_k) { q_rope_out[out_idx1] = f16(n1); q_rope_out[out_idx2] = f16(n2); }
            else       { k_rope_out[out_idx1] = f16(n1); k_rope_out[out_idx2] = f16(n2); }
        }

        i += 64u;
    }
}
