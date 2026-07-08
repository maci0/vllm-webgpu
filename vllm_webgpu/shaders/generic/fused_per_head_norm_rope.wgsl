enable f16;

override HEAD_DIM: u32      = 128u;
override NUM_HEADS: u32     = 32u;
override ROPE_BASE: f32     = 10000.0;
override LN_ROPE_BASE: f32  = 9.210340372;  // = log(ROPE_BASE); host sets this
override HAS_WEIGHT: u32    = 1u;   // 0 for weightless variant (Gemma V heads)
override WG_SIZE: u32       = 64u;
// GEMMA_NORM=1: Gemma-style (1+w) scale; GEMMA_NORM=0: standard w scale.
override GEMMA_NORM: u32    = 0u;
// ROTARY_DIM: number of dimensions to apply RoPE to (rest pass through unchanged).
// Set to HEAD_DIM for full RoPE (default), or HEAD_DIM * partial_rotary_factor for partial.
override ROTARY_DIM: u32    = HEAD_DIM;
// INTERLEAVED=1: pairs are (2i, 2i+1) — used by some models (Qwen3.5 mrope_interleaved).
// INTERLEAVED=0: pairs are (i, i+half) — standard convention (Llama, Gemma3, Qwen3).
override INTERLEAVED: u32   = 0u;
// INPUT_OFFSET: element offset into the input buffer. Set to Q_DIM when reading K
// from a fused QKV buffer, 0 for standalone Q or K buffers.
override INPUT_OFFSET: u32  = 0u;
// USE_FREQ_BUF=1: read precomputed inv_freq from binding 4 instead of computing inline.
// Enables YaRN and other scaled RoPE variants via CPU-side frequency precomputation.
override USE_FREQ_BUF: u32  = 0u;
// ATTN_SCALE: applied as (ATTN_SCALE * cos(angle), ATTN_SCALE * sin(angle)).
// Set to YaRN mscale (0.1 * ln(factor) + 1.0) when USE_FREQ_BUF=1; leave at 1.0 otherwise.
override ATTN_SCALE: f32    = 1.0;

var<workgroup> shared_sq:    array<f32, 64>;
// shared_input caches HEAD_DIM f32 values for reuse in phase 2.
// WGSL allows override constants as workgroup array sizes.
// HEAD_DIM ≤ 256 (1024 bytes) keeps us well within the 16384-byte limit.
var<workgroup> shared_input: array<f32, HEAD_DIM>;

@group(0) @binding(0) var<storage, read>       input        : array<f16>;
@group(0) @binding(1) var<storage, read>       weight       : array<f16>;  // [num_heads, head_dim] or unused
@group(0) @binding(2) var<storage, read>       positions    : array<u32>;
@group(0) @binding(3) var<storage, read_write> output       : array<f16>;
@group(0) @binding(4) var<storage, read>       inv_freq_buf : array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(local_invocation_id) lid  : vec3<u32>,
    @builtin(workgroup_id)        wgid : vec3<u32>,
) {
    let seq_idx  = wgid.y;
    let head_idx = wgid.x;
    let tid      = lid.x;
    let half     = HEAD_DIM / 2u;
    let in_base  = INPUT_OFFSET + (seq_idx * NUM_HEADS + head_idx) * HEAD_DIM;
    let out_base = (seq_idx * NUM_HEADS + head_idx) * HEAD_DIM;
    let eps      = 1e-6f;

    // --- Phase 1: per-head RMSNorm, caching f32 values for reuse in phase 2 ---
    var sq_sum: f32 = 0.0;
    var col = tid;
    loop {
        if (col >= HEAD_DIM) { break; }
        let v = f32(input[in_base + col]);
        shared_input[col] = v;
        sq_sum += v * v;
        col += WG_SIZE;
    }
    shared_sq[tid] = sq_sum;
    workgroupBarrier();  // covers both shared_sq and shared_input writes

    var stride = WG_SIZE / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) { shared_sq[tid] += shared_sq[tid + stride]; }
        workgroupBarrier();
        stride /= 2u;
    }

    let rms_inv = inverseSqrt(shared_sq[0] / f32(HEAD_DIM) + eps);
    let w_base  = head_idx * HEAD_DIM;

    // --- Phase 2: apply norm weight then RoPE, reading from shared cache ---
    // Supports two pairing conventions:
    //   INTERLEAVED=0 (default): pairs (i, half+i)  — Llama/Gemma/Qwen3 convention
    //   INTERLEAVED=1:           pairs (2i, 2i+1)   — Qwen3.5 mrope_interleaved
    let pos       = f32(positions[seq_idx]);
    let rot_half  = ROTARY_DIM / 2u;  // pair boundary for rotary dims
    var i = tid;
    loop {
        if (i >= half) { break; }

        var n1: f32;
        var n2: f32;
        var out_idx1: u32;
        var out_idx2: u32;

        if (INTERLEAVED == 0u) {
            // Standard: pairs are (i, half+i)
            n1 = shared_input[i]        * rms_inv;
            n2 = shared_input[half + i] * rms_inv;
            if (HAS_WEIGHT != 0u) {
                let w1 = select(f32(weight[w_base + i]),        1.0 + f32(weight[w_base + i]),        GEMMA_NORM != 0u);
                let w2 = select(f32(weight[w_base + half + i]), 1.0 + f32(weight[w_base + half + i]), GEMMA_NORM != 0u);
                n1 *= w1; n2 *= w2;
            }
            out_idx1 = out_base + i;
            out_idx2 = out_base + half + i;
        } else {
            // Interleaved: pairs are (2i, 2i+1)
            n1 = shared_input[i * 2u]       * rms_inv;
            n2 = shared_input[i * 2u + 1u]  * rms_inv;
            if (HAS_WEIGHT != 0u) {
                let w1 = select(f32(weight[w_base + i * 2u]),       1.0 + f32(weight[w_base + i * 2u]),       GEMMA_NORM != 0u);
                let w2 = select(f32(weight[w_base + i * 2u + 1u]),  1.0 + f32(weight[w_base + i * 2u + 1u]), GEMMA_NORM != 0u);
                n1 *= w1; n2 *= w2;
            }
            out_idx1 = out_base + i * 2u;
            out_idx2 = out_base + i * 2u + 1u;
        }

        if (i < rot_half) {
            // RoPE applied: theta uses ROTARY_DIM for correct frequency
            var theta_i: f32;
            if (USE_FREQ_BUF == 1u) {
                theta_i = inv_freq_buf[i];
            } else {
                theta_i = exp(-f32(i * 2u) / f32(ROTARY_DIM) * LN_ROPE_BASE);
            }
            let angle   = pos * theta_i;
            let cos_v   = ATTN_SCALE * cos(angle);
            let sin_v   = ATTN_SCALE * sin(angle);
            output[out_idx1] = f16(n1 * cos_v - n2 * sin_v);
            output[out_idx2] = f16(n2 * cos_v + n1 * sin_v);
        } else {
            output[out_idx1] = f16(n1);
            output[out_idx2] = f16(n2);
        }
        i += WG_SIZE;
    }
}
