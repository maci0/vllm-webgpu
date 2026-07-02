enable f16;

override HEAD_DIM: u32      = 128u;
override NUM_HEADS: u32     = 32u;
override ROPE_BASE: f32     = 10000.0;
override LN_ROPE_BASE: f32  = 9.210340372;  // = log(ROPE_BASE); host sets this
override HAS_WEIGHT: u32    = 1u;   // 0 for weightless variant (Gemma V heads)
override WG_SIZE: u32       = 64u;

var<workgroup> shared_sq:    array<f32, 64>;
var<workgroup> shared_input: array<f32, 128>;  // HEAD_DIM elements as f32

@group(0) @binding(0) var<storage, read>       input     : array<f16>;
@group(0) @binding(1) var<storage, read>       weight    : array<f16>;  // [num_heads, head_dim] or unused
@group(0) @binding(2) var<storage, read>       positions : array<u32>;
@group(0) @binding(3) var<storage, read_write> output    : array<f16>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(local_invocation_id) lid  : vec3<u32>,
    @builtin(workgroup_id)        wgid : vec3<u32>,
) {
    let seq_idx  = wgid.y;
    let head_idx = wgid.x;
    let tid      = lid.x;
    let half     = HEAD_DIM / 2u;
    let base     = (seq_idx * NUM_HEADS + head_idx) * HEAD_DIM;
    let eps      = 1e-6f;

    // --- Phase 1: per-head RMSNorm, caching f32 values for reuse in phase 2 ---
    var sq_sum: f32 = 0.0;
    var col = tid;
    loop {
        if (col >= HEAD_DIM) { break; }
        let v = f32(input[base + col]);
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
    // Loop so each thread covers HEAD_DIM/2 / WG_SIZE pairs (handles HEAD_DIM > 2*WG_SIZE).
    let pos = f32(positions[seq_idx]);
    var i = tid;
    loop {
        if (i >= half) { break; }
        let theta_i = exp(-f32(i * 2u) / f32(HEAD_DIM) * LN_ROPE_BASE);
        let angle   = pos * theta_i;
        let cos_v   = cos(angle);
        let sin_v   = sin(angle);

        var n1 = shared_input[i]        * rms_inv;
        var n2 = shared_input[half + i] * rms_inv;
        if (HAS_WEIGHT != 0u) {
            n1 *= f32(weight[w_base + i]);
            n2 *= f32(weight[w_base + half + i]);
        }

        output[base + i]        = f16(n1 * cos_v - n2 * sin_v);
        output[base + half + i] = f16(n2 * cos_v + n1 * sin_v);
        i += WG_SIZE;
    }
}
