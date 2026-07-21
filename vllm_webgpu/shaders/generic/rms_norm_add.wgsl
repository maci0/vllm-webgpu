enable f16;

// rms_norm_add.wgsl — fused RMSNorm + residual-add (post-norm pattern).
//
// Computes:
//   normed[i] = rms_norm(branch_out, weight)[i]   (kept in registers, not written globally)
//   out[i]    = residual[i] + normed[i]
//
// Designed for post-norm architectures (OLMo-2) where the branch output is
// normalized before being added back to the residual stream:
//   residual + rms_norm(attn_out) → new_residual
//   residual + rms_norm(ffn_out)  → new_residual
//
// Replaces the 2-dispatch sequence: rms_norm → add.
// Register-tile path stores branch_out in thread-local registers, skipping the
// second global read in the fallback path.
// VALS_PER_THREAD=0: two-pass global re-read fallback for HIDDEN_DIM > 4096.
//
// Dispatch (num_tokens, 1, 1).

override HIDDEN_DIM:      u32 = 4096u;
override WG_SIZE:         u32 = 256u;
override VALS_PER_THREAD: u32 = 16u;   // HIDDEN_DIM / WG_SIZE; max 16; 0 = fallback

@group(0) @binding(0) var<storage, read>       residual   : array<f16>;
@group(0) @binding(1) var<storage, read>       branch_out : array<f16>;
@group(0) @binding(2) var<storage, read>       weight     : array<f16>;
@group(0) @binding(3) var<storage, read_write> out        : array<f16>;

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) lid:  vec3<u32>,
    @builtin(workgroup_id)        wgid: vec3<u32>,
) {
    let row  = wgid.x;
    let tid  = lid.x;
    let base = row * HIDDEN_DIM;
    let eps  = 1e-6f;

    if (VALS_PER_THREAD > 0u) {
        // Register-tile path: store branch_out in registers to avoid a second global read.
        var local_v: array<f32, 16>;
        var sq_sum: f32 = 0.0;
        var col = tid;
        for (var i = 0u; i < VALS_PER_THREAD; i++) {
            if (col < HIDDEN_DIM) {
                let v = f32(branch_out[base + col]);
                local_v[i] = v;
                sq_sum += v * v;
                col += WG_SIZE;
            }
        }
        shared_sum[tid] = sq_sum;
        workgroupBarrier();

        var stride = WG_SIZE / 2u;
        loop {
            if (stride == 0u) { break; }
            if (tid < stride) { shared_sum[tid] += shared_sum[tid + stride]; }
            workgroupBarrier();
            stride /= 2u;
        }

        let rms_inv = inverseSqrt(shared_sum[0] / f32(HIDDEN_DIM) + eps);

        col = tid;
        for (var i = 0u; i < VALS_PER_THREAD; i++) {
            if (col < HIDDEN_DIM) {
                let normed = local_v[i] * rms_inv * f32(weight[col]);
                out[base + col] = f16(f32(residual[base + col]) + normed);
                col += WG_SIZE;
            }
        }
    } else {
        // Two-pass global re-read fallback for HIDDEN_DIM > WG_SIZE * 16.
        var sq_sum: f32 = 0.0;
        var col = tid;
        loop {
            if (col >= HIDDEN_DIM) { break; }
            let v = f32(branch_out[base + col]);
            sq_sum += v * v;
            col += WG_SIZE;
        }
        shared_sum[tid] = sq_sum;
        workgroupBarrier();

        var stride = WG_SIZE / 2u;
        loop {
            if (stride == 0u) { break; }
            if (tid < stride) { shared_sum[tid] += shared_sum[tid + stride]; }
            workgroupBarrier();
            stride /= 2u;
        }

        let rms_inv = inverseSqrt(shared_sum[0] / f32(HIDDEN_DIM) + eps);

        col = tid;
        loop {
            if (col >= HIDDEN_DIM) { break; }
            let normed = f32(branch_out[base + col]) * rms_inv * f32(weight[col]);
            out[base + col] = f16(f32(residual[base + col]) + normed);
            col += WG_SIZE;
        }
    }
}
