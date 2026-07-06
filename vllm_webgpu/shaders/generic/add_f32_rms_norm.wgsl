enable f16;

// add_f32_rms_norm.wgsl — fused f32 residual-add + RMSNorm for Gemma4 pipeline.
//
// Computes:
//   residual_out[i] = a[i] + SCALE * f32(b[i])    (f32 += SCALE * f16)
//   normed_out[i]   = rms_norm_f32in(residual_out, weight)[i]
//
// Equivalent to Gemma4's add_f32 → rms_norm_f32in sequence, fused into one dispatch.
// GEMMA_NORM=1: (1+w) weight scale (Gemma-style); GEMMA_NORM=0: standard w.
//
// Dispatch (num_tokens, 1, 1).

override HIDDEN_DIM:      u32 = 4096u;
override WG_SIZE:         u32 = 256u;
override VALS_PER_THREAD: u32 = 16u;   // HIDDEN_DIM / WG_SIZE; 0 = fallback loop
override GEMMA_NORM:      u32 = 1u;    // Gemma uses (1+w) scaling
override SCALE:           f32 = 1.0;   // residual contribution scale (layer_output_scale)

@group(0) @binding(0) var<storage, read>       a            : array<f32>;
@group(0) @binding(1) var<storage, read>       b            : array<f16>;
@group(0) @binding(2) var<storage, read>       weight       : array<f16>;
@group(0) @binding(3) var<storage, read_write> residual_out : array<f32>;
@group(0) @binding(4) var<storage, read_write> normed_out   : array<f16>;

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
        var local_v: array<f32, 16>;
        var sq_sum: f32 = 0.0;
        var col = tid;
        for (var i = 0u; i < VALS_PER_THREAD; i++) {
            if (col < HIDDEN_DIM) {
                let v = a[base + col] + SCALE * f32(b[base + col]);
                local_v[i] = v;
                residual_out[base + col] = v;
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
                let w_eff = select(f32(weight[col]), 1.0 + f32(weight[col]), GEMMA_NORM != 0u);
                normed_out[base + col] = f16(clamp(local_v[i] * rms_inv * w_eff, -65504.0, 65504.0));
                col += WG_SIZE;
            }
        }
    } else {
        var sq_sum: f32 = 0.0;
        var col = tid;
        loop {
            if (col >= HIDDEN_DIM) { break; }
            let v = a[base + col] + SCALE * f32(b[base + col]);
            residual_out[base + col] = v;
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
            let v2 = residual_out[base + col];
            let w_eff = select(f32(weight[col]), 1.0 + f32(weight[col]), GEMMA_NORM != 0u);
            normed_out[base + col] = f16(clamp(v2 * rms_inv * w_eff, -65504.0, 65504.0));
            col += WG_SIZE;
        }
    }
}
