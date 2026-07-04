enable f16;

// rms_norm_f32in.wgsl — RMSNorm reading f32 input, writing f16 output.
// Used when the residual stream is stored in f32 (Gemma4 f32-residual mode).
//
// output[i] = weight[i] * input[i] / rms(input)   (f16 result)

override HIDDEN_DIM:      u32 = 4096u;
override WG_SIZE:         u32 = 256u;
override VALS_PER_THREAD: u32 = 16u;   // HIDDEN_DIM / WG_SIZE; 0 = fallback path
// GEMMA_NORM=1: Gemma-style (1+w) scale; GEMMA_NORM=0: standard w scale.
override GEMMA_NORM:      u32 = 0u;

var<workgroup> shared_sum: array<f32, 256>;

@group(0) @binding(0) var<storage, read>       input  : array<f32>;   // f32 residual
@group(0) @binding(1) var<storage, read>       weight : array<f16>;
@group(0) @binding(2) var<storage, read_write> output : array<f16>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) lid  : vec3<u32>,
    @builtin(workgroup_id)        wgid : vec3<u32>,
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
                let v = input[base + col];
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
                let w_eff = select(f32(weight[col]), 1.0 + f32(weight[col]), GEMMA_NORM != 0u);
                output[base + col] = f16(clamp(local_v[i] * rms_inv * w_eff, -65504.0, 65504.0));
                col += WG_SIZE;
            }
        }
    } else {
        var sq_sum: f32 = 0.0;
        var col = tid;
        loop {
            if (col >= HIDDEN_DIM) { break; }
            let v = input[base + col];
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
            let w_eff2 = select(f32(weight[col]), 1.0 + f32(weight[col]), GEMMA_NORM != 0u);
            output[base + col] = f16(clamp(input[base + col] * rms_inv * w_eff2, -65504.0, 65504.0));
            col += WG_SIZE;
        }
    }
}
