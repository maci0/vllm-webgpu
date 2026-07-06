enable f16;

// add_rms_norm.wgsl — fused residual-add + RMSNorm.
//
// Computes:
//   residual_out[i] = a[i] + b[i]
//   normed_out[i]   = rms_norm(residual_out, weight)[i]
//
// Saves 1 dispatch + 2 global reads vs separate add → rms_norm.
// Register-tiled path stores residual in thread-local registers during pass 1,
// avoiding a second global read in pass 2.
// VALS_PER_THREAD=0: two-pass global re-read fallback for HIDDEN_DIM > 4096.
//
// Dispatch (num_tokens, 1, 1).
// GEMMA_NORM=1: (1+w) scale; GEMMA_NORM=0: standard w (Llama/Qwen3).

override HIDDEN_DIM:      u32 = 4096u;
override WG_SIZE:         u32 = 256u;
override VALS_PER_THREAD: u32 = 16u;   // HIDDEN_DIM / WG_SIZE, max 16; 0 = fallback
override GEMMA_NORM:      u32 = 0u;

@group(0) @binding(0) var<storage, read>       a            : array<f16>;
@group(0) @binding(1) var<storage, read>       b            : array<f16>;
@group(0) @binding(2) var<storage, read>       weight       : array<f16>;
@group(0) @binding(3) var<storage, read_write> residual_out : array<f16>;
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
        // Register-tile path: store residual in registers, skip pass-2 global read.
        var local_v: array<f32, 16>;
        var sq_sum: f32 = 0.0;
        var col = tid;
        for (var i = 0u; i < VALS_PER_THREAD; i++) {
            if (col < HIDDEN_DIM) {
                let v = f32(a[base + col]) + f32(b[base + col]);
                local_v[i] = v;
                residual_out[base + col] = f16(v);
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
        // Two-pass global re-read fallback for HIDDEN_DIM > WG_SIZE * 16.
        var sq_sum: f32 = 0.0;
        var col = tid;
        loop {
            if (col >= HIDDEN_DIM) { break; }
            let v = f32(a[base + col]) + f32(b[base + col]);
            residual_out[base + col] = f16(v);
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
            let v2 = f32(residual_out[base + col]);
            let w_eff = select(f32(weight[col]), 1.0 + f32(weight[col]), GEMMA_NORM != 0u);
            normed_out[base + col] = f16(clamp(v2 * rms_inv * w_eff, -65504.0, 65504.0));
            col += WG_SIZE;
        }
    }
}
