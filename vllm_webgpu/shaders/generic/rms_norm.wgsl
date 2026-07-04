enable f16;

override HIDDEN_DIM: u32    = 4096u;
override WG_SIZE: u32      = 256u;
// GEMMA_NORM=1: Gemma-style (1+w) scale; GEMMA_NORM=0: standard w scale (Llama/Qwen3).
override GEMMA_NORM: u32   = 0u;
// Maximum values stored in registers per thread. Host must set this to HIDDEN_DIM / WG_SIZE.
// For HIDDEN_DIM=2560, WG_SIZE=256: VALS_PER_THREAD=10. Max supported: 16 (4096 / 256).
// For HIDDEN_DIM > 4096 (e.g. 8192), set VALS_PER_THREAD=0 to disable register-tiling
// and fall back to the two-pass global re-read.
override VALS_PER_THREAD: u32 = 16u;

var<workgroup> shared_sum: array<f32, 256>;

// Register-tiled storage: thread stores up to VALS_PER_THREAD f32 values locally,
// avoiding a second global read in pass 2.
// WGSL allows override constants as function-scope array sizes.

@group(0) @binding(0) var<storage, read>       input  : array<f16>;
@group(0) @binding(1) var<storage, read>       weight : array<f16>;
@group(0) @binding(2) var<storage, read_write> output : array<f16>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id)  lid  : vec3<u32>,
    @builtin(workgroup_id)         wgid : vec3<u32>,
) {
    let row  = wgid.x;
    let tid  = lid.x;
    let base = row * HIDDEN_DIM;
    let eps  = 1e-6f;

    if (VALS_PER_THREAD > 0u) {
        // Register-tile path: store input values in registers, skip pass-2 global read.
        // Sufficient for HIDDEN_DIM ≤ WG_SIZE * VALS_PER_THREAD (e.g. ≤4096 for WG=256, V=16).
        // Array size fixed at 16 (max needed); VALS_PER_THREAD is the loop bound at runtime.
        var local_v: array<f32, 16>;
        var sq_sum: f32 = 0.0;
        var col = tid;
        for (var i = 0u; i < VALS_PER_THREAD; i++) {
            if (col < HIDDEN_DIM) {
                let v = f32(input[base + col]);
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

        // Pass 2: read from registers — zero global memory traffic for input.
        col = tid;
        for (var i = 0u; i < VALS_PER_THREAD; i++) {
            if (col < HIDDEN_DIM) {
                let w_eff = select(f32(weight[col]), 1.0 + f32(weight[col]), GEMMA_NORM != 0u);
                output[base + col] = f16(clamp(local_v[i] * rms_inv * w_eff, -65504.0, 65504.0));
                col += WG_SIZE;
            }
        }
    } else {
        // Two-pass global re-read fallback (for HIDDEN_DIM > WG_SIZE * max register slots).
        var sq_sum: f32 = 0.0;
        var col = tid;
        loop {
            if (col >= HIDDEN_DIM) { break; }
            let v = f32(input[base + col]);
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
            let normed = f32(input[base + col]) * rms_inv;
            let w_eff2 = select(f32(weight[col]), 1.0 + f32(weight[col]), GEMMA_NORM != 0u);
            output[base + col] = f16(clamp(normed * w_eff2, -65504.0, 65504.0));
            col += WG_SIZE;
        }
    }
}
