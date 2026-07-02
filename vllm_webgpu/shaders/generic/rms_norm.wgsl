enable f16;

override HIDDEN_DIM: u32 = 4096u;
override WG_SIZE: u32    = 256u;

var<workgroup> shared_sum: array<f32, 256>;

// Note: a shared_cache for input values was removed. It required array<f32, HIDDEN_DIM>
// in workgroup memory, which overflows the 16384-byte WebGPU minimum limit for
// HIDDEN_DIM > 4096 (Llama-3-70B: 8192, Gemma-4-27B: 5376). Re-reading from global
// memory in the output pass is correct; the GPU L2 cache covers most of the cost.

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

    // Pass 1: accumulate sum-of-squares
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

    // Parallel reduction
    var stride = WG_SIZE / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        workgroupBarrier();
        stride /= 2u;
    }

    let rms_inv = inverseSqrt(shared_sum[0] / f32(HIDDEN_DIM) + eps);

    // Pass 2: normalize and write — re-read from global input
    col = tid;
    loop {
        if (col >= HIDDEN_DIM) { break; }
        let normed = f32(input[base + col]) * rms_inv;
        output[base + col] = f16(normed * f32(weight[col]));
        col += WG_SIZE;
    }
}
