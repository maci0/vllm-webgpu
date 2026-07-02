enable f16;

override HIDDEN_DIM: u32 = 4096u;
override WG_SIZE: u32 = 256u;

var<workgroup> shared_sum:   array<f32, 256>;
var<workgroup> shared_cache: array<f32, 4096>;

@group(0) @binding(0) var<storage, read>       input  : array<f16>;
@group(0) @binding(1) var<storage, read>       weight : array<f16>;
@group(0) @binding(2) var<storage, read_write> output : array<f16>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) gid  : vec3<u32>,
    @builtin(local_invocation_id)  lid  : vec3<u32>,
    @builtin(workgroup_id)         wgid : vec3<u32>,
) {
    let row   = wgid.x;
    let tid   = lid.x;
    let base  = row * HIDDEN_DIM;
    let eps   = 1e-6f;

    // Accumulate sum-of-squares and cache promoted f32 values for reuse in output pass
    var sq_sum: f32 = 0.0;
    var col = tid;
    var local_idx = 0u;
    loop {
        if (col >= HIDDEN_DIM) { break; }
        let v = f32(input[base + col]);
        shared_cache[tid + local_idx * WG_SIZE] = v;
        sq_sum += v * v;
        col += WG_SIZE;
        local_idx += 1u;
    }
    shared_sum[tid] = sq_sum;
    workgroupBarrier();

    // Parallel reduction in shared memory
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

    // Write normalized + weighted output, reading from shared cache instead of global input
    col = tid;
    local_idx = 0u;
    loop {
        if (col >= HIDDEN_DIM) { break; }
        let normed = shared_cache[tid + local_idx * WG_SIZE] * rms_inv;
        output[base + col] = f16(normed * f32(weight[col]));
        col += WG_SIZE;
        local_idx += 1u;
    }
}
