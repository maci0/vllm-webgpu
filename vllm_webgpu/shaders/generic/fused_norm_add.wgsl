enable f16;

override HIDDEN_DIM: u32 = 4096u;
override WG_SIZE: u32    = 256u;

var<workgroup> shared_sum: array<f32, 256>;

@group(0) @binding(0) var<storage, read>       residual : array<f16>;
@group(0) @binding(1) var<storage, read>       hidden   : array<f16>;
@group(0) @binding(2) var<storage, read>       weight   : array<f16>;
@group(0) @binding(3) var<storage, read_write> output   : array<f16>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id)  lid  : vec3<u32>,
    @builtin(workgroup_id)         wgid : vec3<u32>,
) {
    let row  = wgid.x;
    let tid  = lid.x;
    let base = row * HIDDEN_DIM;
    let eps  = 1e-6f;

    var sq_sum: f32 = 0.0;
    var col = tid;
    loop {
        if (col >= HIDDEN_DIM) { break; }
        let v = f32(residual[base + col]) + f32(hidden[base + col]);
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
        let v = f32(residual[base + col]) + f32(hidden[base + col]);
        output[base + col] = f16(v * rms_inv * f32(weight[col]));
        col += WG_SIZE;
    }
}
