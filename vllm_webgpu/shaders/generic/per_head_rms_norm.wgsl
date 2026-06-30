enable f16;

override HEAD_DIM: u32  = 128u;
override NUM_HEADS: u32 = 32u;
override WG_SIZE: u32   = 128u;

var<workgroup> shared_sum: array<f32, 128>;

@group(0) @binding(0) var<storage, read>       input  : array<f16>;
@group(0) @binding(1) var<storage, read>       weight : array<f16>;
@group(0) @binding(2) var<storage, read_write> output : array<f16>;

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(global_invocation_id) gid  : vec3<u32>,
    @builtin(local_invocation_id)  lid  : vec3<u32>,
    @builtin(workgroup_id)         wgid : vec3<u32>,
) {
    let row  = wgid.y;
    let head = wgid.x;
    let tid  = lid.x;
    let base = (row * NUM_HEADS + head) * HEAD_DIM;
    let eps  = 1e-6f;

    var sq_sum: f32 = 0.0;
    var col = tid;
    loop {
        if (col >= HEAD_DIM) { break; }
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

    let rms_inv = inverseSqrt(shared_sum[0] / f32(HEAD_DIM) + eps);
    let w_base  = head * HEAD_DIM;

    col = tid;
    loop {
        if (col >= HEAD_DIM) { break; }
        output[base + col] = f16(f32(input[base + col]) * rms_inv * f32(weight[w_base + col]));
        col += WG_SIZE;
    }
}
