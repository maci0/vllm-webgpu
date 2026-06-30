enable f16;

override HEAD_DIM: u32  = 256u;
override NUM_HEADS: u32 = 8u;
override WG_SIZE: u32   = 128u;

var<workgroup> shared_sq: array<f32, 128>;

@group(0) @binding(0) var<storage, read>       input  : array<f16>;
@group(0) @binding(1) var<storage, read_write> output : array<f16>;

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(local_invocation_id) lid  : vec3<u32>,
    @builtin(workgroup_id)        wgid : vec3<u32>,
) {
    let seq_idx  = wgid.y;
    let head_idx = wgid.x;
    let tid      = lid.x;
    let base     = (seq_idx * NUM_HEADS + head_idx) * HEAD_DIM;
    let eps      = 1e-6f;

    var sq_sum: f32 = 0.0;
    var col = tid;
    loop {
        if (col >= HEAD_DIM) { break; }
        let v = f32(input[base + col]);
        sq_sum += v * v;
        col += WG_SIZE;
    }
    shared_sq[tid] = sq_sum;
    workgroupBarrier();

    var stride = WG_SIZE / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) { shared_sq[tid] += shared_sq[tid + stride]; }
        workgroupBarrier();
        stride /= 2u;
    }

    let rms_inv = inverseSqrt(shared_sq[0] / f32(HEAD_DIM) + eps);
    col = tid;
    loop {
        if (col >= HEAD_DIM) { break; }
        output[base + col] = f16(f32(input[base + col]) * rms_inv);
        col += WG_SIZE;
    }
}
