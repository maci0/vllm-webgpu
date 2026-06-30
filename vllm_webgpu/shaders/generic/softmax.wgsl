enable f16;

override SEQ_LEN: u32 = 128u;
// BATCH (num rows) is handled by dispatch workgroups, not a shader constant.

var<workgroup> sh_max: array<f32, 256>;
var<workgroup> sh_sum: array<f32, 256>;

@group(0) @binding(0) var<storage, read>       input  : array<f16>;
@group(0) @binding(1) var<storage, read_write> output : array<f16>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) lid  : vec3<u32>,
    @builtin(workgroup_id)        wgid : vec3<u32>,
) {
    let row  = wgid.x;
    let tid  = lid.x;
    let base = row * SEQ_LEN;

    // Pass 1: each thread finds its local max over the elements it owns
    var local_max: f32 = -1e30;
    var col = tid;
    loop {
        if (col >= SEQ_LEN) { break; }
        local_max = max(local_max, f32(input[base + col]));
        col += 256u;
    }
    sh_max[tid] = local_max;
    workgroupBarrier();

    // Parallel reduction: find global max
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            sh_max[tid] = max(sh_max[tid], sh_max[tid + stride]);
        }
        workgroupBarrier();
        stride /= 2u;
    }

    let row_max = sh_max[0];

    // Pass 2: compute exp(x - max) per element and accumulate local sum
    var local_sum: f32 = 0.0;
    col = tid;
    loop {
        if (col >= SEQ_LEN) { break; }
        let e = exp(f32(input[base + col]) - row_max);
        output[base + col] = f16(e);
        local_sum += e;
        col += 256u;
    }
    sh_sum[tid] = local_sum;
    workgroupBarrier();

    // Parallel reduction: sum
    stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            sh_sum[tid] += sh_sum[tid + stride];
        }
        workgroupBarrier();
        stride /= 2u;
    }

    let row_sum = sh_sum[0];

    // Pass 3: normalize
    col = tid;
    loop {
        if (col >= SEQ_LEN) { break; }
        output[base + col] = f16(f32(output[base + col]) / row_sum);
        col += 256u;
    }
}
