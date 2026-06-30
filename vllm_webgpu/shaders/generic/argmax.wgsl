enable f16;

override VOCAB_SIZE: u32 = 32000u;
override WG_SIZE: u32    = 256u;

var<workgroup> sh_max: array<f32, 256>;
var<workgroup> sh_idx: array<u32, 256>;

@group(0) @binding(0) var<storage, read>       logits : array<f16>;
@group(0) @binding(1) var<storage, read_write> result : array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    var local_max: f32 = -1e30;
    var local_idx: u32 = 0u;

    var i = tid;
    loop {
        if (i >= VOCAB_SIZE) { break; }
        let v = f32(logits[i]);
        if (v > local_max) { local_max = v; local_idx = i; }
        i += WG_SIZE;
    }

    sh_max[tid] = local_max;
    sh_idx[tid] = local_idx;
    workgroupBarrier();

    var stride = WG_SIZE / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            if (sh_max[tid + stride] > sh_max[tid]) {
                sh_max[tid] = sh_max[tid + stride];
                sh_idx[tid] = sh_idx[tid + stride];
            }
        }
        workgroupBarrier();
        stride /= 2u;
    }

    if (tid == 0u) { result[0] = sh_idx[0]; }
}
