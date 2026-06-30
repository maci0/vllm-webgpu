enable f16;

override VOCAB_SIZE: u32 = 32000u;
override K: u32          = 50u;
override WG_SIZE: u32    = 256u;

var<workgroup> sh_max:    array<f32, 256>;
var<workgroup> sh_idx:    array<u32, 256>;
// bool is not storable in workgroup memory; use u32 with 0=false, 1=true
var<workgroup> sh_masked: array<u32, 256>;

@group(0) @binding(0) var<storage, read>       logits   : array<f16>;
@group(0) @binding(1) var<storage, read_write> topk_idx : array<u32>;
@group(0) @binding(2) var<storage, read_write> topk_val : array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;

    // Initialize mask to unmasked (0 = not selected)
    var i = tid;
    loop {
        if (i >= 256u) { break; }
        sh_masked[i] = 0u;
        i += WG_SIZE;
    }
    workgroupBarrier();

    for (var k = 0u; k < K; k++) {
        // Find max among non-masked entries
        var local_max: f32 = -1e30;
        var local_idx: u32 = 0u;
        var j = tid;
        loop {
            if (j >= VOCAB_SIZE) { break; }
            if (sh_masked[j % WG_SIZE] == 0u) {
                let v = f32(logits[j]);
                if (v > local_max) { local_max = v; local_idx = j; }
            }
            j += WG_SIZE;
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

        if (tid == 0u) {
            topk_idx[k] = sh_idx[0];
            topk_val[k] = sh_max[0];
            sh_masked[sh_idx[0] % WG_SIZE] = 1u;
        }
        workgroupBarrier();
    }
}
