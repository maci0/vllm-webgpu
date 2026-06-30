enable f16;

override VOCAB_SIZE: u32 = 32000u;
override K: u32          = 50u;
override WG_SIZE: u32    = 256u;

var<workgroup> sh_max: array<f32, 256>;
var<workgroup> sh_idx: array<u32, 256>;

// mask is a storage buffer (not workgroup) so it correctly tracks all VOCAB_SIZE indices.
// Workgroup-memory masks alias when VOCAB_SIZE > WG_SIZE (j % WG_SIZE collision).
@group(0) @binding(0) var<storage, read>            logits   : array<f16>;
@group(0) @binding(1) var<storage, read_write>      topk_idx : array<u32>;
@group(0) @binding(2) var<storage, read_write>      topk_val : array<f32>;
@group(0) @binding(3) var<storage, read_write>      mask     : array<u32>;  // [VOCAB_SIZE], 0=available 1=selected

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;

    // Initialize mask to 0 (all available)
    var i = tid;
    loop {
        if (i >= VOCAB_SIZE) { break; }
        mask[i] = 0u;
        i += WG_SIZE;
    }
    // storageBarrier() is required for storage buffer visibility across threads;
    // workgroupBarrier() only covers var<workgroup> memory.
    storageBarrier();

    for (var k = 0u; k < K; k++) {
        // Find max among non-masked entries
        var local_max: f32 = -1e30;
        var local_idx: u32 = 0u;
        var j = tid;
        loop {
            if (j >= VOCAB_SIZE) { break; }
            if (mask[j] == 0u) {
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
            mask[sh_idx[0]] = 1u;  // mark this index as selected
        }
        // storageBarrier() so all threads see mask[sh_idx[0]] = 1 before next K pass.
        storageBarrier();
    }
}
