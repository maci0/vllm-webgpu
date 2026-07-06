enable f16;

// argmax_f16.wgsl — GPU argmax over a [N] f16 array.
//
// Dispatch (1, 1, 1): one workgroup of 256 threads.
// Each thread strides through N elements computing a local (max_val, argmax).
// Then 8-step tree reduction in shared memory gives the global argmax.
// Writes one u32 index to output[0].
//
// For N=262144: 256 threads × 1024 elements each = 262144. O(N/WG_SIZE + log WG_SIZE).
// Bandwidth: reads N F16 = N×2 bytes. For N=262144: 512KB vs prior CPU path (same read + PCIe).

override N: u32 = 262144u;

@group(0) @binding(0) var<storage, read>       logits : array<f16>;
@group(0) @binding(1) var<storage, read_write> result : array<u32>;  // [1] argmax index

var<workgroup> sh_max: array<f32, 256>;
var<workgroup> sh_idx: array<u32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;

    // Each thread finds local (max_val, argmax) over its stride of N/256 elements.
    var local_max: f32 = -1e30f;
    var local_idx: u32 = 0u;
    var i = tid;
    loop {
        if (i >= N) { break; }
        let v = f32(logits[i]);
        if (v > local_max) { local_max = v; local_idx = i; }
        i += 256u;
    }
    sh_max[tid] = local_max;
    sh_idx[tid] = local_idx;
    workgroupBarrier();

    // Tree reduction: merge (max, idx) pairs, keeping track of argmax.
    var stride = 128u;
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
        result[0] = sh_idx[0];
    }
}
