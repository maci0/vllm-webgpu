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

    // Pass 1: Online scan (Milakov-Divanov) — compute per-thread (local_m, local_d)
    // WITHOUT writing to output.  Fuses the max-scan and sum-scan into one input read.
    // Identity: when running max updates m_old -> m_new, rescale running sum by exp(m_old-m_new).
    // Correctness: each thread accumulates all its elements before any merge, so (local_m, local_d)
    // represents the exact (max, normaliser) for this thread's subset.
    var local_m: f32 = -1e30;
    var local_d: f32 = 0.0;
    var col = tid;
    loop {
        if (col >= SEQ_LEN) { break; }
        let xi    = f32(input[base + col]);
        let m_new = max(local_m, xi);
        local_d   = local_d * exp(local_m - m_new) + exp(xi - m_new);
        local_m   = m_new;
        col += 256u;
    }
    sh_max[tid] = local_m;
    sh_sum[tid] = local_d;
    workgroupBarrier();

    // Parallel reduction: merge (max, sum) pairs using the same online identity.
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            let ma    = sh_max[tid];
            let mb    = sh_max[tid + stride];
            let da    = sh_sum[tid];
            let db    = sh_sum[tid + stride];
            let m_new = max(ma, mb);
            sh_sum[tid] = da * exp(ma - m_new) + db * exp(mb - m_new);
            sh_max[tid] = m_new;
        }
        workgroupBarrier();
        stride /= 2u;
    }

    let global_m = sh_max[0];
    let global_d = sh_sum[0];

    // Pass 2: Normalize — re-read input and write exact softmax(xi).
    // exp(xi - global_m) / global_d is correct for every xi regardless of thread count.
    // Two total input reads (pass 1 + pass 2) vs three in the original three-pass algorithm.
    col = tid;
    loop {
        if (col >= SEQ_LEN) { break; }
        let xi = f32(input[base + col]);
        output[base + col] = f16(exp(xi - global_m) / global_d);
        col += 256u;
    }
}
