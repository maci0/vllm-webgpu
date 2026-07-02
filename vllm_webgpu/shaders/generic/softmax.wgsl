enable f16;

override SEQ_LEN: u32 = 128u;
// BATCH (num rows) is handled by dispatch workgroups, not a shader constant.

var<workgroup> sh_max:  array<f32, 256>;
var<workgroup> sh_sum:  array<f32, 256>;
// sh_lmax: per-thread running max at write time; needed to correct output[] in normalize pass.
var<workgroup> sh_lmax: array<f32, 256>;

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

    // Online softmax pass: single scan over input, maintaining (local_m, local_d) pair.
    // Identity: when updating max from m_old to m_new, rescale running sum by exp(m_old - m_new).
    // This fuses pass 1 (max scan) and pass 2 (exp+sum) into one input read.
    var local_m: f32 = -1e30;
    var local_d: f32 = 0.0;
    var col = tid;
    loop {
        if (col >= SEQ_LEN) { break; }
        let xi    = f32(input[base + col]);
        let m_new = max(local_m, xi);
        local_d   = local_d * exp(local_m - m_new) + exp(xi - m_new);
        local_m   = m_new;
        // Write exp(xi - local_m_at_write_time); corrected to global max in normalize pass.
        output[base + col] = f16(exp(xi - local_m));
        col += 256u;
    }
    sh_max[tid]  = local_m;
    sh_sum[tid]  = local_d;
    sh_lmax[tid] = local_m;  // save per-thread final max for correction factor
    workgroupBarrier();

    // Parallel reduction: merge (max, sum) pairs using the online softmax identity.
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            let ma = sh_max[tid];
            let mb = sh_max[tid + stride];
            let da = sh_sum[tid];
            let db = sh_sum[tid + stride];
            let m_new = max(ma, mb);
            sh_sum[tid] = da * exp(ma - m_new) + db * exp(mb - m_new);
            sh_max[tid] = m_new;
        }
        workgroupBarrier();
        stride /= 2u;
    }

    let global_m = sh_max[0];
    let global_d = sh_sum[0];
    // Correction: output[col] = exp(xi - local_m); multiply by exp(local_m - global_m)
    // to get exp(xi - global_m), then divide by global_d.
    let corr = exp(sh_lmax[tid] - global_m);

    // Normalize pass: apply correction and divide by global sum.
    col = tid;
    loop {
        if (col >= SEQ_LEN) { break; }
        output[base + col] = f16(f32(output[base + col]) * corr / global_d);
        col += 256u;
    }
}
