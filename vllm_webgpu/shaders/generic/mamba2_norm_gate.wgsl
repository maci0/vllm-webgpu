enable f16;

// mamba2_norm_gate.wgsl — Grouped gated RMSNorm for Mamba-2 mixer output.
//
// Implements Mixer2RMSNormGated.forward_native (vLLM) for the TP=1 case:
//
//   For each group g in [0, N_GROUPS):
//     gate_x[g*G + i] = x[g*G + i] * silu(gate[g*G + i])
//     variance        = mean_i(gate_x[g*G + i]^2)
//     y[g*G + i]      = gate_x[g*G + i] / sqrt(variance + eps) * weight[g*G + i]
//
// where G = MAMBA_INT / N_GROUPS (group size).
//
// The two-pass approach recomputes silu(gate) in the second pass to avoid
// storing GROUP_SIZE gated values in workgroup memory.
//
// Dispatch: (N_GROUPS, 1, 1)
// Workgroup: (WG_SIZE, 1, 1)

override MAMBA_INT: u32 = 7680u;  // NUM_HEADS * HEAD_DIM
override N_GROUPS:  u32 = 8u;
override WG_SIZE:   u32 = 256u;

@group(0) @binding(0) var<storage, read>       x_in    : array<f16>; // [MAMBA_INT]
@group(0) @binding(1) var<storage, read>       gate_in : array<f16>; // [MAMBA_INT]
@group(0) @binding(2) var<storage, read>       weight  : array<f16>; // [MAMBA_INT]
@group(0) @binding(3) var<storage, read_write> y_out   : array<f16>; // [MAMBA_INT]

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id)        wgid: vec3<u32>,
    @builtin(local_invocation_id) lid:  vec3<u32>,
) {
    let g   = wgid.x;
    let tid = lid.x;
    let eps = 1e-6f;
    let group_size = MAMBA_INT / N_GROUPS;
    let base       = g * group_size;

    // Pass 1: accumulate sum of squares of gated x.
    var sq_sum = 0.0f;
    var col    = tid;
    loop {
        if (col >= group_size) { break; }
        let idx = base + col;
        let xv  = f32(x_in[idx]);
        let gv  = f32(gate_in[idx]);
        let sg  = gv / (1.0f + exp(-gv));   // silu(gate)
        let gx  = xv * sg;
        sq_sum += gx * gx;
        col += WG_SIZE;
    }
    shared_sum[tid] = sq_sum;
    workgroupBarrier();

    // Tree reduction to compute total sum of squares for this group.
    var stride = WG_SIZE / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) { shared_sum[tid] += shared_sum[tid + stride]; }
        workgroupBarrier();
        stride /= 2u;
    }
    let rms_inv = inverseSqrt(shared_sum[0] / f32(group_size) + eps);

    // Pass 2: recompute gated x, normalize, and apply weight.
    col = tid;
    loop {
        if (col >= group_size) { break; }
        let idx = base + col;
        let xv  = f32(x_in[idx]);
        let gv  = f32(gate_in[idx]);
        let sg  = gv / (1.0f + exp(-gv));
        let gx  = xv * sg;
        let w   = f32(weight[idx]);
        y_out[idx] = f16(clamp(gx * rms_inv * w, -65504.0f, 65504.0f));
        col += WG_SIZE;
    }
}
