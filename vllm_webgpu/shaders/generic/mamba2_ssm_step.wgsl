enable f16;

// mamba2_ssm_step.wgsl — Mamba-2 SSM single-token decode step.
//
// Mathematically verified formula (derived from selective_state_update in vLLM):
//
//   dt_eff   = softplus(dt[h] + dt_bias[h])
//   dA       = exp(A[h] * dt_eff)          A[h] = -exp(A_log[h]) < 0  => dA in (0,1)
//   group    = h * N_GROUPS / NUM_HEADS
//
//   For each d in [0, HEAD_DIM), s in [0, STATE_SIZE):
//     ssm_state[h, d, s] = dA * ssm_state[h, d, s]
//                        + dt_eff * B[group, s] * x[h, d]
//
//   (storageBarrier ensures state is visible before output reads)
//
//   For each d in [0, HEAD_DIM)  (thread tid == d):
//     y[h, d] = sum_s(C[group, s] * ssm_state[h, d, s]) + D[h] * x[h, d]
//
// Input buffer x_B_C layout (flat, f16):
//   [x: NUM_HEADS*HEAD_DIM | B: N_GROUPS*STATE_SIZE | C: N_GROUPS*STATE_SIZE]
//
// State layout: [NUM_HEADS, HEAD_DIM, STATE_SIZE] f32, updated in-place.
//
// Dispatch: (NUM_HEADS, 1, 1)
// Workgroup: (WG_SIZE, 1, 1)  — WG_SIZE must be >= HEAD_DIM

override NUM_HEADS:  u32 = 96u;
override HEAD_DIM:   u32 = 80u;
override STATE_SIZE: u32 = 128u;
override N_GROUPS:   u32 = 8u;
override WG_SIZE:    u32 = 256u;

@group(0) @binding(0) var<storage, read>       x_B_C     : array<f16>; // [NUM_HEADS*HEAD_DIM + 2*N_GROUPS*STATE_SIZE]
@group(0) @binding(1) var<storage, read>       dt_buf    : array<f16>; // [NUM_HEADS]
@group(0) @binding(2) var<storage, read>       A_buf     : array<f32>; // [NUM_HEADS] = -exp(A_log)
@group(0) @binding(3) var<storage, read>       dt_bias   : array<f32>; // [NUM_HEADS]
@group(0) @binding(4) var<storage, read>       D_buf     : array<f32>; // [NUM_HEADS]
@group(0) @binding(5) var<storage, read_write> ssm_state : array<f32>; // [NUM_HEADS, HEAD_DIM, STATE_SIZE]
@group(0) @binding(6) var<storage, read_write> y_out     : array<f16>; // [NUM_HEADS, HEAD_DIM]

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id)        wgid: vec3<u32>,
    @builtin(local_invocation_id) lid:  vec3<u32>,
) {
    let h   = wgid.x;
    let tid = lid.x;

    let mamba_int    = NUM_HEADS * HEAD_DIM;
    let groups_state = N_GROUPS * STATE_SIZE;

    // Compute dt_eff = softplus(dt[h] + dt_bias[h])
    // Numerically stable form: max(x,0) + log(1 + exp(-|x|))
    let dt_raw = f32(dt_buf[h]) + dt_bias[h];
    let dt_eff = select(dt_raw, 0.0f, dt_raw < 0.0f)
               + log(1.0f + exp(-abs(dt_raw)));

    let dA    = exp(A_buf[h] * dt_eff);
    let D_val = D_buf[h];
    let group = h * N_GROUPS / NUM_HEADS;

    let state_base  = h * HEAD_DIM * STATE_SIZE;
    let total_state = HEAD_DIM * STATE_SIZE;

    // Phase 1: update ssm_state for this head.
    // Thread tid processes elements idx = tid, tid+WG_SIZE, tid+2*WG_SIZE, ...
    var idx = tid;
    loop {
        if (idx >= total_state) { break; }
        let d = idx / STATE_SIZE;
        let s = idx % STATE_SIZE;
        let x_val = f32(x_B_C[h * HEAD_DIM + d]);
        let B_val = f32(x_B_C[mamba_int + group * STATE_SIZE + s]);
        let si    = state_base + idx;
        ssm_state[si] = dA * ssm_state[si] + dt_eff * B_val * x_val;
        idx += WG_SIZE;
    }

    // Ensure all state writes are visible before output reads.
    storageBarrier();

    // Phase 2: compute y[h, d] for d = tid (when tid < HEAD_DIM).
    if (tid < HEAD_DIM) {
        let d     = tid;
        let x_val = f32(x_B_C[h * HEAD_DIM + d]);
        let C_base = mamba_int + groups_state + group * STATE_SIZE;
        var acc = 0.0f;
        for (var s = 0u; s < STATE_SIZE; s++) {
            acc += f32(x_B_C[C_base + s]) * ssm_state[state_base + d * STATE_SIZE + s];
        }
        y_out[h * HEAD_DIM + d] = f16(clamp(acc + D_val * x_val, -65504.0f, 65504.0f));
    }
}
