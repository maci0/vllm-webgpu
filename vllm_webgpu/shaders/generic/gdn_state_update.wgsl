enable f16;

// gdn_state_update.wgsl — Gated Delta Networks (GDN) single-token state update.
//
// Implements the Qwen3.5 delta-rule linear attention recurrence for decode.
// One workgroup per value head. K_DIM threads per workgroup.
//
// Per value head h (key head kh = h * NUM_K_HEADS / NUM_V_HEADS):
//   dt = softplus(a[h] + dt_bias[h])
//   decay = exp(-exp(A_log[h]) * dt)
//   q_h = l2_normalize(qkv[Q_BASE + kh*K_DIM : +K_DIM])   [K_DIM f16]
//   k_h = l2_normalize(qkv[K_BASE + kh*K_DIM : +K_DIM])   [K_DIM f16]
//   v_h = qkv[V_BASE + h*V_DIM : +V_DIM]                   [V_DIM f16]
//   beta = sigmoid(q_h * k_h)                              [K_DIM] elementwise
//   delta_k = beta * k_h
//   old_out = state[h] @ q_h                               [V_DIM]
//   state[h] = decay * state[h] - outer(delta_k, old_out) + outer(k_h, v_h)
//   output[h] = state[h] @ q_h                             [V_DIM f16]
//
// State stored as f32 (persistent between forward calls).
// All other values computed in f32 for numerical stability.

override K_DIM:       u32 = 128u;
override V_DIM:       u32 = 128u;
override NUM_K_HEADS: u32 = 16u;
override NUM_V_HEADS: u32 = 32u;
override Q_BASE: u32  = 0u;      // f16 element offset of Q in qkv_buf
override K_BASE: u32  = 2048u;   // f16 element offset of K
override V_BASE: u32  = 4096u;   // f16 element offset of V

@group(0) @binding(0) var<storage, read>       qkv_buf  : array<f16>; // flat QKV
@group(0) @binding(1) var<storage, read>       a_in     : array<f16>; // [NUM_V_HEADS]
@group(0) @binding(2) var<storage, read>       A_log    : array<f16>; // [NUM_V_HEADS]
@group(0) @binding(3) var<storage, read>       dt_bias  : array<f16>; // [NUM_V_HEADS]
@group(0) @binding(4) var<storage, read_write> state    : array<f32>; // [NUM_V_HEADS, K_DIM, V_DIM]
@group(0) @binding(5) var<storage, read_write> output   : array<f16>; // [NUM_V_HEADS, V_DIM]

var<workgroup> sh_q:     array<f32, K_DIM>;
var<workgroup> sh_k:     array<f32, K_DIM>;
var<workgroup> sh_delta: array<f32, K_DIM>;
var<workgroup> sh_old:   array<f32, V_DIM>;
var<workgroup> sh_sq:    array<f32, K_DIM>;  // scratch for reductions + dA broadcast

@compute @workgroup_size(K_DIM, 1, 1)
fn main(
    @builtin(workgroup_id)        wgid: vec3<u32>,
    @builtin(local_invocation_id) lid:  vec3<u32>,
) {
    let vh  = wgid.x;
    let tid = lid.x;
    let kh  = vh * NUM_K_HEADS / NUM_V_HEADS;

    let eps: f32 = 1e-6;
    let kh_q_base = Q_BASE + kh * K_DIM;
    let kh_k_base = K_BASE + kh * K_DIM;
    let vh_v_base = V_BASE + vh * V_DIM;
    let vh_state  = vh * K_DIM * V_DIM;

    // ── Phase 0: compute and broadcast decay ─────────────────────────────────
    if (tid == 0u) {
        let dt_raw = f32(a_in[vh]) + f32(dt_bias[vh]);
        let dt = max(dt_raw, 0.0) + log(1.0 + exp(-abs(dt_raw)));  // softplus
        sh_sq[0] = exp(-exp(f32(A_log[vh])) * dt);  // decay scalar
    }

    // ── Phase 1: L2-normalize q ───────────────────────────────────────────────
    let q_raw = f32(qkv_buf[kh_q_base + tid]);
    sh_sq[tid] = q_raw * q_raw;
    workgroupBarrier();
    var stride = K_DIM / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) { sh_sq[tid] += sh_sq[tid + stride]; }
        workgroupBarrier();
        stride /= 2u;
    }
    let q_inv = inverseSqrt(sh_sq[0] + eps);
    sh_q[tid] = q_raw * q_inv;
    workgroupBarrier();

    // ── Phase 2: L2-normalize k ───────────────────────────────────────────────
    let k_raw = f32(qkv_buf[kh_k_base + tid]);
    sh_sq[tid] = k_raw * k_raw;
    workgroupBarrier();
    stride = K_DIM / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) { sh_sq[tid] += sh_sq[tid + stride]; }
        workgroupBarrier();
        stride /= 2u;
    }
    let k_inv = inverseSqrt(sh_sq[0] + eps);
    sh_k[tid] = k_raw * k_inv;
    workgroupBarrier();

    // ── Phase 0b: re-broadcast decay after reductions clobbered sh_sq ────────
    if (tid == 0u) {
        let dt_raw = f32(a_in[vh]) + f32(dt_bias[vh]);
        let dt = max(dt_raw, 0.0) + log(1.0 + exp(-abs(dt_raw)));
        sh_sq[0] = exp(-exp(f32(A_log[vh])) * dt);
    }
    workgroupBarrier();
    let decay = sh_sq[0];

    // ── Phase 3: beta = sigmoid(q*k), delta_k = beta * k ─────────────────────
    let beta = 1.0 / (1.0 + exp(-(sh_q[tid] * sh_k[tid])));
    sh_delta[tid] = beta * sh_k[tid];
    workgroupBarrier();

    // ── Phase 4: old_out[tid] = state[h][*][tid] @ q ─────────────────────────
    var col_sum: f32 = 0.0;
    for (var k = 0u; k < K_DIM; k++) {
        col_sum += state[vh_state + k * V_DIM + tid] * sh_q[k];
    }
    sh_old[tid] = col_sum;
    workgroupBarrier();

    // ── Phase 5: state update for row k=tid ──────────────────────────────────
    let dk = sh_delta[tid];
    let kn = sh_k[tid];
    for (var v = 0u; v < V_DIM; v++) {
        let idx = vh_state + tid * V_DIM + v;
        state[idx] = decay * state[idx]
            - dk * sh_old[v]
            + kn * f32(qkv_buf[vh_v_base + v]);
    }
    workgroupBarrier();

    // ── Phase 6: output[tid] = updated_state[h][*][tid] @ q ──────────────────
    var out_v: f32 = 0.0;
    for (var k = 0u; k < K_DIM; k++) {
        out_v += state[vh_state + k * V_DIM + tid] * sh_q[k];
    }
    output[vh * V_DIM + tid] = f16(out_v);
}
