enable f16;

// gdn_state_update.wgsl — Gated Delta Networks (GDN) single-token state update.
//
// Mathematically verified formula (empirically confirmed against
// fused_sigmoid_gating_delta_rule_update_cpu from vLLM):
//
//   dt = softplus(a[h] + dt_bias[h])
//   decay = exp(-exp(A_log[h]) * dt)
//   b_gate = sigmoid(b[h])          ← per-head outer-product gate
//
//   q_h = l2_normalize(qkv[Q_BASE + kh*K_DIM : +K_DIM])
//   k_h = l2_normalize(qkv[K_BASE + kh*K_DIM : +K_DIM])
//   v_h = qkv[V_BASE + h*V_DIM : +V_DIM]
//
//   beta = sigmoid(q_h ⊙ k_h)         [K_DIM elementwise]
//   delta_k = beta ⊙ k_h
//   old_out = state[h] @ q_h           [V_DIM]
//   state[h] = decay * state[h] - outer(delta_k, old_out) + b_gate * outer(k_h, v_h)
//   output[h] = (1/sqrt(K_DIM)) * (state[h] @ q_h)    [V_DIM]  ← attention scaling
//
// The 1/sqrt(K_DIM) output scale is confirmed empirically:
//   q=e_0, state[k=0,v=0]=1 → out = 1/sqrt(K_DIM) = 0.0884 for K_DIM=128 ✓
//   q=[1,1,0,...] (2 nonzero), state[k=0,1,v=0]=1 → out = sqrt(2)/sqrt(128) = 0.125 ✓

override K_DIM:       u32 = 128u;
override V_DIM:       u32 = 128u;
override NUM_K_HEADS: u32 = 16u;
override NUM_V_HEADS: u32 = 32u;
override Q_BASE: u32  = 0u;
override K_BASE: u32  = 2048u;
override V_BASE: u32  = 4096u;

@group(0) @binding(0) var<storage, read>       qkv_buf  : array<f16>; // flat QKV
@group(0) @binding(1) var<storage, read>       a_in     : array<f16>; // [NUM_K_HEADS] — dt input per K-head
@group(0) @binding(2) var<storage, read>       b_in     : array<f16>; // [NUM_K_HEADS] — outer-product gate per K-head
@group(0) @binding(3) var<storage, read>       A_log    : array<f32>; // [NUM_V_HEADS] — log eigenvalue per V-head
@group(0) @binding(4) var<storage, read>       dt_bias  : array<f32>; // [NUM_V_HEADS] — dt bias per V-head
@group(0) @binding(5) var<storage, read_write> state    : array<f32>; // [NUM_V_HEADS, K_DIM, V_DIM]
@group(0) @binding(6) var<storage, read_write> output   : array<f16>; // [NUM_V_HEADS, V_DIM]

var<workgroup> sh_q:     array<f32, K_DIM>;
var<workgroup> sh_k:     array<f32, K_DIM>;
var<workgroup> sh_delta: array<f32, K_DIM>;
var<workgroup> sh_old:   array<f32, V_DIM>;
var<workgroup> sh_sq:    array<f32, K_DIM>;

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

    // ── Phase 0b: decay and b_gate after reductions (sh_sq now free) ─────────
    if (tid == 0u) {
        // a_in / b_in are per K-head (num_k_heads entries).
        // A_log / dt_bias are per V-head (num_v_heads entries).
        // When num_v_heads > num_k_heads, multiple V-heads share the same a/b K-head.
        let dt_raw = f32(a_in[kh]) + dt_bias[vh];
        let dt = max(dt_raw, 0.0) + log(1.0 + exp(-abs(dt_raw)));   // softplus
        sh_sq[0] = exp(-exp(A_log[vh]) * dt);                        // decay (per V-head)
        sh_sq[1] = 1.0 / (1.0 + exp(-f32(b_in[kh])));               // b_gate = sigmoid(b) (per K-head)
    }
    workgroupBarrier();
    let decay  = sh_sq[0];
    let b_gate = sh_sq[1];

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
    // state[tid][v] = decay * state[tid][v]
    //               - delta_k[tid] * old_out[v]
    //               + b_gate * k_norm[tid] * v[v]
    let dk = sh_delta[tid];
    let kn = sh_k[tid];
    for (var v = 0u; v < V_DIM; v++) {
        let idx = vh_state + tid * V_DIM + v;
        state[idx] = decay * state[idx]
            - dk * sh_old[v]
            + b_gate * kn * f32(qkv_buf[vh_v_base + v]);
    }
    workgroupBarrier();

    // ── Phase 6: output[tid] = (1/sqrt(K_DIM)) * state_new @ q ──────────────
    let inv_sqrt_k = inverseSqrt(f32(K_DIM));
    var out_v: f32 = 0.0;
    for (var k = 0u; k < K_DIM; k++) {
        out_v += state[vh_state + k * V_DIM + tid] * sh_q[k];
    }
    output[vh * V_DIM + tid] = f16(inv_sqrt_k * out_v);
}
