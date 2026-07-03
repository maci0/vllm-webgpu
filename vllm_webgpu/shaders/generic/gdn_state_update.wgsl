enable f16;

// gdn_state_update.wgsl — Gated Delta Networks (GDN) single-token state update.
//
// Implements the delta rule recurrence for one decode step.
// One workgroup per value head. WG_SIZE threads per workgroup = K_DIM.
//
// For each value head h, k_head = h / (NUM_V_HEADS / NUM_K_HEADS):
//   q_h  = l2_norm(q[k_head])                    [K_DIM]
//   k_h  = l2_norm(k[k_head])                    [K_DIM]
//   beta = sigmoid(q_h * k_h)                    [K_DIM] element-wise
//   delta_k = beta * k_h                         [K_DIM]
//   old_out = state[h] @ q_h                     [V_DIM]  (matrix-vec)
//   state[h] = decay[h] * state[h]
//            - outer(delta_k, old_out)            [K_DIM, V_DIM]
//            + outer(k_h, v[h])                  [K_DIM, V_DIM]
//   out[h] = state[h] @ q_h                      [V_DIM]
//
// Shared memory layout (per workgroup):
//   sh_qnorm  [K_DIM]  f32 — L2-normalised query
//   sh_knorm  [K_DIM]  f32 — L2-normalised key
//   sh_delta  [K_DIM]  f32 — gated key (delta_k)
//   sh_old    [V_DIM]  f32 — old state output (state @ q_h before update)
//   sh_sq     [WG_SIZE] f32 — scratch for reductions
//
// WG_SIZE = K_DIM (typically 128). Thread t handles index t for K_DIM operations
// and also handles index t for V_DIM operations (K_DIM == V_DIM assumed).

override K_DIM:      u32 = 128u;  // key head dimension
override V_DIM:      u32 = 128u;  // value head dimension (must equal K_DIM here)
override NUM_K_HEADS: u32 = 16u;
override NUM_V_HEADS: u32 = 32u;  // must be a multiple of NUM_K_HEADS

// Bindings
@group(0) @binding(0) var<storage, read>       q_all   : array<f16>;  // [NUM_K_HEADS, K_DIM]
@group(0) @binding(1) var<storage, read>       k_all   : array<f16>;  // [NUM_K_HEADS, K_DIM]
@group(0) @binding(2) var<storage, read>       v_all   : array<f16>;  // [NUM_V_HEADS, V_DIM]
@group(0) @binding(3) var<storage, read_write> state   : array<f32>;  // [NUM_V_HEADS, K_DIM, V_DIM]
@group(0) @binding(4) var<storage, read_write> output  : array<f16>;  // [NUM_V_HEADS, V_DIM]
@group(0) @binding(5) var<storage, read>       decay_v : array<f32>;  // [NUM_V_HEADS] per-head decay

var<workgroup> sh_qnorm: array<f32, K_DIM>;
var<workgroup> sh_knorm: array<f32, K_DIM>;
var<workgroup> sh_delta: array<f32, K_DIM>;
var<workgroup> sh_old:   array<f32, V_DIM>;
var<workgroup> sh_sq:    array<f32, K_DIM>;

@compute @workgroup_size(K_DIM, 1, 1)
fn main(
    @builtin(workgroup_id)        wgid: vec3<u32>,
    @builtin(local_invocation_id) lid:  vec3<u32>,
) {
    let vh  = wgid.x;       // value head index
    let tid = lid.x;        // thread = k_dim index (0..K_DIM-1), also v_dim index
    let kh  = vh / (NUM_V_HEADS / NUM_K_HEADS);  // key head for this value head

    let eps: f32 = 1e-6;
    let kh_base = kh * K_DIM;
    let vh_state_base = vh * K_DIM * V_DIM;

    // ── Phase 1: load q and k, compute L2 norms ─────────────────────────────
    let q_raw = f32(q_all[kh_base + tid]);
    let k_raw = f32(k_all[kh_base + tid]);

    // Q norm reduction
    sh_sq[tid] = q_raw * q_raw;
    workgroupBarrier();
    var stride = K_DIM / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) { sh_sq[tid] += sh_sq[tid + stride]; }
        workgroupBarrier();
        stride /= 2u;
    }
    let q_norm_inv = inverseSqrt(sh_sq[0] + eps);
    sh_qnorm[tid] = q_raw * q_norm_inv;
    workgroupBarrier();

    // K norm reduction (reuse sh_sq)
    sh_sq[tid] = k_raw * k_raw;
    workgroupBarrier();
    stride = K_DIM / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) { sh_sq[tid] += sh_sq[tid + stride]; }
        workgroupBarrier();
        stride /= 2u;
    }
    let k_norm_inv = inverseSqrt(sh_sq[0] + eps);
    sh_knorm[tid] = k_raw * k_norm_inv;
    workgroupBarrier();

    // ── Phase 2: compute beta = sigmoid(q_norm * k_norm), delta_k = beta * k_norm ──
    let beta = 1.0 / (1.0 + exp(-(sh_qnorm[tid] * sh_knorm[tid])));
    sh_delta[tid] = beta * sh_knorm[tid];
    workgroupBarrier();

    // ── Phase 3: old_out[v] = sum_k(state[k][v] * q_norm[k]) ──────────────
    // Thread tid computes old_out[tid] by reading column tid of the state matrix.
    // Column tid: state[k * V_DIM + tid] for k = 0..K_DIM-1.
    var col_sum: f32 = 0.0;
    for (var k = 0u; k < K_DIM; k++) {
        col_sum += state[vh_state_base + k * V_DIM + tid] * sh_qnorm[k];
    }
    sh_old[tid] = col_sum;
    workgroupBarrier();

    // ── Phase 4: state update for row k=tid ─────────────────────────────────
    // state[tid][v] = decay * state[tid][v] - delta_k[tid] * old_out[v] + k_norm[tid] * v[v]
    let decay  = decay_v[vh];
    let dk_tid = sh_delta[tid];
    let kn_tid = sh_knorm[tid];
    let v_base = vh * V_DIM;
    for (var v = 0u; v < V_DIM; v++) {
        let s_idx = vh_state_base + tid * V_DIM + v;
        state[s_idx] = decay * state[s_idx]
            - dk_tid * sh_old[v]
            + kn_tid * f32(v_all[v_base + v]);
    }
    workgroupBarrier();

    // ── Phase 5: new output = state_new @ q_norm (same column-read pattern) ─
    var out_sum: f32 = 0.0;
    for (var k = 0u; k < K_DIM; k++) {
        out_sum += state[vh_state_base + k * V_DIM + tid] * sh_qnorm[k];
    }
    output[vh * V_DIM + tid] = f16(out_sum);
}
