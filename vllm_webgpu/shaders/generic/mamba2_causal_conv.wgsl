enable f16;

// mamba2_causal_conv.wgsl — single-token causal depthwise conv1d update for Mamba-2 decode.
//
// Same computation as causal_conv_step.wgsl but includes an additive bias term
// before the SiLU activation, matching causal_conv1d_update() in the vLLM Mamba-2 path.
//
// For decode step t:
//   output[c] = silu(sum_{i=0}^{KERNEL-1} weight[c, i] * history[c, i] + bias[c])
// where history = [conv_state[0..KERNEL-2][c], x[c]]  (oldest first)
// After computation conv_state is updated: shift left, append x[c].
//
// Layout:
//   x:          [CONV_DIM] f16
//   weight:     [CONV_DIM, KERNEL] f16  — depthwise kernel, row-major
//   bias_buf:   [CONV_DIM] f16  — additive bias (bind a dummy buffer when HAS_BIAS=0)
//   conv_state: [KERNEL-1, CONV_DIM] f16  — oldest-first ring buffer
//   output:     [CONV_DIM] f16
//
// Dispatch: (ceil(CONV_DIM / WG_SIZE), 1, 1)

override CONV_DIM: u32 = 9728u;
override KERNEL:   u32 = 4u;
override WG_SIZE:  u32 = 256u;
override HAS_BIAS: u32 = 1u;   // 1: add bias_buf[c]; 0: skip (dummy buffer still required)

@group(0) @binding(0) var<storage, read>       x          : array<f16>;  // [CONV_DIM]
@group(0) @binding(1) var<storage, read>       weight     : array<f16>;  // [CONV_DIM * KERNEL]
@group(0) @binding(2) var<storage, read>       bias_buf   : array<f16>;  // [CONV_DIM]
@group(0) @binding(3) var<storage, read_write> conv_state : array<f16>;  // [(KERNEL-1) * CONV_DIM]
@group(0) @binding(4) var<storage, read_write> output     : array<f16>;  // [CONV_DIM]

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= CONV_DIM) { return; }

    let hist_len = KERNEL - 1u;

    // Dot product: [state[0..K-2][c], x[c]] . weight[c, :]
    var acc: f32 = 0.0;
    for (var i = 0u; i < hist_len; i++) {
        acc += f32(conv_state[i * CONV_DIM + c]) * f32(weight[c * KERNEL + i]);
    }
    acc += f32(x[c]) * f32(weight[c * KERNEL + hist_len]);

    if (HAS_BIAS != 0u) {
        acc += f32(bias_buf[c]);
    }

    // SiLU activation
    let silu_out = acc / (1.0 + exp(-acc));
    output[c] = f16(clamp(silu_out, -65504.0, 65504.0));

    // Shift conv_state left (drop oldest), append x[c] at position hist_len-1.
    // Guard: for hist_len <= 1 (KERNEL <= 2), hist_len-1u would underflow to u32::MAX.
    if (hist_len > 1u) {
        for (var i = 0u; i < hist_len - 1u; i++) {
            conv_state[i * CONV_DIM + c] = conv_state[(i + 1u) * CONV_DIM + c];
        }
    }
    if (hist_len > 0u) {
        conv_state[(hist_len - 1u) * CONV_DIM + c] = x[c];
    }
}
