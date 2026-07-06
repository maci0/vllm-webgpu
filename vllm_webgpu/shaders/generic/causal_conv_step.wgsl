enable f16;

// causal_conv_step.wgsl — single-token causal depthwise conv1d update for GDN decode.
//
// Depthwise conv: each channel has its own kernel (no cross-channel mixing).
// Implements cross-correlation (not convolution): weight[c, i] aligns with
// history[c, i] — no reversal. Matches PyTorch F.conv1d behaviour.
// For decode step t:
//   output[c] = sum_{i=0}^{KERNEL-1} weight[c, i] * history[c, i]
// where history = [conv_state[0..KERNEL-2][c], x[c]]  (oldest first)
// After computation, conv_state is updated: shift left, append x.
//
// Layout:
//   x:          [CONV_DIM] f16  — new input token's projected values
//   weight:     [CONV_DIM, KERNEL] f16 — depthwise kernel weights
//   conv_state: [KERNEL-1, CONV_DIM] f16 — ring buffer of past inputs (oldest first)
//   output:     [CONV_DIM] f16
//
// Dispatch: (ceil(CONV_DIM / WG_SIZE), 1, 1)

override CONV_DIM: u32 = 8192u;
override KERNEL:   u32 = 4u;     // conv kernel size (typically 4)
override WG_SIZE:  u32 = 256u;

@group(0) @binding(0) var<storage, read>       x          : array<f16>;  // [CONV_DIM]
@group(0) @binding(1) var<storage, read>       weight     : array<f16>;  // [CONV_DIM, KERNEL]
@group(0) @binding(2) var<storage, read_write> conv_state : array<f16>;  // [KERNEL-1, CONV_DIM]
@group(0) @binding(3) var<storage, read_write> output     : array<f16>;  // [CONV_DIM]

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= CONV_DIM) { return; }

    let hist_len = KERNEL - 1u;

    // Compute output: dot of [state[0][c], state[1][c], ..., state[K-2][c], x[c]] with weight[c, :]
    var acc: f32 = 0.0;
    for (var i = 0u; i < hist_len; i++) {
        acc += f32(conv_state[i * CONV_DIM + c]) * f32(weight[c * KERNEL + i]);
    }
    acc += f32(x[c]) * f32(weight[c * KERNEL + hist_len]);

    // Raw conv output — no activation here. The gate (z) and norm apply later
    // in linear_attn_norm_gate. Applying SiLU here was incorrect.
    output[c] = f16(clamp(acc, -65504.0, 65504.0));

    // Shift conv_state: drop oldest, append x[c] at position hist_len-1
    for (var i = 0u; i < hist_len - 1u; i++) {
        conv_state[i * CONV_DIM + c] = conv_state[(i + 1u) * CONV_DIM + c];
    }
    if (hist_len > 0u) {
        conv_state[(hist_len - 1u) * CONV_DIM + c] = x[c];
    }
}
