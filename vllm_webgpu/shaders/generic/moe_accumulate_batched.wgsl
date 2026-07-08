enable f16;

// moe_accumulate_batched.wgsl — per-token weighted in-place accumulate for MoE.
//
// Computes: out[t*H + j] += w[t] * b[t*H + j], clamped to f16 range.
// Called once per selected expert after its down_proj output is computed.
// w[t] is the routing weight for token t to this expert (0.0 if token t does
// not route to this expert, so the += is a no-op for those tokens).
//
// N = T * H (total elements = num_tokens * hidden_size)
// H = hidden_size (used to derive token index: t = i / H)
//
// Dispatch: (ceil(N / 256), 1, 1) — one thread per output element.

override N: u32 = 2048u;   // num_tokens * hidden_size
override H: u32 = 2048u;   // hidden_size

@group(0) @binding(0) var<storage, read_write> out: array<f16>;  // [T, H] f16 accumulator
@group(0) @binding(1) var<storage, read>       b:   array<f16>;  // [T, H] f16 expert down_proj output
@group(0) @binding(2) var<storage, read>       w:   array<f32>;  // [T] f32 per-token routing weights

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= N) { return; }
    let t  = i / H;
    let wt = w[t];
    out[i] = f16(clamp(f32(out[i]) + wt * f32(b[i]), -65504.0, 65504.0));
}
