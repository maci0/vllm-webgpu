enable f16;

// moe_accumulate_batched.wgsl — per-token weighted in-place accumulate for MoE.
//
// Computes: out[t*H + j] += w[EXPERT_SLOT * T + t] * b[t*H + j], clamped to f16 range.
// Called once per selected expert after its down_proj output is computed.
//
// The weight buffer w holds [num_unique_experts, T] f32 packed row-major. EXPERT_SLOT
// selects the row for the current expert. Weights for tokens that do not route to this
// expert are 0.0, so the += is a no-op for those tokens.
//
// A single write_buffer fills the entire packed weight array before the expert loop
// starts, avoiding the hazard where all per-iteration write_buffer calls resolve before
// the shared command encoder's compute commands execute (leaving only the last expert's
// weights visible to every dispatch).
//
// N = T * H (total elements = num_tokens * hidden_size)
// H = hidden_size (used to derive token index: t = i / H; T = N / H)
//
// Dispatch: (ceil(N / 256), 1, 1) — one thread per output element.

override N:           u32 = 2048u;   // num_tokens * hidden_size
override H:           u32 = 2048u;   // hidden_size
override EXPERT_SLOT: u32 = 0u;      // row index into the packed [num_unique, T] weight buffer

@group(0) @binding(0) var<storage, read_write> out: array<f16>;  // [T, H] f16 accumulator
@group(0) @binding(1) var<storage, read>       b:   array<f16>;  // [T, H] f16 expert down_proj output
@group(0) @binding(2) var<storage, read>       w:   array<f32>;  // [num_unique, T] f32 packed weights

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= N) { return; }
    let T  = N / H;
    let t  = i / H;
    let wt = w[EXPERT_SLOT * T + t];
    out[i] = f16(clamp(f32(out[i]) + wt * f32(b[i]), -65504.0, 65504.0));
}
