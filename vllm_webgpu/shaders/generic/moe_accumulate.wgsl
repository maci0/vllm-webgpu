enable f16;

// moe_accumulate.wgsl — in-place weighted accumulate for MoE expert outputs.
//
// Computes: out[i] += w_buf[K_IDX] * b[i], clamped to f16 range.
// Called once per selected expert to accumulate its down_proj output into the
// running FFN result buffer.
//
// K_IDX is the slot index (0..K-1) into the pre-written weight buffer.
// Using K_IDX as an override (not SCALE) keeps the weight buffer at runtime
// granularity, so the pipeline cache sees only K=8 unique variants — compiled
// once and reused across all tokens and layers.
//
// Dispatch: (ceil(N / 256), 1, 1) — one thread per output element.

override N:     u32 = 2048u;   // hidden_size (number of elements)
override K_IDX: u32 = 0u;     // which selected-expert slot (0..K-1)

@group(0) @binding(0) var<storage, read_write> out   : array<f16>;  // [N] f16 accumulator
@group(0) @binding(1) var<storage, read>       b     : array<f16>;  // [N] f16 expert down_proj output
@group(0) @binding(2) var<storage, read>       w_buf : array<f32>;  // [K] f32 softmax weights

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= N) { return; }
    let w = w_buf[K_IDX];
    out[i] = f16(clamp(f32(out[i]) + w * f32(b[i]), -65504.0, 65504.0));
}
