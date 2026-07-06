enable f16;

// sigmoid_gate.wgsl — element-wise: output[i] = sigmoid(gate[i]) * value[i]
//
// Used for Qwen3.5 attn_output_gate: the gate projection is gated with sigmoid
// (not SiLU) before multiplying the attention output.
//
// Dispatch: (ceil(N/4/256), 1, 1) — 256 threads, each handles 4 elements (vec4).

override N: u32 = 2048u;   // total element count (must be divisible by 4)

@group(0) @binding(0) var<storage, read>       gate   : array<vec4<f16>>;  // [N/4]
@group(0) @binding(1) var<storage, read>       value  : array<vec4<f16>>;  // [N/4]
@group(0) @binding(2) var<storage, read_write> output : array<vec4<f16>>;  // [N/4]

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= N / 4u) { return; }
    let g4 = vec4<f32>(gate[i]);
    let v4 = vec4<f32>(value[i]);
    let sig4 = vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-g4));
    output[i] = vec4<f16>(clamp(sig4 * v4, vec4<f32>(-65504.0), vec4<f32>(65504.0)));
}
