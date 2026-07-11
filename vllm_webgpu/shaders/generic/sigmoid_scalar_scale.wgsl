enable f16;

// sigmoid_scalar_scale.wgsl — multiply a vector in-place by sigmoid(gate[0]).
//
// Used for the Qwen3.5-MoE shared expert gate: the shared expert output is
// scaled by sigmoid(shared_expert_gate(x)[0]), matching vLLM's
// Qwen2MoeMLP.forward which applies F.sigmoid(self.expert_gate(x)[0]) * out.
//
// Binding 0: gate  — 1-element f16 buffer (pre-sigmoid scalar, output of gate GEMV)
// Binding 1: data  — [N/4] vec4<f16> buffer (shared expert output, scaled in-place)
//
// Dispatch: (ceil(N/4/256), 1, 1) — 256 threads, each handles 4 elements.

override N: u32 = 7168u;   // total element count; must be divisible by 4

@group(0) @binding(0) var<storage, read>       gate : array<f16>;           // [1]
@group(0) @binding(1) var<storage, read_write> data : array<vec4<f16>>;     // [N/4]

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= N / 4u) { return; }
    let g   = f32(gate[0]);
    let sig = 1.0 / (1.0 + exp(-g));
    let v4  = vec4<f32>(data[i]);
    data[i] = vec4<f16>(clamp(vec4<f32>(sig) * v4, vec4<f32>(-65504.0), vec4<f32>(65504.0)));
}
