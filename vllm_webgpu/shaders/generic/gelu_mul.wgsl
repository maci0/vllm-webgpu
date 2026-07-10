enable f16;

// N:          total element count; must be divisible by 4.
// ACTIVATION: 0 = SiLU (default), 1 = x*sigmoid(1.702*x) (SwigluOAI), 2 = ReLU² (Nemotron-3).
//             Matches the ACTIVATION semantic in fused_gate_act.wgsl exactly — both shaders
//             must implement every ACTIVATION value identically.
// CLAMP_MAX:  0 = no clamp; >0 = clamp gate activation (upper only) to this value (swiglu_limit).
//             Matches the CLAMP_MAX semantic in fused_gate_act.wgsl so the f16 and
//             quantized paths are numerically equivalent when extra_gate_consts is set.
// CLAMP_MIN:  0 = no clamp; <0 = clamp up projection symmetrically to [CLAMP_MIN, -CLAMP_MIN].
// UP_BIAS:    additive bias on up projection before multiply (SwigluOAI: 1.0).
// Dispatch ceil(N/4 / 256) workgroups so each thread handles 4 elements.
override N:          u32 = 4096u;
override ACTIVATION: u32 = 0u;  // 0 = SiLU, 1 = x*sigmoid(1.702*x) (SwigluOAI), 2 = ReLU²
override CLAMP_MAX:  f32 = 0.0;  // 0 = no clamp; >0 = clamp gate activation (upper only)
override CLAMP_MIN:  f32 = 0.0;  // 0 = no clamp; <0 = clamp up projection symmetrically
override UP_BIAS:    f32 = 0.0;  // additive bias on up projection (SwigluOAI: 1.0)

@group(0) @binding(0) var<storage, read>       gate   : array<vec4<f16>>;
@group(0) @binding(1) var<storage, read>       up     : array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> output : array<vec4<f16>>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= N / 4u) { return; }
    // SwiGLU / SwigluOAI: activation(gate) * up  (activation on gate_proj, not up_proj)
    // Promote to f32 for exp precision, then back to f16.
    let g4 = vec4<f32>(gate[i]);
    let u4 = vec4<f32>(up[i]);
    var act4: vec4<f32>;
    if (ACTIVATION == 1u) {
        // x * sigmoid(1.702 * x)  (SwigluOAI)
        act4 = g4 * (vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-1.702f * g4)));
    } else if (ACTIVATION == 2u) {
        // ReLU² (squared ReLU, Nemotron-3): max(x, 0)²
        let r4 = max(g4, vec4<f32>(0.0));
        act4 = r4 * r4;
    } else {
        // SiLU: x * sigmoid(x)  (Llama/Qwen default)
        act4 = g4 / (vec4<f32>(1.0) + exp(-g4));
    }
    if (CLAMP_MAX > 0.0) { act4 = min(act4, vec4<f32>(CLAMP_MAX)); }
    var u4_f = u4 + vec4<f32>(UP_BIAS);
    if (CLAMP_MIN < 0.0) { u4_f = clamp(u4_f, vec4<f32>(CLAMP_MIN), vec4<f32>(-CLAMP_MIN)); }
    // Clip before casting: gate × up can overflow f16 (e.g. 18000 × 18000 = 324M >> 65504).
    let product = clamp(act4 * u4_f, vec4<f32>(-65504.0), vec4<f32>(65504.0));
    output[i] = vec4<f16>(product);
}
