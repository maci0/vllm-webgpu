enable f16;

// N:         total element count; must be divisible by 4.
// CLAMP_MAX: 0 = no clamp; >0 = clamp gate activation to this value (swiglu_limit).
//            Matches the CLAMP_MAX semantic in fused_gate_act.wgsl so the f16 and
//            quantized paths are numerically equivalent when extra_gate_consts is set.
// Dispatch ceil(N/4 / 256) workgroups so each thread handles 4 elements.
override N:         u32 = 4096u;
override CLAMP_MAX: f32 = 0.0;  // 0 = no clamp; >0 = clamp gate activation

@group(0) @binding(0) var<storage, read>       gate   : array<vec4<f16>>;
@group(0) @binding(1) var<storage, read>       up     : array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> output : array<vec4<f16>>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= N / 4u) { return; }
    // SwiGLU: silu(gate) * up  (Llama/Qwen convention: activation on gate_proj, not up_proj)
    // Promote to f32 for exp precision, then back to f16.
    let g4 = vec4<f32>(gate[i]);
    let u4 = vec4<f32>(up[i]);
    var silu4 = g4 / (vec4<f32>(1.0) + exp(-g4));
    if (CLAMP_MAX > 0.0) { silu4 = min(silu4, vec4<f32>(CLAMP_MAX)); }
    // Clip before casting: gate × up can overflow f16 (e.g. 18000 × 18000 = 324M >> 65504).
    let product = clamp(silu4 * u4, vec4<f32>(-65504.0), vec4<f32>(65504.0));
    output[i] = vec4<f16>(product);
}
