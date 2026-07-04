enable f16;

// Gemma3/4 FFN activation: tanh-approximate GELU on gate, then gate * up.
// gelu_pytorch_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
// N: total element count; must be divisible by 4.

override N: u32 = 4096u;

@group(0) @binding(0) var<storage, read>       gate   : array<vec4<f16>>;
@group(0) @binding(1) var<storage, read>       up     : array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> output : array<vec4<f16>>;

const SQRT2_OVER_PI: f32 = 0.7978845608028654f;
const GELU_COEF:     f32 = 0.044715f;

fn gelu_tanh(x: f32) -> f32 {
    let inner = SQRT2_OVER_PI * (x + GELU_COEF * x * x * x);
    return 0.5f * x * (1.0f + tanh(inner));
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= N / 4u) { return; }
    let g4 = vec4<f32>(gate[i]);
    let u4 = vec4<f32>(up[i]);
    let gelu4 = vec4<f32>(gelu_tanh(g4.x), gelu_tanh(g4.y), gelu_tanh(g4.z), gelu_tanh(g4.w));
    let product = clamp(gelu4 * u4, vec4<f32>(-65504.0), vec4<f32>(65504.0));
    output[i] = vec4<f16>(product);
}
