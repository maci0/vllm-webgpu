enable f16;

// gelu_mul_fused.wgsl — SwiGLU/GELU from a combined [gate | up] buffer.
// Reads gate from gate_up[0..N-1] and up from gate_up[N..2N-1].
// Output: output[i] = silu(gate[i]) * up[i]   (Qwen3/Llama: SiLU)
// Override GELU=1 for Gemma-style tanh-approximate GELU on gate.
//
// Pairs with fused_gate_up.wgsl which writes the combined gate|up buffer.
// Dispatch: ceil(N/4 / 256) workgroups, each thread handles one vec4.

override N:    u32 = 9728u;   // inter size (NOT 2*inter)
override GELU: u32 = 0u;      // 0=SiLU (Llama/Qwen), 1=tanh-GELU (Gemma)

@group(0) @binding(0) var<storage, read>       gate_up : array<vec4<f16>>;  // [2*N/4] combined
@group(0) @binding(1) var<storage, read_write> output  : array<vec4<f16>>;  // [N/4]

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= N / 4u) { return; }

    let g4 = vec4<f32>(gate_up[i]);            // gate at [0..N-1]
    let u4 = vec4<f32>(gate_up[N / 4u + i]);   // up  at [N..2N-1]

    var act4: vec4<f32>;
    if (GELU == 1u) {
        // tanh-approximate GELU (gelu_pytorch_tanh): Gemma models
        let c: f32 = 0.7978845608f;
        let k: f32 = 0.044715f;
        act4 = 0.5f * g4 * (1.0f + tanh(c * (g4 + k * g4 * g4 * g4)));
    } else {
        // SiLU: g * sigmoid(g)
        act4 = g4 / (vec4<f32>(1.0) + exp(-g4));
    }

    output[i] = vec4<f16>(clamp(act4 * u4, vec4<f32>(-65504.0), vec4<f32>(65504.0)));
}
