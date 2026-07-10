enable f16;

// fused_gate_act.wgsl — gate+up GEMV with inline activation → ffn_act.
//
// Fuses fused_gate_up + gelu_mul_fused into one dispatch:
//   ffn_act[row] = activation(gate_proj(x)[row]) * up_proj(x)[row]
//
// Eliminates the intermediate gate_up scratch buffer and the gelu_mul_fused pass.
//
// Dispatch (N, 1, 1): one workgroup per output row, 256 threads split-K.
// Thread tid processes k = tid*2, tid*2+512, ... (stride 512, 2 f16 per u32).
//
// Overrides:
//   K          — input hidden dim
//   N          — intermediate/output dim (gate and up both have shape [N, K/2] u32)
//   GELU       — 0: SiLU = x*σ(x) (Llama/Qwen), 1: tanh-GELU (Gemma) [legacy]
//   ACTIVATION — 0: SiLU (default), 1: x*sigmoid(1.702*x) (SwigluOAI/GELU approx), 2: ReLU² (Nemotron-3)
//                Takes precedence over GELU when non-zero.
//   CLAMP_MAX  — 0: no clamp; >0: clamp gate activation (upper bound only) to this value (swiglu_limit)
//   CLAMP_MIN  — 0: no clamp; <0: clamp up projection symmetrically to [CLAMP_MIN, -CLAMP_MIN]
//   UP_BIAS    — additive bias applied to up projection before gate*up multiply (SwigluOAI: 1.0)
//
// Bindings:
//   0: x        [K] f16
//   1: gate_w   [N, K/2] u32 (two f16 per u32)
//   2: up_w     [N, K/2] u32
//   3: ffn_act  [N] f16  (output — activated gate·up product)

override K:          u32 = 2560u;
override N:          u32 = 9728u;
override GELU:       u32 = 0u;   // 0 = SiLU, 1 = tanh-GELU (Gemma) [legacy; use ACTIVATION]
override CLAMP_MAX:  f32 = 0.0;  // 0 = no clamp; >0 = clamp gate activation (upper only) to this value
override CLAMP_MIN:  f32 = 0.0;  // 0 = no clamp; <0 = clamp up projection symmetrically to [CLAMP_MIN, -CLAMP_MIN]
override UP_BIAS:    f32 = 0.0;  // additive bias on up projection before multiply (SwigluOAI: 1.0)
override ACTIVATION: u32 = 0u;   // 0 = SiLU, 1 = x*sigmoid(1.702*x) (SwigluOAI), 2 = ReLU²

@group(0) @binding(0) var<storage, read>       x       : array<f16>;
@group(0) @binding(1) var<storage, read>       gate_w  : array<u32>;
@group(0) @binding(2) var<storage, read>       up_w    : array<u32>;
@group(0) @binding(3) var<storage, read_write> ffn_act : array<f16>;

fn activate(x: f32) -> f32 {
    var a: f32;
    if (ACTIVATION == 1u) {
        // GELU: x * Φ(x) approximated as x * sigmoid(1.702 * x)
        a = x * (1.0f / (1.0f + exp(-1.702f * x)));
    } else if (ACTIVATION == 2u) {
        // ReLU² (squared ReLU, Nemotron-3)
        let r = max(x, 0.0f);
        a = r * r;
    } else if (GELU == 1u) {
        // tanh-GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715*x³)))  [legacy]
        let c   = 0.7978845608f;  // sqrt(2/π)
        let val = c * (x + 0.044715f * x * x * x);
        a = 0.5f * x * (1.0f + tanh(val));
    } else {
        // SiLU (default): x * sigmoid(x)
        a = x * (1.0f / (1.0f + exp(-x)));
    }
    if (CLAMP_MAX > 0.0f) { a = min(a, CLAMP_MAX); }
    return a;
}

var<workgroup> sh_gate: array<f32, 256>;
var<workgroup> sh_up:   array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) lid:  vec3<u32>,
    @builtin(workgroup_id)        wgid: vec3<u32>,
) {
    let row = wgid.x;
    let tid = lid.x;

    let row_base = row * K / 2u;
    var g_acc: f32 = 0.0;
    var u_acc: f32 = 0.0;
    var k = tid * 2u;
    loop {
        if (k >= K) { break; }
        let ui  = row_base + k / 2u;
        let gw  = unpack2x16float(gate_w[ui]);
        let uw  = unpack2x16float(up_w[ui]);
        let x0  = f32(x[k]);
        g_acc += gw.x * x0;
        u_acc += uw.x * x0;
        if (k + 1u < K) {
            let x1 = f32(x[k + 1u]);
            g_acc += gw.y * x1;
            u_acc += uw.y * x1;
        }
        k += 512u;
    }

    sh_gate[tid] = g_acc;
    sh_up[tid]   = u_acc;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            sh_gate[tid] += sh_gate[tid + stride];
            sh_up[tid]   += sh_up[tid + stride];
        }
        workgroupBarrier();
        stride /= 2u;
    }

    if (tid == 0u) {
        let g = sh_gate[0];
        var u = sh_up[0] + UP_BIAS;
        if (CLAMP_MIN < 0.0) { u = clamp(u, CLAMP_MIN, -CLAMP_MIN); }
        ffn_act[row] = f16(clamp(activate(g) * u, -65504.0, 65504.0));
    }
}
