enable f16;

// fused_gate_up.wgsl — Compute gate_proj AND up_proj in a single split-K dispatch.
//
// Both projections share the same input x. Computing them together:
//   1. Halves dispatch count vs two separate matmul_quant calls.
//   2. x reads are broadcast (L1-cached across all 256 threads).
//
// Dispatch (N, 1, 1): one workgroup per output row (split-K, same as matmul_quant).
// Thread tid processes k = tid*2, tid*2+512, tid*2+1024, ... (stride 512).
// Output: gate_up_out[0..N-1] = gate, gate_up_out[N..2N-1] = up.
//
// Bindings:
//   0: x        [K] f16
//   1: gate_w   [N, K/2] u32 (two f16 per u32, row-major)
//   2: up_w     [N, K/2] u32
//   3: gate_up  [2*N] f16 output

override K: u32 = 2560u;
override N: u32 = 9728u;   // inter size (output size per projection)

@group(0) @binding(0) var<storage, read>       x        : array<f16>;
@group(0) @binding(1) var<storage, read>       gate_w   : array<u32>;
@group(0) @binding(2) var<storage, read>       up_w     : array<u32>;
@group(0) @binding(3) var<storage, read_write> gate_up  : array<f16>;

var<workgroup> sh_gate: array<f32, 256>;
var<workgroup> sh_up:   array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) lid:  vec3<u32>,
    @builtin(workgroup_id)        wgid: vec3<u32>,
) {
    let row = wgid.x;
    let tid = lid.x;

    // Each thread reads 1 u32 (= 2 f16) from gate and 1 u32 from up per step.
    // x[k], x[k+1] are broadcast (all 256 threads read the same x values).
    // The split-K stride is 512 (= 256 threads × 2 elements per thread per step).
    let row_base = row * K / 2u;  // u32 base for this output row
    var g_acc: f32 = 0.0;
    var u_acc: f32 = 0.0;
    var k = tid * 2u;
    loop {
        if (k >= K) { break; }
        let ui = row_base + k / 2u;
        let gw = unpack2x16float(gate_w[ui]);
        let uw = unpack2x16float(up_w[ui]);
        g_acc += gw.x * f32(x[k]);
        u_acc += uw.x * f32(x[k]);
        if (k + 1u < K) {
            g_acc += gw.y * f32(x[k + 1u]);
            u_acc += uw.y * f32(x[k + 1u]);
        }
        k += 512u;
    }

    sh_gate[tid] = g_acc;
    sh_up[tid]   = u_acc;
    workgroupBarrier();

    // Tree reduction (8 barriers, same cost as matmul_quant split-K).
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
        gate_up[row]     = f16(clamp(sh_gate[0], -65504.0, 65504.0));
        gate_up[N + row] = f16(clamp(sh_up[0],   -65504.0, 65504.0));
    }
}
