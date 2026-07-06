enable f16;

// linear_attn_norm_gate.wgsl — post-GDN per-head RMSNorm + SiLU gate.
//
// Fuses two operations on the GDN output:
//   1. Per-head RMSNorm: normed[h][d] = gdn[h][d] / rms(gdn[h]) * norm_weight[d]
//   2. SiLU gate:        out[h][d] = normed[h][d] * silu(z[h][d])
//
// Matches Qwen3_5RMSNormGated which applies silu(z) = z * sigmoid(z), not sigmoid(z).
//
// Dispatch: (NUM_V_HEADS, 1, 1) — one workgroup per value head, V_DIM threads.
//
// Shared memory: sh_sq[V_DIM] for the per-head RMS reduction.

override NUM_V_HEADS: u32 = 32u;
override V_DIM: u32       = 128u;  // value head dimension

@group(0) @binding(0) var<storage, read>       gdn_in      : array<f16>; // [NUM_V_HEADS, V_DIM]
@group(0) @binding(1) var<storage, read>       norm_weight : array<f16>; // [V_DIM] shared across heads
@group(0) @binding(2) var<storage, read>       z_in        : array<f16>; // [NUM_V_HEADS * V_DIM]
@group(0) @binding(3) var<storage, read_write> output      : array<f16>; // [NUM_V_HEADS, V_DIM]

var<workgroup> sh_sq: array<f32, V_DIM>;

@compute @workgroup_size(V_DIM, 1, 1)
fn main(
    @builtin(workgroup_id)        wgid: vec3<u32>,
    @builtin(local_invocation_id) lid:  vec3<u32>,
) {
    let vh  = wgid.x;
    let tid = lid.x;
    let base = vh * V_DIM;
    let eps = 1e-6f;

    let v = f32(gdn_in[base + tid]);

    sh_sq[tid] = v * v;
    workgroupBarrier();

    var stride = V_DIM / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) { sh_sq[tid] += sh_sq[tid + stride]; }
        workgroupBarrier();
        stride /= 2u;
    }

    let rms_inv = inverseSqrt(sh_sq[0] / f32(V_DIM) + eps);
    let normed = v * rms_inv * f32(norm_weight[tid]);

    // SiLU gate: z * sigmoid(z)
    let z = f32(z_in[base + tid]);
    let gate = z / (1.0 + exp(-z));

    output[base + tid] = f16(clamp(normed * gate, -65504.0, 65504.0));
}
