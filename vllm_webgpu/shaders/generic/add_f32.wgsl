enable f16;

// add_f32.wgsl — residual add with f32 output for numerical stability.
// Used by Gemma4 where large output_norm weights (max 600) cause f16 residual saturation.
//
// output (f32) = a (f32) + SCALE * b (f16)
// N must be divisible by 4 for the vec4 path.

override N:     u32 = 256u;
override SCALE: f32 = 1.0;

@group(0) @binding(0) var<storage, read>       a      : array<vec4<f32>>;  // f32 residual
@group(0) @binding(1) var<storage, read>       b      : array<vec4<f16>>;  // f16 sublayer output
@group(0) @binding(2) var<storage, read_write> output : array<vec4<f32>>;  // f32 residual out

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= N / 4u) { return; }
    output[i] = a[i] + vec4<f32>(b[i]) * vec4<f32>(SCALE);
}
