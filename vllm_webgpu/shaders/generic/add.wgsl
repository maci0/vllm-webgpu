enable f16;

// N: total element count; must be divisible by 4 for vec4 path.
// Dispatch ceil(N/4 / 256) workgroups so each thread handles 4 elements.
// SCALE: applied to b before adding: output = a + SCALE * b.
//   Default SCALE=1.0 is the normal residual add.
//   For Gemma4, set SCALE = layer_output_scale (e.g. 0.053) to stabilise
//   the residual stream and prevent f16 overflow from large norm weights.
override N:     u32 = 256u;
override SCALE: f32 = 1.0;

@group(0) @binding(0) var<storage, read>       a      : array<vec4<f16>>;
@group(0) @binding(1) var<storage, read>       b      : array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> output : array<vec4<f16>>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= N / 4u) { return; }
    let scale4 = vec4<f32>(SCALE);
    let b_scaled = vec4<f32>(b[i]) * scale4;
    // Clamp before casting to prevent inf from scale * large_b.
    output[i] = a[i] + vec4<f16>(clamp(b_scaled, vec4<f32>(-65504.0), vec4<f32>(65504.0)));
}
