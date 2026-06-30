enable f16;

override N: u32 = 4096u;

@group(0) @binding(0) var<storage, read>       residual : array<f16>;
@group(0) @binding(1) var<storage, read>       ple_out  : array<f16>;
@group(0) @binding(2) var<storage, read>       scale    : array<f16>;   // [1] scalar
@group(0) @binding(3) var<storage, read_write> output   : array<f16>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= N) { return; }
    output[i] = f16(f32(residual[i]) + f32(scale[0]) * f32(ple_out[i]));
}
