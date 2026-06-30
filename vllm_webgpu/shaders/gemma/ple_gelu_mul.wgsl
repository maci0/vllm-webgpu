enable f16;

override N: u32 = 4096u;

@group(0) @binding(0) var<storage, read>       gate   : array<f16>;
@group(0) @binding(1) var<storage, read>       up     : array<f16>;
@group(0) @binding(2) var<storage, read_write> output : array<f16>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= N) { return; }
    let u = f32(up[i]);
    output[i] = f16(f32(gate[i]) * (u / (1.0 + exp(-u))));
}
