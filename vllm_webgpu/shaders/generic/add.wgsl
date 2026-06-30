enable f16;

override N: u32 = 256u;

@group(0) @binding(0) var<storage, read>       a      : array<f16>;
@group(0) @binding(1) var<storage, read>       b      : array<f16>;
@group(0) @binding(2) var<storage, read_write> output : array<f16>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < N) {
        output[i] = a[i] + b[i];
    }
}
