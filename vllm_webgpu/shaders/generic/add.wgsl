enable f16;

// N: total element count; must be divisible by 4 for vec4 path.
// Dispatch ceil(N/4 / 256) workgroups (not ceil(N/256)) so each thread handles 4 elements.
override N: u32 = 256u;

@group(0) @binding(0) var<storage, read>       a      : array<vec4<f16>>;
@group(0) @binding(1) var<storage, read>       b      : array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> output : array<vec4<f16>>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i    = gid.x;
    let vec_n = N / 4u;
    if (i < vec_n) {
        output[i] = a[i] + b[i];
    }
}
