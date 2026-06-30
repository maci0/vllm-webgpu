enable f16;

override N: u32   = 256256u;
override CAP: f32 = 30.0;

@group(0) @binding(0) var<storage, read>       input  : array<f16>;
@group(0) @binding(1) var<storage, read_write> output : array<f16>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= N) { return; }
    let v = f32(input[i]) / CAP;
    // tanh via (e^2v - 1)/(e^2v + 1) for precision
    let e2v = exp(2.0 * v);
    output[i] = f16(((e2v - 1.0) / (e2v + 1.0)) * CAP);
}
