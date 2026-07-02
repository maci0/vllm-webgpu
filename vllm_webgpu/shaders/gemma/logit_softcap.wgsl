enable f16;

override N: u32   = 256256u;
override CAP: f32 = 30.0;

@group(0) @binding(0) var<storage, read>       input  : array<f16>;
@group(0) @binding(1) var<storage, read_write> output : array<f16>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= N) { return; }
    let inv_cap: f32 = 1.0 / CAP;
    let v = f32(input[i]) * inv_cap;
    // Numerically stable tanh: branch on |v| to avoid exp overflow.
    // For |v| >= 20: tanh(v) is within 2e-9 of ±1.0, so return ±1.0 exactly.
    // For |v| < 20:  exp(2v) is in (exp(-40), exp(40)) = (4.1e-18, 2.4e17),
    //                always finite in f32, so (e2v-1)/(e2v+1) is well-defined.
    var t: f32;
    if (v >= 20.0) {
        t = 1.0;
    } else if (v <= -20.0) {
        t = -1.0;
    } else {
        let e2v = exp(2.0 * v);
        t = (e2v - 1.0) / (e2v + 1.0);
    }
    output[i] = f16(t * CAP);
}
