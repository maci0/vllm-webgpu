enable f16;

// relu_sq.wgsl — element-wise squared ReLU activation: y = max(0, x)^2.
//
// Used for Nemotron-H MLP layers (mlp_hidden_act = "relu2").
// Matches ReLUSquaredActivation() in vLLM.
//
// Dispatch: (ceil(N / WG_SIZE), 1, 1)

override N:       u32 = 12544u;
override WG_SIZE: u32 = 256u;

@group(0) @binding(0) var<storage, read>       x_in  : array<f16>;
@group(0) @binding(1) var<storage, read_write> y_out : array<f16>;

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= N) { return; }
    let x = f32(x_in[i]);
    let r = max(0.0f, x);
    y_out[i] = f16(r * r);
}
