// f32_scale_inplace.wgsl — multiply f32 buffer elementwise by a scalar constant.
//
// Used by Gemma4 to apply layer_scalar to the full residual after each decoder layer,
// matching vLLM's `hidden_states = hidden_states * self.layer_scalar` (applied once
// per full decoder layer after both attention and FFN sublayers).
//
// Dispatch ceil(N / 256) workgroups of 256 threads, one thread per element.

override N:     u32 = 4096u;
override SCALE: f32 = 1.0;

@group(0) @binding(0) var<storage, read_write> buf: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < N) { buf[i] = buf[i] * SCALE; }
}
