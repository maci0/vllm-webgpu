enable f16;

override HIDDEN_DIM: u32 = 4096u;
// NUM_TOKENS is handled by dispatch workgroups, not a shader constant.
// VEC_DIM: number of vec4<f16> elements per row (= HIDDEN_DIM / 4).
// HIDDEN_DIM must be divisible by 4 (4096, 3584, etc. all satisfy this).

@group(0) @binding(0) var<storage, read>       table     : array<vec4<f16>>;  // [vocab, hidden_dim/4]
@group(0) @binding(1) var<storage, read>       token_ids : array<u32>;
@group(0) @binding(2) var<storage, read_write> output    : array<vec4<f16>>;  // [num_tokens, hidden_dim/4]

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) lid  : vec3<u32>,
    @builtin(workgroup_id)        wgid : vec3<u32>,
) {
    let token_idx = wgid.x;
    let tid       = lid.x;
    let vocab_row = token_ids[token_idx];
    let vec_dim   = HIDDEN_DIM / 4u;
    let src_base  = vocab_row * vec_dim;
    let dst_base  = token_idx * vec_dim;

    var col = tid;
    loop {
        if (col >= vec_dim) { break; }
        output[dst_base + col] = table[src_base + col];
        col += 256u;
    }
}
