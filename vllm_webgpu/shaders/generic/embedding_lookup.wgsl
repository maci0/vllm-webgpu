enable f16;

override HIDDEN_DIM: u32 = 4096u;
override NUM_TOKENS: u32 = 1u;

@group(0) @binding(0) var<storage, read>       table     : array<f16>;  // [vocab, hidden_dim]
@group(0) @binding(1) var<storage, read>       token_ids : array<u32>;
@group(0) @binding(2) var<storage, read_write> output    : array<f16>;  // [num_tokens, hidden_dim]

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) lid  : vec3<u32>,
    @builtin(workgroup_id)        wgid : vec3<u32>,
) {
    let token_idx = wgid.x;
    let tid       = lid.x;
    let vocab_row = token_ids[token_idx];
    let src_base  = vocab_row * HIDDEN_DIM;
    let dst_base  = token_idx * HIDDEN_DIM;

    var col = tid;
    loop {
        if (col >= HIDDEN_DIM) { break; }
        output[dst_base + col] = table[src_base + col];
        col += 256u;
    }
}
