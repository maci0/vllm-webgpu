enable f16;

// embedding_lookup_f32.wgsl — token embedding lookup writing f32 output.
// Used by Gemma4 where the residual stream is stored in f32.
// The embedding table is f16 (standard); output is promoted to f32 for precision.

override HIDDEN_DIM: u32 = 4096u;

@group(0) @binding(0) var<storage, read>       table     : array<vec4<f16>>;  // [vocab, hidden/4] f16
@group(0) @binding(1) var<storage, read>       token_ids : array<u32>;
@group(0) @binding(2) var<storage, read_write> output    : array<f32>;        // [num_tokens, hidden] f32

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
    let dst_base  = token_idx * HIDDEN_DIM;

    // Each thread handles up to HIDDEN_DIM/256 elements (stride for large HIDDEN_DIM).
    var col = tid;
    loop {
        if (col >= vec_dim) { break; }
        let v4 = vec4<f32>(table[src_base + col]);
        output[dst_base + col * 4u    ] = v4.x;
        output[dst_base + col * 4u + 1u] = v4.y;
        output[dst_base + col * 4u + 2u] = v4.z;
        output[dst_base + col * 4u + 3u] = v4.w;
        col += 256u;
    }
}
