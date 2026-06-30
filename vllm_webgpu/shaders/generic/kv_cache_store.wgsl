enable f16;

override BLOCK_SIZE: u32   = 16u;
override NUM_KV_HEADS: u32 = 8u;
override HEAD_DIM: u32     = 128u;

// input: [num_tokens, num_kv_heads, head_dim]  f16
// cache: [num_blocks, block_size, num_kv_heads, head_dim]  f16
// slot_mapping: [num_tokens]  u32  (physical_block * BLOCK_SIZE + slot_within_block)

@group(0) @binding(0) var<storage, read>       kv_in        : array<f16>;
@group(0) @binding(1) var<storage, read_write> kv_cache     : array<f16>;
@group(0) @binding(2) var<storage, read>       slot_mapping : array<u32>;

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(local_invocation_id)  lid  : vec3<u32>,
    @builtin(workgroup_id)         wgid : vec3<u32>,
) {
    let token_idx = wgid.x;
    let head_idx  = wgid.y;
    let tid       = lid.x;   // iterates over HEAD_DIM

    let slot         = slot_mapping[token_idx];
    let block_idx    = slot / BLOCK_SIZE;
    let block_offset = slot % BLOCK_SIZE;

    let src_base = (token_idx * NUM_KV_HEADS + head_idx) * HEAD_DIM;
    let dst_base = ((block_idx * BLOCK_SIZE + block_offset) * NUM_KV_HEADS + head_idx) * HEAD_DIM;

    var col = tid;
    loop {
        if (col >= HEAD_DIM) { break; }
        kv_cache[dst_base + col] = kv_in[src_base + col];
        col += 128u;
    }
}
