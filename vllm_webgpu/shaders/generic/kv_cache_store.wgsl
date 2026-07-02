enable f16;

override BLOCK_SIZE: u32   = 16u;
override NUM_KV_HEADS: u32 = 8u;
override HEAD_DIM: u32     = 128u;

// input: [num_tokens, num_kv_heads, head_dim]  f16
// cache: [num_blocks, block_size, num_kv_heads, head_dim]  f16
// slot_mapping: [num_tokens]  u32  (physical_block * BLOCK_SIZE + slot_within_block)
// HEAD_DIM must be even (all practical models: 64, 128, 256 satisfy this).

// Bind as vec2<f16> (4-byte aligned pairs) to halve the store instruction count.
// The underlying memory layout is identical; this is a pure reinterpret.
@group(0) @binding(0) var<storage, read>       kv_in        : array<vec2<f16>>;
@group(0) @binding(1) var<storage, read_write> kv_cache     : array<vec2<f16>>;
@group(0) @binding(2) var<storage, read>       slot_mapping : array<u32>;

// Halve workgroup size: each thread copies 2 f16 values as one vec2<f16>.
@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(local_invocation_id)  lid  : vec3<u32>,
    @builtin(workgroup_id)         wgid : vec3<u32>,
) {
    let token_idx = wgid.x;
    let head_idx  = wgid.y;
    let tid       = lid.x;   // iterates over HEAD_DIM/2 vec2 pairs

    let slot         = slot_mapping[token_idx];
    let block_idx    = slot / BLOCK_SIZE;
    let block_offset = slot % BLOCK_SIZE;

    // Divide bases by 2 because array element is now vec2<f16> (2 f16s per element)
    let half_dim = HEAD_DIM / 2u;
    let src_base = (token_idx * NUM_KV_HEADS + head_idx) * half_dim;
    let dst_base = ((block_idx * BLOCK_SIZE + block_offset) * NUM_KV_HEADS + head_idx) * half_dim;

    var col = tid;
    loop {
        if (col >= half_dim) { break; }
        kv_cache[dst_base + col] = kv_in[src_base + col];
        col += 64u;
    }
}
