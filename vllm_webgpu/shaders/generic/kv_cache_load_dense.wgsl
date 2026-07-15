enable f16;

// kv_cache_load_dense.wgsl — Load K and V from a paged KV cache into dense buffers.
//
// Inverse of kv_cache_store_both: reads slot-mapped paged cache entries and
// writes them to contiguous token-major buffers for use with flash_attn_prefill.
// Used for KV-shared layers during batch prefill: the shared layer reads from
// the target layer's paged cache rather than its own freshly projected K/V.
//
// Cache layout (same as kv_cache_store_both output):
//   k_cache[(slot * NUM_KV_HEADS + head) * HEAD_DIM/2 + col]  (vec2<f16>)
// where slot = slot_mapping[token_idx].
//
// Dense output layout:
//   k_out[(token_idx * NUM_KV_HEADS + head) * HEAD_DIM/2 + col]  (vec2<f16>)
//
// Dispatch: (num_tokens, NUM_KV_HEADS, 1) — matches kv_cache_store_both.
// Each thread copies 2 F16 values per iteration for both K and V.

override BLOCK_SIZE:   u32 = 16u;
override NUM_KV_HEADS: u32 = 8u;
override HEAD_DIM:     u32 = 128u;

@group(0) @binding(0) var<storage, read>       k_cache      : array<vec2<f16>>;
@group(0) @binding(1) var<storage, read>       v_cache      : array<vec2<f16>>;
@group(0) @binding(2) var<storage, read>       slot_mapping : array<u32>;
@group(0) @binding(3) var<storage, read_write> k_out        : array<vec2<f16>>;
@group(0) @binding(4) var<storage, read_write> v_out        : array<vec2<f16>>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(local_invocation_id) lid:  vec3<u32>,
    @builtin(workgroup_id)        wgid: vec3<u32>,
) {
    let token_idx = wgid.x;
    let head_idx  = wgid.y;
    let tid       = lid.x;

    let slot     = slot_mapping[token_idx];
    let half_dim = HEAD_DIM / 2u;

    // Paged cache: slot is the physical slot index (block_idx * BLOCK_SIZE + block_offset).
    // kv_cache_store_both stores at: (slot * NUM_KV_HEADS + head_idx) * half_dim
    let src_base = (slot * NUM_KV_HEADS + head_idx) * half_dim;
    let dst_base = (token_idx * NUM_KV_HEADS + head_idx) * half_dim;

    var col = tid;
    loop {
        if (col >= half_dim) { break; }
        k_out[dst_base + col] = k_cache[src_base + col];
        v_out[dst_base + col] = v_cache[src_base + col];
        col += 64u;
    }
}
