enable f16;

// kv_cache_store_both.wgsl — Store K and V in a single dispatch.
// Replaces two kv_cache_store calls per layer with one.
// Saves 1 dispatch overhead per transformer layer.
//
// Dispatch: (num_tokens, num_kv_heads, 1) — same as kv_cache_store.
// Each thread copies 2 F16 values (one vec2<f16>) for both K and V.

override BLOCK_SIZE:   u32 = 16u;
override NUM_KV_HEADS: u32 = 8u;
override HEAD_DIM:     u32 = 128u;

@group(0) @binding(0) var<storage, read>       k_in         : array<vec2<f16>>;
@group(0) @binding(1) var<storage, read_write> k_cache      : array<vec2<f16>>;
@group(0) @binding(2) var<storage, read>       v_in         : array<vec2<f16>>;
@group(0) @binding(3) var<storage, read_write> v_cache      : array<vec2<f16>>;
@group(0) @binding(4) var<storage, read>       slot_mapping : array<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(local_invocation_id) lid:  vec3<u32>,
    @builtin(workgroup_id)        wgid: vec3<u32>,
) {
    let token_idx = wgid.x;
    let head_idx  = wgid.y;
    let tid       = lid.x;

    let slot         = slot_mapping[token_idx];
    let block_idx    = slot / BLOCK_SIZE;
    let block_offset = slot % BLOCK_SIZE;

    let half_dim = HEAD_DIM / 2u;
    let src_base = (token_idx * NUM_KV_HEADS + head_idx) * half_dim;
    let dst_base = ((block_idx * BLOCK_SIZE + block_offset) * NUM_KV_HEADS + head_idx) * half_dim;

    var col = tid;
    loop {
        if (col >= half_dim) { break; }
        k_cache[dst_base + col] = k_in[src_base + col];
        v_cache[dst_base + col] = v_in[src_base + col];
        col += 64u;
    }
}
