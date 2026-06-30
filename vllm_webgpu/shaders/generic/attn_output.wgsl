enable f16;

override BLOCK_SIZE: u32   = 16u;
override NUM_Q_HEADS: u32  = 32u;
override NUM_KV_HEADS: u32 = 8u;
override HEAD_DIM: u32     = 128u;
override CTX_LEN: u32      = 4096u;

// scores: [num_q_heads, CTX_LEN]  f16
// V_cache: [num_blocks, block_size, num_kv_heads, head_dim]  f16  (paged)
// block_table: [max_blocks]  u32
// out: [num_q_heads, head_dim]  f16

@group(0) @binding(0) var<storage, read>       scores      : array<f16>;
@group(0) @binding(1) var<storage, read>       V_cache     : array<f16>;
@group(0) @binding(2) var<storage, read>       block_table : array<u32>;
@group(0) @binding(3) var<storage, read_write> out         : array<f16>;

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(local_invocation_id) lid  : vec3<u32>,
    @builtin(workgroup_id)        wgid : vec3<u32>,
) {
    let q_head  = wgid.x;
    let kv_head = q_head / (NUM_Q_HEADS / NUM_KV_HEADS);
    let dim_idx = lid.x;

    if (dim_idx >= HEAD_DIM) { return; }

    var acc: f32 = 0.0;
    var ctx = 0u;
    loop {
        if (ctx >= CTX_LEN) { break; }
        let block_idx = block_table[ctx / BLOCK_SIZE];
        let block_off = ctx % BLOCK_SIZE;
        let v_base    = ((block_idx * BLOCK_SIZE + block_off) * NUM_KV_HEADS + kv_head) * HEAD_DIM;
        acc += f32(scores[q_head * CTX_LEN + ctx]) * f32(V_cache[v_base + dim_idx]);
        ctx += 1u;
    }
    out[q_head * HEAD_DIM + dim_idx] = f16(acc);
}
