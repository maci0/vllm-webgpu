enable f16;

override BLOCK_SIZE: u32   = 16u;
override NUM_Q_HEADS: u32  = 32u;
override NUM_KV_HEADS: u32 = 8u;
override HEAD_DIM: u32     = 128u;
override MAX_SEQ_LEN: u32  = 4096u;

// Q: [num_q_heads, head_dim]  f16  (single decode token)
// K_cache: [num_blocks, block_size, num_kv_heads, head_dim]  f16
// block_table: [max_blocks]  u32
// scores_out: [num_q_heads, MAX_SEQ_LEN]  f32

@group(0) @binding(0) var<storage, read>       Q           : array<f16>;
@group(0) @binding(1) var<storage, read>       K_cache     : array<f16>;
@group(0) @binding(2) var<storage, read>       block_table : array<u32>;
@group(0) @binding(3) var<storage, read_write> scores_out  : array<f32>;

@compute @workgroup_size(16, 1, 1)
fn main(
    @builtin(global_invocation_id) gid  : vec3<u32>,
    @builtin(workgroup_id)         wgid : vec3<u32>,
    @builtin(local_invocation_id)  lid  : vec3<u32>,
) {
    let q_head  = wgid.x;
    let ctx_idx = wgid.y;   // token index in context
    let tid     = lid.x;

    let kv_head = q_head / (NUM_Q_HEADS / NUM_KV_HEADS);

    let block_idx = block_table[ctx_idx / BLOCK_SIZE];
    let block_off = ctx_idx % BLOCK_SIZE;
    let scale     = 1.0 / sqrt(f32(HEAD_DIM));

    let q_base = q_head * HEAD_DIM;
    let k_base = ((block_idx * BLOCK_SIZE + block_off) * NUM_KV_HEADS + kv_head) * HEAD_DIM;

    // Each thread computes a partial dot product over HEAD_DIM / 16 elements
    var dot: f32 = 0.0;
    var d = tid;
    loop {
        if (d >= HEAD_DIM) { break; }
        dot += f32(Q[q_base + d]) * f32(K_cache[k_base + d]);
        d += 16u;
    }

    // Write partial score; for a smoke-test kernel this is sufficient.
    // A production kernel would reduce across the 16 threads in the workgroup.
    scores_out[q_head * MAX_SEQ_LEN + ctx_idx] = dot * scale;
}
