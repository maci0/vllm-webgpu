enable f16;

// fused_qkv.wgsl — Compute Q, K, and V projections in a single dispatch.
//
// Dispatch (Q_DIM + 2*KV_DIM, 1, 1): one workgroup per output row.
//   rows [0,        Q_DIM)      → q_weight
//   rows [Q_DIM,    Q_DIM+KV_DIM) → k_weight
//   rows [Q_DIM+KV_DIM, total)  → v_weight
//
// Output qkv_out: [Q_DIM + 2*KV_DIM] f16, laid out as [Q | K | V].
// fused_per_head_norm_rope reads Q from offset 0, K from offset Q_DIM.
// attn_output / v reads V from offset Q_DIM + KV_DIM.
//
// Bindings:
//   0: x      [K] f16
//   1: q_w    [Q_DIM, K/2] u32  (two f16 per u32)
//   2: k_w    [KV_DIM, K/2] u32
//   3: v_w    [KV_DIM, K/2] u32
//   4: qkv    [Q_DIM + 2*KV_DIM] f16  (output)

override K:      u32 = 2048u;   // hidden size (input)
override Q_DIM:  u32 = 2048u;   // num_heads * head_dim
override KV_DIM: u32 = 512u;    // num_kv_heads * head_dim

@group(0) @binding(0) var<storage, read>       x      : array<f16>;
@group(0) @binding(1) var<storage, read>       q_w    : array<u32>;
@group(0) @binding(2) var<storage, read>       k_w    : array<u32>;
@group(0) @binding(3) var<storage, read>       v_w    : array<u32>;
@group(0) @binding(4) var<storage, read_write> qkv    : array<f16>;

var<workgroup> sh_acc: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) lid:  vec3<u32>,
    @builtin(workgroup_id)        wgid: vec3<u32>,
) {
    let row = wgid.x;
    let tid = lid.x;

    // Determine which weight matrix and local row index.
    var local_row: u32;
    var acc: f32 = 0.0;

    if (row < Q_DIM) {
        // Q projection
        local_row = row;
        let row_base = local_row * K / 2u;
        var k = tid * 2u;
        loop {
            if (k >= K) { break; }
            let w = unpack2x16float(q_w[row_base + k / 2u]);
            acc += w.x * f32(x[k]);
            if (k + 1u < K) { acc += w.y * f32(x[k + 1u]); }
            k += 512u;
        }
    } else if (row < Q_DIM + KV_DIM) {
        // K projection
        local_row = row - Q_DIM;
        let row_base = local_row * K / 2u;
        var k = tid * 2u;
        loop {
            if (k >= K) { break; }
            let w = unpack2x16float(k_w[row_base + k / 2u]);
            acc += w.x * f32(x[k]);
            if (k + 1u < K) { acc += w.y * f32(x[k + 1u]); }
            k += 512u;
        }
    } else {
        // V projection
        local_row = row - Q_DIM - KV_DIM;
        let row_base = local_row * K / 2u;
        var k = tid * 2u;
        loop {
            if (k >= K) { break; }
            let w = unpack2x16float(v_w[row_base + k / 2u]);
            acc += w.x * f32(x[k]);
            if (k + 1u < K) { acc += w.y * f32(x[k + 1u]); }
            k += 512u;
        }
    }

    sh_acc[tid] = acc;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) { sh_acc[tid] += sh_acc[tid + stride]; }
        workgroupBarrier();
        stride /= 2u;
    }

    if (tid == 0u) {
        qkv[row] = f16(clamp(sh_acc[0], -65504.0, 65504.0));
    }
}
