enable f16;

// matmul_quant_mr4.wgsl — tiled GEMM for prefill (M >= 1).
//
// Input X: [M, K], weights: [N, K/2] packed Q4 or [N, K] packed f16.
// Output:  [M, N]
//
// Dispatch: (N, M, 1) — one workgroup per (output_col, input_row) pair.
//
// 256 threads per workgroup cooperate on a single dot-product:
// thread t handles K elements [t, t+256, t+512, ...], tree-reduces to scalar.
// This is 256x more parallel than the prior workgroup_size(1,1,1) implementation.
//
// Arithmetic intensity: K MACs / (K/2 bytes weight + K*2 bytes X) ≈ 0.33 FLOP/byte.
// Memory-bandwidth limited. All 256 threads run concurrently, saturating all CUs.

override K: u32        = 4096u;
override N: u32        = 4096u;
override M: u32        = 1u;
override BLOCK_K: u32  = 32u;
override USE_QUANT: u32 = 1u;
override MR: u32       = 4u;   // unused (kept for API compatibility)
override GROUP_K: u32  = 128u; // quantization group size for USE_QUANT=3 (GPTQ)

@group(0) @binding(0) var<storage, read>       X       : array<f16>;  // [M, K]
@group(0) @binding(1) var<storage, read>       weights : array<u32>;
@group(0) @binding(2) var<storage, read>       scales  : array<f32>;
@group(0) @binding(3) var<storage, read_write> output  : array<f16>;  // [M, N]

var<workgroup> sh_acc: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id)        wgid : vec3<u32>,
    @builtin(local_invocation_id) lid  : vec3<u32>,
) {
    let out_col = wgid.x;   // weight row = output neuron
    let in_row  = wgid.y;   // input token row
    let tid     = lid.x;

    if (out_col >= N || in_row >= M) { return; }

    var acc: f32 = 0.0;

    if (USE_QUANT == 3u) {
        // GPTQ INT4 path: weights[N, K//8] INT32, scales[G, N] f32, zero_point=8.
        // Mirrors the split-K GEMV path in matmul_quant.wgsl (USE_QUANT=3) but adds
        // the in_row dimension so all M input tokens are processed in one dispatch.
        let K8 = K / 8u;
        var q_step = tid;
        loop {
            if (q_step >= K8) { break; }
            let k_base = q_step * 8u;

            // Coalesced read: weights[out_col, q_step] in [N, K//8] layout.
            let q = weights[out_col * K8 + q_step];

            // Scale for this K-group: scales[grp, out_col] in [G, N] layout.
            let grp = q_step / (GROUP_K / 8u);
            let sc  = scales[grp * N + out_col];

            // Unpack 8 nibbles (zero_point = 8) and accumulate.
            acc += (f32(i32( q        & 0xFu) - 8) * sc) * f32(X[in_row * K + k_base      ]);
            acc += (f32(i32((q >>  4u)& 0xFu) - 8) * sc) * f32(X[in_row * K + k_base + 1u]);
            acc += (f32(i32((q >>  8u)& 0xFu) - 8) * sc) * f32(X[in_row * K + k_base + 2u]);
            acc += (f32(i32((q >> 12u)& 0xFu) - 8) * sc) * f32(X[in_row * K + k_base + 3u]);
            acc += (f32(i32((q >> 16u)& 0xFu) - 8) * sc) * f32(X[in_row * K + k_base + 4u]);
            acc += (f32(i32((q >> 20u)& 0xFu) - 8) * sc) * f32(X[in_row * K + k_base + 5u]);
            acc += (f32(i32((q >> 24u)& 0xFu) - 8) * sc) * f32(X[in_row * K + k_base + 6u]);
            acc += (f32(i32((q >> 28u)& 0xFu) - 8) * sc) * f32(X[in_row * K + k_base + 7u]);

            q_step += 256u;
        }
    } else if (USE_QUANT != 0u) {
        // Symmetric Q4 path: one scale per BLOCK_K weights.
        let blocks    = K / BLOCK_K;
        let row_bytes = K / 2u;

        // Each thread strides across blocks at BLOCK_K-granularity.
        // For K=4096, BLOCK_K=32: 128 blocks. Thread 0 handles blocks 0, 256 (mod 128=0 again)
        // → actually each thread handles consecutive blocks stepping by 256.
        // Since 128 < 256, many threads handle 0 or 1 blocks; work is load-balanced fine.
        var blk = tid;
        loop {
            if (blk >= blocks) { break; }
            let scale     = scales[out_col * blocks + blk];
            let blk_start = out_col * row_bytes + blk * (BLOCK_K / 2u);
            var block_acc: f32 = 0.0;
            for (var b = 0u; b < BLOCK_K / 8u; b++) {
                let u  = weights[blk_start / 4u + b];
                let x0 = blk * BLOCK_K + b * 8u;
                let b0 = u & 0xFFu;
                block_acc += f32(i32(b0 & 0x0Fu) - 8) * f32(X[in_row * K + x0    ]);
                block_acc += f32(i32(b0 >> 4u)   - 8) * f32(X[in_row * K + x0 + 1u]);
                let b1 = (u >>  8u) & 0xFFu;
                block_acc += f32(i32(b1 & 0x0Fu) - 8) * f32(X[in_row * K + x0 + 2u]);
                block_acc += f32(i32(b1 >> 4u)   - 8) * f32(X[in_row * K + x0 + 3u]);
                let b2 = (u >> 16u) & 0xFFu;
                block_acc += f32(i32(b2 & 0x0Fu) - 8) * f32(X[in_row * K + x0 + 4u]);
                block_acc += f32(i32(b2 >> 4u)   - 8) * f32(X[in_row * K + x0 + 5u]);
                let b3 = u >> 24u;
                block_acc += f32(i32(b3 & 0x0Fu) - 8) * f32(X[in_row * K + x0 + 6u]);
                block_acc += f32(i32(b3 >> 4u)   - 8) * f32(X[in_row * K + x0 + 7u]);
            }
            acc += block_acc * scale;
            blk += 256u;
        }
    } else {
        // f16 path: each thread strides 2 f16 values at a time across K.
        var k = tid * 2u;
        loop {
            if (k >= K) { break; }
            let w_u32 = weights[(out_col * K + k) / 2u];
            let w_vec = unpack2x16float(w_u32);
            acc += w_vec.x * f32(X[in_row * K + k]);
            if (k + 1u < K) { acc += w_vec.y * f32(X[in_row * K + k + 1u]); }
            k += 512u;
        }
    }

    // 8-step tree reduction (log2(256) = 8 steps).
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
        output[in_row * N + out_col] = f16(sh_acc[0]);
    }
}
