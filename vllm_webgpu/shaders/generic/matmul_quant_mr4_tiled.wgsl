enable f16;

// matmul_quant_mr4_tiled.wgsl — tiled GEMM for large-N matrices (e.g. LM head).
//
// Drop-in companion to matmul_quant_mr4.wgsl for the case where N exceeds the
// WebGPU per-dimension dispatch limit of 65535.  matmul_quant_mr4 dispatches
// (N, M, 1) workgroups — one workgroup per (out_col, in_row) pair — which fails
// when N = vocab_size = 256128.
//
// This shader flips the indexing: dispatch ((N + 255) / 256, M, 1).
// wgid.x is a tile index; thread tid handles out_col = wgid.x * 256 + tid.
// Each thread independently accumulates its own dot product over K with no
// shared-memory reduction.  Total arithmetic is identical; cache behaviour
// differs (256 threads read 256 distinct weight rows rather than cooperating on
// one), but this is correct and acceptable for the single LM-head call.
//
// Input  X:       [M, K]    f16
// Weights:        [N, K/2]  u32 (packed f16 or Q4)
// Scales:         per-quant layout (see USE_QUANT branches)
// Output:         [M, N]    f16

override K: u32        = 4096u;
override N: u32        = 4096u;
override M: u32        = 1u;
override BLOCK_K: u32  = 32u;
override USE_QUANT: u32 = 1u;
override GROUP_K: u32  = 128u;  // GPTQ group size (USE_QUANT=3)

@group(0) @binding(0) var<storage, read>       X       : array<f16>;  // [M, K]
@group(0) @binding(1) var<storage, read>       weights : array<u32>;
@group(0) @binding(2) var<storage, read>       scales  : array<f32>;
@group(0) @binding(3) var<storage, read_write> output  : array<f16>;  // [M, N]

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id)        wgid : vec3<u32>,
    @builtin(local_invocation_id) lid  : vec3<u32>,
) {
    let out_col = wgid.x * 256u + lid.x;   // tiled: each thread = one output column
    let in_row  = wgid.y;                   // input token row

    if (out_col >= N || in_row >= M) { return; }

    var acc: f32 = 0.0;

    if (USE_QUANT == 3u) {
        // GPTQ INT4: weights[N, K//8] INT32, scales[G, N] f32, zero_point=8.
        let K8 = K / 8u;
        for (var q_step = 0u; q_step < K8; q_step++) {
            let k_base = q_step * 8u;
            let q  = weights[out_col * K8 + q_step];
            let grp = q_step / (GROUP_K / 8u);
            let sc  = scales[grp * N + out_col];

            acc += (f32(i32( q        & 0xFu) - 8) * sc) * f32(X[in_row * K + k_base     ]);
            acc += (f32(i32((q >>  4u)& 0xFu) - 8) * sc) * f32(X[in_row * K + k_base + 1u]);
            acc += (f32(i32((q >>  8u)& 0xFu) - 8) * sc) * f32(X[in_row * K + k_base + 2u]);
            acc += (f32(i32((q >> 12u)& 0xFu) - 8) * sc) * f32(X[in_row * K + k_base + 3u]);
            acc += (f32(i32((q >> 16u)& 0xFu) - 8) * sc) * f32(X[in_row * K + k_base + 4u]);
            acc += (f32(i32((q >> 20u)& 0xFu) - 8) * sc) * f32(X[in_row * K + k_base + 5u]);
            acc += (f32(i32((q >> 24u)& 0xFu) - 8) * sc) * f32(X[in_row * K + k_base + 6u]);
            acc += (f32(i32((q >> 28u)& 0xFu) - 8) * sc) * f32(X[in_row * K + k_base + 7u]);
        }
    } else if (USE_QUANT != 0u) {
        // Symmetric Q4: one scale per BLOCK_K weights.
        let blocks    = K / BLOCK_K;
        let row_bytes = K / 2u;
        for (var blk = 0u; blk < blocks; blk++) {
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
        }
    } else {
        // f16 path: weights[N, K] stored as [N, K/2] u32 (two packed f16 per u32).
        for (var k = 0u; k < K; k += 2u) {
            let w_u32 = weights[(out_col * K + k) / 2u];
            let w_vec = unpack2x16float(w_u32);
            acc += w_vec.x * f32(X[in_row * K + k]);
            if (k + 1u < K) { acc += w_vec.y * f32(X[in_row * K + k + 1u]); }
        }
    }

    output[in_row * N + out_col] = f16(acc);
}
