enable f16;

// matmul_quant_mr4.wgsl — tiled matmul for prefill (M >= 1)
// Workgroup computes MR output rows for a single weight column (output col = wgid.y).
// Input X: [M, K], weights: [N, K/2] packed Q4 or [N, K] packed f16.
// Output:  [M, N]

override K: u32        = 4096u;
override N: u32        = 4096u;
override M: u32        = 1u;     // number of input tokens (batch)
override BLOCK_K: u32  = 32u;
override USE_QUANT: u32 = 1u;
override MR: u32       = 4u;    // micro-tile rows per workgroup

@group(0) @binding(0) var<storage, read>       X       : array<f16>;  // [M, K]
@group(0) @binding(1) var<storage, read>       weights : array<u32>;
@group(0) @binding(2) var<storage, read>       scales  : array<f16>;
@group(0) @binding(3) var<storage, read_write> output  : array<f16>;  // [M, N]

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) gid  : vec3<u32>,
    @builtin(workgroup_id)         wgid : vec3<u32>,
) {
    // wgid.x = which micro-tile block of M rows, wgid.y = output column (weight row)
    let out_col   = wgid.y;
    if (out_col >= N) { return; }

    let row_start = wgid.x * MR;

    let blocks    = K / BLOCK_K;
    let row_bytes = K / 2u;

    for (var mr = 0u; mr < MR; mr++) {
        let row = row_start + mr;
        if (row >= M) { break; }

        var acc: f32 = 0.0;

        if (USE_QUANT != 0u) {
            // Q4_K_M path: dequant weight row `out_col`, dot with X row `row`.
            for (var blk = 0u; blk < blocks; blk++) {
                let scale     = f32(scales[out_col * blocks + blk]);
                let blk_start = out_col * row_bytes + blk * (BLOCK_K / 2u);
                for (var b = 0u; b < BLOCK_K / 2u; b++) {
                    let byte_idx = blk_start + b;
                    let packed   = (weights[byte_idx / 4u] >> ((byte_idx % 4u) * 8u)) & 0xFFu;
                    let lo       = f32(i32(packed & 0x0Fu) - 8);
                    let hi       = f32(i32(packed >> 4u) - 8);
                    let k_base   = blk * BLOCK_K + b * 2u;
                    acc += lo * scale * f32(X[row * K + k_base]);
                    acc += hi * scale * f32(X[row * K + k_base + 1u]);
                }
            }
        } else {
            // f16 path: weight row `out_col`, dot with X row `row`.
            for (var k = 0u; k < K; k += 2u) {
                let w_u32 = weights[(out_col * K + k) / 2u];
                let w_vec = unpack2x16float(w_u32);  // returns vec2<f32>
                acc += w_vec.x * f32(X[row * K + k]);
                if (k + 1u < K) { acc += w_vec.y * f32(X[row * K + k + 1u]); }
            }
        }

        output[row * N + out_col] = f16(acc);
    }
}
