enable f16;

// matmul_quant.wgsl — GEMV for decode (M=1)
// Weight layout for Q4_K_M (USE_QUANT=1):
//   weights: [N, K/2] packed as u8 in u32 array (two 4-bit nibbles per byte)
//   scales:  [N, K/BLOCK_K] as f16
// Weight layout for f16 (USE_QUANT=0):
//   weights: [N, K] packed as f16 in u32 array (two f16 values per u32)

override K: u32         = 4096u;
override N: u32         = 4096u;
override BLOCK_K: u32   = 32u;   // Q4_K_M block size
override USE_QUANT: u32 = 1u;    // 1=Q4_K_M, 0=f16

// Note: workgroup shared memory for x[] was removed. A static array<f32, 4096> fit the WebGPU
// 16384-byte workgroup limit, but K can be up to intermediate_size (14336 for Llama-3-8B),
// which would require 57344 bytes — 3.5x the minimum limit. Reading x[] directly from global
// memory is correct; the GPU L2 cache amortises the cost across all N threads in the workgroup.

@group(0) @binding(0) var<storage, read>       x       : array<f16>;  // [K]
@group(0) @binding(1) var<storage, read>       weights : array<u32>;  // raw bytes, reinterpreted
@group(0) @binding(2) var<storage, read>       scales  : array<f16>;  // [N, K/BLOCK_K], unused when USE_QUANT=0
@group(0) @binding(3) var<storage, read_write> output  : array<f16>;  // [N]

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let row = gid.x;
    if (row >= N) { return; }

    var acc: f32 = 0.0;

    if (USE_QUANT != 0u) {
        // Q4_K_M path: dequant on the fly
        let blocks    = K / BLOCK_K;
        let row_bytes = K / 2u;
        for (var blk = 0u; blk < blocks; blk++) {
            let scale     = f32(scales[row * blocks + blk]);
            let blk_start = row * row_bytes + blk * (BLOCK_K / 2u);
            for (var b = 0u; b < BLOCK_K / 2u; b++) {
                let byte_idx = blk_start + b;
                let packed   = (weights[byte_idx / 4u] >> ((byte_idx % 4u) * 8u)) & 0xFFu;
                let lo       = f32(i32(packed & 0x0Fu) - 8);
                let hi       = f32(i32(packed >> 4u) - 8);
                let k_base   = blk * BLOCK_K + b * 2u;
                acc += lo * scale * f32(x[k_base]);
                acc += hi * scale * f32(x[k_base + 1u]);
            }
        }
    } else {
        // f16 path: each u32 holds two packed f16 values (lo=bits[15:0], hi=bits[31:16])
        for (var k = 0u; k < K; k += 2u) {
            let w_u32 = weights[(row * K + k) / 2u];
            let w_vec = unpack2x16float(w_u32);
            acc += w_vec.x * f32(x[k]);
            if (k + 1u < K) { acc += w_vec.y * f32(x[k + 1u]); }
        }
    }

    output[row] = f16(acc);
}
