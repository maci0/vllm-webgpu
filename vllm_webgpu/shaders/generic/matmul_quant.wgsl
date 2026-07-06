enable f16;

// matmul_quant.wgsl — GEMV for decode (M=1)
//
// USE_QUANT=0: f16 weights — weights[N, K] packed as f16 in u32 (two f16 per u32)
// USE_QUANT=1: simple Q4 — weights[N, K/2] packed nibbles (biased ±8), scales[N, K/BLOCK_K] f16
// USE_QUANT=2: GGUF Q4_K — weights in raw Q4_K block format (144 bytes per 256-weight block)
//              Block layout: d(f16,2B) dmin(f16,2B) scales_mins(12B) nibbles(128B)
//              Nibbles are unsigned 0-15; dequant = d*scale*nibble - dmin*min (asymmetric)

override K: u32          = 4096u;
override N: u32          = 4096u;
override BLOCK_K: u32    = 32u;     // block size for USE_QUANT=1 (simple Q4)
override USE_QUANT: u32  = 1u;      // 0=f16, 1=Q4, 2=Q4_K, 3=GPTQ, 4=AWQ, 5=FP8, 6=NVFP4
override GROUP_K: u32    = 128u;    // quantization group size for USE_QUANT=3/4/6
override GLOBAL_SCALE: f32 = 1.0;  // per-tensor scale for USE_QUANT=5 (FP8) and 6 (NVFP4)
// SPLIT_K=1 (default): split-K GEMV — all 256 threads work on ONE output row.
// Dispatch (N, 1, 1) workgroups. Within each workgroup, consecutive threads
// read consecutive weight elements → coalesced on all GPU architectures.
// Mathematically optimal for bandwidth-limited GEMV on NVIDIA/AMD/mobile/Metal.
// 8-barrier workgroup reduction finalises the partial sums.
// SPLIT_K=0: legacy row-per-thread. Dispatch ((N+255)/256, 1, 1) workgroups.
override SPLIT_K: u32   = 1u;

@group(0) @binding(0) var<storage, read>       x       : array<f16>;  // [K]
@group(0) @binding(1) var<storage, read>       weights : array<u32>;  // raw bytes as u32
@group(0) @binding(2) var<storage, read>       scales  : array<f16>;  // [N, K/BLOCK_K] for USE_QUANT=1
@group(0) @binding(3) var<storage, read_write> output  : array<f16>;  // [N]

// Read one byte from the weights u32 array at byte offset `byte_off`.
fn rd_byte(byte_off: u32) -> u32 {
    return (weights[byte_off / 4u] >> ((byte_off % 4u) * 8u)) & 0xFFu;
}

// Read a float16 value stored at byte offset `byte_off` (little-endian).
// unpack2x16float interprets the low 16 bits of a u32 as f16 and returns it as f32.
fn rd_f16(byte_off: u32) -> f32 {
    let lo = rd_byte(byte_off);
    let hi = rd_byte(byte_off + 1u);
    return unpack2x16float(lo | (hi << 8u)).x;
}

// Extract 6-bit scale value for sub-block j (0..7) from the 12 scale bytes at sc_base.
// Algorithm matches llama.cpp get_scale_min_k4.
fn q4k_scale(j: u32, sc_base: u32) -> u32 {
    if (j < 4u) {
        return rd_byte(sc_base + j) & 0x3Fu;
    }
    return (rd_byte(sc_base + j + 4u) & 0x0Fu) | ((rd_byte(sc_base + j - 4u) >> 6u) << 4u);
}

// Extract 6-bit min value for sub-block j (0..7).
fn q4k_min(j: u32, sc_base: u32) -> u32 {
    if (j < 4u) {
        return rd_byte(sc_base + j + 4u) & 0x3Fu;
    }
    return (rd_byte(sc_base + j + 4u) >> 4u) | ((rd_byte(sc_base + j) >> 6u) << 4u);
}

// Shared memory for the split-K reduction (256 partial sums).
var<workgroup> sh_acc: array<f32, 256>;

// ── GPU dequant helpers ──────────────────────────────────────────────────────

// FP8 E4M3 (OCP/NVidia format, exponent bias=7) → f32
// Normal:  (-1)^s * 2^(exp-7) * (1 + mant/8)
// Denorm:  (-1)^s * 2^-6 * mant/8
// NaN:     exp==15 (treated as 0)
fn fp8_to_f32(b: u32) -> f32 {
    let sign = select(1.0f, -1.0f, (b & 0x80u) != 0u);
    let exp  = (b >> 3u) & 0xFu;
    let mant = b & 0x7u;
    if (exp == 0u)  { return sign * f32(mant) * (1.0f / 512.0f); }  // 2^-6 * mant/8
    if (exp == 15u) { return 0.0f; }
    return sign * exp2(f32(i32(exp) - 7)) * (1.0f + f32(mant) * 0.125f);
}

// FP4 E2M1 decode: values = [0,0.5,1,1.5,2,3,4,6] × sign bit
fn fp4_to_f32(code: u32) -> f32 {
    let neg = (code & 0x8u) != 0u;
    var v: f32;
    switch code & 0x7u {
        case 0u: { v = 0.0f; }  case 1u: { v = 0.5f; }
        case 2u: { v = 1.0f; }  case 3u: { v = 1.5f; }
        case 4u: { v = 2.0f; }  case 5u: { v = 3.0f; }
        case 6u: { v = 4.0f; }  default: { v = 6.0f; }
    }
    return select(v, -v, neg);
}

// Extract byte at flat byte-offset `bi` from the u32 weights array.
fn rd_byte_at(bi: u32) -> u32 {
    return (weights[bi / 4u] >> ((bi % 4u) * 8u)) & 0xFFu;
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>,
    @builtin(workgroup_id)         wgid: vec3<u32>,
) {
    let tid = lid.x;

    if (SPLIT_K == 1u) {
        // Split-K GEMV: one workgroup per output row.
        // Supports three weight formats:
        //   USE_QUANT=0: f16 weights [N, K], 512 F16 elements per step
        //   USE_QUANT=3: GPTQ INT4  [K//8, N], 8 nibbles per INT32 (symmetric, zero=8)
        //   USE_QUANT=4: AWQ  INT4  [K, N//8], 8 nibbles per INT32 (symmetric, zero=8)
        // All formats achieve 4-8× bandwidth reduction vs naive f16 row-per-thread.

        let row = wgid.x;
        if (row >= N) { return; }

        var acc: f32 = 0.0;

        if (USE_QUANT == 3u) {
            // GPU GPTQ INT4 dequant (split-K, coalesced).
            // Weight layout: [N, K//8] INT32 (original [K//8, N] TRANSPOSED at load time).
            // Coalesced: all threads in a workgroup read weights[row * K8 + t],
            // i.e. 256 consecutive INT32s per step → 256 cache-line-friendly reads.
            // scales: [G, N] F16 where G = K // GROUP_K.
            // zero_point = 8 (symmetric GPTQ).
            let K8 = K / 8u;
            var q_step = tid;
            loop {
                if (q_step >= K8) { break; }
                let k_base = q_step * 8u;

                // Coalesced read: weight[row, q_step] in [N, K//8] layout
                let q = weights[row * K8 + q_step];

                // Scale for this K-group: scales[grp, row] in [G, N] layout
                let grp = q_step / (GROUP_K / 8u);
                let sc  = f32(scales[grp * N + row]);

                // Unpack 8 nibbles and accumulate (zero_point = 8)
                acc += (f32(i32( q        & 0xFu) - 8) * sc) * f32(x[k_base]);
                acc += (f32(i32((q >>  4u)& 0xFu) - 8) * sc) * f32(x[k_base + 1u]);
                acc += (f32(i32((q >>  8u)& 0xFu) - 8) * sc) * f32(x[k_base + 2u]);
                acc += (f32(i32((q >> 12u)& 0xFu) - 8) * sc) * f32(x[k_base + 3u]);
                acc += (f32(i32((q >> 16u)& 0xFu) - 8) * sc) * f32(x[k_base + 4u]);
                acc += (f32(i32((q >> 20u)& 0xFu) - 8) * sc) * f32(x[k_base + 5u]);
                acc += (f32(i32((q >> 24u)& 0xFu) - 8) * sc) * f32(x[k_base + 6u]);
                acc += (f32(i32((q >> 28u)& 0xFu) - 8) * sc) * f32(x[k_base + 7u]);

                q_step += 256u;
            }
        } else if (USE_QUANT == 4u) {
            // GPU AWQ INT4 dequant (split-K, coalesced).
            // Weight layout: [K, N//8] INT32 stored as [K, N//8].
            // AWQ nibble order within each INT32: positions [0,4,1,5,2,6,3,7]
            // → bit shifts [0, 16, 4, 20, 8, 24, 12, 28].
            // scales: [G, N] F16 where G = K // GROUP_K.
            // zero_point = 8 (symmetric AWQ — asymmetric qzeros handled at load time).
            //
            // All 256 threads in workgroup process the SAME output row (split-K).
            // Thread tid handles k-indices: tid*2, tid*2+512, tid*2+1024, ...
            // At each step, reads qweight[k, row//8] — multiple threads may share
            // the same INT32 (8 consecutive rows share one INT32).
            // 4× bandwidth reduction vs F16 weights.
            let N8 = N / 8u;  // INT32 elements per K row
            // AWQ nibble shift for this output row within its INT32 pack
            let awq_pos = row % 8u;
            // Mapping: position → bit shift using [0,4,1,5,2,6,3,7] nibble order
            var awq_shift: u32;
            switch awq_pos {
                case 0u: { awq_shift = 0u; }
                case 1u: { awq_shift = 16u; }
                case 2u: { awq_shift = 4u; }
                case 3u: { awq_shift = 20u; }
                case 4u: { awq_shift = 8u; }
                case 5u: { awq_shift = 24u; }
                case 6u: { awq_shift = 12u; }
                default: { awq_shift = 28u; }  // case 7u
            }
            var k_awq = tid * 2u;
            loop {
                if (k_awq >= K) { break; }
                let grp = k_awq / GROUP_K;
                let sc  = f32(scales[grp * N + row]);

                let q0  = weights[k_awq * N8 + row / 8u];
                let n0  = f32(i32((q0 >> awq_shift) & 0xFu) - 8);
                acc += n0 * sc * f32(x[k_awq]);

                if (k_awq + 1u < K) {
                    let q1 = weights[(k_awq + 1u) * N8 + row / 8u];
                    let sc1 = f32(scales[(k_awq + 1u) / GROUP_K * N + row]);
                    let n1  = f32(i32((q1 >> awq_shift) & 0xFu) - 8);
                    acc += n1 * sc1 * f32(x[k_awq + 1u]);
                }
                k_awq += 512u;
            }
        } else if (USE_QUANT == 5u) {
            // GPU FP8 E4M3 (split-K, coalesced).
            // weights: [N, K] raw F8 bytes packed 4-per-u32 in binding 1.
            // GLOBAL_SCALE: per-tensor F32 scale (override constant).
            // Coalesced: thread t reads bytes at row*K+t*2 and row*K+t*2+1.
            // Every 2 threads share one u32 → 128 u32 reads per step (coalesced).
            let row_base = row * K;
            var k_fp8 = tid * 2u;
            loop {
                if (k_fp8 >= K) { break; }
                let b0 = rd_byte_at(row_base + k_fp8);
                acc += fp8_to_f32(b0) * GLOBAL_SCALE * f32(x[k_fp8]);
                if (k_fp8 + 1u < K) {
                    let b1 = rd_byte_at(row_base + k_fp8 + 1u);
                    acc += fp8_to_f32(b1) * GLOBAL_SCALE * f32(x[k_fp8 + 1u]);
                }
                k_fp8 += 512u;
            }
        } else if (USE_QUANT == 6u) {
            // GPU NVFP4 (split-K, coalesced).
            // weights:  [N, K//2] packed FP4 bytes, 2 FP4 per byte (lo=k, hi=k+1).
            //           Uploaded as packed u32 in binding 1.
            // scales:   [N, K//16] block scales stored as F16 in binding 2.
            //           Each F16 is one scale for a block of 16 K elements.
            // GLOBAL_SCALE: global F32 scale (override constant).
            // Coalesced: thread t reads weight_byte at row*(K//2)+t → 256 consecutive
            //            bytes per step → 64 u32s → all coalesced.
            let K2  = K / 2u;    // weight bytes per row (2 FP4/byte)
            let K16 = K / 16u;   // block scales per row (16 elements/block)
            var k_fp4 = tid * 2u;
            loop {
                if (k_fp4 >= K) { break; }
                // Block scale for this pair of k-elements
                let blk = k_fp4 / 16u;
                let sc  = f32(scales[row * K16 + blk]) * GLOBAL_SCALE;
                // Packed byte: lo nibble = fp4[k_fp4], hi nibble = fp4[k_fp4+1]
                let byte_idx = row * K2 + k_fp4 / 2u;
                let packed   = rd_byte_at(byte_idx);
                acc += fp4_to_f32(packed & 0xFu) * sc * f32(x[k_fp4]);
                if (k_fp4 + 1u < K) {
                    acc += fp4_to_f32((packed >> 4u) & 0xFu) * sc * f32(x[k_fp4 + 1u]);
                }
                k_fp4 += 512u;
            }
        } else {
            // USE_QUANT=0: f16 split-K — 512 F16 elements (256 u32) per step.
            // Coalesced: all threads read consecutive u32 from weights[row, ...].
            let row_base = row * K;
            var k = tid * 2u;
            loop {
                if (k >= K) { break; }
                let wp = unpack2x16float(weights[(row_base + k) / 2u]);
                acc += wp.x * f32(x[k]);
                if (k + 1u < K) { acc += wp.y * f32(x[k + 1u]); }
                k += 512u;
            }
        }

        // Tree reduction: sum 256 partial accumulators.
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
            output[row] = f16(clamp(sh_acc[0], -65504.0, 65504.0));
        }
        return;
    }

    // Row-per-thread GEMV (SPLIT_K=0): dispatch ((N+255)/256, 1, 1) workgroups.
    let row = gid.x;
    if (row >= N) { return; }

    var acc: f32 = 0.0;

    if (USE_QUANT == 2u) {
        // GGUF Q4_K block decoding.
        // Block structure (144 bytes per 256-weight block):
        //   bytes 0-1:   d     (f16, super-block scale for quantized scales)
        //   bytes 2-3:   dmin  (f16, super-block scale for quantized mins)
        //   bytes 4-15:  12 scale/min bytes (8 × 6-bit scales + 8 × 6-bit mins)
        //   bytes 16-143: 128 qs bytes
        //
        // Nibble layout (DEINTERLEAVED per 64-weight group):
        //   qs[grp*32 + p]:  lo nibble → weight at grp*64 + p  (sub-block grp*2)
        //                    hi nibble → weight at grp*64 + p + 32 (sub-block grp*2+1)
        //   grp = 0..3, p = 0..31
        // i.e. within each 64-weight group, first 32 positions use lo nibbles
        // and positions 32..63 use hi nibbles of the same 32 qs bytes.
        let Q4K_BLOCK_BYTES: u32 = 144u;
        let Q4K_BLOCK_WEIGHTS: u32 = 256u;
        let num_blocks = K / Q4K_BLOCK_WEIGHTS;

        for (var blk = 0u; blk < num_blocks; blk++) {
            let blk_byte_off = row * num_blocks * Q4K_BLOCK_BYTES + blk * Q4K_BLOCK_BYTES;

            let d    = rd_f16(blk_byte_off);
            let dmin = rd_f16(blk_byte_off + 2u);
            let sc_base = blk_byte_off + 4u;   // first of 12 scale/min bytes
            let qs_base = blk_byte_off + 16u;  // first qs byte

            // 4 groups of 64 weights, processed as paired sub-blocks.
            // Each group uses 32 qs bytes: lo nibbles → first 32 weights, hi nibbles → next 32.
            for (var grp = 0u; grp < 4u; grp++) {
                let sub0 = grp * 2u;
                let sub1 = grp * 2u + 1u;
                let eff_scale0 = d    * f32(q4k_scale(sub0, sc_base));
                let eff_min0   = dmin * f32(q4k_min  (sub0, sc_base));
                let eff_scale1 = d    * f32(q4k_scale(sub1, sc_base));
                let eff_min1   = dmin * f32(q4k_min  (sub1, sc_base));
                let qs_grp_base = qs_base + grp * 32u;
                let k0 = blk * Q4K_BLOCK_WEIGHTS + grp * 64u;

                for (var p = 0u; p < 32u; p++) {
                    let byte_val = rd_byte(qs_grp_base + p);
                    let q_lo = f32(byte_val & 0x0Fu);   // weight k0 + p       (sub0)
                    let q_hi = f32(byte_val >> 4u);     // weight k0 + p + 32  (sub1)
                    acc += (q_lo * eff_scale0 - eff_min0) * f32(x[k0 + p     ]);
                    acc += (q_hi * eff_scale1 - eff_min1) * f32(x[k0 + p + 32u]);
                }
            }
        }
    } else if (USE_QUANT == 1u) {
        // Simple symmetric Q4 path (for custom Q4 format with one scale per BLOCK_K weights).
        // Each byte holds two unsigned nibbles; dequant: (nibble - 8) * scale.
        let blocks    = K / BLOCK_K;
        let row_bytes = K / 2u;

        for (var blk = 0u; blk < blocks; blk++) {
            let scale     = f32(scales[row * blocks + blk]);
            let blk_start = row * row_bytes + blk * (BLOCK_K / 2u);

            var block_acc: f32 = 0.0;
            for (var b = 0u; b < BLOCK_K / 8u; b++) {
                let u  = weights[blk_start / 4u + b];
                let k0 = blk * BLOCK_K + b * 8u;

                let b0 = u & 0xFFu;
                block_acc += f32(i32(b0 & 0x0Fu) - 8) * f32(x[k0    ]);
                block_acc += f32(i32(b0 >> 4u)   - 8) * f32(x[k0 + 1u]);

                let b1 = (u >>  8u) & 0xFFu;
                block_acc += f32(i32(b1 & 0x0Fu) - 8) * f32(x[k0 + 2u]);
                block_acc += f32(i32(b1 >> 4u)   - 8) * f32(x[k0 + 3u]);

                let b2 = (u >> 16u) & 0xFFu;
                block_acc += f32(i32(b2 & 0x0Fu) - 8) * f32(x[k0 + 4u]);
                block_acc += f32(i32(b2 >> 4u)   - 8) * f32(x[k0 + 5u]);

                let b3 = u >> 24u;
                block_acc += f32(i32(b3 & 0x0Fu) - 8) * f32(x[k0 + 6u]);
                block_acc += f32(i32(b3 >> 4u)   - 8) * f32(x[k0 + 7u]);
            }
            acc += block_acc * scale;
        }
    } else {
        // f16 path: 8-element unroll, 4 consecutive u32 loads per iteration.
        // K must be divisible by 8 (all practical models satisfy this).
        // Each iteration issues 4 128-bit-aligned word loads — optimal for Metal.
        for (var k = 0u; k < K; k += 8u) {
            let base = (row * K + k) / 2u;
            let w0 = unpack2x16float(weights[base]);
            let w1 = unpack2x16float(weights[base + 1u]);
            let w2 = unpack2x16float(weights[base + 2u]);
            let w3 = unpack2x16float(weights[base + 3u]);
            acc += w0.x * f32(x[k])       + w0.y * f32(x[k + 1u]);
            acc += w1.x * f32(x[k + 2u]) + w1.y * f32(x[k + 3u]);
            acc += w2.x * f32(x[k + 4u]) + w2.y * f32(x[k + 5u]);
            acc += w3.x * f32(x[k + 6u]) + w3.y * f32(x[k + 7u]);
        }
    }

    // Clip to f16 range before casting to prevent +inf/-inf which propagates as NaN.
    output[row] = f16(clamp(acc, -65504.0, 65504.0));
}
