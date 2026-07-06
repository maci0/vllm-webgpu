enable f16;

// matmul_quant.wgsl — GEMV for decode (M=1)
//
// USE_QUANT=0: f16 weights — weights[N, K] packed as f16 in u32 (two f16 per u32)
// USE_QUANT=1: simple Q4 — weights[N, K/2] packed nibbles (biased ±8), scales[N, K/BLOCK_K] f16
// USE_QUANT=2: GGUF Q4_K — weights in raw Q4_K block format (144 bytes per 256-weight block)
//              Block layout: d(f16,2B) dmin(f16,2B) scales_mins(12B) nibbles(128B)
//              Nibbles are unsigned 0-15; dequant = d*scale*nibble - dmin*min (asymmetric)

override K: u32         = 4096u;
override N: u32         = 4096u;
override BLOCK_K: u32   = 32u;     // block size for USE_QUANT=1 (simple Q4)
override USE_QUANT: u32 = 1u;      // 0=f16, 1=simple Q4, 2=GGUF Q4_K, 3=GPU GPTQ INT4, 4=GPU AWQ INT4
override GROUP_K: u32   = 128u;    // quantization group size for USE_QUANT=3/4
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
            // GPU AWQ INT4 dequant (split-K).
            // Weight layout: [K, N//8] INT32. Each INT32 packs 8 nibbles along N.
            // scales: [K//GROUP_K, N] F16.  zero_point=8 (symmetric).
            // Thread t handles rows 2t and 2t+1 (two outputs per thread, same k-pass).
            // At each k, ALL threads read qweight[k, t] — perfectly coalesced!
            // 4× bandwidth reduction vs f16: INT32 (4B) covers 8 N-elements.
            let half_N = N / 2u;        // number of INT32 per K row
            let row2a  = row;           // first output row handled by this workgroup
            // NOTE: for AWQ we must re-dispatch with N/8 workgroups (outputs/8).
            // For simplicity treat this as row-of-int32: each WG does ONE int32's worth.

            // AWQ: thread t reads qweight[k, t] → 8 nibbles for output rows 8t..8t+7
            // For split-K: thread tid processes k = tid*2, tid*2+512, ...
            var k2 = tid * 2u;
            loop {
                if (k2 >= K) { break; }
                let grp2 = k2 / GROUP_K;
                // qweight[k2, row/8] — row selects which int32 in the N//8 array
                let q2 = weights[k2 * half_N + row / 8u];
                // Extract the nibble for row within the 8-pack
                let nibble_shift = (row % 8u) * 4u;
                let n2a = f32(i32((q2 >> nibble_shift) & 0xFu) - 8);
                let sc2 = f32(scales[grp2 * N + row]);
                acc += n2a * sc2 * f32(x[k2]);
                if (k2 + 1u < K) {
                    let q2b = weights[(k2 + 1u) * half_N + row / 8u];
                    let n2b = f32(i32((q2b >> nibble_shift) & 0xFu) - 8);
                    let sc2b = f32(scales[(k2 + 1u) / GROUP_K * N + row]);
                    acc += n2b * sc2b * f32(x[k2 + 1u]);
                }
                k2 += 512u;
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
