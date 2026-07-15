enable f16;

// matmul_quant_f32out.wgsl — GEMV for decode (M=1) with f32 output.
//
// Identical to matmul_quant.wgsl in all compute logic; the only difference is
// that output is declared as array<f32> and the final accumulator is written
// directly without casting through f16.  This eliminates the ~0.001 ULP rounding
// error that would occur if the f32 accumulator were narrowed to f16 before
// top-K comparisons (the bug that matmul_quant.wgsl would introduce for router
// logits where expert pairs may differ by less than one f16 ULP).
//
// Intended exclusively for the router projection in DiffusionGemma and any other
// callers that need f32-precision logits from a GEMV over f16 or quantized weights.
// No HAS_BIAS: bias is not relevant for routing projections.
//
// USE_QUANT=0: f16 weights — weights[N, K] packed as f16 in u32 (two f16 per u32)
// USE_QUANT=1: simple Q4 — weights[N, K/2] packed nibbles (biased ±8), scales[N, K/BLOCK_K] f32
// USE_QUANT=2: GGUF Q4_K — weights in raw Q4_K block format (144 bytes per 256-weight block)
// USE_QUANT=3: GPTQ INT4 — weights[N, K//8] INT32, scales[G, N] F32, zero_point=8
// USE_QUANT=4: AWQ  INT4 — weights[K, N//8] INT32, scales[G, N] F32, zero_point=8
// USE_QUANT=5: FP8 E4M3
// USE_QUANT=6: NVFP4
// USE_QUANT=7: Int8 per-channel
// USE_QUANT=8: NF4 (BitsAndBytes)

override K: u32          = 4096u;
override N: u32          = 4096u;
override BLOCK_K: u32    = 32u;     // block size for USE_QUANT=1 (simple Q4)
override USE_QUANT: u32  = 0u;      // 0=f16, 1=Q4, 2=Q4_K, 3=GPTQ, 4=AWQ, 5=FP8, 6=NVFP4, 7=Int8, 8=NF4
override GROUP_K: u32    = 128u;    // quantization group size for USE_QUANT=3/4/6
override GLOBAL_SCALE: f32 = 1.0;  // per-tensor scale for USE_QUANT=5 (FP8) and 6 (NVFP4)
// SPLIT_K=1 (default): split-K GEMV — all 256 threads work on ONE output row.
// Dispatch (N, 1, 1) workgroups. Within each workgroup, consecutive threads
// read consecutive weight elements → coalesced on all GPU architectures.
// 8-barrier workgroup reduction finalises the partial sums.
// SPLIT_K=0: legacy row-per-thread. Dispatch ((N+255)/256, 1, 1) workgroups.
override SPLIT_K: u32   = 1u;
override USE_BF16: u32  = 0u;      // 1=bf16 packed u16→u32 weights (GDN projection layers)

@group(0) @binding(0) var<storage, read>       x       : array<f16>;  // [K]
@group(0) @binding(1) var<storage, read>       weights : array<u32>;  // raw bytes as u32
@group(0) @binding(2) var<storage, read>       scales  : array<f32>;  // [N, K/BLOCK_K] for USE_QUANT=1
@group(0) @binding(3) var<storage, read_write> output  : array<f32>;  // [N] — f32, no f16 narrowing

// Read one byte from the weights u32 array at byte offset `byte_off`.
fn rd_byte(byte_off: u32) -> u32 {
    return (weights[byte_off / 4u] >> ((byte_off % 4u) * 8u)) & 0xFFu;
}

// Read a float16 value stored at byte offset `byte_off` (little-endian).
fn rd_f16(byte_off: u32) -> f32 {
    let lo = rd_byte(byte_off);
    let hi = rd_byte(byte_off + 1u);
    return unpack2x16float(lo | (hi << 8u)).x;
}

// Extract 6-bit scale value for sub-block j (0..7) from the 12 scale bytes at sc_base.
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

fn int8_to_f32(raw: u32) -> f32 {
    return f32(select(i32(raw), i32(raw) - 256, (raw & 0x80u) != 0u));
}

fn nf4_to_f32(code: u32) -> f32 {
    switch code & 0xFu {
        case  0u: { return -1.0f; }
        case  1u: { return -0.6961928010f; }
        case  2u: { return -0.5250730515f; }
        case  3u: { return -0.3949424624f; }
        case  4u: { return -0.2844374180f; }
        case  5u: { return -0.1847791076f; }
        case  6u: { return -0.0911458358f; }
        case  7u: { return  0.0f; }
        case  8u: { return  0.0795822144f; }
        case  9u: { return  0.1609302461f; }
        case 10u: { return  0.2461898923f; }
        case 11u: { return  0.3379294276f; }
        case 12u: { return  0.4407098889f; }
        case 13u: { return  0.5626170039f; }
        case 14u: { return  0.7246159911f; }
        default:  { return  1.0f; }
    }
}

fn fp8_to_f32(b: u32) -> f32 {
    let sign = select(1.0f, -1.0f, (b & 0x80u) != 0u);
    let exp  = (b >> 3u) & 0xFu;
    let mant = b & 0x7u;
    if (exp == 0u)  { return sign * f32(mant) * (1.0f / 512.0f); }
    if (exp == 15u) { return 0.0f; }
    return sign * exp2(f32(i32(exp) - 7)) * (1.0f + f32(mant) * 0.125f);
}

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
        let row = wgid.x;
        if (row >= N) { return; }

        var acc: f32 = 0.0;

        if (USE_QUANT == 3u) {
            let K8 = K / 8u;
            var q_step = tid;
            loop {
                if (q_step >= K8) { break; }
                let k_base = q_step * 8u;
                let q = weights[row * K8 + q_step];
                let grp = q_step / (GROUP_K / 8u);
                let sc  = scales[grp * N + row];
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
            let N8 = N / 8u;
            let awq_pos = row % 8u;
            var awq_shift: u32;
            switch awq_pos {
                case 0u: { awq_shift = 0u; }
                case 1u: { awq_shift = 8u; }
                case 2u: { awq_shift = 16u; }
                case 3u: { awq_shift = 24u; }
                case 4u: { awq_shift = 4u; }
                case 5u: { awq_shift = 12u; }
                case 6u: { awq_shift = 20u; }
                default: { awq_shift = 28u; }
            }
            var k_awq = tid * 2u;
            loop {
                if (k_awq >= K) { break; }
                let grp = k_awq / GROUP_K;
                let sc  = scales[grp * N + row];
                let q0  = weights[k_awq * N8 + row / 8u];
                let n0  = f32(i32((q0 >> awq_shift) & 0xFu) - 8);
                acc += n0 * sc * f32(x[k_awq]);
                if (k_awq + 1u < K) {
                    let q1 = weights[(k_awq + 1u) * N8 + row / 8u];
                    let sc1 = scales[(k_awq + 1u) / GROUP_K * N + row];
                    let n1  = f32(i32((q1 >> awq_shift) & 0xFu) - 8);
                    acc += n1 * sc1 * f32(x[k_awq + 1u]);
                }
                k_awq += 512u;
            }
        } else if (USE_QUANT == 5u) {
            let row_base = row * K;
            var k_fp8 = tid * 2u;
            if (GROUP_K == 1u) {
                let ch_scale = scales[row];
                loop {
                    if (k_fp8 >= K) { break; }
                    let b0 = rd_byte_at(row_base + k_fp8);
                    acc += fp8_to_f32(b0) * ch_scale * f32(x[k_fp8]);
                    if (k_fp8 + 1u < K) {
                        let b1 = rd_byte_at(row_base + k_fp8 + 1u);
                        acc += fp8_to_f32(b1) * ch_scale * f32(x[k_fp8 + 1u]);
                    }
                    k_fp8 += 512u;
                }
            } else {
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
            }
        } else if (USE_QUANT == 6u) {
            let K2  = K / 2u;
            let K16 = K / 16u;
            var k_fp4 = tid * 2u;
            loop {
                if (k_fp4 >= K) { break; }
                let blk = k_fp4 / 16u;
                let sc  = scales[row * K16 + blk] * GLOBAL_SCALE;
                let byte_idx = row * K2 + k_fp4 / 2u;
                let packed   = rd_byte_at(byte_idx);
                acc += fp4_to_f32(packed & 0xFu) * sc * f32(x[k_fp4]);
                if (k_fp4 + 1u < K) {
                    acc += fp4_to_f32((packed >> 4u) & 0xFu) * sc * f32(x[k_fp4 + 1u]);
                }
                k_fp4 += 512u;
            }
        } else if (USE_QUANT == 7u) {
            let scale    = scales[row];
            let row_base = row * K;
            var k_i8 = tid * 2u;
            loop {
                if (k_i8 >= K) { break; }
                acc += int8_to_f32(rd_byte_at(row_base + k_i8)) * scale * f32(x[k_i8]);
                if (k_i8 + 1u < K) {
                    acc += int8_to_f32(rd_byte_at(row_base + k_i8 + 1u)) * scale * f32(x[k_i8 + 1u]);
                }
                k_i8 += 512u;
            }
        } else if (USE_QUANT == 8u) {
            let K2  = K / 2u;
            let GK  = GROUP_K;
            var k_nf4 = tid * 2u;
            loop {
                if (k_nf4 >= K) { break; }
                let blk     = k_nf4 / GK;
                let sc      = scales[row * (K / GK) + blk];
                let byte_b  = rd_byte_at(row * K2 + k_nf4 / 2u);
                acc += nf4_to_f32(byte_b & 0xFu) * sc * f32(x[k_nf4]);
                if (k_nf4 + 1u < K) {
                    acc += nf4_to_f32((byte_b >> 4u) & 0xFu) * sc * f32(x[k_nf4 + 1u]);
                }
                k_nf4 += 512u;
            }
        } else {
            // USE_QUANT=0: f16 or bf16, 512 elements per step.
            let row_base = row * K;
            var k = tid * 2u;
            loop {
                if (k >= K) { break; }
                let raw_w = weights[(row_base + k) / 2u];
                var wx: f32; var wy: f32;
                if (USE_BF16 == 1u) {
                    wx = bitcast<f32>((raw_w & 0xFFFFu) << 16u);
                    wy = bitcast<f32>((raw_w >> 16u) << 16u);
                } else {
                    let wp = unpack2x16float(raw_w);
                    wx = wp.x; wy = wp.y;
                }
                acc += wx * f32(x[k]);
                if (k + 1u < K) { acc += wy * f32(x[k + 1u]); }
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
        // Write f32 directly — no narrowing to f16, no clamp to f16 range.
        // This preserves the full f32 accumulator for top-K comparisons.
        if (tid == 0u) {
            output[row] = sh_acc[0];
        }
        return;
    }

    // Row-per-thread GEMV (SPLIT_K=0): dispatch ((N+255)/256, 1, 1) workgroups.
    let row = gid.x;
    if (row >= N) { return; }

    var acc: f32 = 0.0;

    if (USE_QUANT == 2u) {
        let Q4K_BLOCK_BYTES: u32 = 144u;
        let Q4K_BLOCK_WEIGHTS: u32 = 256u;
        let num_blocks = K / Q4K_BLOCK_WEIGHTS;

        for (var blk = 0u; blk < num_blocks; blk++) {
            let blk_byte_off = row * num_blocks * Q4K_BLOCK_BYTES + blk * Q4K_BLOCK_BYTES;
            let d    = rd_f16(blk_byte_off);
            let dmin = rd_f16(blk_byte_off + 2u);
            let sc_base = blk_byte_off + 4u;
            let qs_base = blk_byte_off + 16u;

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
                    let q_lo = f32(byte_val & 0x0Fu);
                    let q_hi = f32(byte_val >> 4u);
                    acc += (q_lo * eff_scale0 - eff_min0) * f32(x[k0 + p     ]);
                    acc += (q_hi * eff_scale1 - eff_min1) * f32(x[k0 + p + 32u]);
                }
            }
        }
    } else if (USE_QUANT == 1u) {
        let blocks    = K / BLOCK_K;
        let row_bytes = K / 2u;

        for (var blk = 0u; blk < blocks; blk++) {
            let scale     = scales[row * blocks + blk];
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
    } else if (USE_QUANT == 3u) {
        let K8 = K / 8u;
        for (var q_step = 0u; q_step < K8; q_step++) {
            let k_base = q_step * 8u;
            let q   = weights[row * K8 + q_step];
            let grp = q_step / (GROUP_K / 8u);
            let sc  = scales[grp * N + row];
            acc += (f32(i32( q        & 0xFu) - 8) * sc) * f32(x[k_base]);
            acc += (f32(i32((q >>  4u)& 0xFu) - 8) * sc) * f32(x[k_base + 1u]);
            acc += (f32(i32((q >>  8u)& 0xFu) - 8) * sc) * f32(x[k_base + 2u]);
            acc += (f32(i32((q >> 12u)& 0xFu) - 8) * sc) * f32(x[k_base + 3u]);
            acc += (f32(i32((q >> 16u)& 0xFu) - 8) * sc) * f32(x[k_base + 4u]);
            acc += (f32(i32((q >> 20u)& 0xFu) - 8) * sc) * f32(x[k_base + 5u]);
            acc += (f32(i32((q >> 24u)& 0xFu) - 8) * sc) * f32(x[k_base + 6u]);
            acc += (f32(i32((q >> 28u)& 0xFu) - 8) * sc) * f32(x[k_base + 7u]);
        }
    } else if (USE_QUANT == 4u) {
        let N8 = N / 8u;
        let awq_pos = row % 8u;
        var awq_shift: u32;
        switch awq_pos {
            case 0u: { awq_shift = 0u; }
            case 1u: { awq_shift = 8u; }
            case 2u: { awq_shift = 16u; }
            case 3u: { awq_shift = 24u; }
            case 4u: { awq_shift = 4u; }
            case 5u: { awq_shift = 12u; }
            case 6u: { awq_shift = 20u; }
            default: { awq_shift = 28u; }
        }
        for (var k = 0u; k < K; k++) {
            let grp = k / GROUP_K;
            let sc  = scales[grp * N + row];
            let q   = weights[k * N8 + row / 8u];
            let n   = f32(i32((q >> awq_shift) & 0xFu) - 8);
            acc += n * sc * f32(x[k]);
        }
    } else {
        // f16/bf16 path: 8-element unroll.
        for (var k = 0u; k < K; k += 8u) {
            let base = (row * K + k) / 2u;
            if (USE_BF16 == 1u) {
                let r0 = weights[base];
                let r1 = weights[base + 1u];
                let r2 = weights[base + 2u];
                let r3 = weights[base + 3u];
                acc += bitcast<f32>((r0 & 0xFFFFu) << 16u) * f32(x[k]);
                acc += bitcast<f32>((r0 >> 16u) << 16u) * f32(x[k + 1u]);
                acc += bitcast<f32>((r1 & 0xFFFFu) << 16u) * f32(x[k + 2u]);
                acc += bitcast<f32>((r1 >> 16u) << 16u) * f32(x[k + 3u]);
                acc += bitcast<f32>((r2 & 0xFFFFu) << 16u) * f32(x[k + 4u]);
                acc += bitcast<f32>((r2 >> 16u) << 16u) * f32(x[k + 5u]);
                acc += bitcast<f32>((r3 & 0xFFFFu) << 16u) * f32(x[k + 6u]);
                acc += bitcast<f32>((r3 >> 16u) << 16u) * f32(x[k + 7u]);
            } else {
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
    }

    // Write f32 directly — no narrowing to f16, no clamp to f16 range.
    output[row] = acc;
}
