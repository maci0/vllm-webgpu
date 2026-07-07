enable f16;

// moe_expert_down_accum.wgsl — fused GEMV + weighted accumulate for MoE down projection.
//
// Computes: accum[row] += w_buf[K_IDX] * (act @ down_w[row, :])
//
// This fuses the per-expert down projection matmul with the MoE weighted
// accumulation, saving one GPU dispatch per selected expert compared to the
// separate matmul_quant + moe_accumulate pair.
//
// Uses split-K GEMV (same pattern as matmul_quant SPLIT_K=1, USE_QUANT=0):
// - One workgroup per output row — dispatch (N, 1, 1)
// - 256 threads per workgroup; each thread handles K/256 input elements
// - Coalesced weight reads: all threads read consecutive u32s from one row
// - Shared-memory tree reduction finalizes partial sums
// - Thread 0 reads the routing weight and adds to the f16 accumulator
//
// Only supports f16 weights (USE_QUANT=0). For quantized down projections,
// fall back to separate matmul_quant + moe_accumulate dispatches.

override K: u32     = 14336u;  // intermediate size (act / down_w column count)
override N: u32     = 4096u;   // hidden size (down_w row count / accum length)
override K_IDX: u32 = 0u;     // which selected-expert slot in w_buf (0..top_k-1)

@group(0) @binding(0) var<storage, read>       act:    array<f16>;  // [K] post-activation input
@group(0) @binding(1) var<storage, read>       down_w: array<u32>;  // [N, K] f16 packed (2 f16/u32)
@group(0) @binding(2) var<storage, read_write> accum:  array<f16>;  // [N] f16 running accumulator
@group(0) @binding(3) var<storage, read>       w_buf:  array<f32>;  // [top_k] routing weights

var<workgroup> sh_acc: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) lid:  vec3<u32>,
    @builtin(workgroup_id)        wgid: vec3<u32>,
) {
    let row = wgid.x;
    let tid = lid.x;
    if (row >= N) { return; }

    // Each thread accumulates K/256 partial dot-product terms.
    // Consecutive threads read consecutive u32 words from the same weight row,
    // giving fully coalesced memory access (identical to matmul_quant SPLIT_K=1).
    var local_acc: f32 = 0.0;
    let row_base = row * K;
    var k = tid * 2u;
    loop {
        if (k >= K) { break; }
        let raw_w = down_w[(row_base + k) / 2u];
        let wp = unpack2x16float(raw_w);
        local_acc += wp.x * f32(act[k]);
        if (k + 1u < K) { local_acc += wp.y * f32(act[k + 1u]); }
        k += 512u;
    }

    // 8-stage tree reduction: sum 256 partial accumulators into sh_acc[0].
    sh_acc[tid] = local_acc;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) { sh_acc[tid] += sh_acc[tid + stride]; }
        workgroupBarrier();
        stride /= 2u;
    }

    // Thread 0: weighted add into the f16 accumulator.
    if (tid == 0u) {
        let weight = w_buf[K_IDX];
        let cur    = f32(accum[row]);
        accum[row] = f16(clamp(cur + weight * sh_acc[0], -65504.0, 65504.0));
    }
}
