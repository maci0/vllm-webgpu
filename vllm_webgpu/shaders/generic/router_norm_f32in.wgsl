enable f16;

// router_norm_f32in.wgsl — Gemma4Router input preprocessing.
//
// Implements the three-step transform applied before the router projection:
//   1. RMSNorm without a learned weight (pure normalization)
//   2. Multiply by ROOT_SIZE = 1/sqrt(hidden_size)
//   3. Multiply elementwise by the learned per-dimension router.scale
//
// This matches vLLM's Gemma4Router.forward:
//   x = self.norm(x)                    # no-weight RMSNorm
//   x = x * self.root_size              # 1/sqrt(hidden_size)
//   x = x * self.scale                  # learned per-dim scale
//
// Input:  f32 residual (the pre-MoE residual stream, not pre_feedforward_layernorm_2 output)
// Weight: f16 router.scale (per-dimension learned scale, shape [hidden_size])
// Output: f16 preprocessed router input, ready for the projection matmul
//
// Dispatch (num_tokens, 1, 1).

override HIDDEN_DIM:      u32 = 4096u;
override WG_SIZE:         u32 = 256u;
override VALS_PER_THREAD: u32 = 16u;   // HIDDEN_DIM / WG_SIZE; 0 = two-pass fallback
override ROOT_SIZE:       f32 = 1.0;   // 1/sqrt(hidden_size), set by host

var<workgroup> shared_sum: array<f32, 256>;

@group(0) @binding(0) var<storage, read>       input  : array<f32>;   // f32 residual
@group(0) @binding(1) var<storage, read>       scale  : array<f16>;   // router.scale
@group(0) @binding(2) var<storage, read_write> output : array<f16>;   // preprocessed f16

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) lid  : vec3<u32>,
    @builtin(workgroup_id)        wgid : vec3<u32>,
) {
    let row  = wgid.x;
    let tid  = lid.x;
    let base = row * HIDDEN_DIM;
    let eps  = 1e-6f;

    if (VALS_PER_THREAD > 0u) {
        // Register-tile path: store input in registers, skip second global read.
        var local_v: array<f32, 16>;
        var sq_sum: f32 = 0.0;
        var col = tid;
        for (var i = 0u; i < VALS_PER_THREAD; i++) {
            if (col < HIDDEN_DIM) {
                let v = input[base + col];
                local_v[i] = v;
                sq_sum += v * v;
                col += WG_SIZE;
            }
        }
        shared_sum[tid] = sq_sum;
        workgroupBarrier();
        var stride = WG_SIZE / 2u;
        loop {
            if (stride == 0u) { break; }
            if (tid < stride) { shared_sum[tid] += shared_sum[tid + stride]; }
            workgroupBarrier();
            stride /= 2u;
        }
        // rms_inv incorporates ROOT_SIZE so we avoid a separate multiply per element.
        let rms_inv = inverseSqrt(shared_sum[0] / f32(HIDDEN_DIM) + eps) * ROOT_SIZE;
        col = tid;
        for (var i = 0u; i < VALS_PER_THREAD; i++) {
            if (col < HIDDEN_DIM) {
                let normed = local_v[i] * rms_inv * f32(scale[col]);
                output[base + col] = f16(clamp(normed, -65504.0, 65504.0));
                col += WG_SIZE;
            }
        }
    } else {
        // Two-pass global re-read fallback for HIDDEN_DIM > WG_SIZE * 16.
        var sq_sum: f32 = 0.0;
        var col = tid;
        loop {
            if (col >= HIDDEN_DIM) { break; }
            let v = input[base + col];
            sq_sum += v * v;
            col += WG_SIZE;
        }
        shared_sum[tid] = sq_sum;
        workgroupBarrier();
        var stride = WG_SIZE / 2u;
        loop {
            if (stride == 0u) { break; }
            if (tid < stride) { shared_sum[tid] += shared_sum[tid + stride]; }
            workgroupBarrier();
            stride /= 2u;
        }
        let rms_inv = inverseSqrt(shared_sum[0] / f32(HIDDEN_DIM) + eps) * ROOT_SIZE;
        col = tid;
        loop {
            if (col >= HIDDEN_DIM) { break; }
            let normed = input[base + col] * rms_inv * f32(scale[col]);
            output[base + col] = f16(clamp(normed, -65504.0, 65504.0));
            col += WG_SIZE;
        }
    }
}
