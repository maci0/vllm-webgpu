enable f16;

// rms_norm_add_f32_rms_norm.wgsl — double-norm fusion for Gemma4 sublayer pairs.
//
// Fuses the two-dispatch sequence used at each sublayer boundary:
//   1. tmp        = rms_norm(delta_in, post_weight)       f16 in → f16 tmp (scratch)
//   2. residual  += SCALE * tmp                           f32 residual update
//      normed_out = rms_norm_f32in(residual, pre_weight)  f32 in → f16 out
//
// The intermediate tmp is kept in thread registers, never written to global memory.
// Two reductions are performed using the same shared_sum scratch array.
//
// Algorithm (register-tiled path, VALS_PER_THREAD > 0):
//   Phase 1: load delta_in into local_v[], accumulate sq_sum1
//   Reduce   sq_sum1 → rms_inv1
//   Phase 2: normed1 = local_v * rms_inv1 * post_weight_eff
//            r = residual_in + SCALE * normed1   (f32 add)
//            residual_out = r                    (write updated residual)
//            local_v[i]   = r                    (reuse for phase 3)
//            accumulate sq_sum2 from r
//   Reduce   sq_sum2 → rms_inv2
//   Phase 3: normed_out = clamp(local_v * rms_inv2 * pre_weight_eff)  f16
//
// Fallback path (VALS_PER_THREAD=0, for HIDDEN_DIM > 4096):
//   Phase 1: load delta_in, accumulate sq_sum1 (no register storage)
//   Reduce   sq_sum1 → rms_inv1
//   Phase 2: re-read delta_in, apply first norm, update residual_out, accumulate sq_sum2
//   Reduce   sq_sum2 → rms_inv2
//   Phase 3: re-read residual_out, apply second norm, write normed_out
//
// Bindings:
//   0: delta_in     [HIDDEN_DIM] f16  — attn output or FFN output (first norm input)
//   1: post_weight  [HIDDEN_DIM] f16  — post_attention_layernorm or post_feedforward weight
//   2: residual_in  [HIDDEN_DIM] f32  — running f32 residual (x_buf or inter-layer residual)
//   3: pre_weight   [HIDDEN_DIM] f16  — pre_feedforward_layernorm or next input_layernorm weight
//   4: residual_out [HIDDEN_DIM] f32  — updated residual (read_write, distinct from residual_in)
//   5: normed_out   [HIDDEN_DIM] f16  — second norm output (for next matmul)
//
// Dispatch (num_tokens, 1, 1) — one workgroup per token (row).

override HIDDEN_DIM:      u32 = 4096u;
override WG_SIZE:         u32 = 256u;
override VALS_PER_THREAD: u32 = 16u;   // HIDDEN_DIM / WG_SIZE; 0 = fallback loop
override GEMMA_NORM:      u32 = 1u;    // 1: (1+w) Gemma style; 0: standard w
override SCALE:           f32 = 1.0;   // residual contribution scale (layer_output_scale)

@group(0) @binding(0) var<storage, read>       delta_in    : array<f16>;
@group(0) @binding(1) var<storage, read>       post_weight : array<f16>;
@group(0) @binding(2) var<storage, read>       residual_in : array<f32>;
@group(0) @binding(3) var<storage, read>       pre_weight  : array<f16>;
@group(0) @binding(4) var<storage, read_write> residual_out: array<f32>;
@group(0) @binding(5) var<storage, read_write> normed_out  : array<f16>;

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) lid:  vec3<u32>,
    @builtin(workgroup_id)        wgid: vec3<u32>,
) {
    let row  = wgid.x;
    let tid  = lid.x;
    let base = row * HIDDEN_DIM;
    let eps  = 1e-6f;

    if (VALS_PER_THREAD > 0u) {
        // ── Register-tiled path ──────────────────────────────────────────────
        // Phase 1: load delta_in into registers, accumulate sq_sum1 for first norm.
        var local_v: array<f32, 16>;
        var sq_sum: f32 = 0.0;
        var col = tid;
        for (var i = 0u; i < VALS_PER_THREAD; i++) {
            if (col < HIDDEN_DIM) {
                let v = f32(delta_in[base + col]);
                local_v[i] = v;
                sq_sum += v * v;
                col += WG_SIZE;
            }
        }
        shared_sum[tid] = sq_sum;
        workgroupBarrier();

        // Reduction 1: tree-reduce to get sq_sum1 in shared_sum[0].
        var stride = WG_SIZE / 2u;
        loop {
            if (stride == 0u) { break; }
            if (tid < stride) { shared_sum[tid] += shared_sum[tid + stride]; }
            workgroupBarrier();
            stride /= 2u;
        }

        let rms_inv1 = inverseSqrt(shared_sum[0] / f32(HIDDEN_DIM) + eps);
        // Every thread reads shared_sum[0] above, and Phase 2 below overwrites
        // shared_sum[tid] -- including tid 0. Without this barrier a fast thread
        // can land its sq_sum2 write on shared_sum[0] before a slower thread has
        // read rms_inv1 from it.
        workgroupBarrier();

        // Phase 2: apply first norm, fold into residual add, accumulate sq_sum2.
        // local_v[i] is overwritten with the updated f32 residual value for Phase 3.
        var sq_sum2: f32 = 0.0;
        col = tid;
        for (var i = 0u; i < VALS_PER_THREAD; i++) {
            if (col < HIDDEN_DIM) {
                let w1_eff = select(f32(post_weight[col]), 1.0 + f32(post_weight[col]),
                                   GEMMA_NORM != 0u);
                let normed1 = local_v[i] * rms_inv1 * w1_eff;
                let r = residual_in[base + col] + SCALE * normed1;
                residual_out[base + col] = r;
                local_v[i] = r;
                sq_sum2 += r * r;
                col += WG_SIZE;
            }
        }
        shared_sum[tid] = sq_sum2;
        workgroupBarrier();

        // Reduction 2: tree-reduce to get sq_sum2 in shared_sum[0].
        stride = WG_SIZE / 2u;
        loop {
            if (stride == 0u) { break; }
            if (tid < stride) { shared_sum[tid] += shared_sum[tid + stride]; }
            workgroupBarrier();
            stride /= 2u;
        }

        let rms_inv2 = inverseSqrt(shared_sum[0] / f32(HIDDEN_DIM) + eps);

        // Phase 3: apply second norm, write f16 output.
        col = tid;
        for (var i = 0u; i < VALS_PER_THREAD; i++) {
            if (col < HIDDEN_DIM) {
                let w2_eff = select(f32(pre_weight[col]), 1.0 + f32(pre_weight[col]),
                                   GEMMA_NORM != 0u);
                normed_out[base + col] = f16(clamp(local_v[i] * rms_inv2 * w2_eff,
                                                   -65504.0, 65504.0));
                col += WG_SIZE;
            }
        }
    } else {
        // ── Two-pass fallback path (HIDDEN_DIM > WG_SIZE * max register slots) ──
        // Phase 1: read delta_in, accumulate sq_sum1.
        var sq_sum: f32 = 0.0;
        var col = tid;
        loop {
            if (col >= HIDDEN_DIM) { break; }
            let v = f32(delta_in[base + col]);
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

        let rms_inv1 = inverseSqrt(shared_sum[0] / f32(HIDDEN_DIM) + eps);
        // Same read-then-overwrite hazard on shared_sum[0] as the register-tiled
        // path above.
        workgroupBarrier();

        // Phase 2: re-read delta_in, apply first norm, update residual, accumulate sq_sum2.
        var sq_sum2: f32 = 0.0;
        col = tid;
        loop {
            if (col >= HIDDEN_DIM) { break; }
            let v = f32(delta_in[base + col]);
            let w1_eff = select(f32(post_weight[col]), 1.0 + f32(post_weight[col]),
                               GEMMA_NORM != 0u);
            let normed1 = v * rms_inv1 * w1_eff;
            let r = residual_in[base + col] + SCALE * normed1;
            residual_out[base + col] = r;
            sq_sum2 += r * r;
            col += WG_SIZE;
        }
        shared_sum[tid] = sq_sum2;
        workgroupBarrier();

        stride = WG_SIZE / 2u;
        loop {
            if (stride == 0u) { break; }
            if (tid < stride) { shared_sum[tid] += shared_sum[tid + stride]; }
            workgroupBarrier();
            stride /= 2u;
        }

        let rms_inv2 = inverseSqrt(shared_sum[0] / f32(HIDDEN_DIM) + eps);

        // Phase 3: re-read residual_out (written in phase 2), apply second norm.
        col = tid;
        loop {
            if (col >= HIDDEN_DIM) { break; }
            let r = residual_out[base + col];
            let w2_eff = select(f32(pre_weight[col]), 1.0 + f32(pre_weight[col]),
                               GEMMA_NORM != 0u);
            normed_out[base + col] = f16(clamp(r * rms_inv2 * w2_eff, -65504.0, 65504.0));
            col += WG_SIZE;
        }
    }
}
