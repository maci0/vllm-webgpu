enable f16;

override HEAD_DIM: u32      = 128u;
override NUM_HEADS: u32     = 32u;
override ROPE_BASE: f32     = 10000.0;
override LN_ROPE_BASE: f32  = 9.210340372;  // = log(ROPE_BASE); host sets this
// USE_FREQ_BUF=1: read precomputed inv_freq from binding 3 instead of computing inline.
// Enables YaRN and other scaled RoPE variants via CPU-side frequency precomputation.
override USE_FREQ_BUF: u32  = 0u;
// YARN_MSCALE: YaRN post-rope multiplier applied to both Q and K.
// Set to mscale (0.1 * ln(factor) + 1.0) when USE_FREQ_BUF=1; leave at 1.0 otherwise.
override YARN_MSCALE: f32   = 1.0;
// INPUT_OFFSET: element offset into the input buffer. Set to Q_DIM when reading K
// from a fused QKV buffer, 0 for standalone Q or K buffers.
override INPUT_OFFSET: u32  = 0u;
// OUTPUT_OFFSET: element offset into the output buffer. Defaults to 0 so that
// output is always written at the start of the destination buffer regardless of
// INPUT_OFFSET. Set equal to INPUT_OFFSET only when input and output share the
// same buffer layout.
override OUTPUT_OFFSET: u32 = 0u;

@group(0) @binding(0) var<storage, read>       input        : array<f16>;
@group(0) @binding(1) var<storage, read>       positions    : array<u32>;
@group(0) @binding(2) var<storage, read_write> output       : array<f16>;
@group(0) @binding(3) var<storage, read>       inv_freq_buf : array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) gid  : vec3<u32>,
    @builtin(local_invocation_id)  lid  : vec3<u32>,
    @builtin(workgroup_id)         wgid : vec3<u32>,
) {
    let seq_idx  = wgid.x;
    let head_idx = wgid.y;
    let half     = HEAD_DIM / 2u;
    let tid      = lid.x;   // iterates over [0, half)

    let in_base  = INPUT_OFFSET  + (seq_idx * NUM_HEADS + head_idx) * HEAD_DIM;
    let out_base = OUTPUT_OFFSET + (seq_idx * NUM_HEADS + head_idx) * HEAD_DIM;
    let pos  = f32(positions[seq_idx]);

    // Loop so HEAD_DIM > 2*WG_SIZE is handled correctly (e.g. HEAD_DIM=256 with WG_SIZE=64).
    var i = tid;
    loop {
        if (i >= half) { break; }
        var theta_i: f32;
        if (USE_FREQ_BUF == 1u) {
            theta_i = inv_freq_buf[i];
        } else {
            theta_i = exp(-f32(i * 2u) / f32(HEAD_DIM) * LN_ROPE_BASE);
        }
        let angle   = pos * theta_i;
        let cos_v   = YARN_MSCALE * cos(angle);
        let sin_v   = YARN_MSCALE * sin(angle);

        let x1 = f32(input[in_base + i]);
        let x2 = f32(input[in_base + half + i]);

        output[out_base + i]        = f16(x1 * cos_v - x2 * sin_v);
        output[out_base + half + i] = f16(x2 * cos_v + x1 * sin_v);
        i += 64u;
    }
}
