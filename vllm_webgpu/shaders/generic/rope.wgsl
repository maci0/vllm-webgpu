enable f16;

override HEAD_DIM: u32  = 128u;
override NUM_HEADS: u32 = 32u;
override ROPE_BASE: f32 = 10000.0;

@group(0) @binding(0) var<storage, read>       input     : array<f16>;
@group(0) @binding(1) var<storage, read>       positions : array<u32>;
@group(0) @binding(2) var<storage, read_write> output    : array<f16>;

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

    if (tid >= half) { return; }

    let pos     = f32(positions[seq_idx]);
    let theta_i = pow(ROPE_BASE, -f32(tid * 2u) / f32(HEAD_DIM));
    let angle   = pos * theta_i;
    let cos_v   = cos(angle);
    let sin_v   = sin(angle);

    let base = (seq_idx * NUM_HEADS + head_idx) * HEAD_DIM;
    let x1   = f32(input[base + tid]);
    let x2   = f32(input[base + half + tid]);

    output[base + tid]        = f16(x1 * cos_v - x2 * sin_v);
    output[base + half + tid] = f16(x2 * cos_v + x1 * sin_v);
}
