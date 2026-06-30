enable f16;

override HIDDEN_DIM: u32 = 2048u;
override PLE_DIM: u32    = 64u;

@group(0) @binding(0) var<storage, read>       hidden     : array<f16>;  // [seq, hidden_dim]
@group(0) @binding(1) var<storage, read>       ple_embed  : array<f16>;  // [seq, ple_dim]
@group(0) @binding(2) var<storage, read>       ple_proj_w : array<f16>;  // [hidden_dim, ple_dim]
@group(0) @binding(3) var<storage, read_write> output     : array<f16>;  // [seq, hidden_dim]

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) lid  : vec3<u32>,
    @builtin(workgroup_id)        wgid : vec3<u32>,
) {
    let seq_idx = wgid.x;
    let tid     = lid.x;
    // Compute: output[seq] = hidden[seq] + ple_embed[seq] @ ple_proj_w^T
    var col = tid;
    loop {
        if (col >= HIDDEN_DIM) { break; }
        var dot: f32 = 0.0;
        for (var p = 0u; p < PLE_DIM; p++) {
            dot += f32(ple_embed[seq_idx * PLE_DIM + p]) * f32(ple_proj_w[col * PLE_DIM + p]);
        }
        output[seq_idx * HIDDEN_DIM + col] = f16(f32(hidden[seq_idx * HIDDEN_DIM + col]) + dot);
        col += 256u;
    }
}
