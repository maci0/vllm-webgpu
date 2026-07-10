enable f16;

// f16_to_f32.wgsl — upcast f16 elements to f32.
//
// Input:  src [N_ELEMS] f16
// Output: dst [N_ELEMS] f32
//
// Dispatch: (ceil(N_ELEMS / 256), 1, 1) — one thread per element.

override N_ELEMS: u32 = 256u;

@group(0) @binding(0) var<storage, read>       src: array<f16>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < N_ELEMS) {
        dst[gid.x] = f32(src[gid.x]);
    }
}
