enable f16;

// gumbel_sample.wgsl — GPU temperature sampling via the Gumbel-max trick.
//
// Gumbel-max: sample from softmax(logits/T) by computing
//   argmax(logit[i]/T + Gumbel(i)) = argmax(logit[i]/T - log(-log(u[i])))
// where u[i] ~ Uniform(0,1) are pre-generated random numbers.
//
// Same dispatch as argmax_f16: (1, 1, 1) one workgroup of 256 threads.
// Input: noise buffer with N uniform random floats in (0,1).
// Output: result[0] = sampled token index.

override N:           u32 = 262144u;
override TEMPERATURE: f32 = 1.0;       // sampling temperature (> 0)
override INV_TEMP:    f32 = 1.0;       // 1.0 / TEMPERATURE (precomputed)

@group(0) @binding(0) var<storage, read>       logits : array<f16>;
@group(0) @binding(1) var<storage, read>       noise  : array<f32>;  // [N] uniform(0,1)
@group(0) @binding(2) var<storage, read_write> result : array<u32>;  // [1] sampled index

var<workgroup> sh_max: array<f32, 256>;
var<workgroup> sh_idx: array<u32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;

    var local_max: f32 = -1e30f;
    var local_idx: u32 = 0u;
    var i = tid;
    loop {
        if (i >= N) { break; }
        // Gumbel noise: -log(-log(u + eps)) where u ~ Uniform(0,1)
        let u   = clamp(noise[i], 1e-7f, 1.0f - 1e-7f);
        let g   = -log(-log(u));
        let val = f32(logits[i]) * INV_TEMP + g;
        if (val > local_max) { local_max = val; local_idx = i; }
        i += 256u;
    }
    sh_max[tid] = local_max;
    sh_idx[tid] = local_idx;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            if (sh_max[tid + stride] > sh_max[tid]) {
                sh_max[tid] = sh_max[tid + stride];
                sh_idx[tid] = sh_idx[tid + stride];
            }
        }
        workgroupBarrier();
        stride /= 2u;
    }

    if (tid == 0u) {
        result[0] = sh_idx[0];
    }
}
