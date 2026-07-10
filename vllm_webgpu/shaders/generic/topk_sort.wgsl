// topk_sort.wgsl — GPU top-K selection for MoE router (multi-token).
//
// Input:  logits [T, N_EXPERTS] f32  — router logits for T tokens (f32 for stable ranking)
// Output: topk_idx     [T, K] u32   — top-K expert indices per token
//         topk_weights [T, K] f32   — softmax-normalized expert weights per token
//
// Dispatch: (T, 1, 1) — one workgroup per token.
//   workgroup_id.x = token index (t).
//   Each workgroup uses 256 threads to load N_EXPERTS logits into shared memory,
//   then thread 0 runs sequential top-K selection (K ≤ 8, N_EXPERTS ≤ 256).
//   Thread 0 writes K results to topk_idx[t*K .. t*K+K-1] and topk_weights[...].

override N_EXPERTS: u32 = 128u;
override K: u32         = 8u;    // top-K experts to select

@group(0) @binding(0) var<storage, read>       logits      : array<f32>;  // [T * N_EXPERTS]
@group(0) @binding(1) var<storage, read_write> topk_idx    : array<u32>;  // [T * K] output indices
@group(0) @binding(2) var<storage, read_write> topk_weights: array<f32>;  // [T * K] softmax weights

var<workgroup> sh_logits:  array<f32, 256>;  // local copy of logits (up to 256 experts)
var<workgroup> sh_topk_v:  array<f32, 8>;    // top-K values (up to 8)
var<workgroup> sh_topk_i:  array<u32, 8>;    // top-K indices

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id)        wgid: vec3<u32>,
    @builtin(local_invocation_id) lid:  vec3<u32>,
) {
    let tid       = lid.x;
    let token_idx = wgid.x;  // one workgroup per token

    // Load this token's logits into shared memory
    let logit_base = token_idx * N_EXPERTS;
    if (tid < N_EXPERTS) {
        sh_logits[tid] = logits[logit_base + tid];
    } else {
        sh_logits[tid] = -1e30f;  // sentinel for unused slots
    }
    workgroupBarrier();

    // Thread 0 does sequential top-K selection (tiny: K ≤ 8, N_EXPERTS ≤ 256)
    if (tid == 0u) {
        // Initialize to sentinel
        for (var k = 0u; k < K; k++) {
            sh_topk_v[k] = -1e30f;
            sh_topk_i[k] = 0u;
        }
        // Selection sort: find top-K one by one
        for (var k = 0u; k < K; k++) {
            var best_v = -1e30f;
            var best_i = 0u;
            for (var e = 0u; e < N_EXPERTS; e++) {
                // Skip already-selected
                var already = false;
                for (var prev = 0u; prev < k; prev++) {
                    if (sh_topk_i[prev] == e) { already = true; break; }
                }
                if (!already && sh_logits[e] > best_v) {
                    best_v = sh_logits[e];
                    best_i = e;
                }
            }
            sh_topk_v[k] = best_v;
            sh_topk_i[k] = best_i;
        }
        // Compute softmax over selected K logits
        var max_v = -1e30f;
        for (var k = 0u; k < K; k++) { max_v = max(max_v, sh_topk_v[k]); }
        var sum_e = 0.0f;
        for (var k = 0u; k < K; k++) { sum_e += exp(sh_topk_v[k] - max_v); }
        // Write outputs at this token's offset
        let out_base = token_idx * K;
        for (var k = 0u; k < K; k++) {
            topk_idx[out_base + k]     = sh_topk_i[k];
            topk_weights[out_base + k] = exp(sh_topk_v[k] - max_v) / sum_e;
        }
    }
}
