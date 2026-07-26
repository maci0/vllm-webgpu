"""NumPy reference forward for tiny Llama-family / OLMo-2 decode+prefill checks."""
from __future__ import annotations

import numpy as np


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x32 = x.astype(np.float32)
    rms = np.sqrt(np.mean(x32 ** 2) + eps)
    return (x32 / rms) * weight.astype(np.float32)


def silu(x: np.ndarray) -> np.ndarray:
    x32 = x.astype(np.float32)
    return x32 / (1.0 + np.exp(-x32))


def rope_vec(x: np.ndarray, pos: int, n_heads: int, head_dim: int, base: float = 10000.0) -> np.ndarray:
    """Apply standard (half-split) RoPE to a flat [n_heads * head_dim] vector."""
    h = x.astype(np.float32).reshape(n_heads, head_dim)
    half = head_dim // 2
    theta = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    freqs = float(pos) * theta
    cos_f = np.cos(freqs)
    sin_f = np.sin(freqs)
    x1, x2 = h[:, :half], h[:, half:]
    out = np.empty_like(h)
    out[:, :half] = x1 * cos_f - x2 * sin_f
    out[:, half:] = x2 * cos_f + x1 * sin_f
    return out.reshape(-1)


def gqa_attn(
    q: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    n_q: int,
    n_kv: int,
    head_dim: int,
) -> np.ndarray:
    """Single-query GQA over a K/V cache of shape [T, n_kv, head_dim]."""
    scale = head_dim ** -0.5
    qh = q.astype(np.float32).reshape(n_q, head_dim)
    T = k_cache.shape[0]
    n_rep = n_q // n_kv
    out = np.zeros((n_q, head_dim), dtype=np.float32)
    for i in range(n_q):
        kv_i = i // n_rep
        scores = (qh[i] @ k_cache[:, kv_i, :].T) * scale
        scores -= scores.max()
        w = np.exp(scores)
        w /= w.sum()
        out[i] = w @ v_cache[:, kv_i, :]
    return out.reshape(-1)


def mlp_silu(x: np.ndarray, gate_w: np.ndarray, up_w: np.ndarray, down_w: np.ndarray) -> np.ndarray:
    """SwiGLU MLP: down(silu(gate @ x) * (up @ x)). Weights are [N, K]."""
    x32 = x.astype(np.float32)
    gate = gate_w.astype(np.float32) @ x32
    up = up_w.astype(np.float32) @ x32
    return down_w.astype(np.float32) @ (silu(gate) * up)


def llama_layer(
    x: np.ndarray,
    w: dict,
    pos: int,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    *,
    n_q: int,
    n_kv: int,
    head_dim: int,
    apply_rope: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One Llama pre-norm layer. Returns (x_out, k_cache, v_cache)."""
    q_dim = n_q * head_dim
    kv_dim = n_kv * head_dim

    normed = rms_norm(x, w["input_layernorm"])
    q = w["q_proj"] @ normed.astype(np.float32)
    k = w["k_proj"] @ normed.astype(np.float32)
    v = w["v_proj"] @ normed.astype(np.float32)

    if "q_norm" in w:
        q = rms_norm(q, w["q_norm"])
    if "k_norm" in w:
        k = rms_norm(k, w["k_norm"])

    if apply_rope:
        q = rope_vec(q, pos, n_q, head_dim)
        k = rope_vec(k, pos, n_kv, head_dim)

    k_row = k.astype(np.float32).reshape(1, n_kv, head_dim)
    v_row = v.astype(np.float32).reshape(1, n_kv, head_dim)
    k_cache = np.concatenate([k_cache, k_row], axis=0) if k_cache.size else k_row
    v_cache = np.concatenate([v_cache, v_row], axis=0) if v_cache.size else v_row

    attn = gqa_attn(q, k_cache, v_cache, n_q, n_kv, head_dim)
    o = w["o_proj"].astype(np.float32) @ attn.astype(np.float32)
    x = x.astype(np.float32) + o

    ffn_in = rms_norm(x, w["post_attention_layernorm"])
    x = x + mlp_silu(ffn_in, w["gate_proj"], w["up_proj"], w["down_proj"])
    return x, k_cache, v_cache


def olmo2_layer(
    x: np.ndarray,
    w: dict,
    pos: int,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    *,
    n_q: int,
    n_kv: int,
    head_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One OLMo-2 post-norm layer with full-vector q/k RMSNorm."""
    q = w["q_proj"].astype(np.float32) @ x.astype(np.float32)
    k = w["k_proj"].astype(np.float32) @ x.astype(np.float32)
    v = w["v_proj"].astype(np.float32) @ x.astype(np.float32)

    q = rms_norm(q, w["q_norm"])
    k = rms_norm(k, w["k_norm"])
    q = rope_vec(q, pos, n_q, head_dim)
    k = rope_vec(k, pos, n_kv, head_dim)

    k_row = k.astype(np.float32).reshape(1, n_kv, head_dim)
    v_row = v.astype(np.float32).reshape(1, n_kv, head_dim)
    k_cache = np.concatenate([k_cache, k_row], axis=0) if k_cache.size else k_row
    v_cache = np.concatenate([v_cache, v_row], axis=0) if v_cache.size else v_row

    attn = gqa_attn(q, k_cache, v_cache, n_q, n_kv, head_dim)
    o = w["o_proj"].astype(np.float32) @ attn.astype(np.float32)
    x = x.astype(np.float32) + rms_norm(o, w["post_attention_layernorm"])

    ffn = mlp_silu(x, w["gate_proj"], w["up_proj"], w["down_proj"])
    x = x + rms_norm(ffn, w["post_feedforward_layernorm"])
    return x, k_cache, v_cache


def prefill_logits(
    token_ids: np.ndarray,
    embed: np.ndarray,
    layers_w: list[dict],
    final_norm: np.ndarray,
    lm_head: np.ndarray,
    *,
    n_q: int,
    n_kv: int,
    head_dim: int,
    layer_fn,
    nope_layers: frozenset[int] | None = None,
) -> np.ndarray:
    """Run a short prefill and return float32 logits for the last token."""
    nope = nope_layers or frozenset()
    caches = [
        (np.zeros((0, n_kv, head_dim), dtype=np.float32),
         np.zeros((0, n_kv, head_dim), dtype=np.float32))
        for _ in layers_w
    ]
    x = None
    # Some BLAS builds emit spurious divide/overflow warnings on small GEMVs.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for pos, tok in enumerate(token_ids.tolist()):
            x = embed[tok].astype(np.float32)
            for i, w in enumerate(layers_w):
                k_c, v_c = caches[i]
                kwargs = dict(n_q=n_q, n_kv=n_kv, head_dim=head_dim)
                if layer_fn is llama_layer:
                    kwargs["apply_rope"] = i not in nope
                x, k_c, v_c = layer_fn(x, w, pos, k_c, v_c, **kwargs)
                caches[i] = (k_c, v_c)
        assert x is not None
        normed = rms_norm(x, final_norm)
        return (lm_head.astype(np.float32) @ normed).astype(np.float32)
