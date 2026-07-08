"""Utility helpers for vllm-webgpu."""
from __future__ import annotations
from pathlib import Path

import numpy as np

SHADERS_DIR = Path(__file__).parent / "shaders"

_OVERHEAD_BYTES = 512 * 1024 * 1024  # 512MB buffer for driver overhead + activations


def sample_token(
    logits_1d: "np.ndarray",
    temperature: float,
    top_p: float = 1.0,
    top_k: int = 0,
) -> int:
    """Sample one token from a 1-D float32 logit vector.

    Applies (in order): temperature scaling, top-k filtering, top-p nucleus
    filtering, then draws from the resulting categorical distribution.
    Returns argmax when temperature <= 1e-5.

    Args:
        logits_1d: 1-D float32 logit vector of length vocab_size.
        temperature: Softmax temperature. Values <= 1e-5 produce greedy argmax.
        top_p: Nucleus probability mass cutoff (0, 1]. 1.0 disables.
        top_k: Keep at most top_k tokens. 0 disables.
    """
    if temperature <= 1e-5:
        return int(np.argmax(logits_1d))

    raw = logits_1d.astype(np.float32)

    # Temperature scaling with numerically stable softmax.
    raw -= raw.max()
    probs = np.exp(raw / temperature).astype(np.float32)
    probs /= probs.sum()

    # Top-k: keep exactly k tokens.
    if top_k > 0:
        k = min(top_k, len(probs))
        top_k_idx = np.argpartition(probs, -k)[-k:]
        out = np.zeros_like(probs)
        out[top_k_idx] = probs[top_k_idx]
        s = out.sum()
        probs = out / s if s > 0 else out

    # Top-p (nucleus): keep the smallest set whose cumulative probability exceeds top_p.
    if 0.0 < top_p < 1.0:
        sorted_idx = np.argsort(probs)[::-1]
        cumsum = np.cumsum(probs[sorted_idx])
        cutoff = max(1, int(np.searchsorted(cumsum, top_p, side="left")) + 1)
        keep = sorted_idx[:cutoff]
        out = np.zeros_like(probs)
        out[keep] = probs[keep]
        s = out.sum()
        probs = out / s if s > 0 else out

    return int(np.random.choice(len(probs), p=probs))
