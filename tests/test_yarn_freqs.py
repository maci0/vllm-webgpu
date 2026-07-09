"""Regression test: compute_yarn_freqs must stay in sync with vLLM's YaRN formula.

This test compares compute_yarn_freqs against vLLM's internal _compute_inv_freq
on a small known config. Any vLLM version bump that changes the YaRN formula will
cause this test to fail, surfacing the divergence before it reaches inference.
"""
from __future__ import annotations

import numpy as np
import pytest


_ROPE_SCALING = {
    "rope_type": "yarn",
    "factor": 4.0,
    "beta_fast": 32,
    "beta_slow": 1,
    "original_max_position_embeddings": 4096,
    "extrapolation_factor": 1.0,
    "attn_factor": 1.0,
    "apply_yarn_scaling": True,
    "truncate": True,
}
_HEAD_DIM = 128
_ROPE_THETA = 10000.0


def _vllm_yarn_freqs(head_dim: int, rope_theta: float, rope_scaling: dict) -> "tuple[np.ndarray, float]":
    """Compute expected inv_freq via vLLM's own YaRN helpers."""
    import torch
    from vllm.model_executor.layers.rotary_embedding.common import (
        yarn_find_correction_range,
        yarn_get_mscale,
        yarn_linear_ramp_mask,
    )

    factor = float(rope_scaling.get("factor", 1.0))
    beta_fast = int(rope_scaling.get("beta_fast", 32))
    beta_slow = int(rope_scaling.get("beta_slow", 1))
    orig_ctx = int(rope_scaling.get("original_max_position_embeddings", 4096))
    extrapolation_factor = float(rope_scaling.get("extrapolation_factor", 1.0))
    attn_factor = float(rope_scaling.get("attn_factor", 1.0))
    apply_yarn_scaling = bool(rope_scaling.get("apply_yarn_scaling", True))
    truncate = bool(rope_scaling.get("truncate", True))

    mscale = (
        float(yarn_get_mscale(factor) * attn_factor)
        if apply_yarn_scaling
        else float(attn_factor)
    )

    pos_freqs = rope_theta ** (
        torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    )
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    low, high = yarn_find_correction_range(
        beta_fast, beta_slow, head_dim, rope_theta, orig_ctx, truncate
    )
    inv_freq_mask = (
        1 - yarn_linear_ramp_mask(low, high, head_dim // 2, dtype=torch.float32)
    ) * extrapolation_factor
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_mask)
        + inv_freq_extrapolation * inv_freq_mask
    )
    return inv_freq.numpy().astype(np.float32), mscale


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("vllm"),
    reason="vllm not installed",
)
def test_yarn_freqs_matches_vllm():
    """compute_yarn_freqs output must match vLLM's internal YaRN formula."""
    from vllm_webgpu.models.base import compute_yarn_freqs

    freqs_ours, mscale_ours = compute_yarn_freqs(
        _HEAD_DIM, _ROPE_THETA, _ROPE_SCALING
    )
    freqs_vllm, mscale_vllm = _vllm_yarn_freqs(
        _HEAD_DIM, _ROPE_THETA, _ROPE_SCALING
    )

    np.testing.assert_allclose(
        freqs_ours, freqs_vllm, rtol=1e-5, atol=1e-7,
        err_msg=(
            "compute_yarn_freqs diverged from vLLM's _compute_inv_freq. "
            "Check if vLLM bumped the YaRN formula and update base.py accordingly."
        ),
    )
    assert abs(mscale_ours - mscale_vllm) < 1e-6, (
        f"mscale mismatch: ours={mscale_ours} vllm={mscale_vllm}"
    )


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("vllm"),
    reason="vllm not installed",
)
def test_yarn_freqs_partial_rope():
    """Partial RoPE (rotary_dim < head_dim) produces the correct output length."""
    from vllm_webgpu.models.base import compute_yarn_freqs

    rotary_dim = 64  # half of head_dim=128
    freqs, _ = compute_yarn_freqs(
        _HEAD_DIM, _ROPE_THETA, _ROPE_SCALING, rotary_dim=rotary_dim
    )
    assert freqs.shape == (rotary_dim // 2,), (
        f"expected ({rotary_dim // 2},) freqs, got {freqs.shape}"
    )
