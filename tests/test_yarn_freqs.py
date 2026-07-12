"""Smoke tests for compute_yarn_freqs against vLLM's YaRN implementation.

compute_yarn_freqs delegates directly to
YaRNScalingRotaryEmbedding._compute_inv_freq via a minimal stub built with
object.__new__, so it tracks vLLM's implementation automatically. These tests
verify that the delegation and parameter mapping are correct.

The reference helper _vllm_yarn_freqs below calls _compute_inv_freq via the
same stub pattern; it exists as an independent cross-check so that any mistake
in the attribute mapping inside compute_yarn_freqs surfaces as a test failure.
"""
from __future__ import annotations

import numpy as np
import pytest


_ROPE_THETA = 10000.0

_ROPE_SCALING = {
    "rope_type": "yarn",
    "rope_theta": _ROPE_THETA,
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


def _vllm_yarn_freqs(head_dim: int, rope_theta: float, rope_scaling: dict) -> "tuple[np.ndarray, float]":
    """Compute expected inv_freq by calling YaRNScalingRotaryEmbedding._compute_inv_freq directly.

    Constructs a minimal stub instance via object.__new__ to avoid triggering
    the full __init__ (which builds the cos/sin cache), then calls the private
    method so this reference always tracks any formula changes in vLLM.
    """
    from vllm.model_executor.layers.rotary_embedding.common import yarn_get_mscale
    from vllm.model_executor.layers.rotary_embedding.yarn_scaling_rope import (
        YaRNScalingRotaryEmbedding,
    )

    factor               = float(rope_scaling.get("factor", 1.0))
    beta_fast            = int(rope_scaling.get("beta_fast", 32))
    beta_slow            = int(rope_scaling.get("beta_slow", 1))
    orig_ctx             = int(rope_scaling.get("original_max_position_embeddings", 4096))
    extrapolation_factor = float(rope_scaling.get("extrapolation_factor", 1.0))
    attn_factor          = float(rope_scaling.get("attn_factor", 1.0))
    apply_yarn_scaling   = bool(rope_scaling.get("apply_yarn_scaling", True))

    inst = object.__new__(YaRNScalingRotaryEmbedding)
    inst.base                    = rope_theta
    inst.rotary_dim              = head_dim
    inst.beta_fast               = beta_fast
    inst.beta_slow               = beta_slow
    inst.max_position_embeddings = orig_ctx
    inst.truncate                = bool(rope_scaling.get("truncate", True))
    inst.extrapolation_factor    = extrapolation_factor

    inv_freq = inst._compute_inv_freq(factor)

    mscale = (
        float(yarn_get_mscale(factor) * attn_factor)
        if apply_yarn_scaling
        else float(attn_factor)
    )

    return inv_freq.numpy().astype(np.float32), mscale


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("vllm"),
    reason="vllm not installed",
)
def test_yarn_freqs_matches_vllm():
    """compute_yarn_freqs must produce the same output as the independent reference.

    Both sides call _compute_inv_freq via the same object.__new__ stub pattern,
    but compute_yarn_freqs goes through the public wrapper while _vllm_yarn_freqs
    calls the method directly. Any attribute-mapping mistake in the wrapper will
    surface here.
    """
    from vllm_webgpu.models.base import compute_yarn_freqs

    freqs_ours, mscale_ours = compute_yarn_freqs(
        _HEAD_DIM, _ROPE_SCALING
    )
    freqs_vllm, mscale_vllm = _vllm_yarn_freqs(
        _HEAD_DIM, _ROPE_THETA, _ROPE_SCALING
    )

    np.testing.assert_allclose(
        freqs_ours, freqs_vllm, rtol=1e-5, atol=1e-7,
        err_msg=(
            "compute_yarn_freqs attribute mapping diverged from _vllm_yarn_freqs. "
            "Check the stub attribute assignments in base.py::compute_yarn_freqs."
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
        _HEAD_DIM, _ROPE_SCALING, rotary_dim=rotary_dim
    )
    assert freqs.shape == (rotary_dim // 2,), (
        f"expected ({rotary_dim // 2},) freqs, got {freqs.shape}"
    )
