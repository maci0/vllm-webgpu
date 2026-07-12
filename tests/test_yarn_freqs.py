"""Parity tests for compute_yarn_freqs against vLLM's YaRN implementation.

compute_yarn_freqs inlines the arithmetic from
YaRNScalingRotaryEmbedding._compute_inv_freq using only public vLLM helpers,
because calling _compute_inv_freq directly requires a fully constructed instance
whose __init__ allocates a potentially hundreds-of-MB cos/sin cache.

The reference helper _vllm_yarn_freqs below uses object.__new__ to build a
minimal stub and calls _compute_inv_freq on it, equivalent to reading the
inv_freq that __init__ would have stored, without paying for the cache.

Version pin: validated against vLLM 0.24.0. Run these tests after each vLLM
bump to catch formula drift in yarn_scaling_rope.py::_compute_inv_freq.
When vLLM exports a standalone public inv_freq helper, replace compute_yarn_freqs
with that call and simplify this test accordingly.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

# vLLM version this test was validated against.
_VALIDATED_VLLM_VERSION = "0.24.0"


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
    """compute_yarn_freqs must produce the same output as the vLLM reference."""
    import vllm
    if vllm.__version__ != _VALIDATED_VLLM_VERSION:
        warnings.warn(
            f"test_yarn_freqs_matches_vllm was validated against vLLM "
            f"{_VALIDATED_VLLM_VERSION}; running against {vllm.__version__}. "
            "If the test fails, check yarn_scaling_rope.py::_compute_inv_freq for "
            "formula changes and update compute_yarn_freqs in vllm_webgpu/models/base.py.",
            stacklevel=1,
        )

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
        _HEAD_DIM, _ROPE_SCALING, rotary_dim=rotary_dim
    )
    assert freqs.shape == (rotary_dim // 2,), (
        f"expected ({rotary_dim // 2},) freqs, got {freqs.shape}"
    )
