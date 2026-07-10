"""Utility helpers for vllm-webgpu."""
from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
import torch
# Use the internal module-level function apply_top_k_top_p_pytorch rather than
# the public dispatcher apply_top_k_top_p. The dispatcher only passes
# allow_cpu_sync=True inside the is_cpu() branch, which is never entered for
# PlatformEnum.OOT. Without that flag the pure top-k path falls through to a
# full sort instead of the partial top-k optimisation, losing performance on
# every batch-1 top-k-only decode step. apply_top_k_top_p_pytorch is a
# public module-level function in vllm.v1.sample.ops.topk_topp_sampler that
# calls apply_top_k_only directly and is therefore the correct choice for
# WebGPU's CPU-backed tensor workflow.
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch, random_sample

SHADERS_DIR = Path(__file__).parent / "shaders"


def sample_token(
    logits_1d: np.ndarray,
    temperature: float,
    top_p: float = 1.0,
    top_k: int = 0,
    seed: int | None = None,
) -> int:
    """Sample one token from a 1-D float32 logit vector.

    Applies (in order): temperature scaling, top-k filtering, top-p nucleus
    filtering, then draws from the resulting categorical distribution.
    Returns argmax when temperature < 1e-5.

    Args:
        logits_1d: 1-D float32 logit vector of length vocab_size.
        temperature: Softmax temperature. Values < 1e-5 produce greedy argmax.
        top_p: Nucleus probability mass cutoff (0, 1]. 1.0 disables.
        top_k: Keep at most top_k tokens. 0 disables.
        seed: Optional per-request RNG seed from SamplingParams. When set,
            a seeded torch.Generator is passed to random_sample so that
            the draw is deterministic and reproducible across runs.
    """
    if temperature < 1e-5:
        return int(logits_1d.argmax())

    logits_t = torch.as_tensor(logits_1d, dtype=torch.float32).unsqueeze(0)
    logits_t = logits_t / temperature
    k_t = torch.tensor([top_k]) if top_k > 0 else None
    p_t = torch.tensor([top_p]) if 0.0 < top_p < 1.0 else None
    # allow_cpu_sync=True enables the faster partial-topk path (apply_top_k_only)
    # when only top-k is needed (p_t is None). WebGPU tensors always live on CPU.
    # apply_top_k_top_p_pytorch returns logits unchanged when both k and p are None.
    filtered = apply_top_k_top_p_pytorch(logits_t, k_t, p_t, allow_cpu_sync=True)

    generators = {0: torch.Generator().manual_seed(seed)} if seed is not None else {}

    return random_sample(filtered.softmax(dim=-1), generators).item()
