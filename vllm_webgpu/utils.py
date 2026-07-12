"""Utility helpers for vllm-webgpu."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
import torch
# vLLM v1 sampling internals verified against vllm>=0.24,<0.25.
# These paths have no stability guarantees; a patch release may move or rename
# them. Pin vllm in pyproject.toml and run CI against the exact pinned version.
# Update this comment and pyproject.toml when bumping the vLLM version.
#
# Use the internal module-level function apply_top_k_top_p_pytorch rather than
# the public dispatcher apply_top_k_top_p. The dispatcher only passes
# allow_cpu_sync=True inside the is_cpu() branch, which is never entered for
# PlatformEnum.OOT. Without that flag the pure top-k path falls through to a
# full sort instead of the partial top-k optimisation, losing performance on
# every batch-1 top-k-only decode step. apply_top_k_top_p_pytorch is a
# public module-level function in vllm.v1.sample.ops.topk_topp_sampler that
# calls apply_top_k_only directly when allow_cpu_sync=True and no top-p threshold
# is active (p is None); for combined top-k+top-p requests it still uses the sort
# path, which is acceptable given WebGPU's single-request batch size. It is therefore
# the correct choice for WebGPU's CPU-backed tensor workflow.
# When vLLM fixes apply_top_k_top_p to pass allow_cpu_sync=True for OOT
# platforms, replace the two imports below with:
#   from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p, random_sample
# and change the call in sample_token from apply_top_k_top_p_pytorch(...) to:
#   filtered = apply_top_k_top_p(logits_t, k_t, p_t)
# No other changes needed; the dispatcher will select the fast path automatically.
from vllm.v1.sample.ops.topk_topp_sampler import (
    apply_top_k_top_p_pytorch,
    random_sample,
)
SHADERS_DIR = Path(__file__).parent / "shaders"


@lru_cache(maxsize=16)
def zero_bytes(n: int) -> bytes:
    """Return a cached immutable bytes object of n zero bytes.

    Used to zero-initialise GPU buffers without allocating a new object on
    every call. The cache is keyed by byte count, so each unique size is
    allocated exactly once for the lifetime of the process.

    maxsize=16 bounds memory: fixed callers in model_runner/base use 2-4
    distinct sizes; variable callers (e.g. diffusion_gemma per-request) are
    bounded by LRU eviction rather than accumulating stale entries forever.
    """
    return bytes(n)


def sample_token(
    logits_1d: np.ndarray,
    temperature: float,
    top_p: float = 1.0,
    top_k: int = 0,
    generator: torch.Generator | None = None,
    use_fp64_gumbel: bool = False,
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
        generator: Optional per-request torch.Generator. The caller is
            responsible for seeding it once and passing the same object on
            every decode step so the RNG state advances correctly between
            steps. When None, sampling is non-deterministic.
        use_fp64_gumbel: When True, Gumbel noise is sampled in fp64 for
            higher numerical precision. Mirrors ModelConfig.use_fp64_gumbel.
    """
    if temperature < 1e-5:  # matches vLLM's _SAMPLING_EPS = 1e-5 in vllm/v1/sample/sampler.py
        return logits_1d.argmax().item()

    logits_t = torch.as_tensor(logits_1d, dtype=torch.float32).unsqueeze(0)
    logits_t = logits_t / temperature
    k_t = torch.tensor([top_k]) if top_k > 0 else None
    p_t = torch.tensor([top_p]) if 0.0 < top_p < 1.0 else None
    # allow_cpu_sync=True enables the faster partial-topk path (apply_top_k_only)
    # when only top-k is needed (p_t is None). WebGPU tensors always live on CPU.
    # apply_top_k_top_p_pytorch returns logits unchanged when both k and p are None.
    filtered = apply_top_k_top_p_pytorch(logits_t, k_t, p_t, allow_cpu_sync=True)

    generators = {0: generator} if generator is not None else {}

    return random_sample(filtered.softmax(dim=-1), generators, use_fp64_gumbel=use_fp64_gumbel).item()
