"""Utility helpers for vllm-webgpu."""
from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
from vllm.utils.mem_constants import MiB_bytes

SHADERS_DIR = Path(__file__).parent / "shaders"

OVERHEAD_BYTES = 512 * MiB_bytes  # 512 MiB buffer for driver overhead + activations


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
    import torch
    # apply_top_k_top_p_pytorch is not in the stable vLLM public API, but the
    # public apply_top_k_top_p does not propagate allow_cpu_sync=True for OOT
    # platforms. We call the impl directly to get O(n) topk instead of O(n log n)
    # sort for large vocabs at batch=1.
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch, random_sample

    if temperature < 1e-5:
        return int(logits_1d.argmax())

    logits_t = torch.as_tensor(logits_1d, dtype=torch.float32).unsqueeze(0)
    logits_t = logits_t / temperature
    k_t = torch.tensor([top_k]) if top_k > 0 else None
    p_t = torch.tensor([top_p]) if 0.0 < top_p < 1.0 else None
    filtered = apply_top_k_top_p_pytorch(logits_t, k_t, p_t, allow_cpu_sync=True)

    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        generators: dict[int, torch.Generator] = {0: generator}
    else:
        generators = {}

    return random_sample(filtered.softmax(dim=-1, dtype=torch.float32), generators).item()
