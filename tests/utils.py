"""Test utility helpers shared across test modules."""
from __future__ import annotations

import numpy as np


def _fp8_e4m3_to_f32(raw: "np.ndarray") -> "np.ndarray":
    """Decode FP8 E4M3 (OCP format, exponent bias=7) byte array to float32.

    Uses torch's native float8_e4m3fn dtype for correct OCP semantics.
    Input: uint8 array of any shape. Output: float32 array, same shape.
    """
    import torch
    flat = np.ascontiguousarray(raw).ravel().view(np.uint8)
    return torch.frombuffer(bytearray(flat.tobytes()), dtype=torch.float8_e4m3fn).to(torch.float32).numpy().reshape(raw.shape)
