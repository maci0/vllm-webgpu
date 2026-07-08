"""Utility helpers for vllm-webgpu."""
from __future__ import annotations
from pathlib import Path

SHADERS_DIR = Path(__file__).parent / "shaders"

_OVERHEAD_BYTES = 512 * 1024 * 1024  # 512MB buffer for driver overhead + activations
