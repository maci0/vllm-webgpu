"""Utility helpers for vllm-webgpu."""
from __future__ import annotations
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SHADERS_DIR = Path(__file__).parent / "shaders"

_OVERHEAD_BYTES = 512 * 1024 * 1024  # 512MB buffer for driver overhead + activations


def shaders_dir(subdir: str = "generic") -> Path:
    return SHADERS_DIR / subdir
