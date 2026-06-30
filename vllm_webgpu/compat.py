"""Compatibility patches for vLLM + vllm-webgpu version mismatches."""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)
_APPLIED = False


def apply_compat_patches() -> None:
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True
    # Add patches here as vLLM API mismatches surface.
    # Pattern: check for the issue, patch it, log at DEBUG level.
    logger.debug("vllm-webgpu compat patches applied (none active)")
