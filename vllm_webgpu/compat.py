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


def reset_compat_patches() -> None:
    """Reset the applied flag so apply_compat_patches() runs again.

    Intended for test isolation only: pytest runs all tests in a single process,
    so the module-level _APPLIED flag persists across tests. Call this in a
    fixture or test teardown to ensure patches are re-applied in each test that
    needs a clean slate.
    """
    global _APPLIED
    _APPLIED = False
