"""Version-pinned assertions for private vLLM symbols used by vllm-webgpu.

These tests catch upstream renames or moves of internal APIs before they
silently break at runtime. The pinned range is declared in pyproject.toml:
  vllm>=0.24,<0.25

When bumping the vLLM pin, re-run this file first and fix any failures before
updating pyproject.toml.

Background: apply_top_k_top_p_pytorch and random_sample are not part of
vLLM's documented public API. They live in
vllm.v1.sample.ops.topk_topp_sampler and are used in place of the public
dispatcher apply_top_k_top_p because the dispatcher never passes
allow_cpu_sync=True for PlatformEnum.OOT, which forces a full sort instead of
the faster partial top-k path. Once vLLM fixes that dispatcher for OOT
platforms, these two imports can be replaced with the public API and these
assertions can be removed.

Until then, the assertions below act as an early-warning system.
"""
import importlib
import importlib.util
import pytest


def _attr_exists(module_path: str, attr: str) -> bool:
    """Return True if module_path.attr is importable without errors."""
    spec = importlib.util.find_spec(module_path)
    if spec is None:
        return False
    mod = importlib.import_module(module_path)
    return hasattr(mod, attr)


@pytest.mark.parametrize("module_path,symbol", [
    (
        "vllm.v1.sample.ops.topk_topp_sampler",
        "apply_top_k_top_p_pytorch",
    ),
    (
        "vllm.v1.sample.ops.topk_topp_sampler",
        "random_sample",
    ),
])
def test_vllm_private_symbol_exists(module_path, symbol):
    """Assert that the private vLLM symbol used in vllm_webgpu/utils.py still
    exists at the expected module path under the pinned vLLM version range
    (>=0.24,<0.25).

    Failure here means a vLLM patch release moved or renamed the symbol.
    Fix: update the import in vllm_webgpu/utils.py to the new location and
    bump the vLLM pin in pyproject.toml, then update this file.
    """
    pytest.importorskip("vllm", reason="vllm not installed, skipping compat check")
    assert _attr_exists(module_path, symbol), (
        f"vLLM private symbol {module_path}.{symbol} no longer exists. "
        f"The import in vllm_webgpu/utils.py must be updated to match the "
        f"current vLLM version. See the comment block at the top of utils.py "
        f"for the migration path."
    )
