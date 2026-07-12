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
import inspect
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


def test_modelopt_extract_quant_algo_drift():
    """Detect drift in _extract_modelopt_quant_algo (vllm 0.24).

    vllm_webgpu/quant/weight_loader.py._detect_mx_quant contains a local
    copy of the quant_method/quant_algo extraction logic from
    ModelOptQuantConfigBase._extract_modelopt_quant_algo
    (vllm/model_executor/layers/quantization/modelopt.py). That class imports
    CUDA kernels at module scope, making it permanently unimportable on WebGPU.

    This test imports modelopt under try/except (expected to fail on WebGPU),
    and when it succeeds, compares the function's source lines against the
    known-good two-branch pattern so a vLLM bump that changes the parsing
    logic is caught by CI rather than silently diverging in the local copy.

    Pinned against vLLM 0.24. When bumping, diff the two branches in
    _extract_modelopt_quant_algo against _detect_mx_quant in weight_loader.py.
    """
    pytest.importorskip("vllm", reason="vllm not installed, skipping compat check")

    try:
        from vllm.model_executor.layers.quantization import modelopt as _modelopt_mod
    except (ImportError, RuntimeError):
        pytest.skip("modelopt not importable on this platform (expected on WebGPU/CPU)")

    # Locate the extraction method on whichever base class vLLM 0.24 uses.
    cls = None
    for attr in ("ModelOptQuantConfigBase", "ModelOptFp8Config"):
        cls = getattr(_modelopt_mod, attr, None)
        if cls is not None:
            break
    assert cls is not None, (
        "Could not find ModelOptQuantConfigBase or ModelOptFp8Config in "
        "vllm.model_executor.layers.quantization.modelopt. "
        "Diff _detect_mx_quant in weight_loader.py against the new upstream class."
    )

    method = getattr(cls, "_extract_modelopt_quant_algo", None)
    assert method is not None, (
        f"{cls.__name__} no longer has _extract_modelopt_quant_algo. "
        "Diff _detect_mx_quant in weight_loader.py against the updated upstream."
    )

    src = inspect.getsource(method)
    # The two-branch pattern: 'quantization' key present vs. top-level quant_algo.
    # These string fragments are stable identifiers; a refactor that changes the
    # branch structure will fail this check and require a manual diff.
    for fragment in ('quantization', 'quant_algo'):
        assert fragment in src, (
            f"_extract_modelopt_quant_algo no longer references {fragment!r}. "
            "The hf_quant_config.json parsing logic changed upstream. "
            "Diff _detect_mx_quant in vllm_webgpu/quant/weight_loader.py "
            "against the updated _extract_modelopt_quant_algo and update the copy."
        )
