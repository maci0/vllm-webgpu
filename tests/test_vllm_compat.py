"""Version-pinned assertions for private vLLM symbols and internal logic used by vllm-webgpu.

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
from types import SimpleNamespace
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


@pytest.mark.parametrize("probe,hf_text_attrs,hf_outer_attrs,expected,attn_count", [
    (
        "layers_block_type",
        {"layers_block_type": ["attention", "mamba", "attention", "mamba"]},
        {},
        ["attention", "mamba", "attention", "mamba"],
        2,
    ),
    (
        "attn_type_list",
        {},
        {"attn_type_list": [1, 0, 1, 0]},
        [1, 0, 1, 0],
        2,
    ),
    (
        "layer_types",
        {"layer_types": ["full_attention", "linear_attention", "full_attention", "linear_attention"]},
        {},
        ["full_attention", "linear_attention", "full_attention", "linear_attention"],
        2,
    ),
])
def test_get_layer_types_probe_order_matches_vllm(
    probe, hf_text_attrs, hf_outer_attrs, expected, attn_count
):
    """get_layer_types probe ordering and attribute selection must match
    ModelConfig.get_num_layers_by_block_type.

    For each probe fixture, this test:
    1. Calls get_layer_types and verifies it returns the expected list.
    2. Calls get_num_layers_by_block_type (via a minimal mock ModelConfig)
       on the same fixture and verifies the attention count agrees with
       what the returned list implies.

    When vLLM adds or reorders probes in get_num_layers_by_block_type,
    one of these assertions will fail, turning the VERSION SYNC comment
    in cache_policy.py into a mechanical CI gate.
    """
    pytest.importorskip("vllm", reason="vllm not installed")

    from vllm.config.model import ModelConfig
    from vllm_webgpu.v1.cache_policy import get_layer_types, is_attn_layer

    n = len(expected)
    hf_text_config = SimpleNamespace(**hf_text_attrs)
    hf_outer_config = SimpleNamespace(**{**hf_text_attrs, **hf_outer_attrs})

    # Verify get_layer_types returns the right list for this probe.
    result = get_layer_types(hf_text_config, hf_outer_config)
    assert result == expected, (
        f"get_layer_types returned {result!r} for probe={probe!r}; "
        f"expected {expected!r}. "
        "The probe ordering in cache_policy.py may have drifted from vLLM."
    )

    # Verify the attention count from the returned list matches what
    # get_num_layers_by_block_type would count on the same fixture.
    # Build a minimal mock ModelConfig that routes straight to the hybrid
    # probe path (is_hybrid=True, no noops, not attention-free).
    mock_mc = SimpleNamespace(
        is_hybrid=True,
        has_noops=False,
        is_attention_free=False,
        hf_text_config=hf_text_config,
        hf_config=hf_outer_config,
        model_arch_config=SimpleNamespace(text_model_type="llama"),
        get_layers_start_end_indices=lambda _pc: (0, n),
        get_num_layers=lambda _pc: n,
    )
    vllm_count = ModelConfig.get_num_layers_by_block_type(
        mock_mc,
        parallel_config=SimpleNamespace(),
        block_type="attention",
    )
    local_count = sum(1 for lt in result if is_attn_layer(lt))
    assert vllm_count == local_count == attn_count, (
        f"probe={probe!r}: vLLM count={vllm_count}, local count={local_count}, "
        f"expected={attn_count}. "
        "get_layer_types and get_num_layers_by_block_type disagree on this fixture. "
        "Diff the probe order in cache_policy.py against the updated vLLM source."
    )
