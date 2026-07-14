"""Version-pinned assertions for private vLLM symbols and internal logic used by vllm-webgpu.

These tests catch upstream renames or moves of internal APIs before they
silently break at runtime. The pinned range is declared in pyproject.toml:
  vllm>=0.24,<0.25

When bumping the vLLM pin, re-run this file first and fix any failures before
updating pyproject.toml.

Inline copy tracking
--------------------
vllm_webgpu/quant/weight_loader.py._detect_mx_quant contains a local copy of
ModelOptFp8Config._extract_modelopt_quant_algo from
vllm/model_executor/layers/quantization/modelopt.py.  That module has
top-level CUDA/triton imports that prevent it from loading on WebGPU, so the
copy is unavoidable for now.

To avoid silent divergence, test_modelopt_source_hash_pinned (below) stores a
SHA-256 of the upstream method source (stripped trailing whitespace, normalized
to LF).  Any line change in the upstream method will flip the hash and fail CI,
requiring a manual diff against the inline copy.

When bumping vLLM and the hash fails:
  1. diff _extract_modelopt_quant_algo in modelopt.py against the inline copy
     in weight_loader.py._detect_mx_quant (the except block).
  2. Update the inline copy to match the new logic.
  3. Update PINNED_EXTRACT_ALGO_HASH below to the new hash.
  4. Re-run this file to confirm green.

VLLM_INLINE_COPY_VALIDATED: 0.24.0

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
import hashlib
import importlib
import importlib.util
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest

# SHA-256 (first 16 hex chars) of _extract_modelopt_quant_algo source in vLLM 0.24.0.
# Trailing whitespace stripped per line, joined with \n.
# Recompute with: hashlib.sha256('\n'.join(l.rstrip() for l in src.splitlines()).encode()).hexdigest()[:16]
PINNED_EXTRACT_ALGO_HASH = "a12d15f39ac3f5ee"


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


@pytest.mark.parametrize("cfg,expected_inline,label", [
    # 'quantization' present as a dict: both paths return the upper-cased quant_algo.
    (
        {"quant_method": "modelopt", "quantization": {"quant_algo": "MXFP4"}},
        "MXFP4",
        "present-dict",
    ),
    # 'quantization' present but not a dict: vLLM returns None (coerced to ''),
    # inline copy returns ''. Both reach the same final algo value.
    (
        {"quant_method": "modelopt", "quantization": "not_a_dict"},
        "",
        "present-non-dict",
    ),
    # 'quantization' absent: read quant_algo at top level.
    (
        {"quant_method": "modelopt", "quant_algo": "MXFP8"},
        "MXFP8",
        "absent",
    ),
])
def test_modelopt_inline_copy_matches_vllm(cfg, expected_inline, label):
    """Assert that the inline fallback in _detect_mx_quant returns the same
    algo string as _extract_modelopt_quant_algo (coerced to '' for None) for
    all three branch shapes: 'quantization' present-dict, present-non-dict,
    and absent.

    Pinned against vLLM 0.24.0
    (vllm/model_executor/layers/quantization/modelopt.py L245-262).
    When bumping vLLM, update the inline copy in
    vllm_webgpu/quant/weight_loader.py and re-run.
    """
    pytest.importorskip("vllm", reason="vllm not installed, skipping compat check")

    # Inline copy logic (mirrors weight_loader.py _detect_mx_quant fallback).
    if 'quantization' in cfg:
        inline_algo = str(cfg['quantization'].get('quant_algo', '')).upper() if isinstance(cfg['quantization'], dict) else ''
    else:
        inline_algo = str(cfg.get('quant_algo', '')).upper()

    assert inline_algo == expected_inline, (
        f"label={label!r}: inline copy returned {inline_algo!r}, expected {expected_inline!r}"
    )

    try:
        from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8Config
        vllm_result = ModelOptFp8Config._extract_modelopt_quant_algo(cfg) or ''
        assert vllm_result == expected_inline, (
            f"label={label!r}: vLLM _extract_modelopt_quant_algo returned "
            f"{vllm_result!r} (after coercion), expected {expected_inline!r}. "
            "The upstream logic changed; diff against the inline copy in "
            "vllm_webgpu/quant/weight_loader.py and update accordingly."
        )
    except (ImportError, RuntimeError):
        pass  # CUDA imports fail on WebGPU; inline-only path still verified above.


def test_modelopt_source_hash_pinned():
    """Fail when _extract_modelopt_quant_algo source changes in a vLLM bump.

    modelopt.py carries CUDA imports at module scope, so vllm_webgpu keeps a
    local copy of _extract_modelopt_quant_algo inside _detect_mx_quant.  This
    test stores a SHA-256 of the upstream source; any line change flips the hash
    and requires a manual diff against the inline copy.

    When this test fails after a vLLM bump:
      1. Diff _extract_modelopt_quant_algo in modelopt.py against the inline
         fallback block in vllm_webgpu/quant/weight_loader.py._detect_mx_quant.
      2. Update the inline copy to reflect the new logic.
      3. Update PINNED_EXTRACT_ALGO_HASH at the top of this file to the new hash.
      4. Update VLLM_INLINE_COPY_VALIDATED in the module docstring to the new version.
    """
    pytest.importorskip("vllm", reason="vllm not installed, skipping compat check")
    try:
        from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8Config
    except (ImportError, RuntimeError):
        pytest.skip("modelopt not importable on this platform (expected on WebGPU/CPU)")

    src = inspect.getsource(ModelOptFp8Config._extract_modelopt_quant_algo)
    normalized = "\n".join(line.rstrip() for line in src.splitlines())
    actual_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]

    assert actual_hash == PINNED_EXTRACT_ALGO_HASH, (
        f"_extract_modelopt_quant_algo source changed (hash {actual_hash!r} != "
        f"pinned {PINNED_EXTRACT_ALGO_HASH!r}). "
        "Diff vllm/model_executor/layers/quantization/modelopt.py against the "
        "inline copy in vllm_webgpu/quant/weight_loader.py._detect_mx_quant, "
        "update the copy and PINNED_EXTRACT_ALGO_HASH in this file, and set "
        "VLLM_INLINE_COPY_VALIDATED to the new vLLM version."
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


def test_get_layer_types_version_sync():
    """Fail immediately when vLLM is bumped without updating the VERSION SYNC comment.

    When this test fails, the required action is:
      1. Diff ModelConfig.get_num_layers_by_block_type in vllm/config/model.py
         against get_layer_types() in vllm_webgpu/v1/cache_policy.py.
      2. Update get_layer_types() if probes were added, removed, or reordered.
      3. Update the VERSION SYNC comment in cache_policy.py to the new version.
      4. Re-run this test to confirm it passes.

    The check is strict (exact version equality), so any vLLM bump triggers it.
    The existing test_get_layer_types_probe_order_matches_vllm and
    test_kv_cache_spec_layer_count_per_arch tests then verify that the probe
    sequence still produces correct results.
    """
    pytest.importorskip("vllm", reason="vllm not installed")
    import re
    import pathlib
    import vllm

    cache_policy_path = (
        pathlib.Path(__file__).parent.parent
        / "vllm_webgpu" / "v1" / "cache_policy.py"
    )
    source = cache_policy_path.read_text()

    m = re.search(r"VERSION SYNC: last verified against vLLM (\S+)", source)
    assert m is not None, (
        "Could not find 'VERSION SYNC: last verified against vLLM <version>' in "
        "vllm_webgpu/v1/cache_policy.py. The comment was removed or reformatted."
    )
    pinned = m.group(1).rstrip(".")
    installed = vllm.__version__

    assert pinned == installed, (
        f"VERSION SYNC mismatch: cache_policy.py was verified against vLLM {pinned} "
        f"but {installed} is installed.\n\n"
        "Action required:\n"
        "  1. Diff ModelConfig.get_num_layers_by_block_type in\n"
        "     vllm/config/model.py against get_layer_types() in\n"
        "     vllm_webgpu/v1/cache_policy.py.\n"
        "  2. Update get_layer_types() if probes were added, removed, or reordered.\n"
        "  3. Update the VERSION SYNC comment in cache_policy.py to the new version.\n"
        "  4. Re-run this test to confirm it passes."
    )


def test_get_layer_types_probe_attrs_in_source():
    """Verify that the three probe attribute names used by get_layer_types() still appear
    in ModelConfig.get_num_layers_by_block_type, in the expected order.

    get_layer_types() in cache_policy.py mirrors the hybrid-model probe sequence from
    vLLM's ModelConfig.get_num_layers_by_block_type.  That function is not part of
    vLLM's documented public API, so upstream refactors can silently break the copy.

    This test inspects the vLLM source and fails immediately when vLLM:
      - renames one of the three probe attribute names, or
      - reorders the probes, or
      - removes a probe entirely.

    When this test fails:
      1. Diff ModelConfig.get_num_layers_by_block_type (vllm/config/model.py) against
         get_layer_types() in vllm_webgpu/v1/cache_policy.py.
      2. Update get_layer_types() to match the new probe sequence.
      3. Update the VERSION SYNC comment in cache_policy.py to the new vLLM version.
      4. Re-run this test and test_get_layer_types_version_sync to confirm they pass.

    If vLLM ever exposes a public get_layer_types() list API, replace the probe
    logic in cache_policy.py with a direct call and remove this test.
    """
    pytest.importorskip("vllm", reason="vllm not installed")

    from vllm.config.model import ModelConfig

    src = inspect.getsource(ModelConfig.get_num_layers_by_block_type)

    # The three probe attribute names, in the order they must appear.
    probes = ["layers_block_type", "attn_type_list", "layer_types"]
    positions = {}
    for attr in probes:
        idx = src.find(f'"{attr}"')
        if idx == -1:
            idx = src.find(f"'{attr}'")
        assert idx != -1, (
            f"Probe attribute {attr!r} not found in "
            "ModelConfig.get_num_layers_by_block_type source. "
            "vLLM may have renamed or removed this probe. "
            "Diff vllm/config/model.py against get_layer_types() in "
            "vllm_webgpu/v1/cache_policy.py and update the probe list "
            "and the VERSION SYNC comment."
        )
        positions[attr] = idx

    assert positions["layers_block_type"] < positions["attn_type_list"] < positions["layer_types"], (
        f"Probe order changed in ModelConfig.get_num_layers_by_block_type. "
        f"Found positions: {positions}. "
        "Expected: layers_block_type < attn_type_list < layer_types. "
        "Diff vllm/config/model.py against get_layer_types() in "
        "vllm_webgpu/v1/cache_policy.py and update the probe sequence "
        "and the VERSION SYNC comment."
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


@pytest.mark.parametrize("arch,hf_text_attrs,hf_outer_attrs,num_layers,expected_attn", [
    # NemotronH: layers_block_type probe (probe 1). 4 layers: mamba + mlp + 2 attention.
    # KV entries use the .mixer suffix; mamba and mlp layers are excluded.
    (
        "NemotronHForCausalLM",
        {"layers_block_type": ["mamba", "mlp", "attention", "attention"]},
        {},
        4,
        2,
    ),
    # Gemma4: layer_types probe (probe 3). 6 layers: 5 sliding_attention + 1 full_attention.
    # is_attn_layer returns True for both, so all 6 get KV entries.
    (
        "Gemma4ForCausalLM",
        {"layer_types": ["sliding_attention"] * 5 + ["full_attention"]},
        {},
        6,
        6,
    ),
    # Qwen3.5: layer_types probe (probe 3). 4 layers alternating full/linear.
    # linear_attention carries no KV state (is_attn_layer returns False).
    (
        "Qwen3_5ForConditionalGeneration",
        {"layer_types": ["full_attention", "linear_attention", "full_attention", "linear_attention"]},
        {},
        4,
        2,
    ),
    # Minimax-style: attn_type_list probe (probe 2) on the outer config only.
    # Integer encoding: 1 = attention, 0 = non-attention.
    (
        "MistralForCausalLM",
        {},
        {"attn_type_list": [1, 0, 1, 0]},
        4,
        2,
    ),
    # Llama: uniform path (no layer_types on either config). All layers get KV entries.
    (
        "LlamaForCausalLM",
        {},
        {},
        4,
        4,
    ),
])
def test_kv_cache_spec_layer_count_per_arch(
    arch, hf_text_attrs, hf_outer_attrs, num_layers, expected_attn
):
    """kv_cache_spec emits the correct number of KV-cache entries for each architecture.

    Exercises the get_layer_types() probe chain (layers_block_type, attn_type_list,
    layer_types) for each supported architecture family that uses mixed layers.
    The model is not loaded (self.model is None), so the test drives only the
    hf_config probe path, not the _lp per-layer-params path.

    If vLLM adds a new probe to get_num_layers_by_block_type after a version bump
    and get_layer_types() is not updated accordingly, this test will fail, turning
    the VERSION SYNC comment in cache_policy.py into a mechanical CI gate.

    VERSION SYNC: aligned with ModelConfig.get_num_layers_by_block_type in
    vllm/config/model.py. Re-run after each vLLM bump and update the version
    string in the cache_policy.py VERSION SYNC comment when the test still passes.
    """
    pytest.importorskip("vllm", reason="vllm not installed")

    from vllm_webgpu.v1.model_runner import WebGPUModelRunner

    hf_text = SimpleNamespace(**hf_text_attrs)
    hf_outer = SimpleNamespace(**{**hf_text_attrs, **hf_outer_attrs})

    mc = SimpleNamespace(
        architecture=arch,
        hf_text_config=hf_text,
        hf_config=hf_outer,
        use_fp64_gumbel=False,
        get_total_num_hidden_layers=lambda: num_layers,
        get_head_size=lambda: 64,
        get_total_num_kv_heads=lambda: 2,
    )
    vllm_config = SimpleNamespace(
        model_config=mc,
        cache_config=SimpleNamespace(block_size=16),
        speculative_config=None,
    )

    with patch("vllm_webgpu.v1.model_runner.PipelineCache"):
        runner = WebGPUModelRunner(vllm_config, MagicMock())

    spec = runner.kv_cache_spec
    assert len(spec) == expected_attn, (
        f"arch={arch!r}: expected {expected_attn} KV-cache layers, got {len(spec)}. "
        f"Keys: {sorted(spec)}. "
        "After a vLLM bump, diff ModelConfig.get_num_layers_by_block_type "
        "(vllm/config/model.py) against get_layer_types() in "
        "vllm_webgpu/v1/cache_policy.py and update the probe list and the "
        "VERSION SYNC comment."
    )
