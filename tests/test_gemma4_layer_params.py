"""Regression guard for _build_layer_params_from_config vs vLLM Gemma4 formulas.

The per-layer head_dim / num_kv_heads / KV-sharing formulas in
vllm_webgpu/models/gemma4.py are transcribed from vLLM's Gemma4DecoderLayer
and Gemma4Attention constructors. This test:

  1. Pins the installed vLLM version.  If the version changes, the test
     fails loudly so a developer must re-audit the upstream formulas.

  2. Verifies that specific formula strings are still present in the vLLM
     source.  If upstream reformulates the logic (even in the same version)
     the string checks fire.

  3. Independently computes expected per-layer params using those same
     formulas (applied in pure Python, no GPU required) and asserts they
     match what _build_layer_params_from_config produces.

Source references pinned to vllm 0.24.0:
  vllm/model_executor/models/gemma4.py
    Gemma4Attention.__init__    L461-483  (KV sharing)
    Gemma4DecoderLayer.__init__ L559-580  (head_dim / num_kv_heads)
    Gemma4DecoderLayer.__init__ L599-608  (intermediate_size)
"""
from __future__ import annotations

import inspect
import pathlib
import re
import types

import pytest

# Read the canonical pin from the source module rather than maintaining a
# separate copy that could silently drift.
_GEMMA4_SRC = (
    pathlib.Path(__file__).parent.parent / "vllm_webgpu" / "models" / "gemma4.py"
)
_m = re.search(
    r'_EXPECTED_VLLM_VERSION\s*=\s*["\'](.+?)["\']',
    _GEMMA4_SRC.read_text(),
)
assert _m, "Could not locate _EXPECTED_VLLM_VERSION in vllm_webgpu/models/gemma4.py"
_PINNED_VLLM_VERSION = _m.group(1)

# ---------------------------------------------------------------------------
# Source-level guards
# ---------------------------------------------------------------------------

# Only formulas vllm_webgpu still transcribes are pinned here. The per-layer
# head_dim / num_key_value_heads dispatch is no longer among them: 0.29 extracted
# it into `gemma4_layer_config`, which `_build_layer_params_from_config` now
# calls, so a change there reaches us directly instead of silently diverging.
# The delegation itself is pinned, since reverting it to an inline formula would
# otherwise go unnoticed.
_VLLM_DECODER_LAYER_PATTERNS = [
    # delegation to the shared resolver
    "layer_config = gemma4_layer_config(config, layer_idx)",
    "head_dim = layer_config.head_dim",
    "num_kv_heads = layer_config.num_key_value_heads",
    # intermediate_size boundary
    "first_kv_shared_layer_idx = config.num_hidden_layers - getattr(",
    # intermediate_size doubling
    "layer_intermediate_size = config.intermediate_size * (",
]

_VLLM_ATTENTION_PATTERNS = [
    # KV-sharing boundary (~L463)
    "first_kv_shared_layer_idx = config.num_hidden_layers - num_kv_shared_layers",
    # reversed-search (~L470)
    "len(prev_layers) - 1 - prev_layers[::-1].index(current_layer_type)",
]


def _import_vllm_gemma4():
    from vllm.model_executor.models import gemma4 as _mod
    return _mod


def test_vllm_version_pin():
    """Fail loudly when vLLM is bumped so a developer re-audits the formulas."""
    import vllm
    assert vllm.__version__ == _PINNED_VLLM_VERSION, (
        f"vLLM was bumped from {_PINNED_VLLM_VERSION} to {vllm.__version__!r}. "
        "Re-audit vllm/model_executor/models/gemma4.py "
        "(Gemma4Attention.__init__, Gemma4DecoderLayer.__init__, and "
        "gemma4_layer_config in transformers_utils/configs/gemma4.py) "
        "and update _build_layer_params_from_config + this test."
    )


def test_vllm_decoder_layer_source_patterns():
    """Assert the per-layer head_dim / intermediate_size formulas in vLLM source
    are unchanged from what _build_layer_params_from_config transcribed."""
    mod = _import_vllm_gemma4()
    src = inspect.getsource(mod.Gemma4DecoderLayer.__init__)
    for pattern in _VLLM_DECODER_LAYER_PATTERNS:
        assert pattern in src, (
            f"vLLM Gemma4DecoderLayer.__init__ no longer contains:\n  {pattern!r}\n"
            "The upstream formula changed. Update _build_layer_params_from_config "
            "and this test to match the new vLLM logic."
        )


def test_vllm_attention_source_patterns():
    """Assert the KV-sharing formulas in vLLM source are unchanged."""
    mod = _import_vllm_gemma4()
    src = inspect.getsource(mod.Gemma4Attention.__init__)
    for pattern in _VLLM_ATTENTION_PATTERNS:
        assert pattern in src, (
            f"vLLM Gemma4Attention.__init__ no longer contains:\n  {pattern!r}\n"
            "The upstream KV-sharing formula changed. Update "
            "_build_layer_params_from_config and this test."
        )


# ---------------------------------------------------------------------------
# Reference implementation (same formulas, applied in pure Python)
# ---------------------------------------------------------------------------

def _vllm_reference_layer_params(config, num_layers: int) -> list[dict]:
    """Derive per-layer params using the same formulas as vLLM.

    Deliberately an independent transcription, not a call into vLLM: it is what
    gives `test_build_layer_params_matches_vllm_reference` something to compare
    against now that `_build_layer_params_from_config` delegates the head_dim /
    num_key_value_heads dispatch to `gemma4_layer_config`.

    Source (vLLM 0.29.0):
      vllm/transformers_utils/configs/gemma4.py  gemma4_layer_config
      vllm/model_executor/models/gemma4.py       Gemma4DecoderLayer.__init__
      vllm/model_executor/models/gemma4.py       Gemma4Attention.__init__
    """
    # Replicate Gemma4Attention.__init__ L461-463
    num_kv_shared_layers = getattr(config, "num_kv_shared_layers", 0)
    first_kv_shared = num_layers - num_kv_shared_layers

    results = []
    for layer_idx in range(num_layers):
        # --- gemma4_layer_config (transformers_utils/configs/gemma4.py) ---
        layer_type = config.layer_types[layer_idx]
        is_full_attention = layer_type == "full_attention"

        if is_full_attention:
            head_dim = getattr(config, "global_head_dim", None) or config.head_dim
        else:
            head_dim = config.head_dim

        use_k_eq_v = is_full_attention and getattr(config, "attention_k_eq_v", False)
        if use_k_eq_v:
            num_kv_heads = (
                getattr(config, "num_global_key_value_heads", None)
                or config.num_key_value_heads
            )
        else:
            num_kv_heads = config.num_key_value_heads

        # --- Gemma4Attention.__init__ (KV sharing) ---
        if num_kv_shared_layers > 0 and layer_idx >= first_kv_shared:
            is_kv_shared = True
            prev_layers = config.layer_types[:first_kv_shared]
            current_layer_type = config.layer_types[layer_idx]
            kv_shared_target = (
                len(prev_layers) - 1 - prev_layers[::-1].index(current_layer_type)
            )
        else:
            is_kv_shared = False
            kv_shared_target = -1

        # --- Gemma4DecoderLayer.__init__ (intermediate_size) ---
        is_kv_shared_layer = layer_idx >= first_kv_shared > 0
        use_double_wide_mlp = (
            getattr(config, "use_double_wide_mlp", False) and is_kv_shared_layer
        )
        intermediate_size = config.intermediate_size * (2 if use_double_wide_mlp else 1)

        results.append({
            "head_dim": head_dim,
            "num_kv_heads": num_kv_heads,
            "is_kv_shared": is_kv_shared,
            "kv_shared_target": kv_shared_target,
            "intermediate_size": intermediate_size,
        })
    return results


# ---------------------------------------------------------------------------
# Configs under test
# ---------------------------------------------------------------------------

def _make_config(**overrides):
    """Minimal Gemma4-like config object."""
    cfg = types.SimpleNamespace(
        # 12 layers: 5 sliding + 1 full repeated twice
        num_hidden_layers=12,
        layer_types=(["sliding_attention"] * 5 + ["full_attention"]) * 2,
        hidden_size=2048,
        head_dim=256,
        global_head_dim=512,
        num_attention_heads=8,
        num_key_value_heads=8,
        num_global_key_value_heads=1,
        attention_k_eq_v=True,
        intermediate_size=4096,
        num_kv_shared_layers=0,
        use_double_wide_mlp=False,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


CONFIGS = {
    "no_sharing_no_keqv": _make_config(
        num_kv_shared_layers=0,
        attention_k_eq_v=False,
    ),
    "no_sharing_keqv": _make_config(
        num_kv_shared_layers=0,
        attention_k_eq_v=True,
    ),
    "kv_shared_2_layers": _make_config(
        num_kv_shared_layers=2,
        attention_k_eq_v=True,
    ),
    "kv_shared_dwm": _make_config(
        num_kv_shared_layers=4,
        use_double_wide_mlp=True,
        attention_k_eq_v=True,
    ),
    "uniform_no_global": _make_config(
        layer_types=["sliding_attention"] * 12,
        attention_k_eq_v=False,
        global_head_dim=256,  # same as head_dim — no diff
        num_global_key_value_heads=8,
        num_kv_shared_layers=0,
    ),
}


# ---------------------------------------------------------------------------
# Parametrised comparison test
# ---------------------------------------------------------------------------

def _assert_layer_params_match_reference(cfg, actual):
    """Shared assertion helper: verify actual params match the vLLM reference."""
    num_layers = cfg.num_hidden_layers
    expected = _vllm_reference_layer_params(cfg, num_layers)
    assert len(actual) == num_layers
    for i, (exp, got) in enumerate(zip(expected, actual, strict=True)):
        lt = cfg.layer_types[i]
        ctx = f"layer {i} (type={lt!r})"
        assert got["head_dim"] == exp["head_dim"], (
            f"{ctx}: head_dim {got['head_dim']} != vLLM {exp['head_dim']}"
        )
        assert got["num_kv_heads"] == exp["num_kv_heads"], (
            f"{ctx}: num_kv_heads {got['num_kv_heads']} != vLLM {exp['num_kv_heads']}"
        )
        assert got["is_kv_shared"] == exp["is_kv_shared"], (
            f"{ctx}: is_kv_shared {got['is_kv_shared']} != vLLM {exp['is_kv_shared']}"
        )
        assert got["kv_shared_target"] == exp["kv_shared_target"], (
            f"{ctx}: kv_shared_target {got['kv_shared_target']} != vLLM {exp['kv_shared_target']}"
        )
        assert got["intermediate_size"] == exp["intermediate_size"], (
            f"{ctx}: intermediate_size {got['intermediate_size']} != vLLM {exp['intermediate_size']}"
        )
        assert got["q_dim"] == cfg.num_attention_heads * exp["head_dim"], f"{ctx}: q_dim mismatch"
        assert got["kv_dim"] == exp["num_kv_heads"] * exp["head_dim"], f"{ctx}: kv_dim mismatch"
        if lt == "full_attention" and cfg.attention_k_eq_v:
            assert got["has_v_proj"] is False, f"{ctx}: expected has_v_proj=False for k_eq_v"
        else:
            assert got["has_v_proj"] is True, f"{ctx}: expected has_v_proj=True"


@pytest.mark.parametrize("name", list(CONFIGS))
def test_build_layer_params_matches_vllm_reference(name):
    """Per-layer head_dim / num_kv_heads / KV-sharing from
    _build_layer_params_from_config must match the vLLM reference formulas
    for every combination of Gemma4 config flags."""
    from vllm_webgpu.models.gemma4 import _build_layer_params_from_config

    cfg = CONFIGS[name]
    actual = _build_layer_params_from_config(cfg)
    _assert_layer_params_match_reference(cfg, actual)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_gemma4_layer_params_matches_vllm_reference(name):
    """_gemma4_layer_params (the extracted pure formula helper) must produce the
    same results as the vLLM reference for every config combination."""
    from vllm.transformers_utils.configs.gemma4 import gemma4_layer_config

    from vllm_webgpu.models.gemma4 import _gemma4_layer_params

    cfg = CONFIGS[name]
    # Per-layer dims come from vLLM's resolver, the same way
    # _build_layer_params_from_config supplies them in production.
    layer_cfgs = [gemma4_layer_config(cfg, i) for i in range(len(cfg.layer_types))]
    actual = _gemma4_layer_params(
        layer_types=cfg.layer_types,
        num_q_heads=cfg.num_attention_heads,
        head_dims=[lc.head_dim for lc in layer_cfgs],
        kv_heads=[lc.num_key_value_heads for lc in layer_cfgs],
        k_eq_v=getattr(cfg, "attention_k_eq_v", False),
        intermediate_size=cfg.intermediate_size,
        num_kv_shared_layers=getattr(cfg, "num_kv_shared_layers", 0),
        use_dwm=getattr(cfg, "use_double_wide_mlp", False),
    )
    _assert_layer_params_match_reference(cfg, actual)
