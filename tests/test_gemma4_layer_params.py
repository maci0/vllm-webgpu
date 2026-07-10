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
import types
from unittest.mock import MagicMock

import pytest

_PINNED_VLLM_VERSION = "0.24.0"

# ---------------------------------------------------------------------------
# Source-level guards
# ---------------------------------------------------------------------------

_VLLM_DECODER_LAYER_PATTERNS = [
    # head_dim selection (Gemma4DecoderLayer.__init__ ~L563)
    'head_dim = getattr(config, "global_head_dim", config.head_dim)',
    # k_eq_v flag (~L569)
    'use_k_eq_v = self.is_full_attention and getattr(',
    # num_kv_heads for k_eq_v path (~L576)
    '"num_global_key_value_heads", config.num_key_value_heads',
    # num_kv_heads fallback (~L580)
    "num_kv_heads = config.num_key_value_heads",
    # intermediate_size boundary (~L599)
    "first_kv_shared_layer_idx = config.num_hidden_layers - getattr(",
    # intermediate_size doubling (~L606)
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
        "(Gemma4Attention.__init__ L461-483, Gemma4DecoderLayer.__init__ L559-608) "
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

    Source: vllm/model_executor/models/gemma4.py @ 0.24.0
      Gemma4DecoderLayer.__init__ L559-608
      Gemma4Attention.__init__    L461-483
    """
    # Replicate Gemma4Attention.__init__ L461-463
    num_kv_shared_layers = getattr(config, "num_kv_shared_layers", 0)
    first_kv_shared = num_layers - num_kv_shared_layers

    results = []
    for layer_idx in range(num_layers):
        # --- Gemma4DecoderLayer.__init__ L560-580 ---
        layer_type = config.layer_types[layer_idx]
        is_full_attention = layer_type == "full_attention"

        if is_full_attention:
            head_dim = getattr(config, "global_head_dim", config.head_dim)
        else:
            head_dim = config.head_dim

        use_k_eq_v = is_full_attention and getattr(config, "attention_k_eq_v", False)
        if use_k_eq_v:
            num_kv_heads = getattr(
                config, "num_global_key_value_heads", config.num_key_value_heads
            )
        else:
            num_kv_heads = config.num_key_value_heads

        # --- Gemma4Attention.__init__ L462-483 (KV sharing) ---
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

        # --- Gemma4DecoderLayer.__init__ L599-608 (intermediate_size) ---
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

@pytest.mark.parametrize("name", list(CONFIGS))
def test_build_layer_params_matches_vllm_reference(name):
    """Per-layer head_dim / num_kv_heads / KV-sharing from
    _build_layer_params_from_config must match the vLLM reference formulas
    for every combination of Gemma4 config flags."""
    from vllm_webgpu.models.gemma4 import _build_layer_params_from_config

    cfg = CONFIGS[name]
    num_layers = cfg.num_hidden_layers

    expected = _vllm_reference_layer_params(cfg, num_layers)
    actual = _build_layer_params_from_config(cfg, num_layers)

    assert len(actual) == num_layers

    for i, (exp, got) in enumerate(zip(expected, actual)):
        lt = cfg.layer_types[i]
        ctx = f"layer {i} (type={lt!r}, config={name!r})"
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
        # Derived dims must be consistent
        assert got["q_dim"] == cfg.num_attention_heads * exp["head_dim"], (
            f"{ctx}: q_dim mismatch"
        )
        assert got["kv_dim"] == exp["num_kv_heads"] * exp["head_dim"], (
            f"{ctx}: kv_dim mismatch"
        )
        # has_v_proj: full_attention with k_eq_v has no separate V
        if lt == "full_attention" and cfg.attention_k_eq_v:
            assert got["has_v_proj"] is False, f"{ctx}: expected has_v_proj=False for k_eq_v"
        else:
            assert got["has_v_proj"] is True, f"{ctx}: expected has_v_proj=True"
