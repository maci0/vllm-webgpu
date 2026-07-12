from __future__ import annotations
import math
from dataclasses import dataclass
from itertools import batched
from typing import TYPE_CHECKING

import numpy as np

from vllm.utils.math_utils import cdiv
from vllm_webgpu.models.base import BaseWebGPUModel, _vals_per_thread, _vec4_wg, _rows_wg, _H_NAMES
from vllm_webgpu.webgpu.buffer import WebGPUBuffer

if TYPE_CHECKING:
    from vllm_webgpu.webgpu.device import WebGPUDevice
    from vllm_webgpu.webgpu.pipeline import PipelineCache


# Import-time source-anchor assertions for the three formulas transcribed from
# vLLM's Gemma4DecoderLayer and Gemma4Attention into _build_layer_params_from_config.
# If any upstream formula changes, these assertions fire on import rather than
# silently producing wrong per-layer dimensions. OSError is caught for stripped
# installs where getsource() is unavailable; ImportError propagates intentionally.
try:
    import inspect as _inspect
    from vllm.model_executor.models.gemma4 import (
        Gemma4Attention as _Gemma4Attention,
        Gemma4DecoderLayer as _Gemma4DecoderLayer,
    )
    _decoder_src = _inspect.getsource(_Gemma4DecoderLayer.__init__)
    _attn_src = _inspect.getsource(_Gemma4Attention.__init__)

    # Anchor (1): first_kv_shared boundary and chained comparison guard
    # (Gemma4DecoderLayer.__init__ ~L599-602 in vLLM 0.24)
    # Mirrors: `num_layers - getattr(model_config, 'num_kv_shared_layers', 0)` and
    # `(first_kv_shared > 0) and (i >= first_kv_shared)`.
    _A = "first_kv_shared_layer_idx = config.num_hidden_layers - getattr("
    assert _A in _decoder_src, (
        "Gemma4DecoderLayer.__init__ first_kv_shared boundary has changed "
        "(expected near vLLM 0.24 L599). Formula (1) in _build_layer_params_from_config "
        "mirrors `num_layers - getattr(model_config, 'num_kv_shared_layers', 0)`. "
        "Review and update the mirror and this anchor."
    )
    del _A
    _A = "is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0"
    assert _A in _decoder_src, (
        "Gemma4DecoderLayer.__init__ chained comparison `>= ... > 0` for KV-sharing "
        "has changed (expected near vLLM 0.24 L602). The `(first_kv_shared > 0) and "
        "(i >= first_kv_shared)` guard in _build_layer_params_from_config must match. "
        "Update the is_kv_shared check and this anchor."
    )
    del _A

    # Anchor (2): reversed-search KV-sharing target
    # (Gemma4Attention.__init__ ~L469-471 in vLLM 0.24)
    # Mirrors: `len(_prev) - 1 - _prev[::-1].index(lt)`.
    _A = "len(prev_layers) - 1 - prev_layers[::-1].index(current_layer_type)"
    assert _A in _attn_src, (
        "Gemma4Attention.__init__ reversed-search KV-sharing formula has changed "
        "(expected near vLLM 0.24 L469-471). Formula (2) in _build_layer_params_from_config "
        "uses `len(_prev) - 1 - _prev[::-1].index(lt)`. "
        "Review and update the mirror and this anchor."
    )
    del _A

    # Anchor (3): head_dim and num_kv_heads selection by attention type
    # (Gemma4DecoderLayer.__init__ ~L562-580 in vLLM 0.24)
    # Mirrors: global_head_dim for full_attention; num_global_key_value_heads when k_eq_v.
    _A = 'head_dim = getattr(config, "global_head_dim", config.head_dim)'
    assert _A in _decoder_src, (
        "Gemma4DecoderLayer.__init__ global_head_dim selection has changed "
        "(expected near vLLM 0.24 L563). Formula (3) in _build_layer_params_from_config "
        "sets `hd_l = global_hd` for full_attention. Update the mirror and this anchor."
    )
    del _A
    _A = '"num_global_key_value_heads", config.num_key_value_heads'
    assert _A in _decoder_src, (
        "Gemma4DecoderLayer.__init__ num_global_key_value_heads fallback has changed "
        "(expected near vLLM 0.24 L576-577). Formula (3) in _build_layer_params_from_config "
        "uses `global_kv if k_eq_v else default_kv`. Update the mirror and this anchor."
    )
    del _A

    del _inspect, _Gemma4Attention, _Gemma4DecoderLayer, _decoder_src, _attn_src
except OSError:
    pass


@dataclass(frozen=True)
class _RopeConsts:
    """Per-layer RoPE shader constants, precomputed once in __init__.

    Fused shaders (fused_per_head_norm_rope, fused_qk_norm_rope) consume all
    five fields. Plain rope.wgsl only uses rope_base, ln_rope_base, use_freq_buf.
    Using a frozen dataclass over a plain dict avoids the `**spread` allocation
    on the decode hot path and gives typed attribute access.
    """
    rope_base: float
    ln_rope_base: float
    use_freq_buf: int
    rotary_dim: int
    freq_dim: int


def _build_layer_params_from_config(
    model_config,
    num_layers: int,
) -> list[dict]:
    """Build per-layer attention/FFN params from a Gemma4 safetensors config.

    Reads all required fields directly from model_config, applying the same
    defaults as the caller's getattr chains. Transcribes three formulas from
    vLLM's Gemma4 model implementation. Pin these line references when upgrading
    vLLM:

    (1) first_kv_shared boundary:
        vLLM vllm/model_executor/models/gemma4.py line 601 (Gemma4DecoderLayer.__init__)
        ``self.num_layers - getattr(model_config, 'num_kv_shared_layers', 0)``

    (2) kv_shared_target reversed-search generator:
        vLLM vllm/model_executor/models/gemma4.py lines 467-474
        ``prev_layers[::-1].index(current_layer_type)`` (raises ValueError if not found)

    (3) head_dim / num_kv_heads / has_v_proj per attention type:
        vLLM vllm/model_executor/models/gemma4.py lines 561-577
        full_attention uses global_head_dim + num_global_key_value_heads when
        k_eq_v=True (laptop variant), or global_head_dim + num_key_value_heads
        when k_eq_v=False (standard variant); sliding_attention uses default
        head_dim + num_key_value_heads.
    """
    num_q_heads       = model_config.num_attention_heads
    intermediate_size = model_config.intermediate_size
    layer_types       = model_config.layer_types
    default_hd = getattr(model_config, "head_dim", model_config.hidden_size // num_q_heads)
    default_kv        = model_config.num_key_value_heads
    global_hd        = getattr(model_config, "global_head_dim", default_hd)
    global_kv        = getattr(model_config, "num_global_key_value_heads", default_kv)
    k_eq_v           = getattr(model_config, "attention_k_eq_v", False)

    # (1) vLLM gemma4.py L601 (Gemma4DecoderLayer)
    # Mirror vLLM's chained comparison: is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
    # The > 0 guard suppresses doubling when num_kv_shared_layers == num_hidden_layers (first_kv_shared=0),
    # matching Gemma4DecoderLayer line 602 exactly.
    first_kv_shared = num_layers - getattr(model_config, "num_kv_shared_layers", 0)
    use_dwm = getattr(model_config, "use_double_wide_mlp", False)

    lp: list[dict] = []
    for i, lt in enumerate(layer_types):
        is_kv_shared = (first_kv_shared > 0) and (i >= first_kv_shared)
        inter_l = intermediate_size * (2 if use_dwm and is_kv_shared else 1)

        # (2) vLLM gemma4.py L467-474: find last non-shared layer of the same type.
        if is_kv_shared:
            _prev = layer_types[:first_kv_shared]
            try:
                kv_shared_target = len(_prev) - 1 - _prev[::-1].index(lt)
            except ValueError:
                raise ValueError(
                    f"Layer {i} (type={lt!r}) is KV-shared but type {lt!r} was not "
                    f"found in the non-shared prefix {_prev}. Check layer_types config."
                )
        else:
            kv_shared_target = -1

        # (3) vLLM gemma4.py L561-577: select dims by attention type.
        if lt == "full_attention":
            hd_l = global_hd
            nkv_l = global_kv if k_eq_v else default_kv
            hv = not k_eq_v
        else:
            hd_l = default_hd
            nkv_l = default_kv
            hv = True

        lp.append({
            "head_dim":          hd_l,
            "num_q_heads":       num_q_heads,
            "num_kv_heads":      nkv_l,
            "q_dim":             num_q_heads * hd_l,
            "kv_dim":            nkv_l * hd_l,
            "has_v_proj":        hv,
            "intermediate_size": inter_l,
            "is_kv_shared":      is_kv_shared,
            "kv_shared_target":  kv_shared_target,
        })

    # Cross-check: when the k_eq_v (laptop) variant is active, full_attention
    # layers use global_kv heads. Verify that the stored kv_dim is consistent
    # with the global_head_dim and num_global_key_value_heads attributes on
    # model_config. A silent rename of either attribute in vLLM config causes
    # the getattr calls above to fall back to wrong defaults, producing a kv_dim
    # that no longer matches this independent recomputation.
    # Only applicable for k_eq_v=True: k_eq_v=False full_attention layers use
    # default_kv heads (not global_kv), so the check would spuriously fail there.
    if "full_attention" in layer_types and k_eq_v:
        expected_fa_kv_dim = global_hd * global_kv
        actual_fa_kv_dims = {
            p["kv_dim"] for p, lt in zip(lp, layer_types) if lt == "full_attention"
        }
        if expected_fa_kv_dim not in actual_fa_kv_dims:
            raise ValueError(
                f"full_attention kv_dim mismatch: expected {expected_fa_kv_dim} "
                f"(global_head_dim={getattr(model_config, 'global_head_dim', default_hd)!r} "
                f"* num_global_key_value_heads={getattr(model_config, 'num_global_key_value_heads', default_kv)!r}) "
                f"but full_attention layers produced {actual_fa_kv_dims}. "
                "Check whether vLLM renamed global attention config attributes, or "
                "whether the k_eq_v branch in formula (3) is misapplied."
            )

    return lp


class Gemma4WebGPUModel(BaseWebGPUModel):
    """
    Gemma 4 transformer with heterogeneous per-layer attention.

    Gemma4-12B mixes two attention types:
    - Local layers (head_dim=256, 8 KV heads): standard GQA with per-head RMSNorm on V
    - Global layers (head_dim=512, 1 KV head): MQA, no separate V projection (V=K)

    Every 6th layer (indices 5, 11, 17, ...) is a global attention layer.
    Scratch buffers are allocated at maximum dimensions to handle both types.
    """
    # Subclasses that override forward() and never read lp["scale"] set this to True
    # to skip the O(num_layers) scale computation in __init__.
    _skip_attn_scale: bool = False

    # Subclasses whose load_weights() bypasses _prefill_batch_forward() (and
    # therefore never reads _mr4_ok) set this to True so load_weights() skips
    # the _mr4_quant_supported() scan entirely.
    _skip_mr4_scan: bool = False

    # Number of transformer layers per GPU command-buffer chunk in prefill paths.
    # Chunking prevents Metal from timing out on very long prefill sequences.
    # Referenced by both _prefill_batch_forward and _prefill_sequential_fallback
    # so a single change here applies to both.
    _PREFILL_CHUNK: int = 4

    def __init__(self, model_config, wgpu_device: "WebGPUDevice", pipeline_cache: "PipelineCache", block_size: int = 16) -> None:
        _ple = getattr(model_config, 'hidden_size_per_layer_input', None)
        if _ple is not None and _ple > 0:
            raise ValueError(
                'PLE (hidden_size_per_layer_input > 0) is not supported by the WebGPU plugin'
            )
        super().__init__(model_config, wgpu_device, pipeline_cache)
        self.num_layers: int = model_config.num_hidden_layers
        self.num_q_heads: int = model_config.num_attention_heads
        self.hidden_size: int = model_config.hidden_size
        self.intermediate_size: int = model_config.intermediate_size
        self.vocab_size: int = model_config.vocab_size
        # Softcap is optional — Gemma4 uses 30.0, Gemma3 uses None (no cap)
        self.softcap: float | None = getattr(model_config, "final_logit_softcapping", None)
        self.rope_theta: float = getattr(model_config, "rope_theta", 10000.0)
        self.block_size: int = block_size

        # Gemma3 vs Gemma4 capability flags:
        # - GEMMA_NORM=1: Gemma3 norms store weights as deviations from zero (GemmaRMSNorm, (1+w) formula).
        #   Gemma4 uses plain RMSNorm; checkpoint weights are actual scale values (~1.0), so GEMMA_NORM=0.
        # - _apply_v_norm: only Gemma4 applies per-head RMS norm to V before caching
        # Per-layer attention parameters (head_dim, num_kv_heads, q_dim, kv_dim, has_v_proj).
        # Set from _layer_attention_params if available (parsed from GGUF), otherwise derive
        # using the heuristic that every 6th layer (idx%6==5) is global attention.
        raw_lp = getattr(model_config, "_layer_attention_params", None)
        default_hd = getattr(model_config, "head_dim",
                             self.hidden_size // self.num_q_heads)
        default_kv = getattr(model_config, "num_key_value_heads", 1)

        # Gemma4 safetensors: derive per-layer params from layer_types + global_head_dim.
        layer_types = getattr(model_config, "layer_types", None)

        # Capability flag: only Gemma4 applies per-head RMS norm to V before caching.
        # Gemma3 does NOT normalize V; DiffusionGemmaWebGPUModel overrides _decoder_layer and
        # applies V-norm unconditionally there, so this flag is not consulted for that subclass.
        # Prefer an explicit config field so future variants (e.g. gemma4_moe) can opt in/out
        # without this string comparison silently diverging from the DiffusionGemma path.
        _model_type = getattr(model_config, "model_type", "")
        self._apply_v_norm = getattr(
            model_config, "apply_v_norm",
            _model_type == "gemma4",
        )
        # Gemma3 trains norm weights as deviations from zero (GemmaRMSNorm), so the shader
        # must compute (1+w)*x. Gemma4 uses plain RMSNorm; weights are actual scale values.
        self._GEMMA_NORM = 1 if _model_type == "gemma3" else 0
        if raw_lp and len(raw_lp) == self.num_layers:
            # Shallow-copy each entry before adding defaults so the config object
            # is not mutated. A second instantiation from the same config would
            # otherwise find the keys already present and silently skip setdefault.
            self._lp: list[dict] = [
                {
                    **e,
                    "intermediate_size": e.get("intermediate_size", self.intermediate_size),
                    "is_kv_shared": e.get("is_kv_shared", False),
                    "kv_shared_target": e.get("kv_shared_target", -1),
                    "has_v_proj": e.get("has_v_proj", True),
                }
                for e in raw_lp
            ]
        elif layer_types and len(layer_types) == self.num_layers:
            # Build per-layer params from layer_types list (Gemma4 safetensors config).
            # sliding_attention: local GQA, head_dim=default_hd, has_v_proj=True
            # full_attention:    global, head_dim=global_hd; V=K only when attention_k_eq_v=True
            #
            # use_double_wide_mlp: the last num_kv_shared_layers layers have FFN width * 2.
            # is_kv_shared: the last num_kv_shared_layers layers share KV with an earlier
            # layer of the same type (mirrors vLLM Gemma4Attention.is_kv_shared_layer).
            # Formulas transcribed from vLLM -- see _build_layer_params_from_config docstring.
            self._lp = _build_layer_params_from_config(model_config, self.num_layers)
        else:
            # Uniform fallback: all layers use the config defaults.
            # For Gemma3 safetensors (uniform attention) this is correct.
            hd = default_hd
            nkv = default_kv
            uniform_lp = {
                "head_dim":         hd,
                "num_q_heads":      self.num_q_heads,
                "num_kv_heads":     nkv,
                "q_dim":            self.num_q_heads * hd,
                "kv_dim":           nkv * hd,
                "has_v_proj":       True,
                "intermediate_size": self.intermediate_size,
                "is_kv_shared":     False,
                "kv_shared_target": -1,
            }
            self._lp = [uniform_lp.copy() for _ in range(self.num_layers)]

        # Validate even dimensions required by WGSL shaders
        for name, val in [("hidden_size", self.hidden_size),
                          ("intermediate_size", self.intermediate_size)]:
            if val % 4 != 0:
                raise ValueError(f"{name}={val} must be divisible by 4 for vec4<f16> shaders")

        # Compute max dimensions across all layers for scratch buffer sizing
        self._max_q_dim = max(lp["q_dim"] for lp in self._lp)
        self._max_kv_dim = max(lp["kv_dim"] for lp in self._lp)
        self._max_inter = self._scratch_inter_size()
        max_ctx = getattr(model_config, "max_position_embeddings", 8192)
        self._init_scratch_buffers(max_ctx, self._max_q_dim, self._max_kv_dim)

        _vpt = _vals_per_thread(self.hidden_size)
        self._rms_consts = {"HIDDEN_DIM": self.hidden_size, "VALS_PER_THREAD": _vpt, "GEMMA_NORM": self._GEMMA_NORM}

        # Build per-layer rope constants from rope_parameters.
        # rope_parameters maps layer type to {"rope_type", "rope_theta", "partial_rotary_factor"}.
        # Gemma4:  full_attention     → "proportional", rope_theta=1e6, partial_rotary_factor=0.25
        #          sliding_attention  → "default",      rope_theta=10000
        # Models without rope_parameters (Gemma3, test configs) fall back to self.rope_theta with
        # full rotation (partial_rotary_factor=1.0).
        _rope_params_raw = getattr(model_config, "rope_parameters", None)
        _rope_params_map = _rope_params_raw if isinstance(_rope_params_raw, dict) else {}
        _use_freq = int(self._use_freq_buf)
        self._rope_consts: list[_RopeConsts] = []
        for _i, _lp_e in enumerate(self._lp):
            # layer_types may be non-None but shorter than num_layers in the fallback branch; guard prevents IndexError
            _lt = layer_types[_i] if (layer_types and _i < len(layer_types)) else None
            _rp: dict = _rope_params_map.get(_lt, {}) if _lt else {}
            # Flat-format legacy configs (no layer-type keys) return {} for any layer type.
            # Mirror vLLM's override: sliding layers use rope_local_base_freq (default 10000),
            # not the global rope_theta which is the full-attention value (e.g. 1e6).
            if not _rp and _lt == "sliding_attention":
                _rope_base = float(getattr(model_config, "rope_local_base_freq", 10000.0))
            else:
                _rope_base = float(_rp.get("rope_theta", self.rope_theta))
            _partial    = float(_rp.get("partial_rotary_factor", 1.0))
            _rope_type  = str(_rp.get("rope_type", "default"))
            _hd         = _lp_e["head_dim"]
            _rotary_dim = int(_hd * _partial)
            # "proportional" rope: freq exponent denominator = head_dim, not rotary_dim.
            # Matches Gemma4RotaryEmbedding._compute_inv_freq which uses head_size as
            # denominator regardless of partial_rotary_factor.
            _freq_dim   = _hd if _rope_type == "proportional" else _rotary_dim
            self._rope_consts.append(_RopeConsts(
                rope_base=_rope_base,
                ln_rope_base=math.log(_rope_base),
                use_freq_buf=_use_freq,
                rotary_dim=_rotary_dim,
                freq_dim=_freq_dim,
            ))

        # Precompute per-layer flash attention scale so both _prefill_batch_forward
        # and _transformer_layer read a constant rather than recomputing it each call.
        # Gemma4 (apply_v_norm=True): scale=1.0 (V-norm takes the place of QK scaling).
        # Gemma3 / uniform configs: scale = (query_pre_attn_scalar or head_dim) ** -0.5.
        # head_dim differs between local and global layers for Gemma4, so it must come
        # from the per-layer entry rather than a model-level attribute.
        # Skipped for subclasses (e.g. DiffusionGemmaWebGPUModel) that never read lp["scale"].
        if not self._skip_attn_scale:
            if self._apply_v_norm:
                for _lp_e in self._lp:
                    _lp_e["scale"] = 1.0
            else:
                _query_pre_attn_scalar: float | None = getattr(model_config, "query_pre_attn_scalar", None)
                for _lp_e in self._lp:
                    _lp_e["scale"] = (_query_pre_attn_scalar if _query_pre_attn_scalar is not None else _lp_e["head_dim"]) ** -0.5

        # Register q_norm/k_norm tiling transforms so load_weights tiles at upload time,
        # avoiding a GPU roundtrip (to_numpy → tile → re-upload) per weight per layer.
        # Gemma4 checkpoints store shared norm as (head_dim,); the shader expects
        # (num_heads * head_dim,) with each head repeating the same values.
        # Per-layer head_dim and num_heads are captured at registration time.
        for _i, _lp in enumerate(self._lp):
            _p = self._layer_key_prefix(_i)
            _hd = _lp["head_dim"]
            _nq = _lp["num_q_heads"]
            _nkv = _lp["num_kv_heads"]
            self._weight_transforms[f"{_p}.self_attn.q_norm.weight"] = (
                lambda a, hd=_hd, n=_nq: np.tile(a, n) if a.shape == (hd,) else a
            )
            self._weight_transforms[f"{_p}.self_attn.k_norm.weight"] = (
                lambda a, hd=_hd, n=_nkv: np.tile(a, n) if a.shape == (hd,) else a
            )
        # Updated to True/False in load_weights() once weights are known.
        # Defaults to True so tests that bypass load_weights() reach the batch path.
        self._mr4_ok: bool = True

    def _scratch_token_count(self) -> int:
        """Number of tokens to size T-dependent scratch buffers for. Override in subclasses."""
        return 1

    def _scratch_inter_size(self) -> int:
        """Intermediate size for FFN scratch buffers. Override in subclasses."""
        return max(lp["intermediate_size"] for lp in self._lp)

    def _init_pre_buffers(self, max_ctx: int) -> None:
        """Allocate the _pre dict of per-step reuse buffers.

        Extracted so DiffusionGemmaWebGPUModel can call this without duplicating
        the allocation. Any new entry added here is automatically present for all
        subclasses that delegate via super().
        """
        T = self._scratch_token_count()
        H = self.hidden_size
        V = self.vocab_size
        self._pre: dict[str, "WebGPUBuffer"] = {
            "ids":      self._make_buf(T * 4),         # [T] uint32 token ids
            "pos":      self._make_buf(T * 4),         # [T] uint32 positions
            "slot_map": self._make_buf(T * 4),         # [T] uint32 physical slots
            "bt":       self._make_buf(max(4096, cdiv(max_ctx, self.block_size)) * 4),  # block table
            "x":        self._make_buf(T * H * 4),     # [T, H] f32 residual
            "norm_out": self._make_buf(T * H * 2),     # [T, H] f16 final norm
            "logits":   self._make_buf(T * V * 2),     # [T, V] f16 logits
        }
        if self.softcap is not None and self.softcap > 0:
            self._pre["capped"] = self._make_buf(T * V * 2)  # [1, V] f16 softcapped logits (Gemma4)

    def _init_scratch_buffers(self, max_ctx: int, max_q_dim: int, max_kv_dim: int) -> None:
        """Pre-allocate scratch buffers at maximum layer dimensions."""
        T = self._scratch_token_count()
        H = self.hidden_size
        I = self._max_inter

        self._init_pre_buffers(max_ctx)

        self._sc: dict[str, "WebGPUBuffer"] = {
            "normed":     self._make_buf(T * H * 2),                              # f16
            "qkv_buf":    self._make_buf(T * (max_q_dim + 2 * max_kv_dim) * 2),  # f16 [Q|K|V]
            "q_buf":      self._make_buf(T * max_q_dim * 2),                      # f16
            "k_buf":      self._make_buf(T * max_kv_dim * 2),                     # f16
            "v_buf":      self._make_buf(T * max_kv_dim * 2),                     # f16
            "v_normed":   self._make_buf(T * max_kv_dim * 2),                     # f16
            "q_rope":     self._make_buf(T * max_q_dim * 2),  # f16
            "k_rope":     self._make_buf(T * max_kv_dim * 2), # f16
            "attn_out":   self._make_buf(T * max_q_dim * 2),  # f16
            "o_proj_out": self._make_buf(T * H * 2),           # f16
            "gate_buf":   self._make_buf(T * I * 2),           # f16
            "up_buf":     self._make_buf(T * I * 2),           # f16
            "ffn_act":    self._make_buf(T * I * 2),
            "ffn_out":    self._make_buf(T * H * 2),
            # Residual buffers stored in f32 for precision.
            # Gemma4 has output_norm weights up to 600 which cause f16 saturation
            # when accumulated across 48 layers — f32 residuals prevent this.
            "h0":         self._make_buf(T * H * 4),           # f32 (4 bytes)
            "h1":         self._make_buf(T * H * 4),           # f32
            "h2":         self._make_buf(T * H * 4),           # f32
        }
        self._hstate: int = 0

    def _layer_key_prefix(self, layer_idx: int) -> str:
        """Return the weight key prefix for layer i. Subclasses may override."""
        return f"model.layers.{layer_idx}"

    def _embed_key(self) -> str:
        """Return the weight key for the embedding table. Subclasses may override."""
        return "model.embed_tokens.weight"

    def _norm_key(self) -> str:
        """Return the weight key for the final layer norm. Subclasses may override."""
        return "model.norm.weight"


    def _load_layer_scales(self) -> None:
        """Cache layer_scalar values on CPU at load time.

        Avoids 48 GPU→CPU readbacks per token (each to_numpy() is a blocking ~100µs sync).
        Subclasses with a different key scheme should override _layer_key_prefix instead.
        """
        self._layer_scales: list[float] = []
        for i in range(self.num_layers):
            p = self._layer_key_prefix(i)
            ls_buf = self.weights.get(f"{p}.layer_scalar")
            if ls_buf is not None:
                self._layer_scales.append(self._buf_to_numpy(ls_buf).item())
            else:
                self._layer_scales.append(1.0)

    def load_weights(self, path: str, f32_keys: "frozenset[str] | None" = None,
                     skip_prefixes: "frozenset[str] | None" = None) -> None:
        super().load_weights(path, f32_keys=f32_keys, skip_prefixes=skip_prefixes)
        self._load_layer_scales()
        if not self._skip_mr4_scan:
            self._mr4_ok: bool = self._mr4_quant_supported()

    def forward(
        self,
        input_ids: np.ndarray,
        positions: np.ndarray,
        attn_metadata: object,
    ) -> np.ndarray:
        dev = self.wgpu_device.wgpu_device
        num_tokens = len(input_ids)
        hidden = self.hidden_size
        vocab = self.vocab_size

        self._hstate = 0

        self._check_single_sequence(attn_metadata)

        # Prefill path: T>1 tokens use matmul_quant_mr4 (batch GEMM) and flash_attn_prefill.
        if num_tokens > 1:
            return self._prefill_batch_forward(input_ids, positions, attn_metadata, num_tokens)

        ctx_len = int(attn_metadata.max_decode_seq_len)

        # Update pre-allocated buffers via write_buffer — no GPU allocation per step.
        pre = self._pre
        dev.queue.write_buffer(pre["ids"].buf, 0, input_ids.astype(np.uint32, copy=False).tobytes())
        dev.queue.write_buffer(pre["pos"].buf, 0, positions.astype(np.uint32, copy=False).tobytes())
        dev.queue.write_buffer(
            pre["slot_map"].buf, 0,
            np.asarray(attn_metadata.slot_mapping, dtype=np.uint32).tobytes())
        bt_arr = self._bt_arr(attn_metadata)
        dev.queue.write_buffer(pre["bt"].buf, 0, bt_arr.tobytes())

        ids_buf    = pre["ids"]
        pos_buf    = pre["pos"]
        slot_map   = pre["slot_map"]
        bt_buf     = pre["bt"]
        x_buf      = pre["x"]
        norm_out   = pre["norm_out"]
        logits_buf = pre["logits"]

        _rms = self._rms_consts
        sc = self._sc

        with self._batched_dispatch():
            # Embedding lookup → f32 output for f32 residual pipeline
            self._dispatch("embedding_lookup_f32",
                           [self.weights[self._embed_key()], ids_buf, x_buf],
                           {"HIDDEN_DIM": hidden}, (num_tokens, 1, 1))

            # Initial pre-norm for layer 0 (subsequent pre-norms are fused into each
            # layer's final add_f32_rms_norm dispatch).
            self._dispatch("rms_norm_f32in",
                           [x_buf, self.weights[f"{self._layer_key_prefix(0)}.input_layernorm.weight"], sc["normed"]],
                           _rms, (num_tokens, 1, 1))

            normed_x = sc["normed"]
            for i in range(self.num_layers):
                normed_x, x_buf = self._transformer_layer(
                    i, normed_x, x_buf, pos_buf, slot_map, bt_buf, ctx_len, num_tokens)

            # Final norm: reads f32 residual, writes f16 norm_out
            self._dispatch("rms_norm_f32in",
                           [x_buf, self.weights[self._norm_key()], norm_out],
                           _rms, (num_tokens, 1, 1))

            _lm_key, lm_head_w, uq_lm, _lm_base = self._lm_head_parts()
            # vocab_size exceeds the 65535 workgroup-per-dimension limit, so the split-K
            # path is unusable. Force SPLIT_K=0 (row-per-thread) with ceil(vocab/256) WGs.
            self._dispatch("matmul_quant",
                           [norm_out, lm_head_w,
                            self._scales_buf(_lm_key, uq_lm, self._dummy_buf), logits_buf],
                           {"K": hidden, "N": vocab, "USE_QUANT": uq_lm, "SPLIT_K": 0,
                            **self._quant_extra(_lm_base, uq_lm)},
                           _rows_wg(vocab))

            self._dispatch_softcap_and_sample(vocab, logits_buf, pre.get("capped", self._dummy_buf))

        return self._finish_forward(self._greedy_decode)

    def _lm_head_parts(self) -> "tuple[str, object, int, str]":
        """Return (key, weight_buf, uq, base_key) for the LM head.

        Centralises the four-line setup repeated in forward(), _prefill_batch_forward(),
        and _prefill_sequential_fallback() so callers avoid duplicating the pattern.
        """
        key = self._lm_head_key()
        return key, self.weights[key], self._uq_for_key(key), key.removesuffix('.weight')

    def _dispatch_softcap_and_sample(
        self,
        vocab: int,
        logits_buf: "WebGPUBuffer",
        capped_buf: "WebGPUBuffer",
    ) -> None:
        """Dispatch optional logit softcap, greedy argmax, and staging copy.

        Must be called from within an active _batched_dispatch() context.
        Sets _last_logit_buf and _last_vocab as side effects; return value
        is unused by all call sites.
        """
        if self.softcap is not None and self.softcap > 0:
            self._dispatch(
                "logit_softcap", [logits_buf, capped_buf],
                {"VOCAB": vocab, "CAP": float(self.softcap)},
                _rows_wg(vocab),
                shader_subdir="gemma")
            result_buf = capped_buf
        else:
            result_buf = logits_buf
        if self._greedy_decode:
            self._dispatch("argmax_f16", [result_buf, self._ensure_sample_buf()],
                           {"N": vocab}, (1, 1, 1))
            self._copy_sample_to_staging()
        self._last_logit_buf = result_buf
        self._last_vocab     = vocab

    def _mr4_quant_supported(self) -> bool:
        """Return True when every layer weight uses a quant format compatible with matmul_quant_mr4.

        Only USE_QUANT=0 (f16) and USE_QUANT=3 (GPTQ int4) are supported in the batch
        prefill path. Scanning all layers up front catches mixed-quant checkpoints before
        any KV-cache population occurs rather than crashing mid-forward.
        """
        for i in range(self.num_layers):
            p  = self._layer_key_prefix(i)
            lp = self._lp[i]
            is_kv_shared = lp["is_kv_shared"]
            keys = [
                f"{p}.self_attn.q_proj.weight",
                f"{p}.self_attn.o_proj.weight",
                f"{p}.mlp.gate_proj.weight",
                f"{p}.mlp.up_proj.weight",
                f"{p}.mlp.down_proj.weight",
            ]
            if not is_kv_shared:
                keys.append(f"{p}.self_attn.k_proj.weight")
                if lp["has_v_proj"]:
                    keys.append(f"{p}.self_attn.v_proj.weight")
            if any(self._uq_for_key(k) not in (0, 3) for k in keys):
                return False
        return True

    def _batch_gemm(
        self,
        src: "WebGPUBuffer",
        wk: str,
        out_b: "WebGPUBuffer",
        K: int,
        N: int,
        T: int,
    ) -> None:
        """Dispatch matmul_quant_mr4: out[T, N] = src[T, K] @ w[N, K].T.

        Only USE_QUANT=0 (f16) and USE_QUANT=3 (GPTQ int4) are supported.
        Other quant types must use the sequential (GEMV) path.
        """
        uq = self._uq_for_key(wk)
        if uq == 3:
            sc_b = self._scales_buf(wk, uq, self._dummy_buf)
            self._dispatch("matmul_quant_mr4",
                           [src, self.weights[wk], sc_b, out_b],
                           {"K": K, "N": N, "M": T, "USE_QUANT": 3,
                            **self._quant_extra(wk.removesuffix(".weight"), uq)},
                           (N, T, 1))
        elif uq == 0:
            self._dispatch("matmul_quant_mr4",
                           [src, self.weights[wk], self._dummy_buf, out_b],
                           {"K": K, "N": N, "M": T, "USE_QUANT": 0},
                           (N, T, 1))
        else:
            raise RuntimeError(
                f"Batch prefill does not support USE_QUANT={uq} for weight {wk}. "
                f"Only f16 (USE_QUANT=0) and GPTQ int4 (USE_QUANT=3) are handled by "
                f"matmul_quant_mr4. Other quant types must use the sequential path."
            )

    def _prefill_batch_forward(  # noqa: C901
        self,
        input_ids: "np.ndarray",
        positions: "np.ndarray",
        attn_metadata: object,
        T: int,
    ) -> "np.ndarray":
        """Batch prefill: process T prompt tokens in a series of GPU command encoders.

        GEMM ops use matmul_quant_mr4 (T rows at once).
        Attention uses flash_attn_prefill (dense causal Q/K/V, not paged cache).
        Layers are chunked across separate command encoders (4 per submit) to stay
        under Metal's per-command-buffer GPU timeout.
        Returns shape (1, 1) int32 (GPU argmax of last-token logits).
        """
        if not self._mr4_ok:
            return self._prefill_sequential_fallback(input_ids, positions, attn_metadata, T)

        dev = self.wgpu_device.wgpu_device
        hidden = self.hidden_size
        vocab  = self.vocab_size
        max_inter  = self._max_inter
        max_q_dim  = self._max_q_dim
        max_kv_dim = self._max_kv_dim

        # T-token batch buffers. Allocated once per prefill call;
        # allocation cost is negligible vs the GEMM savings.
        b: dict = {
            "x":        self._make_buf(T * hidden * 4),       # f32 embedding residual
            "normed":   self._make_buf(T * hidden * 2),        # f16 normed (pre-attn and pre-FFN)
            "q_buf":    self._make_buf(T * max_q_dim * 2),     # f16 Q projection output
            "k_buf":    self._make_buf(T * max_kv_dim * 2),    # f16 K projection output
            "v_buf":    self._make_buf(T * max_kv_dim * 2),    # f16 V projection output
            "q_rope":   self._make_buf(T * max_q_dim * 2),     # f16 Q after norm+rope
            "k_rope":   self._make_buf(T * max_kv_dim * 2),    # f16 K after norm+rope
            "v_normed": self._make_buf(T * max_kv_dim * 2),    # f16 V after per-head RMS norm
            "attn_out": self._make_buf(T * max_q_dim * 2),     # f16 attention output
            "o_proj":   self._make_buf(T * hidden * 2),         # f16 output projection
            "gate_buf": self._make_buf(T * max_inter * 2),          # f16 FFN gate
            "up_buf":   self._make_buf(T * max_inter * 2),          # f16 FFN up
            "ffn_act":  self._make_buf(T * max_inter * 2),          # f16 activated gate*up
            "ffn_out":  self._make_buf(T * hidden * 2),         # f16 FFN output
            "h0":       self._make_buf(T * hidden * 4),         # f32 residual (rotation slot 0)
            "h1":       self._make_buf(T * hidden * 4),         # f32 residual (rotation slot 1)
            "h2":       self._make_buf(T * hidden * 4),         # f32 residual (rotation slot 2)
            # Single-token scratch for final norm + LM head
            "last_f32":  self._make_buf(hidden * 4),            # f32 last-token residual copy
            "last_norm": self._make_buf(hidden * 2),            # f16 last-token after final norm
            "logits":    self._make_buf(vocab * 2),             # f16 LM head output
        }
        if self.softcap is not None and self.softcap > 0:
            b["capped"] = self._make_buf(vocab * 2)             # f16 softcapped logits (Gemma4)
        slot_map_buf = WebGPUBuffer.from_numpy(
            dev, np.asarray(attn_metadata.slot_mapping, dtype=np.uint32))
        pos_buf = WebGPUBuffer.from_numpy(dev, positions.astype(np.uint32, copy=False))
        ids_buf = WebGPUBuffer.from_numpy(dev, input_ids.astype(np.uint32, copy=False))
        _rms = self._rms_consts

        _hstate  = 0
        _freq_buf = self._rope_freq_buf

        for chunk_idx, chunk_layers in enumerate(batched(range(self.num_layers), self._PREFILL_CHUNK)):
            with self._batched_dispatch():
                if chunk_idx == 0:
                    self._dispatch("embedding_lookup_f32",
                                   [self.weights[self._embed_key()], ids_buf, b["x"]],
                                   {"HIDDEN_DIM": hidden}, (T, 1, 1))
                    self._dispatch(
                        "rms_norm_f32in",
                        [b["x"],
                         self.weights[f"{self._layer_key_prefix(0)}.input_layernorm.weight"],
                         b["normed"]],
                        _rms, (T, 1, 1))
                    normed_x = b["normed"]
                    x_res    = b["x"]

                for i in chunk_layers:
                    lp              = self._lp[i]
                    p               = self._layer_key_prefix(i)
                    q_dim           = lp["q_dim"]
                    kv_dim          = lp["kv_dim"]
                    head_dim        = lp["head_dim"]
                    num_kv_heads    = lp["num_kv_heads"]
                    has_v           = lp["has_v_proj"]
                    inter           = lp["intermediate_size"]
                    is_kv_shared    = lp["is_kv_shared"]
                    kv_shared_target = lp["kv_shared_target"]
                    add_n           = T * hidden
                    gelu_n          = T * inter
                    _ls             = self._layer_scales[i]

                    # Per-layer rope constants (typed dataclass; access fields directly).
                    rc = self._rope_consts[i]

                    residual = b[_H_NAMES[(_hstate + 1) % 3]]
                    out_h    = b[_H_NAMES[(_hstate + 2) % 3]]

                    qw = f"{p}.self_attn.q_proj.weight"

                    # QKV projections (always separate in batch path — no fused_qkv).
                    # For KV-shared layers only Q is used; K and V come from the target cache.
                    self._batch_gemm(normed_x, qw, b["q_buf"], hidden, q_dim, T)
                    if not is_kv_shared:
                        kw = f"{p}.self_attn.k_proj.weight"
                        self._batch_gemm(normed_x, kw, b["k_buf"], hidden, kv_dim, T)
                        if has_v:
                            self._batch_gemm(normed_x, f"{p}.self_attn.v_proj.weight",
                                       b["v_buf"], hidden, kv_dim, T)
                            v_src = b["v_buf"]
                        else:
                            v_src = b["k_buf"]   # global attention: V = K (pre-RoPE)
                    # Resolve which KV pool slot to read/write.
                    # KV-shared layers use the target layer's cache; non-shared use their own.
                    _kv_layer = kv_shared_target if (is_kv_shared and kv_shared_target >= 0) else i
                    k_cache, v_cache = self.kv_pool[_kv_layer]

                    # Per-head RMSNorm + RoPE for Q and K
                    q_norm_w  = self.weights.get(f"{p}.self_attn.q_norm.weight")
                    k_norm_wl = self.weights.get(f"{p}.self_attn.k_norm.weight")

                    if is_kv_shared:
                        # KV-shared layer: Q norm + RoPE only.
                        # K norm, K RoPE, V norm, and KV cache store are skipped.
                        # K and V come from the target layer's paged cache (k_cache, v_cache).
                        if q_norm_w is not None:
                            self._dispatch(
                                "fused_per_head_norm_rope",
                                [b["q_buf"], q_norm_w, pos_buf, b["q_rope"], _freq_buf],
                                {"ROPE_BASE": rc.rope_base, "LN_ROPE_BASE": rc.ln_rope_base,
                                 "USE_FREQ_BUF": rc.use_freq_buf, "ROTARY_DIM": rc.rotary_dim,
                                 "FREQ_DIM": rc.freq_dim, "HEAD_DIM": head_dim,
                                 "NUM_HEADS": self.num_q_heads, "HAS_WEIGHT": 1,
                                 "GEMMA_NORM": self._GEMMA_NORM, "INPUT_OFFSET": 0},
                                (self.num_q_heads, T, 1))
                        else:
                            self._dispatch(
                                "rope",
                                [b["q_buf"], pos_buf, b["q_rope"], _freq_buf],
                                {"ROPE_BASE": rc.rope_base, "LN_ROPE_BASE": rc.ln_rope_base,
                                 "USE_FREQ_BUF": rc.use_freq_buf, "HEAD_DIM": head_dim,
                                 "NUM_HEADS": self.num_q_heads},
                                (T, self.num_q_heads, 1))
                        # Load the target layer's cached K and V into dense buffers so that
                        # flash_attn_prefill can consume them (flash_attn_prefill needs dense K/V).
                        self._dispatch(
                            "kv_cache_load_dense",
                            [k_cache, v_cache, slot_map_buf, b["k_rope"], b["v_normed"]],
                            {"BLOCK_SIZE":   self.block_size,
                             "NUM_KV_HEADS": num_kv_heads,
                             "HEAD_DIM":     head_dim},
                            (T, num_kv_heads, 1))
                        v_for_attn = b["v_normed"]
                    else:
                        # Non-shared: Q + K norm + RoPE, V norm, KV cache store.
                        if q_norm_w is not None and k_norm_wl is not None:
                            # K_SEPARATE=1: Q and K are in separate buffers.
                            # Binding 0=Q buf, binding 6=K buf, INPUT_OFFSET_K=0.
                            self._dispatch(
                                "fused_qk_norm_rope",
                                [b["q_buf"], q_norm_w, k_norm_wl, pos_buf,
                                 b["q_rope"], b["k_rope"], b["k_buf"], _freq_buf],
                                {"ROPE_BASE": rc.rope_base, "LN_ROPE_BASE": rc.ln_rope_base,
                                 "USE_FREQ_BUF": rc.use_freq_buf, "ROTARY_DIM": rc.rotary_dim,
                                 "FREQ_DIM": rc.freq_dim,
                                 "HEAD_DIM":      head_dim,
                                 "NUM_Q_HEADS":   self.num_q_heads,
                                 "NUM_KV_HEADS":  num_kv_heads,
                                 "HAS_WEIGHT":    1,
                                 "GEMMA_NORM":    self._GEMMA_NORM,
                                 "INPUT_OFFSET_K": 0,
                                 "K_SEPARATE":    1},
                                (self.num_q_heads + num_kv_heads, T, 1))
                        else:
                            for src, dst, n_heads, wk in [
                                (b["q_buf"], b["q_rope"], self.num_q_heads,
                                 f"{p}.self_attn.q_norm.weight"),
                                (b["k_buf"], b["k_rope"], num_kv_heads,
                                 f"{p}.self_attn.k_norm.weight"),
                            ]:
                                nw = self.weights.get(wk)
                                if nw is not None:
                                    self._dispatch(
                                        "fused_per_head_norm_rope",
                                        [src, nw, pos_buf, dst, _freq_buf],
                                        {"ROPE_BASE": rc.rope_base, "LN_ROPE_BASE": rc.ln_rope_base,
                                         "USE_FREQ_BUF": rc.use_freq_buf, "ROTARY_DIM": rc.rotary_dim,
                                         "FREQ_DIM": rc.freq_dim, "HEAD_DIM": head_dim,
                                         "NUM_HEADS": n_heads, "HAS_WEIGHT": 1,
                                         "GEMMA_NORM": self._GEMMA_NORM,
                                         "INPUT_OFFSET": 0},
                                        (n_heads, T, 1))
                                else:
                                    self._dispatch(
                                        "rope",
                                        [src, pos_buf, dst, _freq_buf],
                                        {"ROPE_BASE": rc.rope_base, "LN_ROPE_BASE": rc.ln_rope_base,
                                         "USE_FREQ_BUF": rc.use_freq_buf, "HEAD_DIM": head_dim,
                                         "NUM_HEADS": n_heads},
                                        (T, n_heads, 1))

                        # Per-head RMS norm on V before caching (Gemma4 only; not Gemma3).
                        # v_src is b["v_buf"] for local layers and b["k_buf"] for global;
                        # both are standalone buffers so V_IN_OFFSET=0.
                        if self._apply_v_norm:
                            self._dispatch(
                                "per_head_rms_norm_no_weight",
                                [v_src, b["v_normed"]],
                                {"HEAD_DIM": head_dim, "NUM_HEADS": num_kv_heads,
                                 "WG_SIZE": min(head_dim, 128), "V_IN_OFFSET": 0},
                                (num_kv_heads, T, 1), shader_subdir="gemma")
                            v_for_attn = b["v_normed"]
                        else:
                            v_for_attn = v_src

                        # KV cache store (paged, for subsequent decode steps).
                        # V_IN_OFFSET=0: v_for_attn is always a standalone buffer here.
                        self._dispatch(
                            "kv_cache_store_both",
                            [b["k_rope"], k_cache, v_for_attn, v_cache, slot_map_buf],
                            {"BLOCK_SIZE":   self.block_size,
                             "NUM_KV_HEADS": num_kv_heads,
                             "HEAD_DIM":     head_dim,
                             "V_IN_OFFSET":  0},
                            (T, num_kv_heads, 1))

                    # Causal attention over all T query tokens (dense Q/K/V, not paged).
                    # flash_attn_prefill applies causal masking: token t_q attends to [0, t_q].
                    # For KV-shared layers, b["k_rope"] and b["v_normed"] were loaded from
                    # the target layer's paged cache via kv_cache_load_dense above.
                    self._dispatch(
                        "flash_attn_prefill",
                        [b["q_rope"], b["k_rope"], v_for_attn, b["attn_out"]],
                        {"NUM_Q_HEADS":  self.num_q_heads,
                         "NUM_KV_HEADS": num_kv_heads,
                         "HEAD_DIM":     head_dim,
                         "NUM_T":        T,
                         "SCALE":        lp["scale"]},
                        (self.num_q_heads, T, 1))

                    # Output projection (batch GEMM)
                    self._batch_gemm(b["attn_out"], f"{p}.self_attn.o_proj.weight",
                               b["o_proj"], q_dim, hidden, T)

                    # Post-attention norm + residual add + pre-FFN norm.
                    # Gemma4 correct sublayer order:
                    #   residual = x
                    #   hidden   = input_layernorm(x) → attn → o_proj
                    #   hidden   = post_attention_layernorm(hidden)
                    #   residual = residual + hidden
                    #   ffn_in   = pre_feedforward_layernorm(residual)
                    post_attn_w = self.weights.get(
                        f"{p}.post_attention_layernorm.weight")
                    pre_ffn_w   = self.weights.get(
                        f"{p}.pre_feedforward_layernorm.weight")

                    if post_attn_w is not None and pre_ffn_w is not None:
                        self._dispatch(
                            "rms_norm_add_f32_rms_norm",
                            [b["o_proj"], post_attn_w,
                             x_res, pre_ffn_w,
                             residual, b["normed"]],
                            _rms, (T, 1, 1))
                        ffn_normed = b["normed"]
                    else:
                        if post_attn_w is None:
                            raise ValueError(
                                f"Layer {i} missing post_attention_layernorm.weight "
                                "— vLLM creates this norm unconditionally; absence "
                                "indicates a corrupt checkpoint"
                            )
                        raise ValueError(
                            f"Layer {i} missing pre_feedforward_layernorm.weight "
                            "— f32 residual cannot be fed to f16 FFN projection"
                        )

                    # FFN gate + up projections (batch GEMM) + tanh-GELU activation
                    gw_k = f"{p}.mlp.gate_proj.weight"
                    uw_k = f"{p}.mlp.up_proj.weight"
                    self._batch_gemm(ffn_normed, gw_k, b["gate_buf"], hidden, inter, T)
                    self._batch_gemm(ffn_normed, uw_k, b["up_buf"],   hidden, inter, T)
                    self._dispatch(
                        "gelu_mul",
                        [b["gate_buf"], b["up_buf"], b["ffn_act"]],
                        {"N": gelu_n},
                        _vec4_wg(gelu_n),
                        shader_subdir="gemma")

                    # FFN down projection (batch GEMM)
                    self._batch_gemm(b["ffn_act"], f"{p}.mlp.down_proj.weight",
                               b["ffn_out"], inter, hidden, T)

                    # Post-FFN norm + residual add (+ pre-norm for next layer if not last)
                    post_ffw_w = self.weights.get(
                        f"{p}.post_feedforward_layernorm.weight")
                    if post_ffw_w is None:
                        raise ValueError(
                            f"Layer {i} missing post_feedforward_layernorm.weight "
                            "— vLLM applies this norm unconditionally; a missing weight "
                            "indicates a corrupt or incomplete checkpoint."
                        )
                    if i < self.num_layers - 1:
                        next_w = self.weights[
                            f"{self._layer_key_prefix(i + 1)}.input_layernorm.weight"]
                        self._dispatch(
                            "rms_norm_add_f32_rms_norm",
                            [b["ffn_out"], post_ffw_w,
                             residual, next_w,
                             out_h, b["normed"]],
                            _rms, (T, 1, 1))
                    else:
                        # Last layer: no next pre-norm, just update residual.
                        self._dispatch(
                            "rms_norm", [b["ffn_out"], post_ffw_w, b["o_proj"]],
                            _rms, (T, 1, 1))
                        self._dispatch(
                            "add_f32", [residual, b["o_proj"], out_h],
                            {"N": add_n}, _vec4_wg(add_n))

                    # Apply layer_scalar to the full f32 residual (matches vLLM).
                    if abs(_ls - 1.0) > 1e-6:
                        self._dispatch(
                            "f32_scale_inplace", [out_h],
                            {"N": add_n, "SCALE": _ls},
                            (cdiv(add_n, 256), 1, 1))

                    if i < self.num_layers - 1:
                        normed_x = b["normed"]
                    x_res    = out_h
                    _hstate  = (_hstate + 2) % 3

        # Extract last token, apply final norm, run LM head.
        # copy_buffer_to_buffer is a GPU-side operation (no CPU round-trip).
        _lm_key, lm_head_w, uq_lm, _lm_base = self._lm_head_parts()
        with self._batched_dispatch():
            last_byte_offset = (T - 1) * hidden * 4   # f32: 4 bytes per element
            enc = self._active_encoder
            assert enc is not None
            enc.copy_buffer_to_buffer(
                x_res.buf, last_byte_offset,
                b["last_f32"].buf, 0,
                hidden * 4,
            )
            self._dispatch(
                "rms_norm_f32in",
                [b["last_f32"], self.weights[self._norm_key()], b["last_norm"]],
                _rms, (1, 1, 1))
            self._dispatch(
                "matmul_quant",
                [b["last_norm"], lm_head_w,
                 self._scales_buf(_lm_key, uq_lm, self._dummy_buf), b["logits"]],
                {"K": hidden, "N": vocab, "USE_QUANT": uq_lm, "SPLIT_K": 0,
                 **self._quant_extra(_lm_base, uq_lm)},
                _rows_wg(vocab))

            self._dispatch_softcap_and_sample(vocab, b["logits"], b.get("capped", self._dummy_buf))

        return self._finish_forward(self._greedy_decode)

    def _prefill_sequential_fallback(
        self,
        input_ids: "np.ndarray",
        positions: "np.ndarray",
        attn_metadata: object,
        T: int,
    ) -> "np.ndarray":
        """Process T prefill tokens one at a time via the decode-path infrastructure.

        Used when projection weights use a quant format not supported by
        matmul_quant_mr4 (anything other than f16 or GPTQ int4). Each token runs
        through _transformer_layer with num_tokens=1, building the KV cache
        incrementally to maintain causal attention. Only the last token's logits
        are returned.
        """
        dev = self.wgpu_device.wgpu_device
        pre = self._pre
        sc  = self._sc
        hidden = self.hidden_size
        vocab  = self.vocab_size
        _rms   = self._rms_consts
        bt_arr = self._bt_arr(attn_metadata)
        # Serialize once: bt_arr does not change across token iterations.
        # Calling tobytes() inside the loop would re-allocate the bytes object T times.
        _bt_bytes = bt_arr.tobytes()

        for t in range(T):
            self._hstate = 0
            tok_pos = int(positions[t])
            tok_ctx = tok_pos + 1

            ids_t  = input_ids[t : t + 1]
            pos_t  = positions[t : t + 1]
            slot_t = np.asarray(attn_metadata.slot_mapping[t : t + 1], dtype=np.uint32)

            dev.queue.write_buffer(pre["ids"].buf,      0, ids_t.astype(np.uint32, copy=False).tobytes())
            dev.queue.write_buffer(pre["pos"].buf,      0, pos_t.astype(np.uint32, copy=False).tobytes())
            dev.queue.write_buffer(pre["slot_map"].buf, 0, slot_t.tobytes())
            dev.queue.write_buffer(pre["bt"].buf,       0, _bt_bytes)

            normed_x = sc["normed"]
            x_buf    = pre["x"]
            for chunk_idx, chunk_layers in enumerate(batched(range(self.num_layers), self._PREFILL_CHUNK)):
                with self._batched_dispatch():
                    if chunk_idx == 0:
                        self._dispatch(
                            "embedding_lookup_f32",
                            [self.weights[self._embed_key()], pre["ids"], pre["x"]],
                            {"HIDDEN_DIM": hidden}, (1, 1, 1))
                        self._dispatch(
                            "rms_norm_f32in",
                            [pre["x"],
                             self.weights[f"{self._layer_key_prefix(0)}.input_layernorm.weight"],
                             sc["normed"]],
                            _rms, (1, 1, 1))

                    for layer_idx in chunk_layers:
                        normed_x, x_buf = self._transformer_layer(
                            layer_idx, normed_x, x_buf,
                            pre["pos"], pre["slot_map"], pre["bt"], tok_ctx, 1,
                        )

        # Final norm and LM head on the last token's hidden state.
        _lm_key, lm_head_w, uq_lm, _lm_base = self._lm_head_parts()
        with self._batched_dispatch():
            self._dispatch(
                "rms_norm_f32in",
                [x_buf, self.weights[self._norm_key()], pre["norm_out"]],
                _rms, (1, 1, 1))
            self._dispatch(
                "matmul_quant",
                [pre["norm_out"], lm_head_w,
                 self._scales_buf(_lm_key, uq_lm, self._dummy_buf), pre["logits"]],
                {"K": hidden, "N": vocab, "USE_QUANT": uq_lm, "SPLIT_K": 0,
                 **self._quant_extra(_lm_base, uq_lm)},
                _rows_wg(vocab))

            self._dispatch_softcap_and_sample(vocab, pre["logits"], pre.get("capped", self._dummy_buf))

        return self._finish_forward(self._greedy_decode)

    def _transformer_layer(
        self,
        layer_idx: int,
        normed_x: "WebGPUBuffer",
        x_buf: "WebGPUBuffer",
        pos_buf: "WebGPUBuffer",
        slot_map: "WebGPUBuffer",
        bt_buf: "WebGPUBuffer",
        ctx_len: int,
        num_tokens: int,
    ) -> "tuple[WebGPUBuffer, WebGPUBuffer]":
        """Returns (normed_out, raw_out) — normed_out is sc['normed'] for next layer."""
        sc = self._sc
        lp = self._lp[layer_idx]
        hidden = self.hidden_size
        p = self._layer_key_prefix(layer_idx)
        q_dim = lp["q_dim"]
        kv_dim = lp["kv_dim"]
        head_dim = lp["head_dim"]
        num_kv_heads = lp["num_kv_heads"]
        has_v = lp["has_v_proj"]
        inter = lp["intermediate_size"]
        # Per-layer scalar from GGUF (layer_scalar weight, e.g. ~0.97 or ~0.053 depending on model).
        # Applied to the full residual once after both attn and FFN sublayers, matching vLLM:
        #   hidden_states = hidden_states * self.layer_scalar
        # Cached at load_weights() — no GPU-to-CPU readback per token.
        _ls = self._layer_scales[layer_idx]
        # KV-shared layers reuse the target layer's KV cache (mirrors vLLM is_kv_shared_layer).
        is_kv_shared    = lp["is_kv_shared"]
        kv_shared_target = lp["kv_shared_target"]

        residual = sc[_H_NAMES[(self._hstate + 1) % 3]]
        out = sc[_H_NAMES[(self._hstate + 2) % 3]]
        add_n = num_tokens * hidden
        gelu_n = num_tokens * inter
        _rms = self._rms_consts

        _kv_layer = kv_shared_target if (is_kv_shared and kv_shared_target >= 0) else layer_idx
        k_cache, v_cache = self.kv_pool[_kv_layer]

        with self._batched_dispatch(label=f"L{layer_idx:02d}"):
            # normed_x already pre-normalized by caller (or previous layer's fused add_f32_rms_norm).

            # QKV projections: fused for f16 local layers; separate for quantized weights or
            # global layers (has_v=False, no v_proj weight).
            qw = f"{p}.self_attn.q_proj.weight"
            kw = f"{p}.self_attn.k_proj.weight"
            uq_q = self._uq_for_key(qw)
            uq_k = self._uq_for_key(kw)
            if has_v and not is_kv_shared:
                vw = f"{p}.self_attn.v_proj.weight"
                uq_v = self._uq_for_key(vw)
                _use_fused_qkv = (uq_q == 0 and uq_k == 0 and uq_v == 0)
            else:
                _use_fused_qkv = False

            # _use_fused_qkv is already False when is_kv_shared=True because the else
            # branch sets it False for both not-has_v and is_kv_shared cases.
            if _use_fused_qkv:
                # All f16, non-shared: single fused_qkv dispatch → sc["qkv_buf"] laid out as [Q | K | V].
                self._dispatch("fused_qkv",
                               [normed_x, self.weights[qw], self.weights[kw], self.weights[vw],
                                sc["qkv_buf"]],
                               {"K": hidden, "Q_DIM": q_dim, "KV_DIM": kv_dim},
                               (q_dim + 2 * kv_dim, 1, 1))
                _q_src = sc["qkv_buf"]       # Q at element offset 0
                _k_src = sc["qkv_buf"]       # K at element offset q_dim
                _v_src = sc["qkv_buf"]       # V at element offset q_dim + kv_dim
                _v_src_offset = q_dim + kv_dim
            else:
                # Separate projections (quantized weights, global attention, or KV-shared layer).
                # KV-shared layers only need Q; K and V come from the target layer's KV cache.
                # Initialize defaults so all three names are bound regardless of is_kv_shared.
                _v_src_offset = 0
                _v_src = sc["v_buf"]
                _k_src = sc["k_buf"]
                self._dispatch("matmul_quant",
                               [normed_x, self.weights[qw],
                                self._scales_buf(qw, uq_q, self._dummy_buf), sc["q_buf"]],
                               {"K": hidden, "N": q_dim, "USE_QUANT": uq_q,
                                **self._quant_extra(f"{p}.self_attn.q_proj", uq_q)},
                               (q_dim, 1, 1))
                if not is_kv_shared:
                    self._dispatch("matmul_quant",
                                   [normed_x, self.weights[kw],
                                    self._scales_buf(kw, uq_k, self._dummy_buf), sc["k_buf"]],
                                   {"K": hidden, "N": kv_dim, "USE_QUANT": uq_k,
                                    **self._quant_extra(f"{p}.self_attn.k_proj", uq_k)},
                                   (kv_dim, 1, 1))
                    if has_v:
                        self._dispatch("matmul_quant",
                                       [normed_x, self.weights[vw],
                                        self._scales_buf(vw, uq_v, self._dummy_buf), sc["v_buf"]],
                                       {"K": hidden, "N": kv_dim, "USE_QUANT": uq_v,
                                        **self._quant_extra(f"{p}.self_attn.v_proj", uq_v)},
                                       (kv_dim, 1, 1))
                        _v_src = sc["v_buf"]
                    else:
                        _v_src = sc["k_buf"]  # global attention: V = K
                    _k_src = sc["k_buf"]
                    _v_src_offset = 0
                _q_src = sc["q_buf"]

            # Per-head RMSNorm + RoPE for Q and K.
            # When both norm weights exist: fused_qk_norm_rope handles both in one dispatch.
            #   Fused QKV path: K_SEPARATE=0, both Q and K in qkv_buf (INPUT_OFFSET_K=q_dim).
            #   Separate buffer path: K_SEPARATE=1, Q in q_buf (binding 0), K in k_buf (binding 6).
            q_norm_w = self.weights.get(f"{p}.self_attn.q_norm.weight")
            k_norm_w_l = self.weights.get(f"{p}.self_attn.k_norm.weight")
            _freq_buf = self._rope_freq_buf
            # Per-layer rope constants (typed dataclass; access fields directly).
            rc = self._rope_consts[layer_idx]

            if is_kv_shared:
                # KV-shared layer (last N sliding-attention layers in laptop Gemma4 variant):
                # Q gets norm + RoPE; K norm, K RoPE, V norm, and KV cache store are all
                # skipped. K and V for attention come from the target layer's KV cache.
                # Matches vLLM Gemma4Attention.forward when is_kv_shared_layer=True.
                if q_norm_w is not None:
                    self._dispatch("fused_per_head_norm_rope",
                                   [_q_src, q_norm_w, pos_buf, sc["q_rope"], _freq_buf],
                                   {"ROPE_BASE": rc.rope_base, "LN_ROPE_BASE": rc.ln_rope_base,
                                    "USE_FREQ_BUF": rc.use_freq_buf, "ROTARY_DIM": rc.rotary_dim,
                                    "FREQ_DIM": rc.freq_dim, "HEAD_DIM": head_dim,
                                    "NUM_HEADS": self.num_q_heads, "HAS_WEIGHT": 1,
                                    "GEMMA_NORM": self._GEMMA_NORM, "INPUT_OFFSET": 0},
                                   (self.num_q_heads, num_tokens, 1))
                else:
                    self._dispatch("rope", [_q_src, pos_buf, sc["q_rope"], _freq_buf],
                                   {"ROPE_BASE": rc.rope_base, "LN_ROPE_BASE": rc.ln_rope_base,
                                    "USE_FREQ_BUF": rc.use_freq_buf, "HEAD_DIM": head_dim,
                                    "NUM_HEADS": self.num_q_heads},
                                   (num_tokens, self.num_q_heads, 1))
            else:
                # Non-shared: Q + K norm + RoPE, then V norm and KV cache store.
                if q_norm_w is not None and k_norm_w_l is not None:
                    _k_separate = 0 if _use_fused_qkv else 1
                    _k_in_offset = q_dim if _use_fused_qkv else 0
                    _k_bind = sc["qkv_buf"] if _use_fused_qkv else sc["k_buf"]
                    self._dispatch("fused_qk_norm_rope",
                                   [_q_src, q_norm_w, k_norm_w_l, pos_buf,
                                    sc["q_rope"], sc["k_rope"], _k_bind, _freq_buf],
                                   {"ROPE_BASE": rc.rope_base, "LN_ROPE_BASE": rc.ln_rope_base,
                                    "USE_FREQ_BUF": rc.use_freq_buf, "ROTARY_DIM": rc.rotary_dim,
                                    "FREQ_DIM": rc.freq_dim,
                                    "HEAD_DIM": head_dim,
                                    "NUM_Q_HEADS": self.num_q_heads,
                                    "NUM_KV_HEADS": num_kv_heads,
                                    "HAS_WEIGHT": 1,
                                    "GEMMA_NORM": self._GEMMA_NORM,
                                    "INPUT_OFFSET_K": _k_in_offset,
                                    "K_SEPARATE": _k_separate},
                                   (self.num_q_heads + num_kv_heads, num_tokens, 1))
                else:
                    # Fallback: separate per-head norm+rope or plain rope for each of Q and K.
                    _k_in_off = q_dim if _use_fused_qkv else 0
                    for src, dst, n_heads, w_key, in_off in [
                        (_q_src, sc["q_rope"], self.num_q_heads, f"{p}.self_attn.q_norm.weight", 0),
                        (_k_src, sc["k_rope"], num_kv_heads, f"{p}.self_attn.k_norm.weight", _k_in_off),
                    ]:
                        norm_w = self.weights.get(w_key)
                        if norm_w is not None:
                            self._dispatch("fused_per_head_norm_rope",
                                           [src, norm_w, pos_buf, dst, _freq_buf],
                                           {"ROPE_BASE": rc.rope_base, "LN_ROPE_BASE": rc.ln_rope_base,
                                            "USE_FREQ_BUF": rc.use_freq_buf, "ROTARY_DIM": rc.rotary_dim,
                                            "FREQ_DIM": rc.freq_dim, "HEAD_DIM": head_dim,
                                            "NUM_HEADS": n_heads, "HAS_WEIGHT": 1,
                                            "GEMMA_NORM": self._GEMMA_NORM,
                                            "INPUT_OFFSET": in_off},
                                           (n_heads, num_tokens, 1))
                        elif not _use_fused_qkv:
                            # Plain rope from standalone buffer (no offset needed).
                            self._dispatch("rope", [src, pos_buf, dst, _freq_buf],
                                           {"ROPE_BASE": rc.rope_base, "LN_ROPE_BASE": rc.ln_rope_base,
                                            "USE_FREQ_BUF": rc.use_freq_buf, "HEAD_DIM": head_dim,
                                            "NUM_HEADS": n_heads},
                                           (num_tokens, n_heads, 1))
                        else:
                            # Rope from fused QKV buffer at in_off — use HAS_WEIGHT=0 variant.
                            self._dispatch("fused_per_head_norm_rope",
                                           [src, src, pos_buf, dst, _freq_buf],
                                           {"ROPE_BASE": rc.rope_base, "LN_ROPE_BASE": rc.ln_rope_base,
                                            "USE_FREQ_BUF": rc.use_freq_buf, "ROTARY_DIM": rc.rotary_dim,
                                            "FREQ_DIM": rc.freq_dim, "HEAD_DIM": head_dim,
                                            "NUM_HEADS": n_heads, "HAS_WEIGHT": 0,
                                            "GEMMA_NORM": 0, "INPUT_OFFSET": in_off},
                                           (n_heads, num_tokens, 1))

                # Per-head RMSNorm (no weight) on V before caching — Gemma4 only.
                # Gemma3 does NOT apply V normalization (no v_norm weight in the model).
                # V_IN_OFFSET is non-zero when V lives inside the fused qkv_buf.
                if self._apply_v_norm:
                    self._dispatch("per_head_rms_norm_no_weight", [_v_src, sc["v_normed"]],
                                   {"HEAD_DIM": head_dim, "NUM_HEADS": num_kv_heads,
                                    "WG_SIZE": min(head_dim, 128),
                                    "V_IN_OFFSET": _v_src_offset},
                                   (num_kv_heads, num_tokens, 1), shader_subdir="gemma")
                    v_to_cache = sc["v_normed"]
                    _v_cache_offset = 0
                else:
                    v_to_cache = _v_src  # Gemma3: use V directly without normalization
                    _v_cache_offset = _v_src_offset  # 0 when not fused; q_dim+kv_dim when fused

                # Fused K+V cache store — single dispatch saves 1 overhead per layer.
                self._dispatch("kv_cache_store_both",
                               [sc["k_rope"], k_cache, v_to_cache, v_cache, slot_map],
                               {"BLOCK_SIZE": self.block_size, "NUM_KV_HEADS": num_kv_heads,
                                "HEAD_DIM": head_dim, "V_IN_OFFSET": _v_cache_offset},
                               (num_tokens, num_kv_heads, 1))

            # Always use flash_attn_decode for single-token decode.
            # The 65535 limit applied to attn_score's dispatch dimension; flash_attn_decode
            # loops internally and has no dispatch dimension limit.
            self._dispatch("flash_attn_decode",
                           [sc["q_rope"], k_cache, v_cache, bt_buf, sc["attn_out"]],
                           {"BLOCK_SIZE": self.block_size, "NUM_Q_HEADS": self.num_q_heads,
                            "NUM_KV_HEADS": num_kv_heads, "HEAD_DIM": head_dim,
                            "CTX_LEN": ctx_len,
                            "SCALE": lp["scale"]},
                           (self.num_q_heads, 1, 1))

            # Output projection → sc["o_proj_out"]
            ow = f"{p}.self_attn.o_proj.weight"
            uq = self._uq_for_key(ow)
            self._dispatch("matmul_quant",
                           [sc["attn_out"], self.weights[ow],
                            self._scales_buf(ow, uq, self._dummy_buf), sc["o_proj_out"]],
                           {"K": q_dim, "N": hidden, "USE_QUANT": uq,
                            **self._quant_extra(f"{p}.self_attn.o_proj", uq)},
                           (hidden, 1, 1))

            # Correct Gemma4 attention sublayer (matches HF Gemma3DecoderLayer.forward):
            #   residual = x
            #   hidden = input_layernorm(x)     → attn → o_proj
            #   hidden = post_attention_layernorm(hidden)   ← norm on ATTN OUTPUT (before residual)
            #   residual = residual + hidden                ← residual add AFTER norm
            #
            # When both post_attn_norm and pre_ffn_norm are present, fuse them with
            # rms_norm_add_f32_rms_norm to keep the intermediate in registers only.
            post_attn_norm_w = self.weights.get(f"{p}.post_attention_layernorm.weight")
            pre_ffn_norm_w = self.weights.get(f"{p}.pre_feedforward_layernorm.weight")
            if post_attn_norm_w is not None and pre_ffn_norm_w is not None:
                # Fused: rms_norm(o_proj_out, post_attn_w) + residual_add + rms_norm(residual, pre_ffn_w)
                # Eliminates 1 dispatch vs the rms_norm → add_f32_rms_norm pair.
                # No SCALE here: vLLM does not apply layer_scalar at the attention sublayer.
                self._dispatch("rms_norm_add_f32_rms_norm",
                               [sc["o_proj_out"], post_attn_norm_w,
                                x_buf, pre_ffn_norm_w,
                                residual, sc["normed"]],
                               _rms, (num_tokens, 1, 1))
                ffn_normed = sc["normed"]
            else:
                if post_attn_norm_w is None:
                    raise ValueError(
                        f"Layer {layer_idx} missing post_attention_layernorm.weight "
                        "— vLLM creates this norm unconditionally; absence indicates "
                        "a corrupt checkpoint"
                    )
                raise ValueError(
                    f"Layer {layer_idx} missing pre_feedforward_layernorm.weight "
                    "— f32 residual cannot be fed to f16 FFN projection"
                )

            # Gate + up projection
            # Fused gate+up (f16 only); Gemma uses tanh-GELU.
            gw_k = f"{p}.mlp.gate_proj.weight"
            uw_k = f"{p}.mlp.up_proj.weight"
            uq_g = self._uq_for_key(gw_k)
            uq_u = self._uq_for_key(uw_k)
            if uq_g == 0 and uq_u == 0:
                self._dispatch("fused_gate_act",
                               [ffn_normed, self.weights[gw_k], self.weights[uw_k],
                                sc["ffn_act"]],
                               {"K": hidden, "N": inter, "GELU": 1}, (inter, 1, 1))
            else:
                for out_b, proj, w_k, uq2 in [
                        (sc["gate_buf"], "gate_proj", gw_k, uq_g),
                        (sc["up_buf"],   "up_proj",   uw_k, uq_u)]:
                    self._dispatch("matmul_quant",
                                   [ffn_normed, self.weights[w_k],
                                    self._scales_buf(w_k, uq2, self._dummy_buf), out_b],
                                   {"K": hidden, "N": inter, "USE_QUANT": uq2,
                                    **self._quant_extra(f"{p}.mlp.{proj}", uq2)},
                                   (inter, 1, 1))
                self._dispatch("gelu_mul", [sc["gate_buf"], sc["up_buf"], sc["ffn_act"]],
                               {"N": gelu_n}, _vec4_wg(gelu_n),
                               shader_subdir="gemma")

            # Down projection → sc["ffn_out"]
            w_k = f"{p}.mlp.down_proj.weight"
            uq = self._uq_for_key(w_k)
            self._dispatch("matmul_quant",
                           [sc["ffn_act"], self.weights[w_k],
                            self._scales_buf(w_k, uq, self._dummy_buf), sc["ffn_out"]],
                           {"K": inter, "N": hidden, "USE_QUANT": uq,
                            **self._quant_extra(f"{p}.mlp.down_proj", uq)},
                           (hidden, 1, 1))

            # Post-FFN norm on FFN output (before residual add), then fused residual + next pre-norm.
            # When post_ffw_w and next input_layernorm both exist (all non-last layers),
            # fuse into rms_norm_add_f32_rms_norm to keep the intermediate in registers.
            post_ffw_w = self.weights.get(f"{p}.post_feedforward_layernorm.weight")
            if post_ffw_w is None:
                raise ValueError(
                    f"Layer {layer_idx} missing post_feedforward_layernorm.weight "
                    "— vLLM applies this norm unconditionally; a missing weight "
                    "indicates a corrupt or incomplete checkpoint."
                )
            if layer_idx < self.num_layers - 1:
                next_w = self.weights[f"{self._layer_key_prefix(layer_idx + 1)}.input_layernorm.weight"]
                # Fused: rms_norm(ffn_out, post_ffw_w) + residual_add + rms_norm(residual, next_w)
                # SCALE=1.0 (default): layer_scalar applied separately below via f32_scale_inplace.
                # RMSNorm is scale-invariant so normed_out is correct even after scaling out.
                self._dispatch("rms_norm_add_f32_rms_norm",
                               [sc["ffn_out"], post_ffw_w,
                                residual, next_w,
                                out, sc["normed"]],
                               _rms, (num_tokens, 1, 1))
            else:
                # Last layer: no next pre-norm, just update residual.
                self._dispatch("rms_norm", [sc["ffn_out"], post_ffw_w, sc["o_proj_out"]],
                               _rms, (num_tokens, 1, 1))
                ffn_delta = sc["o_proj_out"]
                self._dispatch("add_f32", [residual, ffn_delta, out],
                               {"N": add_n}, _vec4_wg(add_n))

            # Apply layer_scalar to the full residual once per decoder layer.
            # Matches vLLM: hidden_states = hidden_states * self.layer_scalar,
            # which scales (x + delta_attn + delta_ffn), not just the deltas.
            if abs(_ls - 1.0) > 1e-6:
                self._dispatch("f32_scale_inplace", [out],
                               {"N": add_n, "SCALE": _ls},
                               (cdiv(add_n, 256), 1, 1))

        self._hstate = (self._hstate + 2) % 3
        return sc["normed"], out
