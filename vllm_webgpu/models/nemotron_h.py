from __future__ import annotations
import math
from typing import TYPE_CHECKING

import numpy as np

from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator, is_conv_state_dim_first
from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM as _NemotronHForCausalLM
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm_webgpu.models.base import BaseWebGPUModel, _vals_per_thread, _vec4_wg, _rows_wg, _H_NAMES
from vllm_webgpu.webgpu.buffer import WebGPUBuffer, _ELEM_BYTES, assert_elem_bytes_stable

if TYPE_CHECKING:
    from collections.abc import Callable
    from vllm_webgpu.webgpu.device import WebGPUDevice
    from vllm_webgpu.webgpu.pipeline import PipelineCache

logger = init_logger(__name__)

# Verify _ELEM_BYTES values used to size conv/SSM state buffers in _init_mamba_states.
# The conv_bytes formula hard-codes f16 and the ssm_bytes formula hard-codes f32 because
# the WGSL shaders run at fixed precision and cannot switch dtype at runtime.
assert_elem_bytes_stable()

# Shared transform for all Mamba A_log weights: -exp(A_log) converts the log-space
# parameter to the negative-real value expected by the Mamba SSM kernel.
# Defined once at module scope so the same function object is reused across all
# Mamba layers rather than allocating a new lambda per layer during __init__.
def _a_log_transform(arr: "np.ndarray") -> "np.ndarray":
    return -np.exp(arr)


# Verify that the upstream mapper fields match the snapshot this code was written
# against (vLLM 0.24.0). Catches upstream changes at import time.
_mapper = _NemotronHForCausalLM.hf_to_vllm_mapper
if _mapper.orig_to_new_prefix != {"backbone": "model"}:
    raise AssertionError(
        f"NemotronHForCausalLM.hf_to_vllm_mapper.orig_to_new_prefix changed upstream: "
        f"{_mapper.orig_to_new_prefix!r}. Review load_weights before removing this assertion."
    )
if _mapper.orig_to_new_substr != {"A_log": "A", "embeddings": "embed_tokens"}:
    raise AssertionError(
        f"NemotronHForCausalLM.hf_to_vllm_mapper.orig_to_new_substr changed upstream: "
        f"{_mapper.orig_to_new_substr!r}. Review load_weights before removing this assertion."
    )
# Behavioral check: verify the mapper's actual output, not just its fields.
# A new regex or suffix rule added to the mapper could silently rename keys
# (e.g. 'model.norm_f.weight') even if the field equality checks above still pass.
if _mapper.apply_list(["backbone.layers.0.mixer.A_log"]) != ["model.layers.0.mixer.A"]:
    raise AssertionError(
        "NemotronHForCausalLM.hf_to_vllm_mapper no longer produces the expected key "
        "for 'backbone.layers.0.mixer.A_log'. A new mapping rule may have been added. "
        "Review load_weights and update this assertion."
    )
if _mapper.apply_list(["backbone.embeddings.weight"]) != ["model.embed_tokens.weight"]:
    raise AssertionError(
        "NemotronHForCausalLM.hf_to_vllm_mapper no longer produces the expected key "
        "for 'backbone.embeddings.weight'. A new mapping rule may have been added. "
        "Review load_weights and update this assertion."
    )
del _mapper

# Layout invariant (enforced at runtime by _validate_mamba_weights): confirm
# MambaMixer2's in_proj layout still matches the formula used by in_proj_dim
# (mamba_int + conv_dim + mamba_num_heads).
# WebGPU always uses tp=1, so the active branch is always MergedColumnParallelLinear
# (mamba_mixer2.py L328-340), with output_sizes=[intermediate_size, intermediate_size,
# groups_ssm_state_size, groups_ssm_state_size, num_heads] (5 entries summing to
# 2*intermediate_size + 2*groups_ssm_state_size + num_heads).
# The ColumnParallelLinear branch (mamba_mixer2.py L353-355) uses a single
# output_size=intermediate_size + self.conv_dim + self.num_heads, which expands to
# the same total (intermediate_size + (intermediate_size + 2*groups_ssm_state_size)
# + num_heads). On each vLLM bump, diff BOTH branches:
#   - MergedColumnParallelLinear: mamba_mixer2.py L328-340 (output_sizes list)
#   - ColumnParallelLinear:       mamba_mixer2.py L353-355 (output_size expression)
# If either branch adds a new output group, in_proj_dim and the scratch buffer
# at _make_buf(IPD * 2) will be wrong until updated here.
# NOTE: _validate_mamba_weights is the authoritative runtime guard. It checks
# the actual in_proj.weight shape from the loaded checkpoint against in_proj_dim.

# conv_dim is computed here from config params using the same formula as MambaMixer2
# (mamba_mixer2.py L313: conv_dim = intermediate_size + 2 * groups_ssm_state_size).
# For tp=1, extra_groups_for_head_shards returns 0, so this exactly matches
# mamba2_state_shape. _validate_mamba_weights provides the authoritative runtime
# guard by checking the actual in_proj.weight shape.

# vLLM 0.24 does not export _resolve_intermediate_size. Local copy mirrors
# NemotronHMLPDecoderLayer.__init__ L286-292 (vLLM 0.24). On a future vLLM bump,
# run `grep -n '_resolve_intermediate_size' .venv/lib/python*/site-packages/vllm/model_executor/models/nemotron_h.py`
# to check whether the symbol is now exported at module scope (not indented inside a
# class); if so, replace this definition with
# `from vllm.model_executor.models.nemotron_h import _resolve_intermediate_size`.
def _resolve_intermediate_size(v, idx: int) -> int:
    if isinstance(v, list):
        return v[0] if len(v) == 1 else v[idx]
    return v


# Correctness of _resolve_intermediate_size is verified in
# tests/test_nemotron_h_model.py::test_resolve_intermediate_size.


# USE_QUANT values returned by _uq_for_key for each quantization scheme.
# 0 = F16 (no quantization), 3 = GPTQ int4, 4 = AWQ sym int4,
# 5 = fp8_gpu, 6 = nvfp4_gpu, 7 = int8_gpu, 8 = nf4_gpu.
# Values 5-8 use the non-AWQ GPU-side byte-concat path in _pack_attn_weights;
# the _is_awq == 4 check is the sole gate selecting AWQ-specific unpacking.


class NemotronHWebGPUModel(BaseWebGPUModel):
    """
    Nemotron-H hybrid Mamba-2 SSM / Attention model (WebGPU decode backend).

    Architecture: NemotronHForCausalLM from vLLM / HuggingFace.
    Each layer is one of:
      M = Mamba-2 SSM  (in_proj -> conv -> SSM -> gated_norm -> out_proj)
      * = Attention     (qkv_proj -> flash_attn -> o_proj)
      - = MLP-only      (up_proj -> relu^2 -> down_proj)
      E = MoE (not implemented; raises at runtime)

    The layer sequence is encoded in config.hybrid_override_pattern as a
    string of the characters above, one per layer.

    Weight key remapping: HuggingFace checkpoints use 'backbone.' prefix and
    store all mixer weights under '.mixer.' regardless of block type. Keys are
    remapped to the vLLM-canonical 'model.' prefix; no per-type renaming is needed.
    """

    def __init__(
        self,
        model_config,
        wgpu_device: "WebGPUDevice",
        pipeline_cache: "PipelineCache",
        block_size: int = 16,
    ) -> None:
        super().__init__(model_config, wgpu_device, pipeline_cache)

        self.hidden_size: int = model_config.hidden_size
        self.num_layers: int = model_config.num_hidden_layers
        self.vocab_size: int = model_config.vocab_size

        # Attention layer parameters
        self.num_q_heads: int = model_config.num_attention_heads
        self.num_kv_heads: int = model_config.num_key_value_heads
        hd = getattr(model_config, "head_dim", None)
        self.head_dim: int = hd if hd is not None else self.hidden_size // self.num_q_heads
        self._attn_scale: float = self.head_dim ** -0.5
        self._q_dim: int = self.num_q_heads * self.head_dim
        self._k_dim: int = self.num_kv_heads * self.head_dim
        # Mamba-2 parameters
        self.mamba_num_heads: int = model_config.mamba_num_heads
        self.mamba_head_dim: int = model_config.mamba_head_dim
        if self.mamba_head_dim > 256:
            raise NotImplementedError(
                f"mamba_head_dim={self.mamba_head_dim} exceeds WG_SIZE=256; "
                "mamba2_ssm_step Phase 2 requires WG_SIZE >= HEAD_DIM. "
                "Increase WG_SIZE in the shader and this dispatch, or add a "
                "second dispatch for the remaining elements."
            )
        # mamba_int: the Mamba "intermediate size" = num_heads * head_dim
        self.mamba_int: int = self.mamba_num_heads * self.mamba_head_dim
        self.n_groups: int = model_config.n_groups
        self.ssm_state_size: int = model_config.ssm_state_size
        self.conv_kernel: int = model_config.conv_kernel
        # conv_dim uses the same formula as MambaMixer2 (mamba_mixer2.py L313):
        # conv_dim = intermediate_size + 2 * n_groups * state_size.
        # For tp=1 extra_groups_for_head_shards returns 0, so this is exact.
        # On each vLLM bump, verify the formula against mamba_mixer2.py L313.
        # _validate_mamba_weights is the primary runtime guard and must remain.
        self.conv_dim: int = self.mamba_int + 2 * self.n_groups * self.ssm_state_size
        # in_proj output: [gate (mamba_int) | x_B_C (conv_dim) | dt (mamba_num_heads)]
        # Derived from MambaMixer2's two in_proj branches (both produce the same total
        # for tp=1; WebGPU never runs tp>1):
        #   MergedColumnParallelLinear (mamba_mixer2.py L328-340):
        #     output_sizes = [intermediate_size, intermediate_size,
        #                     groups_ssm_state_size, groups_ssm_state_size, num_heads]
        #     total = 2*intermediate_size + 2*groups_ssm_state_size + num_heads
        #   ColumnParallelLinear (mamba_mixer2.py L353-355):
        #     output_size = intermediate_size + conv_dim + num_heads
        #                 = intermediate_size + (intermediate_size +
        #                   2*groups_ssm_state_size) + num_heads  (same total)
        # On each vLLM bump, diff BOTH branches to catch a new output group.
        # _validate_mamba_weights is the primary runtime guard and must remain.
        self.in_proj_dim: int = (
            self.mamba_int + self.conv_dim + self.mamba_num_heads
        )

        self.block_size: int = block_size

        self._layer_types: list[str] = model_config.layers_block_type
        # Length invariant is enforced by NemotronHConfig.__init__ asserting
        # len(hybrid_override_pattern) == num_hidden_layers.

        # Register CPU-side A_log → -exp(A) transforms for all Mamba layers.
        # Applied during load_weights before GPU upload, eliminating a per-layer
        # GPU readback+re-upload that _validate_mamba_weights would otherwise
        # require. The transform uses the HF checkpoint key name (backbone. prefix).
        #
        # Per-layer CPU-side scale accumulator for attention layers.
        # For GPTQ (and AWQ) checkpoints the weight loader fires these callbacks
        # with each projection's scale array instead of uploading it to GPU.
        # _pack_attn_weights then stacks q/k/v on CPU in one shot and uploads the
        # packed [G, N_total] array once, eliminating the create_buffer /
        # submit / map_sync / unmap cycle that re-read each layer's scales from GPU.
        # Keys are HF-format scale key names (backbone. prefix, before mapper).
        self._scale_acc: dict[int, dict[str, np.ndarray]] = {}
        self._scale_transforms: dict[str, Callable[[np.ndarray], None]] = {}

        for _i, _lt in enumerate(self._layer_types):
            if _lt == "mamba":
                # -exp(A_log): transforms HF A_log checkpoint values before GPU upload.
                self._weight_transforms[f"backbone.layers.{_i}.mixer.A_log"] = _a_log_transform
            elif _lt == "attention":  # mutually exclusive with mamba; elif avoids redundant test
                _acc: dict = {}
                self._scale_acc[_i] = _acc
                _hf_p = f"backbone.layers.{_i}.mixer"
                for _proj in ("q", "k", "v"):
                    self._scale_transforms[f"{_hf_p}.{_proj}_proj.weight.scales"] = (
                        (lambda arr, k=_proj, acc=_acc: acc.__setitem__(k, arr))
                    )

        # The WebGPU MLP path does not implement bias addition. All known
        # NemotronH checkpoints ship with mlp_bias=False (the default), so
        # this is latent. Fail fast rather than silently produce wrong outputs
        # if a checkpoint with mlp_bias=True is ever loaded.
        if model_config.mlp_bias:
            raise NotImplementedError(
                "NemotronHWebGPUModel does not support mlp_bias=True. "
                "The WebGPU _mlp_layer path omits the up_proj and down_proj "
                "bias additions. Implement bias-add dispatches before using "
                "a checkpoint with mlp_bias=True."
            )

        # The WebGPU _mamba_layer path does not apply in_proj.bias or
        # out_proj.bias. All known NemotronH checkpoints ship with
        # use_bias=False, so this is latent. Fail fast rather than silently
        # produce wrong Mamba outputs if a checkpoint with use_bias=True is
        # ever loaded.
        if model_config.use_bias:
            raise NotImplementedError(
                "NemotronHWebGPUModel does not support use_bias=True. "
                "The WebGPU _mamba_layer path omits in_proj.bias and "
                "out_proj.bias additions. Implement bias-add dispatches "
                "before using a checkpoint with use_bias=True."
            )

        # relu_sq.wgsl hard-codes relu^2 for MLP layers.
        # NemotronHConfig exposes mlp_hidden_act (defaults to "relu2"). Fail fast
        # if a checkpoint uses a different activation to prevent silently wrong outputs.
        if model_config.mlp_hidden_act not in ("relu2", "squared_relu"):
            raise NotImplementedError(
                f"NemotronHWebGPUModel requires mlp_hidden_act=relu2; "
                f"relu_sq.wgsl hard-codes relu^2, got {model_config.mlp_hidden_act!r}."
            )

        # mamba2_causal_conv.wgsl hard-codes SiLU as the conv activation.
        # NemotronHConfig exposes mamba_hidden_act (default "silu"); if a
        # checkpoint sets it to anything else the conv outputs will be silently
        # wrong. Fail fast, consistent with the mlp_bias/use_bias guards above.
        if model_config.mamba_hidden_act != "silu":
            raise NotImplementedError(
                "NemotronHWebGPUModel requires mamba_hidden_act=\"silu\". "
                "mamba2_causal_conv.wgsl hard-codes SiLU as the conv "
                "activation; other activations produce silently wrong outputs."
            )

        # _layer_dispatch has no MoE branch. A checkpoint with 'E' entries in
        # hybrid_override_pattern would load all weights, fill GPU memory, then
        # raise an unguarded NotImplementedError on the first forward pass.
        # Fail here instead, before any GPU allocation happens.
        if "moe" in self._layer_types:
            raise NotImplementedError(
                "NemotronHWebGPUModel does not support MoE layers "
                "(hybrid_override_pattern contains 'E')."
            )

        # Precomputed per-layer intermediate size for heterogeneous MLP configs.
        # Index by layer_idx; 0 for non-MLP layers. Avoids O(num_layers) slice-
        # and-count inside _mlp_layer on every forward pass.
        # Per-layer config override: some NemotronH variants (puzzle-style heterogeneous
        # checkpoints) expose get_nemotron_h_config_for_layer() on the model_config
        # to return per-layer overrides, including a different intermediate_size.
        _get_layer_cfg = getattr(model_config, 'get_nemotron_h_config_for_layer', None)

        # Build per-layer intermediate sizes in O(num_layers) using a running MLP counter.
        # The vLLM canonical expression is:
        #   mlp_index = hybrid_override_pattern[: layer_idx + 1].count("-") - 1
        # (NemotronHMLPDecoderLayer.__init__, vLLM 0.24, lines 280-292)
        # A slice + count() per layer is O(n^2). The running counter below is O(n).
        # VERSION SYNC: verify on each vLLM version bump that this resolution
        # logic has not changed.
        _layer_int_sizes: list[int] = []
        _mlp_count = 0  # 0-indexed MLP position; matches vLLM's mlp_index = count("-") - 1
        for _li, _lt in enumerate(self._layer_types):
            if _lt != "mlp":
                _layer_int_sizes.append(0)
                continue
            if _get_layer_cfg is not None:
                _lcfg = _get_layer_cfg(_li)
                # Per-layer bias check for puzzle (heterogeneous) models.
                # The global guard above only inspects the top-level config;
                # individual layers can carry mlp_bias=True even when the
                # global value is False. vLLM reads the per-layer config at
                # line 299 of its nemotron_h.py (bias=config.mlp_bias after
                # config=layer_config), so we must mirror that here.
                if getattr(_lcfg, 'mlp_bias', False):
                    raise NotImplementedError(
                        f"NemotronHWebGPUModel does not support mlp_bias=True "
                        f"(found on layer {_li}). The WebGPU _mlp_layer path "
                        f"omits the up_proj and down_proj bias additions. "
                        f"Implement bias-add dispatches before using a puzzle "
                        f"checkpoint with per-layer mlp_bias=True."
                    )
                _lact = getattr(_lcfg, 'mlp_hidden_act', model_config.mlp_hidden_act)
                if _lact not in ('relu2', 'squared_relu'):
                    raise NotImplementedError(
                        f"Layer {_li} mlp_hidden_act={_lact!r}; "
                        f"relu_sq.wgsl hard-codes relu^2."
                    )
                # Do not silently fall back to the global intermediate_size
                # when a per-layer config exists but omits the attribute.
                # vLLM accesses config.intermediate_size directly (AttributeError
                # if absent); matching that behavior avoids wrong output for
                # puzzle checkpoints whose per-layer config should have the field
                # but does not due to a misconfigured checkpoint.
                if not hasattr(_lcfg, 'intermediate_size'):
                    raise AttributeError(
                        f"Per-layer config for layer {_li} returned by "
                        f"get_nemotron_h_config_for_layer() has no "
                        f"'intermediate_size' attribute. vLLM would raise "
                        f"AttributeError here; silently substituting the global "
                        f"value would produce wrong results for this puzzle "
                        f"checkpoint. Fix the per-layer config or remove the "
                        f"get_nemotron_h_config_for_layer override."
                    )
                _isize = _resolve_intermediate_size(_lcfg.intermediate_size, _mlp_count)
                _layer_int_sizes.append(_isize)
            else:
                _layer_int_sizes.append(_resolve_intermediate_size(model_config.intermediate_size, _mlp_count))
            _mlp_count += 1  # post-increment: index for the next MLP layer
        self._layer_int_size: list[int] = _layer_int_sizes

        # Persistent Mamba state buffers — allocated in _init_mamba_states()
        # after weights are loaded (device is available from __init__).
        self._conv_states: dict[int, "WebGPUBuffer"] = {}
        self._ssm_states: dict[int, "WebGPUBuffer"] = {}
        self._rms_base: dict = {
            "HIDDEN_DIM": self.hidden_size,
            "VALS_PER_THREAD": _vals_per_thread(self.hidden_size),
        }
        # Cached per-layer pre-norm weights. Set by load_weights; empty list until then.
        # _layer_norm_weights[i] is used in _layer_dispatch to avoid per-step dict lookups.
        # _layer0_norm_w is a convenience alias to [0] also used by _norm0_w property.
        self._layer_norm_weights: "list" = []
        self._layer0_norm_w: "WebGPUBuffer | None" = None
        self._hstate: int = 0
        # Set True during replay_prefix_for_ssm so _attn_layer skips KV writes.
        self._replay_mode: bool = False
        self._init_scratch_buffers()

    @property
    def _norm0_w(self) -> "WebGPUBuffer":
        """Layer-0 pre-norm weight buffer, lazily populated from weights dict.

        Normally set by load_weights; the lazy fallback covers test paths that
        inject model.weights directly without calling load_weights.
        """
        if self._layer0_norm_w is None:
            self._layer0_norm_w = self.weights["model.layers.0.norm.weight"]
        return self._layer0_norm_w

    # ── Scratch buffer allocation ─────────────────────────────────────────────

    def _init_scratch_buffers(self) -> None:
        H   = self.hidden_size
        MI  = self.mamba_int
        CD  = self.conv_dim
        IPD = self.in_proj_dim
        MNH = self.mamba_num_heads
        I   = max(self._layer_int_size)
        V   = self.vocab_size

        max_ctx = self.model_config.max_position_embeddings
        # 4096-block floor provides 4096 * block_size token slots of headroom
        # regardless of max_position_embeddings, covering generation past the
        # declared context limit without reallocation.
        max_bt_blocks = max(4096, cdiv(max_ctx, self.block_size))

        # Fixed pre-allocated decode buffers (zero-alloc hot path for T=1).
        self._pre: dict[str, "WebGPUBuffer"] = {
            "ids":      self._make_buf(4),                # [1] u32
            "slot_map": self._make_buf(4),                # [1] u32
            "bt":       self._make_buf(max_bt_blocks * 4),  # block table
            "x":        self._make_buf(H * 2),       # [H] f16 embedding output
            "norm_out": self._make_buf(H * 2),       # [H] f16 final norm output
            "logits":   self._make_buf(V * 2),       # [V] f16 LM head output
        }

        # Scratch buffers shared across layers.
        self._sc: dict[str, "WebGPUBuffer"] = {
            # 3-buffer residual rotation to prevent aliasing between layers.
            "h0": self._make_buf(H * 2),
            "h1": self._make_buf(H * 2),
            "h2": self._make_buf(H * 2),
            # Pre-normed input for current layer's mixer.
            "normed": self._make_buf(H * 2),
            # Mixer output (all layer types write here before residual add).
            "mixer_out": self._make_buf(H * 2),

            # Mamba-2 intermediates
            "mamba_inproj":  self._make_buf(IPD * 2),  # [in_proj_dim] f16
            "mamba_conv_in": self._make_buf(CD * 2),   # [conv_dim] f16 — x_B_C extracted
            "mamba_conv_out": self._make_buf(CD * 2),  # [conv_dim] f16 — after conv+SiLU
            "mamba_dt":      self._make_buf(MNH * 2),  # [mamba_num_heads] f16 — dt
            "mamba_gate":    self._make_buf(MI * 2),   # [mamba_int] f16 — gate portion
            "mamba_ssm_y":   self._make_buf(MI * 2),   # [mamba_int] f16 — SSM step output
            "mamba_norm_out": self._make_buf(MI * 2),  # [mamba_int] f16 — after gated norm

            # Attention intermediates
            "qkv_buf":     self._make_buf((self._q_dim + 2 * self._k_dim) * 2),
            "q_buf":       self._make_buf(self._q_dim * 2),
            "k_buf":       self._make_buf(self._k_dim * 2),
            "v_buf":       self._make_buf(self._k_dim * 2),
            "attn_out":    self._make_buf(self._q_dim * 2),

            # MLP intermediates
            "up_buf":  self._make_buf(I * 2),
            "ffn_act": self._make_buf(I * 2),
        }

    # ── Mamba state management ────────────────────────────────────────────────

    def _init_mamba_states(self, num_spec: int = 0) -> None:
        """Allocate zero-initialized GPU buffers for each Mamba layer's state.

        num_spec must be 0: the mamba2_causal_conv WGSL shader operates on a
        fixed ring of conv_kernel-1 slots and does not read the extra num_spec
        slots that MambaStateShapeCalculator would add, so num_spec > 0 would
        silently waste VRAM. The guard in model_runner.py enforces this at the
        call site; the assert here makes the contract explicit at the model
        boundary.

        Buffer dtypes are fixed by the WGSL shader precision and are NOT driven
        by ``vllm_config.cache_config.mamba_cache_dtype``:

        - conv state: always f16 (2 bytes/element)
        - SSM state: always f32 (4 bytes/element)

        The shaders are compiled ahead-of-time at these precisions and cannot
        switch dtype at runtime, so ``mamba_cache_dtype`` is intentionally
        ignored on the WebGPU path. The import-time assertion on
        ``_ELEM_BYTES["f16"]`` guards against silent under-allocation if
        the byte-size table is ever refactored.
        """
        assert num_spec == 0, (
            "speculative decoding not yet supported on WebGPU "
            "(mamba2_causal_conv shader uses KERNEL-1 slots)"
        )
        if is_conv_state_dim_first():
            raise NotImplementedError(
                "VLLM_SSM_CONV_STATE_LAYOUT=DS is not supported on the WebGPU "
                "path: mamba2_causal_conv.wgsl hard-codes the SD (state_len-first) "
                "index formula and has no runtime path for DS layout."
            )
        conv_shape, ssm_shape = MambaStateShapeCalculator.mamba2_state_shape(
            tp_world_size=1,
            intermediate_size=self.mamba_int,
            n_groups=self.n_groups,
            num_heads=self.mamba_num_heads,
            head_dim=self.mamba_head_dim,
            state_size=self.ssm_state_size,
            conv_kernel=self.conv_kernel,
            num_spec=num_spec,
        )
        # WebGPU WGSL shaders operate at fixed precision: f16 for the conv
        # state and f32 for the SSM state. These sizes are not configurable
        # via mamba_cache_dtype on the WebGPU path; the shaders are compiled
        # ahead-of-time and cannot switch dtype at runtime.
        conv_bytes = math.prod(conv_shape) * _ELEM_BYTES["f16"]
        ssm_bytes  = math.prod(ssm_shape)  * _ELEM_BYTES["f32"]

        for i, lt in enumerate(self._layer_types):
            if lt != "mamba":
                continue
            self._conv_states[i] = self._make_buf(conv_bytes)
            self._ssm_states[i]  = self._make_buf(ssm_bytes)

    def reset_recurrent_states(self) -> None:
        """Zero all Mamba conv and SSM states. Call before each new request."""
        for buf in (*self._conv_states.values(), *self._ssm_states.values()):
            self._zero_write(buf)

    def save_recurrent_states(self) -> dict:
        """Snapshot all Mamba conv/SSM state buffers to CPU in one GPU readback.

        Returns {"conv": {layer_idx: bytes}, "ssm": {layer_idx: bytes}}.
        """
        bufs = (
            [("conv", i, b) for i, b in self._conv_states.items()] +
            [("ssm",  i, b) for i, b in self._ssm_states.items()]
        )
        return self._readback_recurrent_states(bufs)

    def restore_recurrent_states(self, states: dict) -> None:
        """Write saved state bytes back into Mamba conv/SSM GPU buffers.

        queue.write_buffer enqueues writes without blocking, so all layers
        are uploaded before the next GPU dispatch without an explicit submit.
        """
        dev = self.wgpu_device.wgpu_device
        for i, data in states.get("conv", {}).items():
            dev.queue.write_buffer(self._conv_states[i].buf, 0, data)
        for i, data in states.get("ssm", {}).items():
            dev.queue.write_buffer(self._ssm_states[i].buf, 0, data)

    def replay_prefix_for_ssm(
        self,
        token_ids: np.ndarray,
        block_ids: list,
        start_pos: int = 0,
    ) -> None:
        """Replay tokens to reconstruct Mamba SSM state after preemption.

        When a request is preempted and resumed with prefix-cached KV, the attention
        KV cache already holds the correct K/V for positions 0..pos-1. The caller
        must zero the SSM states with reset_recurrent_states() before calling this.

        This method runs the full model token by token (embedding + all layers) so
        the residual stream is correct for each Mamba layer. KV cache writes are
        skipped because the cache is already populated. Attention reads proceed
        normally using the existing KV data and the block table.

        After return, the Mamba conv/SSM states match what they would be after a
        normal prefill of token_ids starting at sequence position start_pos.

        Args:
            token_ids:  Tokens to replay.
            block_ids:  Block table for this request.
            start_pos:  Sequence position of the first token in token_ids. When
                        prefix caching was active during the original prefill,
                        pass num_computed so that tok_ctx is correct for each
                        attention read against the existing KV cache entries.
        """
        dev = self.wgpu_device.wgpu_device
        pre = self._pre
        sc  = self._sc

        bt_arr = np.array(block_ids, dtype=np.uint32)
        dev.queue.write_buffer(pre["bt"].buf, 0, bt_arr.tobytes())

        self._replay_mode = True
        try:
            for t in range(len(token_ids)):
                self._hstate = 0
                tok_ctx = start_pos + t + 1
                dev.queue.write_buffer(
                    pre["ids"].buf, 0,
                    token_ids[t:t+1].astype(np.uint32, copy=False).tobytes())

                with self._batched_dispatch():
                    self._dispatch(
                        "embedding_lookup",
                        [self.weights["model.embed_tokens.weight"], pre["ids"], pre["x"]],
                        {"HIDDEN_DIM": self.hidden_size},
                        (1, 1, 1),
                    )
                    self._dispatch(
                        "rms_norm",
                        [pre["x"],
                         self._norm0_w,
                         sc["normed"]],
                        self._rms_base,
                        (1, 1, 1),
                    )

                    normed_x = sc["normed"]
                    x_buf    = pre["x"]

                    for i in range(self.num_layers):
                        normed_x, x_buf = self._layer_dispatch(
                            i, normed_x, x_buf,
                            pre["slot_map"], pre["bt"],
                            tok_ctx, 1,
                        )
                    # No final norm or logit needed — SSM state is the goal.
        finally:
            self._replay_mode = False

    # ── Weight loading ────────────────────────────────────────────────────────

    def load_weights(self, path: str, *, num_spec: int = 0) -> None:
        """Load weights with key remapping and Mamba-specific postprocessing."""
        # D, dt_bias, and A_log are F32 in the checkpoint and are read as array<f32>
        # by the SSM shader. The base loader would downcast them to F16, losing 13
        # mantissa bits. A_log near 0 (slow-decay states) would suffer 5-10% relative
        # error, corrupting the SSM state transition coefficient A after -exp().
        # Pass their HF-side key names so they are uploaded as F32 directly.
        # HF prefix is 'backbone.' (mapper swaps it to 'model.').
        f32_keys = frozenset(
            f"backbone.layers.{i}.mixer.{wk}"
            for i, lt in enumerate(self._layer_types)
            if lt == "mamba"
            for wk in ("D", "dt_bias", "A_log")
        )
        # Skip mtp.* keys before any GPU buffer allocation. The raw HF keys use
        # the "mtp." prefix; none are accessed during inference.
        # vLLM's own NemotronHForCausalLM skips these the same way.
        skip_prefixes = frozenset({"mtp."})
        super().load_weights(path, f32_keys=f32_keys, skip_prefixes=skip_prefixes,
                             scale_transforms=self._scale_transforms)
        _sp = tuple(skip_prefixes)
        _missing_transforms = [
            k for k in self._weight_transforms
            if k not in self.weights
            and not k.startswith(_sp)
        ]
        if _missing_transforms:
            raise AssertionError(
                f"Registered weight transforms not consumed: {_missing_transforms}. "
                f"Either the checkpoint key format changed, or the key falls under a "
                f"skipped prefix ({skip_prefixes}) and the transform registration needs "
                f"to be guarded accordingly."
            )
        self.weights = _NemotronHForCausalLM.hf_to_vllm_mapper.apply_dict(self.weights)
        # self.weight_meta was populated by super().load_weights() with HF-prefixed
        # keys (e.g. 'backbone.layers.0.mixer.in_proj'). Remap those keys to match
        # the vLLM key space that the rest of the model uses ('model.*'), then drop
        # entries whose weight was filtered out before loading (e.g. mtp.*).
        if self.weight_meta:
            self.weight_meta = _NemotronHForCausalLM.hf_to_vllm_mapper.apply_dict(self.weight_meta)
            self.weight_meta = {
                k: v for k, v in self.weight_meta.items()
                if (k + ".weight") in self.weights
            }
        self._pack_attn_weights()
        # Release CPU-side scale accumulators and closures; they are only needed
        # during load_weights and are never accessed after _pack_attn_weights returns.
        self._scale_acc.clear()
        self._scale_transforms.clear()
        self._validate_mamba_weights()
        self._init_mamba_states(num_spec)
        # Cache all per-layer pre-norm weight buffers. They are accessed on every
        # forward call (decode, prefill, and replay_prefix_for_ssm); caching
        # avoids repeated dict lookups per step.
        self._layer_norm_weights = [
            self.weights[f"model.layers.{i}.norm.weight"] for i in range(self.num_layers)
        ]
        self._layer0_norm_w = self._layer_norm_weights[0]
        n_mamba = self._layer_types.count("mamba")
        n_attn  = self._layer_types.count("attention")
        logger.info(
            "NemotronH: loaded %d weight tensors (%d Mamba layers, %d attn layers)",
            len(self.weights), n_mamba, n_attn,
        )

    def _pack_attn_weights(self) -> None:
        """Fuse separate q/k/v projection weights into a single qkv_proj buffer.

        HF NemotronH checkpoints store three tensors per attention layer:
          {p}.q_proj.weight, {p}.k_proj.weight, {p}.v_proj.weight

        The forward pass dispatches a single fused matmul against
        {p}.qkv_proj.weight, so the three row-major matrices are concatenated
        along axis 0 (i.e., their flat byte arrays are concatenated in order
        Q, K, V).  The originals are deleted after packing.
        """
        import wgpu as _wgpu_lib

        dev = self.wgpu_device.wgpu_device

        for i, lt in enumerate(self._layer_types):
            if lt != "attention":
                continue
            p = f"model.layers.{i}.mixer"
            q_key = f"{p}.q_proj.weight"
            k_key = f"{p}.k_proj.weight"
            v_key = f"{p}.v_proj.weight"

            if q_key not in self.weights:
                # Already packed or checkpoint uses a different layout.
                continue

            q_nb = self.weights[q_key].nbytes
            k_nb = self.weights[k_key].nbytes
            v_nb = self.weights[v_key].nbytes
            total_nb = q_nb + k_nb + v_nb
            q_s = f"{q_key}.scales"
            k_s = f"{k_key}.scales"
            v_s = f"{v_key}.scales"
            # Scales may either be in the CPU accumulator (GPTQ/AWQ: suppressed
            # individual GPU uploads) or uploaded as individual GPU buffers.
            _sc_acc = self._scale_acc.get(i, {})
            _have_cpu_scales = all(proj in _sc_acc for proj in ("q", "k", "v"))
            has_scales = (
                _have_cpu_scales
                or (q_s in self.weights and k_s in self.weights and v_s in self.weights)
            )

            # AWQ weights are stored K-major as [K, N//8]. GPU-side byte
            # concatenation of three such buffers produces column-block layout
            # (all K rows of Q, then all K rows of K, then V), but the shader
            # indexes weights[k_idx * (N_total//8) + n_group], which requires
            # row-interleaved [K, N_total//8]. With GQA (k_dim < q_dim) every
            # k and v output group reads the wrong nibble words.
            # GPTQ weights are [N, K//8] (N-major after the load-time transpose),
            # so byte concat along the flat sequence is equivalent to axis=0
            # concat and is correct without any special handling.
            _uq_q = self._uq_for_key(q_key)
            _uq_k = self._uq_for_key(k_key)
            _uq_v = self._uq_for_key(v_key)
            if len({_uq_q, _uq_k, _uq_v}) != 1:
                raise ValueError(
                    f"{p}: q/k/v projections have mixed quantization formats "
                    f"(q={_uq_q}, k={_uq_k}, v={_uq_v}). Cannot pack into a "
                    f"single fused QKV buffer safely."
                )
            _is_awq = _uq_q == 4  # awq_sym

            if _is_awq:
                # CPU-side axis=1 concat produces [K, N_total//8] so every row
                # interleaves q, k, v output groups in the order the shader expects.
                # Consolidate three GPU readbacks into one staged copy + single map_sync
                # to avoid two extra GPU round-trips per attention layer during model load.
                q_buf = self.weights[q_key]
                k_buf = self.weights[k_key]
                v_buf = self.weights[v_key]
                _staging_w = dev.create_buffer(
                    size=total_nb,
                    usage=_wgpu_lib.BufferUsage.COPY_DST | _wgpu_lib.BufferUsage.MAP_READ)
                _enc_w = dev.create_command_encoder()
                _enc_w.copy_buffer_to_buffer(q_buf.buf, 0, _staging_w, 0, q_nb)
                _enc_w.copy_buffer_to_buffer(k_buf.buf, 0, _staging_w, q_nb, k_nb)
                _enc_w.copy_buffer_to_buffer(v_buf.buf, 0, _staging_w, q_nb + k_nb, v_nb)
                dev.queue.submit([_enc_w.finish()])
                _staging_w.map_sync(mode=_wgpu_lib.MapMode.READ)
                _raw_w = bytes(_staging_w.read_mapped())
                _staging_w.unmap()
                q_w = np.frombuffer(_raw_w[:q_nb], dtype=np.int32).reshape(q_buf.shape)
                k_w = np.frombuffer(_raw_w[q_nb:q_nb + k_nb], dtype=np.int32).reshape(k_buf.shape)
                v_w = np.frombuffer(_raw_w[q_nb + k_nb:], dtype=np.int32).reshape(v_buf.shape)
                packed_w = np.concatenate([q_w, k_w, v_w], axis=1)
                qkv_raw_buf = WebGPUBuffer.from_numpy(dev, packed_w)
            else:
                # Non-AWQ (GPTQ, FP16, FP8, etc.): GPU-side byte concat is correct.
                # All weight buffers are 4-byte aligned from the loader.
                qkv_raw_buf = self._make_buf(total_nb)
                _enc = dev.create_command_encoder()
                _enc.copy_buffer_to_buffer(self.weights[q_key].buf, 0, qkv_raw_buf.buf, 0, q_nb)
                _enc.copy_buffer_to_buffer(self.weights[k_key].buf, 0, qkv_raw_buf.buf, q_nb, k_nb)
                _enc.copy_buffer_to_buffer(self.weights[v_key].buf, 0, qkv_raw_buf.buf, q_nb + k_nb, v_nb)
                dev.queue.submit([_enc.finish()])

            if has_scales:
                # Scales have shape [G, N] (G = K // group_size, N = output neurons).
                # GPU-side byte concatenation would produce column-block order:
                # [q_group_0..G | k_group_0..G | v_group_0..G], but the shader reads
                # scales[grp * N_total + row], which requires row-interleaved [G, N_total].
                # For G > 1 (e.g. K=4096, group_size=128 -> G=32) every grp > 0 lookup
                # would land in the wrong projection's data. Stack on axis=1 on the CPU.
                if _have_cpu_scales:
                    # CPU path: scales were accumulated by _scale_transforms before GPU
                    # upload. Stack directly without any GPU round-trip.
                    q_sc = _sc_acc["q"]
                    k_sc = _sc_acc["k"]
                    v_sc = _sc_acc["v"]
                else:
                    # GPU path: scales were uploaded individually (non-GPTQ/AWQ formats
                    # or checkpoints that don't match _scale_transforms keys). Read back
                    # via a single staged command encoder to avoid three separate stalls.
                    q_sb = self.weights[q_s]
                    k_sb = self.weights[k_s]
                    v_sb = self.weights[v_s]
                    # Scale buffers are always uploaded as f32 by the weight loader.
                    # Assert before frombuffer to surface format mismatches immediately
                    # rather than silently producing wrong scale values.
                    for _sb_name, _sb in ((q_s, q_sb), (k_s, k_sb), (v_s, v_sb)):
                        if _sb.dtype != "f32":
                            raise ValueError(
                                f"Expected f32 scales for fused QKV, got {_sb.dtype!r} "
                                f"for {_sb_name!r}. Update the frombuffer dtype below."
                            )
                    _q_snb, _k_snb, _v_snb = q_sb.nbytes, k_sb.nbytes, v_sb.nbytes
                    _staging_s = dev.create_buffer(
                        size=_q_snb + _k_snb + _v_snb,
                        usage=_wgpu_lib.BufferUsage.COPY_DST | _wgpu_lib.BufferUsage.MAP_READ)
                    _enc_s = dev.create_command_encoder()
                    _enc_s.copy_buffer_to_buffer(q_sb.buf, 0, _staging_s, 0, _q_snb)
                    _enc_s.copy_buffer_to_buffer(k_sb.buf, 0, _staging_s, _q_snb, _k_snb)
                    _enc_s.copy_buffer_to_buffer(v_sb.buf, 0, _staging_s, _q_snb + _k_snb, _v_snb)
                    dev.queue.submit([_enc_s.finish()])
                    _staging_s.map_sync(mode=_wgpu_lib.MapMode.READ)
                    _raw_s = bytes(_staging_s.read_mapped())
                    _staging_s.unmap()
                    q_sc = np.frombuffer(_raw_s[:_q_snb], dtype=np.float32).reshape(q_sb.shape)
                    k_sc = np.frombuffer(_raw_s[_q_snb:_q_snb + _k_snb], dtype=np.float32).reshape(k_sb.shape)
                    v_sc = np.frombuffer(_raw_s[_q_snb + _k_snb:], dtype=np.float32).reshape(v_sb.shape)
                if q_sc.ndim == 2:
                    # [G, N] layout: concatenate along N axis to get [G, N_total].
                    packed_sc = np.concatenate([q_sc, k_sc, v_sc], axis=1)
                else:
                    # 1D per-channel scales (G=1): axis=0 concat is correct since
                    # grp is always 0, so scales[0*N_total+row] == scales[row].
                    packed_sc = np.concatenate([q_sc, k_sc, v_sc])
                scales_buf = WebGPUBuffer.from_numpy(dev, packed_sc)

            qkv_key = f"{p}.qkv_proj.weight"
            if not _is_awq:
                # Preserve the source weight dtype so _uq_for_key resolves the
                # correct USE_QUANT for GPU-quantized formats (GPTQ i32, FP8/INT8
                # u8 with fmt tag). The shape is not set here; callers derive K
                # and N from model-config constants passed as shader overrides.
                src_dtype = self.weights[q_key].dtype
                qkv_raw_buf.dtype = src_dtype
            self.weights[qkv_key] = qkv_raw_buf

            # Propagate quant_meta from q_proj to qkv_proj so _uq_for_key
            # and _quant_extra find the correct fmt / group_size / global_scale.
            # self.weight_meta (populated and remapped in load_weights) is the
            # authoritative store; __quant_meta__ no longer lives in self.weights.
            qmeta = self.weight_meta
            if qmeta:
                q_base = q_key.removesuffix(".weight")
                k_base = k_key.removesuffix(".weight")
                v_base = v_key.removesuffix(".weight")
                qkv_base = f"{p}.qkv_proj"
                if q_base in qmeta:
                    q_meta_entry = qmeta[q_base]
                    # Verify that k and v share the same metadata as q before
                    # fusing. For FP8 per-tensor quantization each projection
                    # carries an independently calibrated global_scale; copying
                    # only q's entry would silently dequantize k and v rows with
                    # the wrong scale, corrupting every attention layer.
                    for proj_base, proj_label in (
                        (k_base, "k_proj"), (v_base, "v_proj")
                    ):
                        if proj_base in qmeta:
                            proj_entry = qmeta[proj_base]
                            mismatched = {
                                field: (q_meta_entry.get(field), proj_entry.get(field))
                                for field in set(q_meta_entry) | set(proj_entry)
                                if q_meta_entry.get(field) != proj_entry.get(field)
                            }
                            if mismatched:
                                raise ValueError(
                                    f"{p}: q_proj and {proj_label} have mismatched "
                                    f"quant_meta ({mismatched!r}). Fusing them into "
                                    f"a single qkv_proj dispatch would dequantize "
                                    f"{proj_label} rows with q_proj's scale. "
                                    f"Per-tensor FP8 with differing scales is not "
                                    f"supported for fused qkv dispatch."
                                )
                    qmeta[qkv_base] = dict(q_meta_entry)
                    # Remove stale entries for q/k/v_proj; those weight tensors no
                    # longer exist after packing into qkv_proj. Only clean up when
                    # the propagation to qkv_base actually occurred to avoid silently
                    # dropping metadata when q_base is absent (partial quantization).
                    qmeta.pop(k_base, None)
                    qmeta.pop(v_base, None)
                    qmeta.pop(q_base, None)
                else:
                    # q_proj metadata absent; fall back to k or v if available so
                    # _quant_extra gets a valid global_scale / group_size for the
                    # fused qkv_proj dispatch.
                    fallback = qmeta.get(k_base) or qmeta.get(v_base)
                    if fallback is not None:
                        qmeta[qkv_base] = dict(fallback)
                    qmeta.pop(k_base, None)
                    qmeta.pop(v_base, None)

            # Register the packed scales buffer created above (if present).
            if has_scales:
                self.weights[f"{qkv_key}.scales"] = scales_buf
            # Unconditionally remove any individual scale buffers that may remain
            # (handles partial-scale checkpoints where not all three are present).
            for s in (q_s, k_s, v_s):
                self.weights.pop(s, None)

            del self.weights[q_key], self.weights[k_key], self.weights[v_key]

    def _validate_mamba_weights(self) -> None:
        """Validate weights for all Mamba layers.

        Checks:
        1. in_proj.weight shape[0] against in_proj_dim (all Mamba layers).
           Catches a silent formula mismatch if vLLM changes MambaMixer2Tp's
           conv_dim or in_proj output_sizes, which would mis-size scratch buffers.
        2. conv1d.weight element count (shape may be [conv_dim, 1, kernel] or
           [conv_dim, kernel]; both are row-major identical).
        3. Presence and f32 dtype for A, D, and dt_bias. The mamba2_ssm_step
           shader binds all three as array<f32>; a missing key or f16 upload
           produces silent garbage with no GPU-side error.

        The A_log to -exp(A) transform is a CPU-side weight_transform applied
        before GPU upload, so no GPU round-trip is needed here.
        """
        for i, lt in enumerate(self._layer_types):
            if lt != "mamba":
                continue
            p = f"model.layers.{i}.mixer"

            # in_proj.weight: machine-check in_proj_dim formula against the
            # actual checkpoint for every Mamba layer. The formula mirrors
            # MambaMixer2.__init__ in mamba_mixer2.py:
            #   L313: self.conv_dim = intermediate_size + 2*groups_ssm_state_size
            #   MergedColumnParallelLinear branch (L328-339, n_groups%tp==0):
            #     output_sizes = [intermediate_size, intermediate_size,
            #                     groups_ssm_state_size, groups_ssm_state_size,
            #                     num_heads]
            #   ColumnParallelLinear branch (L353-358, n_groups%tp!=0):
            #     output_size = intermediate_size + self.conv_dim + self.num_heads
            # Both branches sum to 2*intermediate_size + 2*groups_ssm_state_size
            # + num_heads, mapped here as mamba_int + conv_dim + mamba_num_heads.
            # A vLLM bump that changes conv_dim or adds an extra output group
            # would silently mis-size mamba_inproj / mamba_conv_in / mamba_dt.
            # Checked for every Mamba layer to catch puzzle-model heterogeneity.
            inproj_key = f"{p}.in_proj.weight"
            if inproj_key not in self.weights:
                raise ValueError(f"{inproj_key} missing from loaded weights")
            # AWQ stores in_proj.weight as [K, N//8]; all other formats
            # (F16, GPTQ, FP8, INT8, NF4, NVFP4) store shape[0] == N.
            if self._uq_for_key(inproj_key) == 4:  # AWQ: shape[0]=K, shape[1]=N//8
                actual = self.weights[inproj_key].shape[1] * 8
                if actual != self.in_proj_dim:
                    raise ValueError(
                        f"{inproj_key} AWQ shape[1]*8={actual} does not "
                        f"match computed in_proj_dim={self.in_proj_dim} "
                        f"(mamba_int={self.mamba_int} + "
                        f"conv_dim={self.conv_dim} + "
                        f"mamba_num_heads={self.mamba_num_heads}). "
                        f"Recheck MambaMixer2Tp output_sizes in "
                        f"mamba_mixer2.py L328-355 against this vLLM version."
                    )
            else:
                actual_inproj_dim = self.weights[inproj_key].shape[0]
                if actual_inproj_dim != self.in_proj_dim:
                    raise ValueError(
                        f"{inproj_key} shape[0]={actual_inproj_dim} does not "
                        f"match computed in_proj_dim={self.in_proj_dim} "
                        f"(mamba_int={self.mamba_int} + "
                        f"conv_dim={self.conv_dim} + "
                        f"mamba_num_heads={self.mamba_num_heads}). "
                        f"Recheck MambaMixer2Tp output_sizes in "
                        f"mamba_mixer2.py L328-355 against this vLLM version."
                    )

            # conv1d.weight: validate element count.
            # Shape may be [conv_dim, 1, kernel] or [conv_dim, kernel]; elements
            # are in the same row-major order in both cases, so no GPU roundtrip needed.
            cw_key = f"{p}.conv1d.weight"
            if cw_key not in self.weights:
                raise ValueError(f"{cw_key} missing from loaded weights")
            expected = self.conv_dim * self.conv_kernel
            actual = math.prod(self.weights[cw_key].shape)
            if actual != expected:
                raise ValueError(
                    f"conv1d.weight layer {i}: got {actual} elements, expected {expected}"
                )

            # SSM parameters: verify presence and f32 dtype.
            # The mamba2_ssm_step shader binds A, D, and dt_bias as array<f32>;
            # a missing key or f16 upload would produce silent garbage.
            for wk in ("A", "D", "dt_bias"):
                key = f"{p}.{wk}"
                if key not in self.weights:
                    raise ValueError(f"{key} missing from loaded weights")
                if self.weights[key].dtype != "f32":
                    raise ValueError(
                        f"{key} must be f32 (shader reads array<f32>), got {self.weights[key].dtype}"
                    )

            # norm.weight and out_proj.weight: used in _mamba_layer steps 4 and 5.
            # A corrupt or incomplete checkpoint missing either would silently pass the
            # checks above and then raise a bare KeyError on the first forward pass.
            for wk in ("norm.weight", "out_proj.weight"):
                key = f"{p}.{wk}"
                if key not in self.weights:
                    raise ValueError(f"{key} missing from loaded weights")

        # Global weights: every forward call hits these unconditionally.
        # A truncated checkpoint that passes all per-layer checks above can still
        # crash with a bare KeyError on the first token if any of these are absent.
        for key in ("model.norm_f.weight", "model.embed_tokens.weight"):
            if key not in self.weights:
                raise ValueError(f"{key} missing from loaded weights")
        lm_key = self._lm_head_key()
        if lm_key not in self.weights:
            raise ValueError(f"{lm_key} missing from loaded weights")

    # ── Forward pass ──────────────────────────────────────────────────────────

    def _finalize_output(self) -> np.ndarray:
        """Record logit buffer and return sampled token or full logits."""
        self._last_logit_buf = self._pre["logits"]
        self._last_vocab = self.vocab_size
        return self._finish_forward(self._greedy_decode)

    def _run_final_norm_and_lm_head(
        self, x_buf: "WebGPUBuffer", num_tokens: int
    ) -> None:
        """Dispatch final RMS norm, LM head matmul, and optional on-GPU argmax.

        Must be called inside a _batched_dispatch() context. num_tokens controls
        the rms_norm workgroup count (1 for decode and per-token prefill steps).
        """
        pre = self._pre
        hidden = self.hidden_size
        vocab = self.vocab_size
        self._dispatch(
            "rms_norm",
            [x_buf, self.weights["model.norm_f.weight"], pre["norm_out"]],
            self._rms_base,
            (num_tokens, 1, 1),
        )
        lm_key = self._lm_head_key()
        lm_head_w = self.weights[lm_key]
        uq = self._uq_for_key(lm_key)
        # Always use SPLIT_K=0 (row-per-thread, ceil(vocab/256) WGs) for the LM
        # head: SPLIT_K=1 dispatches (vocab, 1, 1) WGs which exceeds the 65535
        # per-dimension WebGPU limit for large vocabularies. SPLIT_K=0 supports
        # USE_QUANT values 0 (f16), 3 (gptq_sym), 4 (awq_sym), 5 (fp8_gpu),
        # 6 (nvfp4_gpu), 7 (int8_gpu), 8 (nf4_gpu) — the full set that
        # _uq_for_key() can return.
        self._dispatch(
            "matmul_quant",
            [pre["norm_out"], lm_head_w,
             self._scales_buf(lm_key, uq, self._dummy_buf),
             pre["logits"]],
            {"K": hidden, "N": vocab, "USE_QUANT": uq, "SPLIT_K": 0,
             **self._quant_extra(lm_key.removesuffix(".weight"), uq)},
            _rows_wg(vocab),
        )
        if self._greedy_decode:
            self._dispatch(
                "argmax_f16",
                [pre["logits"], self._ensure_sample_buf()],
                {"N": vocab},
                (1, 1, 1),
            )
            self._copy_sample_to_staging()

    def forward(
        self,
        input_ids: np.ndarray,
        positions: np.ndarray,
        attn_metadata: object,
    ) -> np.ndarray:
        num_tokens = len(input_ids)
        self._hstate = 0

        self._check_single_sequence(attn_metadata)

        if num_tokens > 1:
            return self._prefill_forward(
                input_ids, positions, attn_metadata,
                num_tokens,
            )

        # Decode path (T=1): zero-alloc hot path via pre-allocated buffers.
        dev = self.wgpu_device.wgpu_device
        hidden = self.hidden_size
        ctx_len = int(attn_metadata.max_decode_seq_len)

        pre = self._pre
        sc = self._sc
        dev.queue.write_buffer(pre["ids"].buf, 0, input_ids.astype(np.uint32, copy=False).tobytes())
        dev.queue.write_buffer(
            pre["slot_map"].buf, 0,
            np.asarray(attn_metadata.slot_mapping, dtype=np.uint32).tobytes(),
        )
        bt_arr = self._bt_arr(attn_metadata)
        dev.queue.write_buffer(pre["bt"].buf, 0, bt_arr.tobytes())

        with self._batched_dispatch():
            # Embedding lookup.
            self._dispatch(
                "embedding_lookup",
                [self.weights["model.embed_tokens.weight"], pre["ids"], pre["x"]],
                {"HIDDEN_DIM": hidden},
                (num_tokens, 1, 1),
            )

            # Layer 0 pre-norm (initial case: no residual add yet).
            self._dispatch(
                "rms_norm",
                [pre["x"],
                 self._norm0_w,
                 sc["normed"]],
                self._rms_base,
                (num_tokens, 1, 1),
            )

            normed_x = sc["normed"]
            x_buf    = pre["x"]  # initial residual = embedding

            # normed_x is stale (points to sc["normed"] from the last iteration,
            # which _layer_dispatch marks as unused for the final layer). Only
            # x_buf is used after the loop.
            for i in range(self.num_layers):
                normed_x, x_buf = self._layer_dispatch(
                    i, normed_x, x_buf,
                    pre["slot_map"], pre["bt"],
                    ctx_len, num_tokens,
                )

            # Final norm, LM head, and optional argmax.
            self._run_final_norm_and_lm_head(x_buf, num_tokens)

        return self._finalize_output()

    def _layer_dispatch(
        self,
        layer_idx: int,
        normed_x: "WebGPUBuffer",
        x_buf: "WebGPUBuffer",
        slot_map: "WebGPUBuffer",
        bt_buf: "WebGPUBuffer",
        ctx_len: int,
        num_tokens: int,
    ) -> "tuple[WebGPUBuffer | None, WebGPUBuffer]":
        """Dispatch one Nemotron-H layer (Mamba, Attention, or MLP).

        The layer:
          1. Runs the mixer on the pre-normed input (normed_x).
          2. Fuses residual-add with the next layer's pre-norm (or plain add
             for the final layer so the caller can apply norm_f).

        Returns (normed_for_next_layer, raw_accumulated_residual).
        """
        sc = self._sc
        lt = self._layer_types[layer_idx]
        out = sc[_H_NAMES[(self._hstate + 2) % 3]]
        with self._batched_dispatch(label=f"L{layer_idx:02d}"):
            # _active_encoder is guaranteed non-None by the outer _batched_dispatch() context.
            if lt == "mamba":
                self._mamba_layer(layer_idx, normed_x)
            elif lt == "attention":
                self._attn_layer(
                    layer_idx, normed_x, slot_map, bt_buf, ctx_len, num_tokens
                )
            elif lt == "mlp":
                self._mlp_layer(layer_idx, normed_x)
            else:
                raise NotImplementedError(
                    f"Layer type {lt!r} at index {layer_idx} not implemented"
                )

            # sc["mixer_out"] now holds the mixer result.
            mixer_out = sc["mixer_out"]

            # Fuse add + pre-norm for the next layer (saves one dispatch per layer).
            # For the last layer: plain add; norm_f applied in forward() after the loop.
            if layer_idx < self.num_layers - 1:
                # _layer_norm_weights is populated by load_weights; fall back to
                # dict lookup for test paths that inject weights directly.
                if self._layer_norm_weights:
                    next_norm_w = self._layer_norm_weights[layer_idx + 1]
                else:
                    next_norm_w = self.weights[f"model.layers.{layer_idx + 1}.norm.weight"]
                self._dispatch(
                    "add_rms_norm",
                    [x_buf, mixer_out, next_norm_w, out, sc["normed"]],
                    self._rms_base,
                    (num_tokens, 1, 1),
                )
                normed_out = sc["normed"]
            else:
                self._dispatch(
                    "add",
                    [x_buf, mixer_out, out],
                    {"N": self.hidden_size},
                    _vec4_wg(self.hidden_size),
                )
                normed_out = None  # stale after last layer; norm_f applied in forward()

        self._hstate = (self._hstate + 2) % 3
        return normed_out, out

    # ── Mixer implementations ─────────────────────────────────────────────────

    def _mamba_layer(
        self,
        layer_idx: int,
        normed_x: "WebGPUBuffer",
    ) -> None:
        """Mamba-2 SSM layer.

        Pipeline: in_proj -> [GPU-side extract] -> causal_conv -> SSM step
                  -> grouped gated RMSNorm -> out_proj.
        Result goes to sc["mixer_out"].
        """
        sc  = self._sc
        p   = f"model.layers.{layer_idx}.mixer"
        H   = self.hidden_size
        MI  = self.mamba_int
        CD  = self.conv_dim
        MNH = self.mamba_num_heads
        MHD = self.mamba_head_dim
        NS  = self.ssm_state_size
        NG  = self.n_groups

        # Step 1: in_proj — hidden -> [gate | x_B_C | dt]
        in_w = f"{p}.in_proj.weight"
        uq   = self._uq_for_key(in_w)
        self._dispatch(
            "matmul_quant",
            [normed_x, self.weights[in_w],
             self._scales_buf(in_w, uq, self._dummy_buf), sc["mamba_inproj"]],
            {"K": H, "N": self.in_proj_dim, "USE_QUANT": uq,
             **self._quant_extra(f"{p}.in_proj", uq)},
            (self.in_proj_dim, 1, 1),
        )

        # GPU-side byte copies to extract the three portions of in_proj output.
        # gate:   bytes [0          .. MI*2)        -> sc["mamba_gate"]
        # x_B_C:  bytes [MI*2      .. (MI+CD)*2)   -> sc["mamba_conv_in"]
        # dt:     bytes [(MI+CD)*2 .. (MI+CD+MNH)*2) -> sc["mamba_dt"]
        enc = self._active_encoder
        enc.copy_buffer_to_buffer(
            sc["mamba_inproj"].buf, 0,
            sc["mamba_gate"].buf,   0,
            MI * 2,
        )
        enc.copy_buffer_to_buffer(
            sc["mamba_inproj"].buf, MI * 2,
            sc["mamba_conv_in"].buf, 0,
            CD * 2,
        )
        enc.copy_buffer_to_buffer(
            sc["mamba_inproj"].buf, (MI + CD) * 2,
            sc["mamba_dt"].buf,      0,
            MNH * 2,
        )

        # Step 2: Causal conv1d on x_B_C with SiLU activation.
        conv_w = f"{p}.conv1d.weight"
        conv_b = f"{p}.conv1d.bias"
        has_bias = int(conv_b in self.weights)
        bias_buf = self.weights.get(conv_b, self._dummy_buf)  # dummy when absent
        self._dispatch(
            "mamba2_causal_conv",
            [sc["mamba_conv_in"], self.weights[conv_w], bias_buf,
             self._conv_states[layer_idx], sc["mamba_conv_out"]],
            {"CONV_DIM": CD, "KERNEL": self.conv_kernel,
             "WG_SIZE": 256, "HAS_BIAS": has_bias},
            _rows_wg(CD),
        )

        # Step 3: Mamba-2 SSM state update.
        # x_B_C layout after conv: [x(MI) | B(N_GROUPS*NS) | C(N_GROUPS*NS)]
        self._dispatch(
            "mamba2_ssm_step",
            [sc["mamba_conv_out"], sc["mamba_dt"],
             self.weights[f"{p}.A"], self.weights[f"{p}.dt_bias"],
             self.weights[f"{p}.D"],
             self._ssm_states[layer_idx], sc["mamba_ssm_y"]],
            {"NUM_HEADS": MNH, "HEAD_DIM": MHD,
             "STATE_SIZE": NS, "N_GROUPS": NG, "WG_SIZE": 256},
            (MNH, 1, 1),
        )

        # Step 4: Grouped gated RMSNorm.
        self._dispatch(
            "mamba2_norm_gate",
            [sc["mamba_ssm_y"], sc["mamba_gate"],
             self.weights[f"{p}.norm.weight"], sc["mamba_norm_out"]],
            {"MAMBA_INT": MI, "N_GROUPS": NG, "WG_SIZE": 256},
            (NG, 1, 1),
        )

        # Step 5: out_proj — mamba_int -> hidden.
        out_w = f"{p}.out_proj.weight"
        uq2   = self._uq_for_key(out_w)
        self._dispatch(
            "matmul_quant",
            [sc["mamba_norm_out"], self.weights[out_w],
             self._scales_buf(out_w, uq2, self._dummy_buf), sc["mixer_out"]],
            {"K": MI, "N": H, "USE_QUANT": uq2,
             **self._quant_extra(f"{p}.out_proj", uq2)},
            (H, 1, 1),
        )

    def _attn_layer(
        self,
        layer_idx: int,
        normed_x: "WebGPUBuffer",
        slot_map: "WebGPUBuffer",
        bt_buf: "WebGPUBuffer",
        ctx_len: int,
        num_tokens: int,
    ) -> None:
        """Full-attention layer (no per-head norm in NemotronH).

        Pipeline: qkv_proj -> [extract Q/K/V] -> KV cache -> flash_attn -> o_proj.
        Result goes to sc["mixer_out"].
        """
        sc    = self._sc
        p     = f"model.layers.{layer_idx}.mixer"
        H     = self.hidden_size
        q_dim = self._q_dim
        k_dim = self._k_dim

        # Fused QKV projection.
        qkv_w    = f"{p}.qkv_proj.weight"
        total_qkv = q_dim + 2 * k_dim
        uq = self._uq_for_key(qkv_w)
        self._dispatch(
            "matmul_quant",
            [normed_x, self.weights[qkv_w],
             self._scales_buf(qkv_w, uq, self._dummy_buf), sc["qkv_buf"]],
            {"K": H, "N": total_qkv, "USE_QUANT": uq,
             **self._quant_extra(f"{p}.qkv_proj", uq)},
            (total_qkv, 1, 1),
        )

        # GPU-side extraction: split QKV buffer into Q, K, V.
        enc = self._active_encoder
        enc.copy_buffer_to_buffer(
            sc["qkv_buf"].buf, 0,           sc["q_buf"].buf, 0, q_dim * 2)
        enc.copy_buffer_to_buffer(
            sc["qkv_buf"].buf, q_dim * 2,   sc["k_buf"].buf, 0, k_dim * 2)
        enc.copy_buffer_to_buffer(
            sc["qkv_buf"].buf, (q_dim + k_dim) * 2,
            sc["v_buf"].buf, 0, k_dim * 2)

        # No RoPE: NemotronH uses no rotary position embeddings.

        # Fused KV cache store. Skipped during SSM prefix replay because the
        # cache is already correctly populated from the original prefill.
        k_cache, v_cache = self.kv_pool[layer_idx]
        if not self._replay_mode:
            self._dispatch(
                "kv_cache_store_both",
                [sc["k_buf"], k_cache, sc["v_buf"], v_cache, slot_map],
                {"BLOCK_SIZE": self.block_size, "NUM_KV_HEADS": self.num_kv_heads,
                 "HEAD_DIM": self.head_dim, "V_IN_OFFSET": 0},
                (num_tokens, self.num_kv_heads, 1),
            )

        # Flash attention decode.
        self._dispatch(
            "flash_attn_decode",
            [sc["q_buf"], k_cache, v_cache, bt_buf, sc["attn_out"]],
            {"BLOCK_SIZE": self.block_size, "NUM_Q_HEADS": self.num_q_heads,
             "NUM_KV_HEADS": self.num_kv_heads, "HEAD_DIM": self.head_dim,
             "CTX_LEN": ctx_len, "SCALE": self._attn_scale, "START_BLOCK": 0},
            (self.num_q_heads, 1, 1),
        )

        # Output projection.
        ow  = f"{p}.o_proj.weight"
        uq2 = self._uq_for_key(ow)
        self._dispatch(
            "matmul_quant",
            [sc["attn_out"], self.weights[ow],
             self._scales_buf(ow, uq2, self._dummy_buf), sc["mixer_out"]],
            {"K": q_dim, "N": H, "USE_QUANT": uq2,
             **self._quant_extra(f"{p}.o_proj", uq2)},
            (H, 1, 1),
        )

    def _mlp_layer(
        self,
        layer_idx: int,
        normed_x: "WebGPUBuffer",
    ) -> None:
        """MLP-only layer with squared-ReLU activation (no gate projection).

        Pipeline: up_proj -> relu^2 -> down_proj.
        Result goes to sc["mixer_out"].
        """
        sc  = self._sc
        p   = f"model.layers.{layer_idx}.mixer"
        H   = self.hidden_size
        I   = self._layer_int_size[layer_idx]

        # up_proj: hidden -> intermediate
        uw  = f"{p}.up_proj.weight"
        uq  = self._uq_for_key(uw)
        self._dispatch(
            "matmul_quant",
            [normed_x, self.weights[uw],
             self._scales_buf(uw, uq, self._dummy_buf), sc["up_buf"]],
            {"K": H, "N": I, "USE_QUANT": uq,
             **self._quant_extra(f"{p}.up_proj", uq)},
            (I, 1, 1),
        )

        # relu^2 element-wise activation. WG_SIZE=256 matches _rows_wg's divisor,
        # so the workgroup count is identical; this shader is element-wise, not
        # row-parallel, but the formula (cdiv(N, 256), 1, 1) is the same.
        self._dispatch(
            "relu_sq",
            [sc["up_buf"], sc["ffn_act"]],
            {"N": I, "WG_SIZE": 256},
            _rows_wg(I),
        )

        # down_proj: intermediate -> hidden
        dw  = f"{p}.down_proj.weight"
        uq2 = self._uq_for_key(dw)
        self._dispatch(
            "matmul_quant",
            [sc["ffn_act"], self.weights[dw],
             self._scales_buf(dw, uq2, self._dummy_buf), sc["mixer_out"]],
            {"K": I, "N": H, "USE_QUANT": uq2,
             **self._quant_extra(f"{p}.down_proj", uq2)},
            (H, 1, 1),
        )

    # ── Prefill fallback ──────────────────────────────────────────────────────

    def _prefill_forward(
        self,
        input_ids: np.ndarray,
        positions: np.ndarray,
        attn_metadata: object,
        T: int,
    ) -> np.ndarray:
        """Process T prompt tokens one at a time through the decode path.

        Each token is processed sequentially so the Mamba conv/SSM states
        accumulate correctly. KV cache is filled token-by-token for causal
        attention. Only the last token's logits are returned.
        """
        dev = self.wgpu_device.wgpu_device
        pre = self._pre
        sc  = self._sc
        bt_arr = self._bt_arr(attn_metadata)
        dev.queue.write_buffer(pre["bt"].buf, 0, bt_arr.tobytes())

        for t in range(T):
            self._hstate = 0
            tok_ctx = int(positions[t]) + 1

            dev.queue.write_buffer(
                pre["ids"].buf, 0, input_ids[t:t+1].astype(np.uint32, copy=False).tobytes())
            dev.queue.write_buffer(
                pre["slot_map"].buf, 0,
                np.asarray(attn_metadata.slot_mapping[t:t+1], dtype=np.uint32).tobytes())

            with self._batched_dispatch():
                self._dispatch(
                    "embedding_lookup",
                    [self.weights["model.embed_tokens.weight"], pre["ids"], pre["x"]],
                    {"HIDDEN_DIM": self.hidden_size},
                    (1, 1, 1),
                )
                self._dispatch(
                    "rms_norm",
                    [pre["x"],
                     self._norm0_w,
                     sc["normed"]],
                    self._rms_base,
                    (1, 1, 1),
                )

                normed_x = sc["normed"]
                x_buf    = pre["x"]

                # normed_x is stale on the final iteration (see _layer_dispatch);
                # only x_buf is used after the loop.
                for i in range(self.num_layers):
                    normed_x, x_buf = self._layer_dispatch(
                        i, normed_x, x_buf,
                        pre["slot_map"], pre["bt"],
                        tok_ctx, 1,
                    )

                if t == T - 1:
                    self._run_final_norm_and_lm_head(x_buf, 1)

        return self._finalize_output()

