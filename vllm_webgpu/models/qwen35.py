from __future__ import annotations
import math
from itertools import batched, chain
from typing import TYPE_CHECKING

import numpy as np

from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm_webgpu.models.base import _gemv_wg, _vec4_wg, _H_NAMES
from vllm_webgpu.models.mixtral import MixtralWebGPUModel
import vllm_webgpu.envs as _webgpu_envs

from vllm_webgpu.webgpu.buffer import WebGPUBuffer, _ELEM_BYTES

if TYPE_CHECKING:
    from vllm_webgpu.webgpu.device import WebGPUDevice
    from vllm_webgpu.webgpu.pipeline import PipelineCache

logger = init_logger(__name__)


# Qwen3.5 linear attention layer constants
_LIN_K_HEADS = 16
_LIN_V_HEADS = 32
_LIN_K_DIM = 128
_LIN_V_DIM = 128
_LIN_CONV_KERNEL = 4



class Qwen35WebGPUModel(MixtralWebGPUModel):
    """
    Qwen3.5-9B hybrid inference model — all compute on WebGPU.

    Full-attention layers (every 4th, indices 3/7/11/.../31): standard GQA
    using existing WebGPU kernels (same as LlamaWebGPUModel).

    Linear-attention layers (all others): GDN (Gated Delta Networks) entirely
    on WebGPU using custom WGSL kernels:
      - causal_conv_step.wgsl: single-step causal depthwise convolution
      - gdn_state_update.wgsl: delta-rule SSM state update
      - linear_attn_norm_gate.wgsl: per-head RMSNorm + SiLU gate (z * sigmoid(z))

    Persistent recurrent state (SSM matrix + conv history) lives in GPU buffers
    that are updated in-place each decode step. No CPU↔GPU transfers in the
    hot path.
    """

    def __init__(self, model_config, wgpu_device: "WebGPUDevice", pipeline_cache: "PipelineCache", block_size: int = 16) -> None:
        # Set GDN + MoE + other Qwen3.5-specific attributes BEFORE calling
        # super().__init__(). LlamaWebGPUModel.__init__() calls
        # self._init_scratch_buffers() via Python's dynamic dispatch, which
        # resolves to Qwen35's override. That override uses these values, so they
        # must exist before the super() call returns.

        self._layer_types: list = getattr(model_config, "layer_types", None)
        assert self._layer_types is not None, (
            "Qwen3_5TextConfig.layer_types is None; expected it to be populated "
            "from full_attention_interval at config init time."
        )
        # Partial RoPE: some models only rotate a fraction of head dimensions.
        # partial_rotary_factor=0.25 → rotary_dim = head_dim * 0.25.
        _prf = getattr(model_config, "partial_rotary_factor", 1.0)
        # Read head_dim from model_config directly — self.head_dim not set yet.
        _head_dim_raw = getattr(model_config, "head_dim",
                                model_config.hidden_size // model_config.num_attention_heads)
        self._rotary_dim: int = max(2, int(_head_dim_raw * _prf) // 2 * 2)
        # Interleaved RoPE: pairs (2i, 2i+1) vs standard (i, i+half).
        # Qwen3.5 uses mrope_interleaved=True, stored in rope_parameters dict,
        # not as a top-level config attribute.
        _rope_params = getattr(model_config, "rope_parameters", {}) or {}
        self._rope_interleaved: int = 1 if _rope_params.get("mrope_interleaved", False) else 0
        # Attention output gate: when True, q_proj.weight has shape [2*q_dim, hidden].
        # The first half is Q; the second half is a gate applied as sigmoid(gate)*attn_out
        # before the o_proj. The split is performed at load time by the _make_split weight
        # transform registered in load_weights; the gate half is stored under
        # self_attn.q_gate_proj.weight.
        self._attn_output_gate: bool = getattr(model_config, "attn_output_gate", True)

        # GDN (linear-attention) architecture dimensions from config.
        # Fall back to Qwen3.5-9B defaults if not present.
        self._lin_k_heads: int = getattr(model_config, "linear_num_key_heads", _LIN_K_HEADS)
        self._lin_k_dim: int   = getattr(model_config, "linear_key_head_dim",  _LIN_K_DIM)
        self._lin_v_heads: int = getattr(model_config, "linear_num_value_heads", _LIN_V_HEADS)
        self._lin_v_dim: int   = getattr(model_config, "linear_value_head_dim", _LIN_V_DIM)
        self._lin_conv_kernel: int = getattr(model_config, "linear_conv_kernel_dim", _LIN_CONV_KERNEL)
        # Total QKV packed dimension: Q + K + V. Q_heads == K_heads for GDN, so Q_dim == K_dim.
        self._lin_val_dim: int  = self._lin_v_heads * self._lin_v_dim   # total value dim
        _lin_key_dim: int       = self._lin_k_heads * self._lin_k_dim   # total key dim (= Q dim)
        # GDN QKV buffer offsets (f16 elements); constant across all layers and tokens.
        # Q is always at offset 0 (leading element in packed QKV buffer).
        self._gdn_k_base: int = _lin_key_dim
        self._gdn_v_base: int = 2 * _lin_key_dim
        # Derive conv_dim directly from MambaStateShapeCalculator so CONV_DIM always
        # matches the vLLM formula regardless of future changes to mamba_utils.py.
        # The shape calculator returns a 2-tuple; extract conv_dim from the axis that
        # holds it (axis order depends on VLLM_SSM_CONV_STATE_LAYOUT).
        self._gdn_conv_shape, self._gdn_ssm_shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_world_size=1,
            num_k_heads=self._lin_k_heads, num_v_heads=self._lin_v_heads,
            head_k_dim=self._lin_k_dim, head_v_dim=self._lin_v_dim,
            conv_kernel_size=self._lin_conv_kernel,
        )
        self._lin_conv_dim: int = self._gdn_conv_shape[0] if is_conv_state_dim_first() else self._gdn_conv_shape[1]

        # MoE config (Qwen3.6-35B-A3B and similar MoE variants).
        # When num_experts > 0 the FFN in every layer is a mixture-of-experts block;
        # the standard gate/up/down weights are replaced by a router + per-expert weights.
        self._moe_num_experts: int = getattr(model_config, "num_experts", 0)
        self._moe_k: int           = getattr(model_config, "num_experts_per_tok", 0)
        self._moe_inter: int = v if (v := getattr(model_config, "moe_intermediate_size", None)) is not None else model_config.intermediate_size
        self._moe_shared_inter: int = v if (v := getattr(model_config, "shared_expert_intermediate_size", None)) is not None else model_config.intermediate_size
        # _is_moe is set from config here and may be overridden in load_weights
        # once we can verify against actual weight keys.
        self._is_moe: bool = self._moe_num_experts > 0 and self._moe_k > 0
        # GEMMA_NORM=1 for Gemma safetensors (weights are deviations from 1, mean≈0.2).
        # GEMMA_NORM=0 for standard RMSNorm (weights absolute, mean≈1.0 — MLX format).
        # Determined from the model config: standard HuggingFace Qwen3.5 checkpoints
        # set rms_norm_type="gemma" to indicate the deviation format; MLX checkpoints
        # do not set this field and store absolute weights instead.
        _rms_norm_type = getattr(model_config, "rms_norm_type", None)
        self._gemma_norm: int = 1 if _rms_norm_type == "gemma" else 0

        # GDN_BF16: when set, GDN projection matmuls use bf16-preserved weight buffers
        # (key + "__bf16") instead of the default f16 version. Falls back silently if
        # the __bf16 buffer is absent (model not BF16 or flag off).
        self._gdn_bf16: bool = _webgpu_envs.GDN_BF16

        # linear_attn_norm_gate.wgsl hard-codes SiLU (z * sigmoid(z)). Reject
        # checkpoints that specify a different gate so the mismatch is caught at
        # load time rather than producing wrong outputs silently.
        _otype = getattr(model_config, "output_gate_type", "silu")
        if _otype not in ("silu", "swish"):
            raise NotImplementedError(
                f"Qwen35WebGPUModel requires output_gate_type silu/swish; "
                f"linear_attn_norm_gate.wgsl hard-codes SiLU, got {_otype!r}."
            )

        # Persistent GPU buffers for recurrent state (allocated after load_weights).
        # SSM state:  [NUM_V_HEADS, V_DIM, K_DIM] f32 = 2MB per linear-attn layer
        # Conv state: [CONV_KERNEL-1, CONV_DIM] f16 = 49KB per linear-attn layer
        self._ssm_gpu: list = []   # one WebGPUBuffer per layer (or None for full-attn)
        self._conv_gpu: list = []  # one WebGPUBuffer per layer

        # LlamaWebGPUModel.__init__() sets: num_layers, num_q_heads, num_kv_heads,
        # hidden_size, intermediate_size, vocab_size, head_dim, rope_theta, block_size,
        # _rope_consts (containing LN_ROPE_BASE), _rms_consts (without GEMMA_NORM), runs
        # dimension validation, then calls self._init_scratch_buffers() and
        # self._init_rope_freq_buf().
        super().__init__(model_config, wgpu_device, pipeline_cache, block_size=block_size)

        # Zero-buffer cache for reset_recurrent_states. Reuses the same bytearray
        # objects across resets to avoid allocating ~58 MB of Python heap per reset
        # (27 GDN layers × 2 MB SSM + 49 KB conv each on Qwen3.5-9B).
        # Matches the NemotronHWebGPUModel pattern.
        self._zero_buf_cache: dict[int, bytes] = {}

        # Seed _rms_consts with GEMMA_NORM so all add_rms_norm dispatches (including
        # GDN layers, which read _rms_consts directly) have the constant from the moment
        # the object is constructed. GEMMA_NORM is fixed from model_config.rms_norm_type
        # at construction time and not updated after load.
        self._rms_consts["GEMMA_NORM"] = self._gemma_norm

        # Seed _rope_base so _attn_block can read it before load_weights completes (e.g.
        # unit tests that call _attn_block directly). GEMMA_NORM is fixed from
        # model_config.rms_norm_type at construction time and not updated after load.
        self._rope_base = {
            **self._rope_consts,
            "GEMMA_NORM": self._gemma_norm,
            "ROTARY_DIM": self._rotary_dim,
            "INTERLEAVED": self._rope_interleaved,
        }

        # Mixtral.__init__ reads num_local_experts (0 for Qwen35) and overwrites _is_moe.
        # Re-assert the correct values from Qwen35-specific config fields.
        self._num_experts = self._moe_num_experts
        self._top_k = self._moe_k
        self._is_moe = self._moe_num_experts > 0 and self._moe_k > 0

        if self._is_moe and not hasattr(self, "_topk_idx_staging"):
            # MixtralWebGPUModel.__init__ only allocates staging buffers when it detects
            # _is_moe as True. For Qwen35, Mixtral sees num_local_experts=0 (uses a
            # different config key), so it skips the allocation. Allocate them now.
            import wgpu as _wgpu_lib
            dev = self.wgpu_device.wgpu_device
            self._wgpu_map_read = _wgpu_lib.MapMode.READ
            self._wgpu_staging_usage = _wgpu_lib.BufferUsage.COPY_DST | _wgpu_lib.BufferUsage.MAP_READ
            _staging_sz = max(self._top_k * 4, 8)
            self._topk_idx_staging = dev.create_buffer(
                size=_staging_sz,
                usage=self._wgpu_staging_usage)
            # Match Mixtral's lazy pattern: only allocate when debug logging is active.
            self._topk_w_staging = None
            self._expert_out_zeros = bytearray(self.hidden_size * 2)

        # NOTE: profiling=True is incompatible with MoE forward (per-layer submit breaks
        # _batched_dispatch encoder management). Set profiling=False before forward().

    def _is_full_attn(self, i: int) -> bool:
        if i < len(self._layer_types):
            return self._layer_types[i] == "full_attention"
        raise ValueError(
            f"Layer {i} has no layer_types entry; Qwen3_5Config should always "
            "populate layer_types from full_attention_interval at init time."
        )

    def _init_scratch_buffers(self, max_ctx: int) -> None:
        # Inherit standard _pre (7 keys), _sc (17 keys), and _hstate from parent.
        # Pass qkv_size so the parent allocates qkv_buf at the correct GDN size
        # directly, avoiding an allocate-then-discard cycle on every instantiation.
        super()._init_scratch_buffers(max_ctx, qkv_size=self._lin_conv_dim * 2)

        Q = self.num_q_heads * self.head_dim

        # Attention output gate: silu(gate)*attn_out before o_proj.
        # Only allocated for models with attn_output_gate=True; no shader dispatch
        # unconditionally binds this slot, so the allocation must be conditional.
        if self._attn_output_gate:
            self._sc["q_gate_buf"] = self._make_buf(Q * 2)

        # GDN linear-attention scratch buffers (sized from config, not hardcoded).
        self._sc.update({
            "qkv_conv":     self._make_buf(self._lin_conv_dim * 2),  # post-conv output
            "a_buf":        self._make_buf(self._lin_v_heads * 2),   # in_proj_a output [V_HEADS f16]
            "z_buf":        self._make_buf(self._lin_val_dim * 2),   # in_proj_z output
            "gdn_out":      self._make_buf(self._lin_val_dim * 2),   # GDN attn output
            "gated":        self._make_buf(self._lin_val_dim * 2),   # after norm+gate
            "b_buf":        self._make_buf(self._lin_v_heads * 2),   # in_proj_b output [V_HEADS f16]
        })

        if self._is_moe:
            _moe_act_sz = max(self._moe_inter, self._moe_shared_inter, 1)
            self._moe_act_sz = _moe_act_sz
            self._moe_sc = self._alloc_moe_sc(self._moe_num_experts, self._moe_k, _moe_act_sz)

    def _postprocess_weights(self) -> None:
        """Post-load weight fixups for full-attn layers.

        When attn_output_gate=True, the q_proj.weight split is handled at load time
        via _weight_transforms (registered in load_weights above).  The transform
        returns only the Q half; the gate half is stashed in a local dict and uploaded
        immediately after super().load_weights() returns, with no GPU round-trip.

        For quantized (I8) q_proj weights the loader skips transforms entirely, so the
        weight arrives with its original [2*q_dim, H] shape and q_gate_proj.weight is
        never created.  This method detects that case and raises a clear error rather
        than letting the gate be silently bypassed at inference time.

        Note: q_norm/k_norm tiling is handled at load time via _weight_transforms for
        all checkpoint formats. No GPU readback happens here for correctly split fp16 weights.
        """
        if self._attn_output_gate:
            q_dim = self.num_q_heads * self.head_dim
            for i in range(self.num_layers):
                if not self._is_full_attn(i):
                    continue
                prefix = f"model.layers.{i}.self_attn"
                q_key = f"{prefix}.q_proj.weight"
                gate_key = f"{prefix}.q_gate_proj.weight"
                if gate_key not in self.weights and q_key in self.weights:
                    buf = self.weights[q_key]
                    # Detect unsplit q+gate tensors by shape rather than total element count.
                    # fp16/INT8/FP8: (2*q_dim, hidden) → shape[0] == 2*q_dim
                    # GPTQ:          (2*q_dim, hidden//8) → shape[0] == 2*q_dim
                    # NF4/NVFP4:     (2*q_dim, hidden//2) → shape[0] == 2*q_dim
                    # AWQ:           (hidden, 2*q_dim//8) → shape[1] * 8 == 2*q_dim
                    uq = self._uq_for_key(q_key)
                    shape0_unsplit = len(buf.shape) >= 1 and buf.shape[0] == 2 * q_dim
                    awq_unsplit    = uq == 4 and len(buf.shape) >= 2 and buf.shape[1] * 8 == 2 * q_dim
                    if shape0_unsplit or awq_unsplit:
                        # uq 5/6/7/8: weight is already quantized (fp8_gpu,
                        # nvfp4, int8_gpu, nf4), so advising "load fp16" is wrong.
                        if uq != 0:
                            hint = "Pre-split the q_proj tensor before quantizing."
                        else:
                            hint = ("Load an fp16 checkpoint, or pre-split the "
                                    "q_proj tensor before quantizing.")
                        raise ValueError(
                            f"Layer {i}: q_gate_proj.weight is missing but "
                            f"q_proj.weight has shape {buf.shape}, which matches "
                            "an unsplit combined q+gate tensor. "
                            "The split transform was skipped by the loader. "
                            + hint
                        )

    def _alloc_lin_states(self) -> None:
        """Allocate GPU buffers for persistent GDN recurrent state.

        Called after load_weights. Each linear-attention layer gets:
          - SSM state:  [NUM_V_HEADS * K_DIM * V_DIM] f32 (zero-initialized)
          - Conv state: [(CONV_KERNEL-1) * CONV_DIM] f16 (zero-initialized)

        HuggingFace checkpoints store conv1d weight as [CONV_DIM, 1, KERNEL]
        (standard PyTorch depthwise-conv layout). The GPU shader reads bytes
        identically for both [CONV_DIM, 1, KERNEL] and [CONV_DIM, KERNEL], so no
        reshape is needed.
        """
        dev = self.wgpu_device.wgpu_device

        # gdn_state_update.wgsl lays out SSM state as [NUM_V_HEADS, V_DIM, K_DIM] f32,
        # matching vLLM's gated_delta_net_state_shape convention.
        #
        # Only the total element count (math.prod) is used from conv_shape, not its
        # axis layout. gated_delta_net_state_shape may return (conv_dim, kernel-1) or
        # (kernel-1, conv_dim) depending on VLLM_SSM_CONV_STATE_LAYOUT, but both
        # orderings produce the same product, so the buffer size is correct regardless.
        #
        # _lin_conv_dim is derived from gated_delta_net_state_shape in __init__ rather
        # than re-computed from the config formula, so it is always consistent with the
        # conv_shape returned here. No extra assertion is needed.
        conv_bytes = math.prod(self._gdn_conv_shape) * _ELEM_BYTES["f16"]
        ssm_bytes  = math.prod(self._gdn_ssm_shape)  * _ELEM_BYTES["f32"]

        self._ssm_gpu  = [None] * self.num_layers
        self._conv_gpu = [None] * self.num_layers

        for i in range(self.num_layers):
            if self._is_full_attn(i):
                continue
            self._ssm_gpu[i]  = WebGPUBuffer.empty(dev, max(ssm_bytes, 8))
            self._conv_gpu[i] = WebGPUBuffer.empty(dev, max(conv_bytes, 8))

    def load_weights(self, path: str) -> None:
        # A_log and dt_bias are small per-head arrays originally in bf16 but stored
        # as f16. Keeping them as f32 avoids ~3-bit mantissa loss in the decay
        # computation. Pass their checkpoint key names so they are uploaded as f32
        # directly, without a GPU round-trip. Matches the nemotron_h.py pattern.
        f32_keys = frozenset(
            f"model.layers.{i}.linear_attn.{wk}"
            for i in range(self.num_layers)
            if not self._is_full_attn(i)
            for wk in ("A_log", "dt_bias")
        )

        # Register CPU-side split transforms for q_proj.weight when attn_output_gate
        # is enabled.  The loader calls weight_transforms[name](arr) with the fp16
        # numpy array before uploading; by returning only the Q half here, the gate
        # half is stashed without any GPU round-trip.  _postprocess_weights sees
        # shape[0] == q_dim (not 2*q_dim) for correctly split weights.
        # Quantized q_proj weights are NOT split here — the loader ignores transforms
        # for I8 keys, so the weight lands with its original shape.  _postprocess_weights
        # detects the missing gate key and raises a clear error in that case.
        _q_gate_pending: dict[str, np.ndarray] = {}
        if self._attn_output_gate:
            _q_dim = self.num_q_heads * self.head_dim

            def _make_split(gk, pending=_q_gate_pending):
                def _split(arr):
                    if arr.shape[0] != 2 * _q_dim:
                        return arr  # already split or unexpected shape; pass through
                    # arr is always float16: loader converts BF16/F32 before invoking transforms
                    a = arr.reshape(self.num_q_heads, 2 * self.head_dim, self.hidden_size)
                    q_half    = a[:, :self.head_dim, :].reshape(_q_dim, self.hidden_size)
                    gate_half = a[:, self.head_dim:, :].reshape(_q_dim, self.hidden_size)
                    pending[gk] = gate_half
                    return q_half
                return _split

            for _i in range(self.num_layers):
                if not self._is_full_attn(_i):
                    continue
                _q_key = f"model.layers.{_i}.self_attn.q_proj.weight"
                _g_key = f"model.layers.{_i}.self_attn.q_gate_proj.weight"
                self._weight_transforms[_q_key] = _make_split(_g_key)

        super().load_weights(path, f32_keys=f32_keys)

        # Upload gate halves that were split on CPU during the weight transforms above.
        if _q_gate_pending:
            _dev = self.wgpu_device.wgpu_device
            for _gk, _gate_arr in _q_gate_pending.items():
                self.weights[_gk] = WebGPUBuffer.from_numpy(_dev, _gate_arr)
        self._postprocess_weights()
        self._alloc_lin_states()
        # Confirm MoE detection against actual weight keys.
        # Check any layer rather than pinning to layer 0.
        has_moe_gate = any(k.endswith(".mlp.gate.weight") for k in self.weights)
        if has_moe_gate and not self._is_moe:
            logger.warning(
                "MoE gate weight found but config did not declare num_experts. "
                "MoE scratch buffers were not pre-allocated; disabling MoE path.")
            # Cannot safely enable MoE after _init_scratch_buffers already ran.
        elif not has_moe_gate and self._is_moe:
            logger.info("No MoE gate weight found; disabling MoE path (using dense FFN).")
            self._is_moe = False

    def reset_recurrent_states(self) -> None:
        """Zero out all GDN recurrent GPU buffers (call at start of each new sequence)."""
        dev = self.wgpu_device.wgpu_device
        for buf in chain(self._ssm_gpu, self._conv_gpu):
            if buf is not None:
                zeros = self._zero_buf_cache.get(buf.nbytes)
                if zeros is None:
                    zeros = bytes(buf.nbytes)
                    self._zero_buf_cache[buf.nbytes] = zeros
                dev.queue.write_buffer(buf.buf, 0, zeros)

    def save_recurrent_states(self) -> dict:
        """Snapshot all GDN conv/SSM state buffers to CPU in one GPU readback.

        Returns {"conv": {layer_idx: bytes}, "ssm": {layer_idx: bytes}}.

        SSM state bytes use the vLLM convention: [V_HEADS, V_DIM, K_DIM] f32,
        matching gated_delta_net_state_shape. The shader stores state in the same
        layout, so no transposition is needed on readback.
        """
        bufs: list[tuple[str, int, object]] = []
        for i, buf in enumerate(self._conv_gpu):
            if buf is not None:
                bufs.append(("conv", i, buf))
        for i, buf in enumerate(self._ssm_gpu):
            if buf is not None:
                bufs.append(("ssm", i, buf))
        return self._readback_recurrent_states(bufs)

    def restore_recurrent_states(self, states: dict) -> None:
        """Write saved state bytes back into GDN conv/SSM GPU buffers.

        queue.write_buffer enqueues writes without blocking, so all layers
        are uploaded before the next GPU dispatch without an explicit submit.

        SSM bytes in `states["ssm"]` must use the vLLM convention
        [V_HEADS, V_DIM, K_DIM] f32, which is also the shader's native layout.
        Bytes from save_recurrent_states or from vLLM state tensors can be
        uploaded directly without any transposition.
        """
        dev = self.wgpu_device.wgpu_device
        for i, data in states.get("conv", {}).items():
            dev.queue.write_buffer(self._conv_gpu[i].buf, 0, data)
        for i, data in states.get("ssm", {}).items():
            dev.queue.write_buffer(self._ssm_gpu[i].buf, 0, data)

    def _resolve_gdn_weight(self, key: str):
        """Return (buffer, use_bf16_flag, use_quant) for a GDN projection weight key.

        When GDN_BF16 is active, prefers the bf16-preserved variant (key +
        '__bf16') when present; falls back to the standard f16 buffer.

        Emits a warning when both use_quant != 0 and a bf16 variant is present,
        since the matmul_quant shader cannot handle both flags simultaneously.
        The quantized path takes precedence in that case.
        """
        use_quant = self._uq_for_key(key)
        if self._gdn_bf16:
            bf16_buf = self.weights.get(key + "__bf16")
            if bf16_buf is not None:
                if use_quant != 0:
                    logger.warning(
                        "GDN weight %s is both quantized (USE_QUANT=%d) and "
                        "bf16-preserved; USE_BF16 ignored — quantized path takes "
                        "precedence.", key, use_quant)
                else:
                    return bf16_buf, 1, 0
        return self.weights[key], 0, use_quant

    def _gdn_proj(self, p: str, proj_name: str,
                  in_buf: "WebGPUBuffer", out_buf: "WebGPUBuffer",
                  K: int, N: int) -> None:
        """Run one GDN matmul_quant projection dispatch."""
        wk = f"{p}.{proj_name}.weight"
        w, bf16, uq = self._resolve_gdn_weight(wk)
        qi = self._quant_extra(f"{p}.{proj_name}", uq)
        self._dispatch("matmul_quant",
                       [in_buf, w,
                        self._scales_buf(wk, uq, self._dummy_buf),
                        out_buf],
                       {"K": K, "N": N, "USE_QUANT": uq, "USE_BF16": bf16, **qi},
                       _gemv_wg(N))

    def _gdn_layer_gpu(
        self,
        layer_idx: int,
        normed_x: "WebGPUBuffer",
        x_buf: "WebGPUBuffer",
        num_tokens: int,
    ) -> "tuple[WebGPUBuffer, WebGPUBuffer]":
        """Single-token GDN decode step using pure WebGPU kernels.

        All operations dispatch WGSL compute shaders — no CPU fallback.
        Persistent state (SSM matrix + conv history) lives in GPU buffers
        that are mutated in-place each call.

        Pipeline:
          1. normed_x is passed in pre-normed by caller
          2. matmul_quant(normed_x, qkv_w)   → qkv_buf    [8192 f16]
          3. causal_conv_step(qkv_buf)        → qkv_conv   [8192 f16], updates conv_state
          4. matmul_quant(normed_x, a_proj_w) → a_buf      [V_HEADS f16]
          5a. matmul_quant(normed_x, b_proj_w) → b_buf     [V_HEADS f16]
          5b. matmul_quant(normed_x, z_proj_w) → z_buf     [4096 f16]
          6. gdn_state_update(qkv_conv, a_buf, b_buf, A_log, dt_bias, ssm_state) → gdn_out [val_dim f16], updates ssm_state
          7. linear_attn_norm_gate(gdn,z)     → gated      [4096 f16]
          8. matmul_quant(gated, out_proj)    → out_buf    [hidden f16]
          9. add(x, out_buf)                  → residual
          10. FFN (rms_norm → gate/up → gelu → down → add)
        """
        sc = self._sc
        hidden = self.hidden_size
        p = f"model.layers.{layer_idx}.linear_attn"
        pp = f"model.layers.{layer_idx}"
        add_n = num_tokens * hidden

        residual = sc[_H_NAMES[(self._hstate + 1) % 3]]
        out = sc[_H_NAMES[(self._hstate + 2) % 3]]

        # GDN linear attention has no KV cache — state is in ssm_gpu/conv_gpu buffers.
        # Offsets into flat QKV buffer (f16 elements), precomputed in __init__.
        _rms_h = self._rms_consts

        with self._batched_dispatch(label=f"L{layer_idx:02d}"):
            # normed_x is already the pre-normalized input from the caller.
            cd = self._lin_conv_dim
            vd = self._lin_val_dim
            kh = self._lin_k_heads
            vh = self._lin_v_heads

            # 2. QKV projection: [hidden] → [conv_dim]
            self._gdn_proj(p, "in_proj_qkv", normed_x, sc["qkv_buf"], hidden, cd)

            # 3. Causal conv step: updates conv_state in-place, writes qkv_conv
            conv_w = self.weights[f"{p}.conv1d.weight"]
            self._dispatch("causal_conv_step",
                           [sc["qkv_buf"], conv_w, self._conv_gpu[layer_idx], sc["qkv_conv"]],
                           {"CONV_DIM": cd, "KERNEL": self._lin_conv_kernel, "WG_SIZE": 256},
                           ((cd + 255) // 256, 1, 1))

            # 4. a projection: normed → [V_HEADS] (dt for decay, one per V-head)
            self._gdn_proj(p, "in_proj_a", normed_x, sc["a_buf"], hidden, vh)

            # 5a. b projection: normed → [V_HEADS] (outer-product gate, one per V-head)
            self._gdn_proj(p, "in_proj_b", normed_x, sc["b_buf"], hidden, vh)

            # 5b. z gate projection: normed → [val_dim]
            self._gdn_proj(p, "in_proj_z", normed_x, sc["z_buf"], hidden, vd)

            # 6. GDN state update: updates ssm_state in-place, writes gdn_out
            self._dispatch("gdn_state_update",
                           [sc["qkv_conv"], sc["a_buf"], sc["b_buf"],
                            self.weights[f"{p}.A_log"],
                            self.weights[f"{p}.dt_bias"],
                            self._ssm_gpu[layer_idx], sc["gdn_out"]],
                           {"K_DIM": self._lin_k_dim, "V_DIM": self._lin_v_dim,
                            "NUM_K_HEADS": kh, "NUM_V_HEADS": vh,
                            "Q_BASE": 0, "K_BASE": self._gdn_k_base, "V_BASE": self._gdn_v_base},
                           _gemv_wg(vh))

            # 7. Per-head RMSNorm + SiLU gate (z * sigmoid(z)) → gated
            self._dispatch("linear_attn_norm_gate",
                           [sc["gdn_out"], self.weights[f"{p}.norm.weight"],
                            sc["z_buf"], sc["gated"]],
                           {"NUM_V_HEADS": vh, "V_DIM": self._lin_v_dim},
                           _gemv_wg(vh))

            # 8. Output projection: [val_dim] → [hidden]
            self._gdn_proj(p, "out_proj", sc["gated"], sc["o_proj_out"], vd, hidden)

            # 9+10 fused: add(x, attn_out, residual) + rms_norm(residual, post_attn_norm) → ffn_normed
            self._dispatch("add_rms_norm",
                           [x_buf, sc["o_proj_out"],
                            self.weights[f"{pp}.post_attention_layernorm.weight"],
                            residual, sc["ffn_normed"]],
                           _rms_h, (num_tokens, 1, 1))

            # 10. FFN (MoE or dense)
            ffn_out = self._ffn_dispatch(sc["ffn_normed"], layer_idx, num_tokens)

            if layer_idx < self.num_layers - 1:
                next_w = self.weights[f"model.layers.{layer_idx+1}.input_layernorm.weight"]
                self._dispatch("add_rms_norm",
                               [residual, ffn_out, next_w, out, sc["normed"]],
                               _rms_h, (num_tokens, 1, 1))
                normed_out = sc["normed"]
            else:
                self._dispatch("add", [residual, ffn_out, out],
                               {"N": add_n}, _vec4_wg(add_n))
                normed_out = out  # safe placeholder; callers discard first return on last layer

        self._hstate = (self._hstate + 2) % 3
        return normed_out, out

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
        """Route to GDN or full-attention based on layer type."""
        if self._is_full_attn(layer_idx):
            return super()._transformer_layer(
                layer_idx, normed_x, x_buf, pos_buf, slot_map, bt_buf, ctx_len, num_tokens)
        return self._gdn_layer_gpu(layer_idx, normed_x, x_buf, num_tokens)

    def _moe_ffn_layer(
        self,
        normed_x: "WebGPUBuffer",
        layer_idx: int,
    ) -> None:
        """MoE FFN for Qwen35: delegates to MixtralWebGPUModel._moe_ffn_layer.

        Uses Qwen35 weight key conventions (gate_proj/up_proj/down_proj) and
        dispatches the always-active shared expert. Result accumulates into
        self._moe_sc["expert_out"].
        """
        super()._moe_ffn_layer(
            normed_x, layer_idx,
            bsm_prefix="mlp",
            router_subkey="gate",
            gate_key="gate_proj",
            up_key="up_proj",
            down_key="down_proj",
            expert_inter=self._moe_inter,
            shared_expert_prefix="shared_expert",
            shared_expert_inter=self._moe_shared_inter,
        )

    def _prefill_chunked_forward(
        self,
        input_ids: np.ndarray,
        positions: np.ndarray,
        attn_metadata: object,
        num_tokens: int,
    ) -> np.ndarray:
        """Sequential prefill with chunked GPU submission to avoid Metal GPU timeout.

        Each token is processed in its own single-token forward pass (sequential),
        but groups of _CHUNK tokens share one command encoder. GDN SSM and conv
        states are updated in-place across tokens, which is correct for the
        recurrent formulation. Only the last token's logits are sampled.

        Per-token input buffers (ids, pos, slot_map) are allocated upfront so that
        write_buffer calls do not race with encoder dispatches that reference the
        same buffer from a prior token.
        """
        dev = self.wgpu_device.wgpu_device

        hidden = self.hidden_size
        vocab = self.vocab_size
        # Tokens per command encoder. 6 tokens of 36 Qwen3.5-9B layers
        # generates ~6x less GPU work per submit than the full sequence,
        # keeping each encoder well under Metal's per-command-buffer timeout.
        _CHUNK = 6

        _rms_base = self._rms_consts
        sc = self._sc
        pre = self._pre

        # Allocate one small buffer set per token for ids/pos/slot_map.
        # Shared scratch (sc["normed"], sc["h0/h1/h2"]) is safe to reuse because
        # the GPU executes dispatches within each encoder in submission order.
        bt_arr = self._bt_arr(attn_metadata)
        dev.queue.write_buffer(self._pre["bt"].buf, 0, bt_arr.tobytes())
        bt_buf = self._pre["bt"]

        slot_arr = np.asarray(attn_metadata.slot_mapping, dtype=np.uint32)
        tok_ids_bufs: list = []
        tok_pos_bufs: list = []
        tok_slot_bufs: list = []
        for tc in range(num_tokens):
            tok_ids_bufs.append(WebGPUBuffer.from_numpy(
                dev, input_ids[tc:tc+1].astype(np.uint32, copy=False)))
            tok_pos_bufs.append(WebGPUBuffer.from_numpy(
                dev, positions[tc:tc+1].astype(np.uint32, copy=False)))
            tok_slot_bufs.append(WebGPUBuffer.from_numpy(
                dev, slot_arr[tc:tc+1]))

        greedy = self._greedy_decode

        for chunk_toks in batched(range(num_tokens), _CHUNK):
            # Open one encoder for this chunk.
            # Layer method _batched_dispatch calls become re-entrant no-ops
            # because _active_encoder is already set, so all dispatches land here.
            self._active_encoder = dev.create_command_encoder()

            for tc in chunk_toks:
                ctx_t = int(positions[tc]) + 1
                ids_buf  = tok_ids_bufs[tc]
                pos_buf  = tok_pos_bufs[tc]
                slot_map = tok_slot_bufs[tc]
                # Reset h-state rotation: each token's forward pass starts at h0.
                self._hstate = 0

                # Embedding: token id → hidden state in pre["x"]
                x_buf = pre["x"]
                self._dispatch("embedding_lookup",
                               [self.weights["model.embed_tokens.weight"],
                                ids_buf, x_buf],
                               {"HIDDEN_DIM": hidden}, (1, 1, 1))

                # Pre-norm for layer 0 (subsequent pre-norms fused in add_rms_norm)
                self._dispatch("rms_norm",
                               [x_buf,
                                self.weights["model.layers.0.input_layernorm.weight"],
                                sc["normed"]],
                               _rms_base, (1, 1, 1))
                normed_x = sc["normed"]

                for i in range(self.num_layers):
                    normed_x, x_buf = self._transformer_layer(
                        i, normed_x, x_buf, pos_buf, slot_map, bt_buf,
                        ctx_t, 1)

                # For the last token: final norm, LM head, argmax, staging copy.
                if tc == num_tokens - 1:
                    self._dispatch("rms_norm",
                                   [x_buf, self.weights["model.norm.weight"],
                                    pre["norm_out"]],
                                   _rms_base, (1, 1, 1))
                    self._decode_teardown(pre["norm_out"], pre["logits"], vocab, greedy)

            # Submit all dispatches for this chunk.
            dev.queue.submit([self._active_encoder.finish()])
            self._active_encoder = None

        if greedy:
            tok = self._read_sample_tok()
            return np.array([[tok]], dtype=np.int32)
        return self.logit_readback()

    def forward(
        self,
        input_ids: np.ndarray,
        positions: np.ndarray,
        attn_metadata: object,
    ) -> np.ndarray:
        if self._is_moe:
            return super().forward(input_ids, positions, attn_metadata)

        num_tokens = len(input_ids)

        if num_tokens > 1:
            self._check_single_sequence(attn_metadata)
            return self._prefill_chunked_forward(
                input_ids, positions, attn_metadata, num_tokens)

        return super().forward(input_ids, positions, attn_metadata)

    def _attn_block(
        self,
        layer_idx: int,
        normed_x: "WebGPUBuffer",
        pos_buf: "WebGPUBuffer",
        slot_map: "WebGPUBuffer",
        bt_buf: "WebGPUBuffer",
        ctx_len: int,
        num_tokens: int,
    ) -> "WebGPUBuffer":
        """QKV projections, RoPE, KV-cache store, attention decode, and O projection.

        Extends the parent with Qwen3.5-specific behaviour:
          - attn_output_gate: sigmoid-gated attention output before o_proj.
          - GEMMA_NORM, ROTARY_DIM, INTERLEAVED added to per-head norm+rope shaders.

        fused_qkv is skipped: attn_output_gate requires the gate matmul between QKV
        projections and RoPE, which is incompatible with the parent's fused path.

        Must be called inside an active _batched_dispatch context.
        """
        sc = self._sc
        hidden = self.hidden_size
        p = f"model.layers.{layer_idx}"
        q_dim = self.num_q_heads * self.head_dim

        _uq = self._uq_for_key
        k_cache, v_cache = self.kv_pool[layer_idx]

        # QKV projections (always separate; fused_qkv is incompatible with attn_output_gate).
        _uq_q = _uq(f"{p}.self_attn.q_proj.weight")
        _uq_k = _uq(f"{p}.self_attn.k_proj.weight")
        _uq_v = _uq(f"{p}.self_attn.v_proj.weight")
        _q_src, _k_src, _v_src = self._qkv_proj(normed_x, layer_idx, _uq_q, _uq_k, _uq_v)

        # When attn_output_gate=True, q_proj.weight was split at load time.
        # Compute the gate projection: normed_x → q_gate_buf [q_dim f16].
        # The gate is applied as sigmoid(gate)*attn_out before o_proj (step below).
        gate_w = None
        if self._attn_output_gate:
            gate_wk = f"{p}.self_attn.q_gate_proj.weight"
            gate_w = self.weights.get(gate_wk)
            if gate_w is not None:
                uq_gate = _uq(gate_wk)
                qi_gate = self._quant_extra(f"{p}.self_attn.q_gate_proj", uq_gate)
                self._dispatch("matmul_quant",
                               [normed_x, gate_w,
                                self._scales_buf(gate_wk, uq_gate, self._dummy_buf),
                                sc["q_gate_buf"]],
                               {"K": hidden, "N": q_dim, "USE_QUANT": uq_gate, **qi_gate},
                               _gemv_wg(q_dim))

        # Per-head RMSNorm + RoPE with Qwen3.5-specific constants.
        _q_norm_w = self.weights.get(f"{p}.self_attn.q_norm.weight")
        _k_norm_w = self.weights.get(f"{p}.self_attn.k_norm.weight")
        _freq_buf = self._rope_freq_buf
        _rope_base = self._rope_base
        if _q_norm_w is not None and _k_norm_w is not None:
            # fused_qk_norm_rope with K_SEPARATE=1: Q in _q_src, K in _k_src (separate buffers).
            self._dispatch("fused_qk_norm_rope",
                           [_q_src, _q_norm_w, _k_norm_w, pos_buf,
                            sc["q_rope"], sc["k_rope"], _k_src, _freq_buf],
                           {**_rope_base,
                            "NUM_Q_HEADS": self.num_q_heads,
                            "NUM_KV_HEADS": self.num_kv_heads,
                            "HAS_WEIGHT": 1,
                            "INPUT_OFFSET_K": 0,
                            "K_SEPARATE": 1},
                           (self.num_q_heads + self.num_kv_heads, num_tokens, 1))
        else:
            for src, dst, n_heads, norm_w in [
                (_q_src, sc["q_rope"], self.num_q_heads, _q_norm_w),
                (_k_src, sc["k_rope"], self.num_kv_heads, _k_norm_w),
            ]:
                if norm_w is not None:
                    self._dispatch("fused_per_head_norm_rope",
                                   [src, norm_w, pos_buf, dst, _freq_buf],
                                   {**_rope_base, "NUM_HEADS": n_heads, "HAS_WEIGHT": 1, "INPUT_OFFSET": 0},
                                   (n_heads, num_tokens, 1))
                else:
                    self._dispatch("fused_per_head_norm_rope",
                                   [src, self._dummy_buf, pos_buf, dst, _freq_buf],
                                   {**_rope_base, "NUM_HEADS": n_heads, "HAS_WEIGHT": 0, "INPUT_OFFSET": 0},
                                   (n_heads, num_tokens, 1))

        # Fused K+V cache store. V always lives in its own _v_src buffer (no offset needed).
        self._dispatch("kv_cache_store_both",
                       [sc["k_rope"], k_cache, _v_src, v_cache, slot_map],
                       {"BLOCK_SIZE": self.block_size, "NUM_KV_HEADS": self.num_kv_heads,
                        "HEAD_DIM": self.head_dim, "V_IN_OFFSET": 0},
                       (num_tokens, self.num_kv_heads, 1))

        # Pass ctx_len raw (not through _effective_ctx_len) because this path is
        # only dispatched for full-attention layers. Full-attention layers must
        # attend to the entire context even on models that also have sliding-window
        # layers, so capping by _sw would be incorrect here.
        self._dispatch("flash_attn_decode",
                       [sc["q_rope"], k_cache, v_cache, bt_buf, sc["attn_out"]],
                       {"BLOCK_SIZE": self.block_size, "NUM_Q_HEADS": self.num_q_heads,
                        "NUM_KV_HEADS": self.num_kv_heads, "HEAD_DIM": self.head_dim,
                        "CTX_LEN": ctx_len},
                       (self.num_q_heads, 1, 1))

        # Apply attention output gate: gated = sigmoid(gate) * attn_out.
        # _q_src is free at this point (last read in RoPE), reused as the gated output.
        if gate_w is not None:
            gate_n = num_tokens * q_dim
            self._dispatch("sigmoid_gate",
                           [sc["q_gate_buf"], sc["attn_out"], _q_src],
                           {"N": gate_n}, _vec4_wg(gate_n))
            o_proj_in = _q_src
        else:
            o_proj_in = sc["attn_out"]

        # Output projection.
        w_key = f"{p}.self_attn.o_proj.weight"
        uq = _uq(w_key)
        qi = self._quant_extra(f"{p}.self_attn.o_proj", uq)
        self._dispatch("matmul_quant",
                       [o_proj_in, self.weights[w_key],
                        self._scales_buf(w_key, uq, self._dummy_buf), sc["o_proj_out"]],
                       {"K": q_dim, "N": hidden, "USE_QUANT": uq, **qi},
                       _gemv_wg(hidden))

        return sc["o_proj_out"]


