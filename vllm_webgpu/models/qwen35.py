from __future__ import annotations
import math
from typing import TYPE_CHECKING

import numpy as np

from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm_webgpu.models.base import _gemv_wg, _vec4_wg, _H_NAMES
from vllm_webgpu.models.mixtral import MixtralWebGPUModel
import vllm_webgpu.envs as _webgpu_envs

from vllm_webgpu.webgpu.buffer import WebGPUBuffer

if TYPE_CHECKING:
    from vllm_webgpu.webgpu.device import WebGPUDevice
    from vllm_webgpu.webgpu.pipeline import PipelineCache

logger = init_logger(__name__)


# Qwen3.5-9B fixed architecture constants
_FULL_ATTN_INTERVAL = 4
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
      - linear_attn_norm_gate.wgsl: per-head RMSNorm + sigmoid gate

    Persistent recurrent state (SSM matrix + conv history) lives in GPU buffers
    that are updated in-place each decode step. No CPU↔GPU transfers in the
    hot path.
    """

    def __init__(self, model_config, wgpu_device: "WebGPUDevice", pipeline_cache: "PipelineCache") -> None:
        # Set GDN + MoE + other Qwen3.5-specific attributes BEFORE calling
        # super().__init__(). LlamaWebGPUModel.__init__() calls
        # self._init_scratch_buffers() via Python's dynamic dispatch, which
        # resolves to Qwen35's override. That override uses these values, so they
        # must exist before the super() call returns.

        self._layer_types: list | None = getattr(model_config, "layer_types", None)
        # Partial RoPE: some models only rotate a fraction of head dimensions.
        # partial_rotary_factor=0.25 → rotary_dim = head_dim * 0.25.
        _prf = getattr(model_config, "partial_rotary_factor", None) or 1.0
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
        # before the o_proj. _postprocess_weights splits the weight and stores the gate
        # half under self_attn.q_gate_proj.weight.
        self._attn_output_gate: bool = bool(getattr(model_config, "attn_output_gate", True))

        # GDN (linear-attention) architecture dimensions from config.
        # Fall back to Qwen3.5-9B defaults if not present.
        self._lin_k_heads: int = getattr(model_config, "linear_num_key_heads", _LIN_K_HEADS)
        self._lin_k_dim: int   = getattr(model_config, "linear_key_head_dim",  _LIN_K_DIM)
        self._lin_v_heads: int = getattr(model_config, "linear_num_value_heads", _LIN_V_HEADS)
        self._lin_v_dim: int   = getattr(model_config, "linear_value_head_dim", _LIN_V_DIM)
        self._lin_conv_kernel: int = getattr(model_config, "linear_conv_kernel_dim", _LIN_CONV_KERNEL)
        # Total QKV packed dimension: K + K + V heads (Q_heads = K_heads for GDN)
        self._lin_val_dim: int  = self._lin_v_heads * self._lin_v_dim   # total value dim
        self._lin_key_dim: int  = self._lin_k_heads * self._lin_k_dim   # total key dim (= Q dim)
        self._lin_conv_dim: int = 2 * self._lin_key_dim + self._lin_val_dim  # QKV (Q_dim == K_dim)
        # GDN QKV buffer offsets (f16 elements); constant across all layers and tokens.
        # Q is always at offset 0 (leading element in packed QKV buffer).
        self._gdn_k_base: int = self._lin_key_dim
        self._gdn_v_base: int = 2 * self._lin_key_dim

        # MoE config (Qwen3.6-35B-A3B and similar MoE variants).
        # When num_experts > 0 the FFN in every layer is a mixture-of-experts block;
        # the standard gate/up/down weights are replaced by a router + per-expert weights.
        self._moe_num_experts: int = getattr(model_config, "num_experts", 0)
        self._moe_k: int           = getattr(model_config, "num_experts_per_tok", 0)
        self._moe_inter: int       = getattr(model_config, "moe_intermediate_size", model_config.intermediate_size)
        self._moe_shared_inter: int = (
            getattr(model_config, "shared_expert_intermediate_size", None)
            or model_config.intermediate_size)
        # _is_moe is set from config here and may be overridden in load_weights
        # once we can verify against actual weight keys.
        self._is_moe: bool = self._moe_num_experts > 0 and self._moe_k > 0
        # GEMMA_NORM=1 for safetensors (weights are deviations from 1, mean≈0.2).
        # GEMMA_NORM=0 for MLX format (weights are absolute, mean≈1.0 — +1 already baked in).
        # Detected after load_weights() by checking the first layernorm weight mean.
        self._gemma_norm: int = 1  # default; updated in _postprocess_weights

        # GDN_BF16: when set, GDN projection matmuls use bf16-preserved weight buffers
        # (key + "__bf16") instead of the default f16 version. Falls back silently if
        # the __bf16 buffer is absent (model not BF16 or flag off).
        self._gdn_bf16: bool = _webgpu_envs.GDN_BF16

        # Persistent GPU buffers for recurrent state (allocated after load_weights).
        # SSM state:  [NUM_V_HEADS, K_DIM, V_DIM] f32 = 2MB per linear-attn layer
        # Conv state: [CONV_KERNEL-1, CONV_DIM] f16 = 49KB per linear-attn layer
        self._ssm_gpu: list = []   # one WebGPUBuffer per layer (or None for full-attn)
        self._conv_gpu: list = []  # one WebGPUBuffer per layer

        # LlamaWebGPUModel.__init__() sets: num_layers, num_q_heads, num_kv_heads,
        # hidden_size, intermediate_size, vocab_size, head_dim, rope_theta, block_size,
        # _rope_consts (containing LN_ROPE_BASE), _rms_consts (without GEMMA_NORM), runs
        # dimension validation, then calls self._init_scratch_buffers() and
        # self._init_rope_freq_buf().
        super().__init__(model_config, wgpu_device, pipeline_cache)

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
            self._wgpu_lib = _wgpu_lib
            dev = self.wgpu_device.wgpu_device
            _staging_sz = max(self._top_k * 4, 8)
            self._topk_idx_staging = dev.create_buffer(
                size=_staging_sz,
                usage=_wgpu_lib.BufferUsage.COPY_DST | _wgpu_lib.BufferUsage.MAP_READ)
            self._topk_w_staging = dev.create_buffer(
                size=_staging_sz,
                usage=_wgpu_lib.BufferUsage.COPY_DST | _wgpu_lib.BufferUsage.MAP_READ)

        # NOTE: profiling=True is incompatible with MoE forward (per-layer submit breaks
        # _batched_dispatch encoder management). Set profiling=False before forward().

    def _is_full_attn(self, i: int) -> bool:
        if self._layer_types is not None and i < len(self._layer_types):
            return self._layer_types[i] == "full_attention"
        return (i + 1) % _FULL_ATTN_INTERVAL == 0

    def _init_scratch_buffers(self, max_ctx: int) -> None:
        # Inherit standard _pre (7 keys), _sc (17 keys), and _hstate from parent.
        super()._init_scratch_buffers(max_ctx)

        dev = self.wgpu_device.wgpu_device
        Q = self.num_q_heads * self.head_dim

        def mk(n: int) -> "WebGPUBuffer":
            return WebGPUBuffer.empty(dev, max(n, 8))

        # qkv_buf: parent sizes for full-attn (Q + 2*KV); GDN layers need
        # lin_conv_dim (K+K+V heads packed). Replace with the larger GDN size so
        # both full-attn and GDN layers can reuse the same buffer.
        self._sc["qkv_buf"] = mk(self._lin_conv_dim * 2)

        # Attention output gate: silu(gate)*attn_out before o_proj.
        # Not present in the parent; added only for models with attn_output_gate=True,
        # but always allocated so dispatch bindings are stable.
        self._sc["q_gate_buf"] = mk(Q * 2)

        # GDN linear-attention scratch buffers (sized from config, not hardcoded).
        self._sc.update({
            "qkv_conv":     mk(self._lin_conv_dim * 2),  # post-conv output
            "a_buf":        mk(self._lin_k_heads * 2),   # in_proj_a output [K_HEADS f16]
            "z_buf":        mk(self._lin_val_dim * 2),   # in_proj_z output
            "gdn_out":      mk(self._lin_val_dim * 2),   # GDN attn output
            "gated":        mk(self._lin_val_dim * 2),   # after norm+gate
            "b_buf":        mk(self._lin_k_heads * 2),   # in_proj_b output [K_HEADS f16]
        })

        if self._is_moe:
            H = self.hidden_size
            _moe_act_sz = max(self._moe_inter, self._moe_shared_inter, 1)
            self._moe_act_sz = _moe_act_sz
            self._moe_sc: dict[str, "WebGPUBuffer"] = {
                "router_out":   mk(self._moe_num_experts * 2),  # [N_E] f16 router logits
                "topk_idx":     mk(self._moe_k * 4),            # [K] u32 expert indices
                "topk_w":       mk(self._moe_k * 4),            # [K] f32 softmax weights
                "expert_act":   mk(_moe_act_sz * 2),            # [max_inter] f16 gate*up activated
                "expert_out":   mk(H * 2),                      # [hidden] f16 accumulated output
                "expert_tmp":   mk(H * 2),                      # [hidden] f16 per-expert temp
            }

    def _postprocess_weights(self) -> None:
        """Post-load weight transformations for full-attn layers:

        1. Detect GEMMA_NORM format: deviation (safetensors, mean≈0) vs absolute (MLX, mean≈1).
        2. When attn_output_gate=True: split q_proj.weight [2*q_dim, hidden] into
           q_proj.weight [q_dim, hidden] (Q part) and q_gate_proj.weight [q_dim, hidden]
           (gate part). The gate is applied as silu(gate)*attn_out before o_proj.

        Note: q_norm/k_norm tiling is handled at load time via _weight_transforms
        (registered in LlamaWebGPUModel.__init__) for all checkpoint formats.
        """
        dev = self.wgpu_device.wgpu_device

        # Detect norm weight format from the first input_layernorm weight.
        for ln_i in range(min(self.num_layers, 4)):
            ln_w = self.weights.get(f"model.layers.{ln_i}.input_layernorm.weight")
            if ln_w is not None:
                mean_abs = float(np.abs(ln_w.to_numpy().view(np.float16)).mean())
                self._gemma_norm = 0 if mean_abs > 0.7 else 1
                break

        for i in range(self.num_layers):
            if not self._is_full_attn(i):
                continue
            p = f"model.layers.{i}"

            # Split fused Q+gate weight when attn_output_gate=True.
            # HF: q_proj(h).view(batch, seq, num_heads, head_dim*2) → chunk(2, dim=-1)
            #   → first head_dim per head = Q, last head_dim per head = gate
            # Weight shape [2*q_dim, hidden] stored as [num_heads, 2*head_dim, hidden].
            # Query rows (interleaved): head_h[:head_dim] = rows [h*2*hd : h*2*hd+hd]
            # Gate rows (interleaved):  head_h[head_dim:] = rows [h*2*hd+hd : (h+1)*2*hd]
            if self._attn_output_gate:
                q_dim = self.num_q_heads * self.head_dim
                hd = self.head_dim
                q_proj_key = f"{p}.self_attn.q_proj.weight"
                buf = self.weights.get(q_proj_key)
                if buf is not None and buf.shape[0] == 2 * q_dim:
                    if self._uq_for_key(q_proj_key) != 0:
                        raise RuntimeError(
                            f"Layer {i}: quantized q_proj with shape [2*q_dim, H] and "
                            f"attn_output_gate=True is not supported. The interleaved "
                            f"Q+gate rows cannot be split. Use fp16 weights or pre-split "
                            f"the checkpoint offline."
                        )
                    arr = buf.to_numpy().view(np.float16).reshape(self.num_q_heads, 2 * hd, self.hidden_size)
                    q_arr = np.ascontiguousarray(arr[:, :hd, :].reshape(q_dim, self.hidden_size))
                    gate_arr = np.ascontiguousarray(arr[:, hd:, :].reshape(q_dim, self.hidden_size))
                    self.weights[q_proj_key] = WebGPUBuffer.from_numpy(dev, q_arr)
                    gate_key = f"{p}.self_attn.q_gate_proj.weight"
                    self.weights[gate_key] = WebGPUBuffer.from_numpy(dev, gate_arr)

        self._rms_consts["GEMMA_NORM"] = self._gemma_norm

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

        # Use vLLM's canonical shape calculator (same pattern as nemotron_h.py).
        # gdn_state_update.wgsl lays out state as [NUM_V_HEADS, K_DIM, V_DIM] f32,
        # while vLLM returns (num_v_heads, head_v_dim, head_k_dim), with the inner two
        # dims transposed relative to the shader's stride order.  The byte
        # allocation is identical regardless (multiplication is commutative), so the
        # buffers sized here are correct.  The stride difference only matters inside
        # the shader's index arithmetic, which is a separate concern.
        conv_shape, ssm_shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_world_size=1,
            num_k_heads=self._lin_k_heads,
            num_v_heads=self._lin_v_heads,
            head_k_dim=self._lin_k_dim,
            head_v_dim=self._lin_v_dim,
            conv_kernel_size=self._lin_conv_kernel,
        )
        ssm_bytes  = math.prod(ssm_shape) * 4   # f32
        conv_bytes = math.prod(conv_shape) * 2   # f16

        self._ssm_gpu  = [None] * self.num_layers
        self._conv_gpu = [None] * self.num_layers

        for i in range(self.num_layers):
            if self._is_full_attn(i):
                continue
            self._ssm_gpu[i]  = WebGPUBuffer.empty(dev, ssm_bytes)
            self._conv_gpu[i] = WebGPUBuffer.empty(dev, conv_bytes)

            p = f"model.layers.{i}.linear_attn"

            # Upgrade SSM parameter precision: A_log and dt_bias are small per-head
            # arrays originally in bf16 but stored as f16. Keeping them as f32 avoids
            # ~3-bit mantissa loss in the decay computation.
            for key_suffix in ("A_log", "dt_bias"):
                key = f"{p}.{key_suffix}"
                w = self.weights.get(key)
                if w is None:
                    continue
                f16_np = w.to_numpy().view(np.float16)
                f32_np = f16_np.astype(np.float32)
                self.weights[key] = WebGPUBuffer.from_numpy(dev, f32_np)

    def load_weights(self, path: str) -> None:
        super().load_weights(path)
        self._postprocess_weights()
        self._alloc_lin_states()
        # Confirm MoE detection against actual weight keys.
        # Check any layer rather than pinning to layer 0.
        has_moe_gate = any("mlp.gate.weight" in k for k in self.weights)
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
        for lst in (self._ssm_gpu, self._conv_gpu):
            for buf in lst:
                if buf is not None:
                    dev.queue.write_buffer(buf.buf, 0, bytearray(buf.nbytes))

    def save_recurrent_states(self) -> dict:
        """Snapshot all GDN conv/SSM state buffers to CPU in one GPU readback.

        All layer buffers are copied into a single staging buffer in one command
        encoder submission, avoiding N separate GPU-to-CPU round trips.
        Returns {"conv": {layer_idx: bytes}, "ssm": {layer_idx: bytes}}.
        """
        import wgpu
        dev = self.wgpu_device.wgpu_device

        bufs: list[tuple[str, int, object]] = []
        for i, buf in enumerate(self._conv_gpu):
            if buf is not None:
                bufs.append(("conv", i, buf))
        for i, buf in enumerate(self._ssm_gpu):
            if buf is not None:
                bufs.append(("ssm", i, buf))

        if not bufs:
            return {"conv": {}, "ssm": {}}

        offsets: list[int] = []
        total = 0
        for _, _, buf in bufs:
            offsets.append(total)
            total += buf.nbytes

        staging = dev.create_buffer(
            size=max(total, 4),
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        enc = dev.create_command_encoder()
        for (_, _, buf), off in zip(bufs, offsets):
            enc.copy_buffer_to_buffer(buf.buf, 0, staging, off, buf.nbytes)
        dev.queue.submit([enc.finish()])

        staging.map_sync(mode=wgpu.MapMode.READ)
        raw = bytes(staging.read_mapped())
        staging.unmap()

        result: dict[str, dict] = {"conv": {}, "ssm": {}}
        for (kind, i, buf), off in zip(bufs, offsets):
            result[kind][i] = raw[off : off + buf.nbytes]
        return result

    def restore_recurrent_states(self, states: dict) -> None:
        """Write saved state bytes back into GDN conv/SSM GPU buffers.

        queue.write_buffer enqueues writes without blocking, so all layers
        are uploaded before the next GPU dispatch without an explicit submit.
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
          1. rms_norm(x)                   → normed        [hidden f16]
          2. matmul_quant(normed, qkv_w)   → qkv_buf       [8192 f16]
          3. causal_conv_step(qkv_buf)     → qkv_conv      [8192 f16], updates conv_state
          4. matmul_quant(x, a_proj_w)     → a_buf         [32 f16]
          5. matmul_quant(x, z_proj_w)     → z_buf         [4096 f16]
          6. gdn_state_update(qkv_conv, a) → gdn_out       [4096 f16], updates ssm_state
          7. linear_attn_norm_gate(gdn,z)  → gated         [4096 f16]
          8. matmul_quant(gated, out_proj) → out_buf       [hidden f16]
          9. add(x, out_buf)               → residual
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
        k_base = self._gdn_k_base
        v_base = self._gdn_v_base

        _rms_h = self._rms_consts

        with self._batched_dispatch(label=f"L{layer_idx:02d}"):
            # normed_x is already the pre-normalized input from the caller.
            cd = self._lin_conv_dim
            vd = self._lin_val_dim
            kh = self._lin_k_heads
            kd = self._lin_k_dim
            vh = self._lin_v_heads
            vdh = self._lin_v_dim

            # 2. QKV projection: [hidden] → [conv_dim]
            _wk_qkv = f"{p}.in_proj_qkv.weight"
            _w_qkv, _bf16_qkv, _uq_qkv = self._resolve_gdn_weight(_wk_qkv)
            _qi_qkv = self._quant_extra(f"{p}.in_proj_qkv", _uq_qkv)
            self._dispatch("matmul_quant",
                           [normed_x, _w_qkv,
                            self._scales_buf(_wk_qkv, _uq_qkv, self._dummy_scales_buf),
                            sc["qkv_buf"]],
                           {"K": hidden, "N": cd, "USE_QUANT": _uq_qkv, "USE_BF16": _bf16_qkv,
                            **_qi_qkv},
                           _gemv_wg(cd))

            # 3. Causal conv step: updates conv_state in-place, writes qkv_conv
            conv_w = self.weights[f"{p}.conv1d.weight"]
            self._dispatch("causal_conv_step",
                           [sc["qkv_buf"], conv_w, self._conv_gpu[layer_idx], sc["qkv_conv"]],
                           {"CONV_DIM": cd, "KERNEL": self._lin_conv_kernel, "WG_SIZE": 256},
                           ((cd + 255) // 256, 1, 1))

            # 4. a projection: normed → [K_HEADS] (dt for decay)
            _wk_a = f"{p}.in_proj_a.weight"
            _w_a, _bf16_a, _uq_a = self._resolve_gdn_weight(_wk_a)
            _qi_a = self._quant_extra(f"{p}.in_proj_a", _uq_a)
            self._dispatch("matmul_quant",
                           [normed_x, _w_a,
                            self._scales_buf(_wk_a, _uq_a, self._dummy_scales_buf),
                            sc["a_buf"]],
                           {"K": hidden, "N": kh, "USE_QUANT": _uq_a, "USE_BF16": _bf16_a,
                            **_qi_a},
                           _gemv_wg(kh))

            # 5a. b projection: normed → [K_HEADS] (outer-product gate)
            _wk_b = f"{p}.in_proj_b.weight"
            _w_b, _bf16_b, _uq_b = self._resolve_gdn_weight(_wk_b)
            _qi_b = self._quant_extra(f"{p}.in_proj_b", _uq_b)
            self._dispatch("matmul_quant",
                           [normed_x, _w_b,
                            self._scales_buf(_wk_b, _uq_b, self._dummy_scales_buf),
                            sc["b_buf"]],
                           {"K": hidden, "N": kh, "USE_QUANT": _uq_b, "USE_BF16": _bf16_b,
                            **_qi_b},
                           _gemv_wg(kh))

            # 5b. z gate projection: normed → [val_dim]
            _wk_z = f"{p}.in_proj_z.weight"
            _w_z, _bf16_z, _uq_z = self._resolve_gdn_weight(_wk_z)
            _qi_z = self._quant_extra(f"{p}.in_proj_z", _uq_z)
            self._dispatch("matmul_quant",
                           [normed_x, _w_z,
                            self._scales_buf(_wk_z, _uq_z, self._dummy_scales_buf),
                            sc["z_buf"]],
                           {"K": hidden, "N": vd, "USE_QUANT": _uq_z, "USE_BF16": _bf16_z,
                            **_qi_z},
                           _gemv_wg(vd))

            # 6. GDN state update: updates ssm_state in-place, writes gdn_out
            self._dispatch("gdn_state_update",
                           [sc["qkv_conv"], sc["a_buf"], sc["b_buf"],
                            self.weights[f"{p}.A_log"],
                            self.weights[f"{p}.dt_bias"],
                            self._ssm_gpu[layer_idx], sc["gdn_out"]],
                           {"K_DIM": kd, "V_DIM": vdh,
                            "NUM_K_HEADS": kh, "NUM_V_HEADS": vh,
                            "Q_BASE": 0, "K_BASE": k_base, "V_BASE": v_base},
                           _gemv_wg(vh))

            # 7. Per-head RMSNorm + sigmoid gate → gated
            self._dispatch("linear_attn_norm_gate",
                           [sc["gdn_out"], self.weights[f"{p}.norm.weight"],
                            sc["z_buf"], sc["gated"]],
                           {"NUM_V_HEADS": vh, "V_DIM": vdh},
                           _gemv_wg(vh))

            # 8. Output projection: [val_dim] → [hidden]
            _wk_out = f"{p}.out_proj.weight"
            _w_out, _bf16_out, _uq_out = self._resolve_gdn_weight(_wk_out)
            _qi_out = self._quant_extra(f"{p}.out_proj", _uq_out)
            self._dispatch("matmul_quant",
                           [sc["gated"], _w_out,
                            self._scales_buf(_wk_out, _uq_out, self._dummy_scales_buf),
                            sc["o_proj_out"]],
                           {"K": vd, "N": hidden, "USE_QUANT": _uq_out, "USE_BF16": _bf16_out,
                            **_qi_out},
                           _gemv_wg(hidden))

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
        bt_buf = WebGPUBuffer.from_numpy(dev, bt_arr)

        tok_ids_bufs: list = []
        tok_pos_bufs: list = []
        tok_slot_bufs: list = []
        for tc in range(num_tokens):
            tok_ids_bufs.append(WebGPUBuffer.from_numpy(
                dev, input_ids[tc:tc+1].astype(np.uint32)))
            tok_pos_bufs.append(WebGPUBuffer.from_numpy(
                dev, positions[tc:tc+1].astype(np.uint32)))
            tok_slot_bufs.append(WebGPUBuffer.from_numpy(
                dev, np.array([attn_metadata.slot_mapping[tc]], dtype=np.uint32)))

        greedy = self._greedy_decode
        if greedy:
            self._ensure_sample_buf()

        for chunk_start in range(0, num_tokens, _CHUNK):
            chunk_end = min(chunk_start + _CHUNK, num_tokens)

            # Open one encoder for this chunk.
            # Layer method _batched_dispatch calls become re-entrant no-ops
            # because _active_encoder is already set, so all dispatches land here.
            self._active_encoder = dev.create_command_encoder()

            for tc in range(chunk_start, chunk_end):
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

        if hasattr(attn_metadata, "block_tables") and len(attn_metadata.block_tables) > 1:
            raise RuntimeError("multi-sequence batching not supported in this build")

        if num_tokens > 1:
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
        _q_src, _k_src, _v_src = self._qkv_proj(normed_x, layer_idx)

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
                                self._scales_buf(gate_wk, uq_gate, self._dummy_scales_buf),
                                sc["q_gate_buf"]],
                               {"K": hidden, "N": q_dim, "USE_QUANT": uq_gate, **qi_gate},
                               _gemv_wg(q_dim))

        # Per-head RMSNorm + RoPE with Qwen3.5-specific constants.
        _q_norm_w = self.weights.get(f"{p}.self_attn.q_norm.weight")
        _k_norm_w = self.weights.get(f"{p}.self_attn.k_norm.weight")
        _freq_buf = self._rope_freq_buf
        _rope_base = {**self._rope_consts,
                      "GEMMA_NORM": self._gemma_norm,
                      "ROTARY_DIM": self._rotary_dim,
                      "INTERLEAVED": self._rope_interleaved}
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
                                   {**_rope_base, "NUM_HEADS": n_heads, "HAS_WEIGHT": 1},
                                   (n_heads, num_tokens, 1))
                else:
                    self._dispatch("fused_per_head_norm_rope",
                                   [src, self._dummy_scales_buf, pos_buf, dst, _freq_buf],
                                   {**_rope_base, "NUM_HEADS": n_heads, "HAS_WEIGHT": 0},
                                   (n_heads, num_tokens, 1))

        # Fused K+V cache store. V always lives in its own _v_src buffer (no offset needed).
        self._dispatch("kv_cache_store_both",
                       [sc["k_rope"], k_cache, _v_src, v_cache, slot_map],
                       {"BLOCK_SIZE": self.block_size, "NUM_KV_HEADS": self.num_kv_heads,
                        "HEAD_DIM": self.head_dim, "V_IN_OFFSET": 0},
                       (num_tokens, self.num_kv_heads, 1))

        self._dispatch("flash_attn_decode",
                       [sc["q_rope"], k_cache, v_cache, bt_buf, sc["attn_out"]],
                       {"BLOCK_SIZE": self.block_size, "NUM_Q_HEADS": self.num_q_heads,
                        "NUM_KV_HEADS": self.num_kv_heads, "HEAD_DIM": self.head_dim,
                        "CTX_LEN": self._effective_ctx_len(ctx_len)},
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
                        self._scales_buf(w_key, uq, self._dummy_scales_buf), sc["o_proj_out"]],
                       {"K": q_dim, "N": hidden, "USE_QUANT": uq, **qi},
                       _gemv_wg(hidden))

        return sc["o_proj_out"]


