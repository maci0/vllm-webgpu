from __future__ import annotations
import logging
import os
from typing import TYPE_CHECKING

import numpy as np

from vllm_webgpu.models.llama import LlamaWebGPUModel, _gemv_wg
from vllm_webgpu.models.mixtral import MixtralWebGPUModel

if TYPE_CHECKING:
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.device import WebGPUDevice
    from vllm_webgpu.webgpu.pipeline import PipelineCache

logger = logging.getLogger(__name__)


# Qwen3.5-9B fixed architecture constants
_FULL_ATTN_INTERVAL = 4
_LIN_K_HEADS = 16
_LIN_V_HEADS = 32
_LIN_K_DIM = 128
_LIN_V_DIM = 128
_LIN_CONV_KERNEL = 4



class Qwen35WebGPUModel(LlamaWebGPUModel):
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
        _prf = getattr(model_config, "partial_rotary_factor", 1.0) or 1.0
        # Read head_dim from model_config directly — self.head_dim not set yet.
        _head_dim_raw = getattr(model_config, "head_dim",
                                model_config.hidden_size // model_config.num_attention_heads)
        self._rotary_dim: int = max(2, int(_head_dim_raw * _prf))
        if self._rotary_dim % 2 != 0:
            self._rotary_dim -= 1
        # Interleaved RoPE: pairs (2i, 2i+1) vs standard (i, i+half).
        # Qwen3.5 uses mrope_interleaved=True.
        self._rope_interleaved: int = 1 if getattr(model_config, "mrope_interleaved", False) else 0
        # Attention output gate: when True, q_proj.weight has shape [2*q_dim, hidden].
        # The first half is Q; the second half is a gate applied as silu(gate)*attn_out
        # before the o_proj. _postprocess_weights splits the weight and stores the gate
        # half under self_attn.q_gate_proj.weight.
        self._attn_output_gate: bool = bool(getattr(model_config, "attn_output_gate", False))

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
        self._lin_conv_dim: int = self._lin_key_dim + self._lin_key_dim + self._lin_val_dim  # QKV

        # MoE config (Qwen3.6-35B-A3B and similar MoE variants).
        # When num_experts > 0 the FFN in every layer is a mixture-of-experts block;
        # the standard gate/up/down weights are replaced by a router + per-expert weights.
        self._moe_num_experts: int = getattr(model_config, "num_experts", 0)
        self._moe_k: int           = getattr(model_config, "num_experts_per_tok", 0)
        self._moe_inter: int       = getattr(model_config, "moe_intermediate_size", 0)
        self._moe_shared_inter: int = getattr(
            model_config, "shared_expert_intermediate_size", model_config.intermediate_size)
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
        self._gdn_bf16: bool = os.environ.get("GDN_BF16", "0") == "1"

        # Persistent GPU buffers for recurrent state (allocated after load_weights).
        # SSM state:  [NUM_V_HEADS, K_DIM, V_DIM] f32 = 2MB per linear-attn layer
        # Conv state: [CONV_KERNEL-1, CONV_DIM] f16 = 49KB per linear-attn layer
        self._ssm_gpu: list = []   # one WebGPUBuffer per layer (or None for full-attn)
        self._conv_gpu: list = []  # one WebGPUBuffer per layer

        # LlamaWebGPUModel.__init__() sets: num_layers, num_q_heads, num_kv_heads,
        # hidden_size, intermediate_size, vocab_size, head_dim, rope_theta, block_size,
        # _ln_rope_theta, _rms_consts (without GEMMA_NORM), runs dimension validation,
        # then calls self._init_scratch_buffers() and self._init_rope_freq_buf().
        super().__init__(model_config, wgpu_device, pipeline_cache)

        # NOTE: profiling=True is incompatible with MoE forward (per-layer submit breaks
        # _batched_dispatch encoder management). Set profiling=False before forward().

    @property
    def _lm_head_weight(self) -> "WebGPUBuffer":
        return (
            self.weights.get("lm_head.weight")
            or self.weights.get("model.lm_head.weight")
            or self.weights["model.embed_tokens.weight"]
        )

    def _is_full_attn(self, i: int) -> bool:
        if self._layer_types is not None:
            return self._layer_types[i] == "full_attention"
        return (i + 1) % _FULL_ATTN_INTERVAL == 0

    def _init_scratch_buffers(self, max_ctx: int) -> None:
        from vllm_webgpu.webgpu.buffer import WebGPUBuffer

        # Inherit standard _pre (7 keys), _sc (17 keys), and _hstate from parent.
        super()._init_scratch_buffers(max_ctx)

        dev = self.wgpu_device.wgpu_device
        rw = self._rw_flags()
        T = 1
        Q = self.num_q_heads * self.head_dim

        def mk(n: int) -> "WebGPUBuffer":
            return WebGPUBuffer.empty(dev, n, usage=rw)

        # qkv_buf: parent sizes for full-attn (Q + 2*KV); GDN layers need
        # lin_conv_dim (K+K+V heads packed). Replace with the larger GDN size so
        # both full-attn and GDN layers can reuse the same buffer.
        self._sc["qkv_buf"] = mk(self._lin_conv_dim * 2)

        # Attention output gate: silu(gate)*attn_out before o_proj.
        # Not present in the parent; added only for models with attn_output_gate=True,
        # but always allocated so dispatch bindings are stable.
        self._sc["q_gate_buf"] = mk(T * Q * 2)

        # GDN linear-attention scratch buffers (sized from config, not hardcoded).
        self._sc.update({
            "qkv_conv":     mk(self._lin_conv_dim * 2),  # post-conv output
            "a_buf":        mk(self._lin_k_heads * 2),   # in_proj_a output [K_HEADS f16]
            "z_buf":        mk(self._lin_val_dim * 2),   # in_proj_z output
            "gdn_out":      mk(self._lin_val_dim * 2),   # GDN attn output
            "gated":        mk(self._lin_val_dim * 2),   # after norm+gate
            "b_buf":        mk(self._lin_k_heads * 2),   # in_proj_b output [K_HEADS f16]
            # Dummy binding-2 scales buffer for USE_QUANT=0 dispatches. Prevents
            # sc["normed"] from being silently aliased as a scales buffer, which
            # would corrupt output if a dispatch is promoted to USE_QUANT=1/2.
            "dummy_scales": mk(8),
        })

        if self._is_moe:
            H = self.hidden_size
            _moe_act_sz = max(self._moe_inter, self._moe_shared_inter, 1)
            self._moe_sc: dict[str, "WebGPUBuffer"] = {
                "router_out":   mk(self._moe_num_experts * 2),  # [N_E] f16 router logits
                "topk_idx":     mk(self._moe_k * 4),            # [K] u32 expert indices
                "topk_w":       mk(self._moe_k * 4),            # [K] f32 softmax weights
                "expert_gate":  mk(_moe_act_sz * 2),            # [max_inter] f16 gate proj
                "expert_up":    mk(_moe_act_sz * 2),            # [max_inter] f16 up proj
                "expert_act":   mk(_moe_act_sz * 2),            # [max_inter] f16 gate*up activated
                "expert_out":   mk(H * 2),                      # [hidden] f16 accumulated output
                "expert_tmp":   mk(H * 2),                      # [hidden] f16 per-expert temp
                "moe_w_buf":    mk(self._moe_k * 4),            # [K] f32 written before each encoder
                "dummy_scales": mk(8),                          # fallback scales binding
            }

    @property
    def _num_experts(self) -> int:
        """Alias expected by MixtralWebGPUModel._moe_ffn_layer."""
        return self._moe_num_experts

    @property
    def _top_k(self) -> int:
        """Alias expected by MixtralWebGPUModel._moe_ffn_layer."""
        return self._moe_k

    def _postprocess_weights(self) -> None:
        """Post-load weight transformations for full-attn layers:

        0. Detect GEMMA_NORM format: deviation (safetensors, mean≈0) vs absolute (MLX, mean≈1).
        1. Tile q_norm/k_norm via super()._postprocess_weights() (handles all layers gracefully).
        2. When attn_output_gate=True: split q_proj.weight [2*q_dim, hidden] into
           q_proj.weight [q_dim, hidden] (Q part) and q_gate_proj.weight [q_dim, hidden]
           (gate part). The gate is applied as silu(gate)*attn_out before o_proj.
        """
        from vllm_webgpu.webgpu.buffer import WebGPUBuffer
        dev = self.wgpu_device.wgpu_device
        rw = self._rw_flags()

        # Tile q_norm/k_norm weights via the parent implementation. Linear-attn layers
        # have no norm keys, so the parent loop skips them gracefully.
        super()._postprocess_weights()

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
                if buf is not None and len(buf.shape) >= 1 and buf.shape[0] == 2 * q_dim:
                    arr = buf.to_numpy().view(np.float16).reshape(self.num_q_heads, 2 * hd, buf.shape[1])
                    q_arr = np.ascontiguousarray(arr[:, :hd, :].reshape(q_dim, buf.shape[1]))
                    gate_arr = np.ascontiguousarray(arr[:, hd:, :].reshape(q_dim, buf.shape[1]))
                    self.weights[q_proj_key] = WebGPUBuffer.from_numpy(dev, q_arr, usage=rw)
                    gate_key = f"{p}.self_attn.q_gate_proj.weight"
                    self.weights[gate_key] = WebGPUBuffer.from_numpy(dev, gate_arr, usage=rw)

        self._rms_consts["GEMMA_NORM"] = self._gemma_norm

    def _alloc_lin_states(self) -> None:
        """Allocate GPU buffers for persistent GDN recurrent state.

        Called after load_weights. Each linear-attention layer gets:
          - SSM state:  [NUM_V_HEADS * K_DIM * V_DIM] f32 (zero-initialized)
          - Conv state: [(CONV_KERNEL-1) * CONV_DIM] f16 (zero-initialized)

        conv1d weight from MLX has shape [8192, 4, 1]; reshape the last dim
        so the shader reads [CONV_DIM, KERNEL] = [8192, 4] correctly.
        """
        from vllm_webgpu.webgpu.buffer import WebGPUBuffer

        dev = self.wgpu_device.wgpu_device
        rw = self._rw_flags()

        ssm_bytes  = self._lin_v_heads * self._lin_k_dim * self._lin_v_dim * 4   # f32
        conv_bytes = (self._lin_conv_kernel - 1) * self._lin_conv_dim * 2        # f16

        self._ssm_gpu  = [None] * self.num_layers
        self._conv_gpu = [None] * self.num_layers

        for i in range(self.num_layers):
            if self._is_full_attn(i):
                continue
            self._ssm_gpu[i]  = WebGPUBuffer.empty(dev, ssm_bytes,  usage=rw)
            self._conv_gpu[i] = WebGPUBuffer.empty(dev, conv_bytes, usage=rw)

            # conv1d weight from HuggingFace has shape [CONV_DIM, 1, KERNEL] (standard
            # PyTorch depthwise conv). The shader expects [CONV_DIM, KERNEL] (flat 2D).
            # Reshape by dropping the middle size-1 dim (groups/in_channels dimension).
            p = f"model.layers.{i}.linear_attn"
            conv_w_key = f"{p}.conv1d.weight"
            w = self.weights.get(conv_w_key)
            if w is not None and len(w.shape) == 3 and w.shape[1] == 1:
                # [CONV_DIM, 1, KERNEL] → [CONV_DIM, KERNEL]: drop the middle 1.
                arr = w.to_numpy().view(np.float16).reshape(w.shape[0], w.shape[2])
                self.weights[conv_w_key] = WebGPUBuffer.from_numpy(dev, arr, usage=rw)

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
                self.weights[key] = WebGPUBuffer.from_numpy(dev, f32_np, usage=rw)

    def load_weights(self, path: str) -> None:
        super().load_weights(path)
        self._postprocess_weights()
        self._alloc_lin_states()
        # Confirm MoE detection against actual weight keys.
        has_moe_gate = "model.layers.0.mlp.gate.weight" in self.weights
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
        for buf in self._ssm_gpu:
            if buf is not None:
                dev.queue.write_buffer(buf.buf, 0, bytes(buf.buf.size))
        for buf in self._conv_gpu:
            if buf is not None:
                dev.queue.write_buffer(buf.buf, 0, bytes(buf.buf.size))

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
        inter = self.intermediate_size
        p = f"model.layers.{layer_idx}.linear_attn"
        pp = f"model.layers.{layer_idx}"
        add_n = num_tokens * hidden

        h_names = ["h0", "h1", "h2"]
        residual = sc[h_names[(self._hstate + 1) % 3]]
        out = sc[h_names[(self._hstate + 2) % 3]]

        # GDN linear attention has no KV cache — state is in ssm_gpu/conv_gpu buffers.
        # Offsets into flat QKV buffer (f16 elements)
        q_base = 0                                  # Q: [NUM_K_HEADS × K_DIM]
        k_base = self._lin_k_heads * self._lin_k_dim  # K starts after Q
        v_base = self._lin_key_dim * 2                # V starts after Q+K

        _rms_h = self._rms_consts

        with self._batched_dispatch(label=f"L{layer_idx:02d}"):
            # normed_x is already the pre-normalized input from the caller.
            cd = self._lin_conv_dim
            vd = self._lin_val_dim
            kh = self._lin_k_heads
            kd = self._lin_k_dim
            vh = self._lin_v_heads
            vdh = self._lin_v_dim

            # Helper: resolve weight buffer, preferring the bf16-preserved variant when
            # GDN_BF16 is active. Returns (buffer, use_bf16_flag).
            def _gdn_w(key: str):
                if self._gdn_bf16:
                    bf16_buf = self.weights.get(key + "__bf16")
                    if bf16_buf is not None:
                        return bf16_buf, 1
                return self.weights[key], 0

            # 2. QKV projection: [hidden] → [conv_dim]
            _w_qkv, _bf16_qkv = _gdn_w(f"{p}.in_proj_qkv.weight")
            self._dispatch("matmul_quant",
                           [normed_x, _w_qkv, sc["dummy_scales"], sc["qkv_buf"]],
                           {"K": hidden, "N": cd, "USE_QUANT": 0, "USE_BF16": _bf16_qkv},
                           (cd, 1, 1))

            # 3. Causal conv step: updates conv_state in-place, writes qkv_conv
            conv_w = self.weights[f"{p}.conv1d.weight"]
            self._dispatch("causal_conv_step",
                           [sc["qkv_buf"], conv_w, self._conv_gpu[layer_idx], sc["qkv_conv"]],
                           {"CONV_DIM": cd, "KERNEL": self._lin_conv_kernel, "WG_SIZE": 256},
                           ((cd + 255) // 256, 1, 1))

            # 4. a projection: normed → [K_HEADS] (dt for decay)
            _w_a, _bf16_a = _gdn_w(f"{p}.in_proj_a.weight")
            self._dispatch("matmul_quant",
                           [normed_x, _w_a, sc["dummy_scales"], sc["a_buf"]],
                           {"K": hidden, "N": kh, "USE_QUANT": 0, "USE_BF16": _bf16_a},
                           (kh, 1, 1))

            # 5a. b projection: normed → [K_HEADS] (outer-product gate)
            _w_b, _bf16_b = _gdn_w(f"{p}.in_proj_b.weight")
            self._dispatch("matmul_quant",
                           [normed_x, _w_b, sc["dummy_scales"], sc["b_buf"]],
                           {"K": hidden, "N": kh, "USE_QUANT": 0, "USE_BF16": _bf16_b},
                           (kh, 1, 1))

            # 5b. z gate projection: normed → [val_dim]
            _w_z, _bf16_z = _gdn_w(f"{p}.in_proj_z.weight")
            self._dispatch("matmul_quant",
                           [normed_x, _w_z, sc["dummy_scales"], sc["z_buf"]],
                           {"K": hidden, "N": vd, "USE_QUANT": 0, "USE_BF16": _bf16_z},
                           (vd, 1, 1))

            # 6. GDN state update: updates ssm_state in-place, writes gdn_out
            self._dispatch("gdn_state_update",
                           [sc["qkv_conv"], sc["a_buf"], sc["b_buf"],
                            self.weights[f"{p}.A_log"],
                            self.weights[f"{p}.dt_bias"],
                            self._ssm_gpu[layer_idx], sc["gdn_out"]],
                           {"K_DIM": kd, "V_DIM": vdh,
                            "NUM_K_HEADS": kh, "NUM_V_HEADS": vh,
                            "Q_BASE": q_base, "K_BASE": k_base, "V_BASE": v_base},
                           (vh, 1, 1))

            # 7. Per-head RMSNorm + sigmoid gate → gated
            self._dispatch("linear_attn_norm_gate",
                           [sc["gdn_out"], self.weights[f"{p}.norm.weight"],
                            sc["z_buf"], sc["gated"]],
                           {"NUM_V_HEADS": vh, "V_DIM": vdh},
                           (vh, 1, 1))

            # 8. Output projection: [val_dim] → [hidden]
            _w_out, _bf16_out = _gdn_w(f"{p}.out_proj.weight")
            self._dispatch("matmul_quant",
                           [sc["gated"], _w_out, sc["dummy_scales"], sc["o_proj_out"]],
                           {"K": vd, "N": hidden, "USE_QUANT": 0, "USE_BF16": _bf16_out},
                           (hidden, 1, 1))

            # 9+10 fused: add(x, attn_out, residual) + rms_norm(residual, post_attn_norm) → ffn_normed
            self._dispatch("add_rms_norm",
                           [x_buf, sc["o_proj_out"],
                            self.weights[f"{pp}.post_attention_layernorm.weight"],
                            residual, sc["ffn_normed"]],
                           _rms_h, (num_tokens, 1, 1))

            # 10. FFN (MoE or dense)
            if self._is_moe:
                self._moe_ffn_layer(sc["ffn_normed"], layer_idx)
                ffn_out = self._moe_sc["expert_out"]
            else:
                gw_k = f"{pp}.mlp.gate_proj.weight"
                uw_k = f"{pp}.mlp.up_proj.weight"
                self._dispatch("fused_gate_act",
                               [sc["ffn_normed"], self.weights[gw_k], self.weights[uw_k],
                                sc["ffn_act"]],
                               {"K": hidden, "N": inter, "GELU": 0}, (inter, 1, 1))

                w_k = f"{pp}.mlp.down_proj.weight"
                self._dispatch("matmul_quant",
                               [sc["ffn_act"], self.weights[w_k],
                                sc["dummy_scales"], sc["ffn_out"]],
                               {"K": inter, "N": hidden, "USE_QUANT": 0},
                               (hidden, 1, 1))
                ffn_out = sc["ffn_out"]

            if layer_idx < self.num_layers - 1:
                next_w = self.weights[f"model.layers.{layer_idx+1}.input_layernorm.weight"]
                self._dispatch("add_rms_norm",
                               [residual, ffn_out, next_w, out, sc["normed"]],
                               _rms_h, (num_tokens, 1, 1))
            else:
                self._dispatch("add", [residual, ffn_out, out],
                               {"N": add_n}, ((add_n // 4 + 255) // 256, 1, 1))

        self._hstate = (self._hstate + 2) % 3
        return sc["normed"], out

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
        MixtralWebGPUModel._moe_ffn_layer(
            self, normed_x, layer_idx,
            bsm_prefix="mlp",
            router_subkey="gate",
            gate_key="gate_proj",
            up_key="up_proj",
            down_key="down_proj",
            expert_inter=self._moe_inter,
            shared_expert_prefix="shared_expert",
            shared_expert_inter=self._moe_shared_inter,
        )

    def _forward_moe(
        self,
        input_ids: np.ndarray,
        positions: np.ndarray,
        attn_metadata: object,
    ) -> np.ndarray:
        """MoE forward pass — manages command encoders explicitly.

        Unlike the standard forward(), this does NOT use an outer _batched_dispatch
        context manager. Instead, self._active_encoder is set manually so that
        layer methods behave re-entrantly (their inner _batched_dispatch calls are
        no-ops when _active_encoder is already set).

        _moe_ffn_layer() flushes and replaces _active_encoder mid-layer to
        handle the CPU readback required for expert index selection.
        """
        dev = self.wgpu_device.wgpu_device
        num_tokens = len(input_ids)
        hidden = self.hidden_size
        self._hstate = 0
        sc = self._sc

        if hasattr(attn_metadata, "block_tables") and len(attn_metadata.block_tables) > 1:
            raise RuntimeError("multi-sequence batching not supported in this build")

        if num_tokens > 1:
            raise NotImplementedError(
                "MoE prefill not supported; only decode is implemented")

        ids_buf, pos_buf, slot_map, bt_buf, x_buf, norm_out, logits_buf, ctx_len = \
            self._decode_setup(input_ids, positions, attn_metadata)
        vocab = self.vocab_size

        _rms_base = self._rms_consts

        # Start the first command encoder manually.
        # Layer methods see _active_encoder is not None → their _batched_dispatch
        # calls become re-entrant no-ops, recording into this encoder.
        self._active_encoder = dev.create_command_encoder()

        self._dispatch(
            "embedding_lookup",
            [self.weights["model.embed_tokens.weight"], ids_buf, x_buf],
            {"HIDDEN_DIM": hidden}, (num_tokens, 1, 1))
        self._dispatch(
            "rms_norm",
            [x_buf, self.weights["model.layers.0.input_layernorm.weight"], sc["normed"]],
            _rms_base, (num_tokens, 1, 1))

        normed_x = sc["normed"]
        for i in range(self.num_layers):
            if self._is_full_attn(i):
                normed_x, x_buf = self._transformer_layer(
                    i, normed_x, x_buf, pos_buf, slot_map, bt_buf, ctx_len, num_tokens)
            else:
                normed_x, x_buf = self._gdn_layer_gpu(i, normed_x, x_buf, num_tokens)

        self._dispatch(
            "rms_norm",
            [x_buf, self.weights["model.norm.weight"], norm_out],
            _rms_base, (num_tokens, 1, 1))

        self._decode_teardown(norm_out, logits_buf, vocab, greedy=True)

        # Submit the final encoder and release it.
        dev.queue.submit([self._active_encoder.finish()])
        self._active_encoder = None

        tok = self._read_sample_tok()
        return np.array([[tok]], dtype=np.int32)

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
        from vllm_webgpu.webgpu.buffer import WebGPUBuffer

        dev = self.wgpu_device.wgpu_device
        rw = self._rw_flags()

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
        bt_buf = WebGPUBuffer.from_numpy(dev, bt_arr, usage=rw)

        tok_ids_bufs: list = []
        tok_pos_bufs: list = []
        tok_slot_bufs: list = []
        for tc in range(num_tokens):
            tok_ids_bufs.append(WebGPUBuffer.from_numpy(
                dev, input_ids[tc:tc+1].astype(np.uint32), usage=rw))
            tok_pos_bufs.append(WebGPUBuffer.from_numpy(
                dev, positions[tc:tc+1].astype(np.uint32), usage=rw))
            tok_slot_bufs.append(WebGPUBuffer.from_numpy(
                dev, np.array([attn_metadata.slot_mapping[tc]], dtype=np.uint32),
                usage=rw))

        self._ensure_sample_buf(vocab)

        for chunk_start in range(0, num_tokens, _CHUNK):
            chunk_end = min(chunk_start + _CHUNK, num_tokens)

            # Open one encoder for this chunk.
            # Layer method _batched_dispatch calls become re-entrant no-ops
            # because _active_encoder is already set, so all dispatches land here.
            self._active_encoder = dev.create_command_encoder()

            for tc in range(chunk_start, chunk_end):
                ctx_t = int(attn_metadata.slot_mapping[tc]) + 1
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
                    if self._is_full_attn(i):
                        normed_x, x_buf = self._transformer_layer(
                            i, normed_x, x_buf, pos_buf, slot_map, bt_buf,
                            ctx_t, 1)
                    else:
                        normed_x, x_buf = self._gdn_layer_gpu(
                            i, normed_x, x_buf, 1)

                # For the last token: final norm, LM head, argmax, staging copy.
                if tc == num_tokens - 1:
                    self._dispatch("rms_norm",
                                   [x_buf, self.weights["model.norm.weight"],
                                    pre["norm_out"]],
                                   _rms_base, (1, 1, 1))
                    self._dispatch("matmul_quant",
                                   [pre["norm_out"], self._lm_head_weight,
                                    pre["norm_out"], pre["logits"]],
                                   {"K": hidden, "N": vocab,
                                    "USE_QUANT": 0, "SPLIT_K": 0},
                                   ((vocab + 255) // 256, 1, 1))
                    self._dispatch("argmax_f16",
                                   [pre["logits"], self._gpu_sample_tok],
                                   {"N": vocab}, (1, 1, 1))
                    self._copy_sample_to_staging()

            # Submit all dispatches for this chunk.
            dev.queue.submit([self._active_encoder.finish()])
            self._active_encoder = None

        self._last_logit_buf = pre["logits"]
        self._last_vocab = vocab
        tok = self._read_sample_tok()
        return np.array([[tok]], dtype=np.int32)

    def forward(
        self,
        input_ids: np.ndarray,
        positions: np.ndarray,
        attn_metadata: object,
    ) -> np.ndarray:
        if self._is_moe:
            return self._forward_moe(input_ids, positions, attn_metadata)

        dev = self.wgpu_device.wgpu_device
        num_tokens = len(input_ids)
        hidden = self.hidden_size
        self._hstate = 0

        if hasattr(attn_metadata, "block_tables") and len(attn_metadata.block_tables) > 1:
            raise RuntimeError("multi-sequence batching not supported in this build")

        ctx_len = int(attn_metadata.max_decode_seq_len
                      if attn_metadata.max_decode_seq_len is not None
                      else num_tokens)
        if ctx_len <= 0:
            ctx_len = num_tokens

        # Prefill (num_tokens > 1): process tokens sequentially but batch CHUNK
        # tokens per command encoder to avoid Metal's per-command-buffer GPU timeout.
        # GDN SSM state is updated in-place on the GPU; sequential order is preserved
        # because dispatches within an encoder execute in submission order.
        if num_tokens > 1:
            return self._prefill_chunked_forward(
                input_ids, positions, attn_metadata, num_tokens)

        pre = self._pre
        dev.queue.write_buffer(pre["ids"].buf, 0, input_ids.astype(np.uint32).tobytes())
        dev.queue.write_buffer(pre["pos"].buf, 0, positions.astype(np.uint32).tobytes())
        dev.queue.write_buffer(
            pre["slot_map"].buf, 0,
            np.array(attn_metadata.slot_mapping, dtype=np.uint32).tobytes())
        bt_arr = self._bt_arr(attn_metadata)
        dev.queue.write_buffer(pre["bt"].buf, 0, bt_arr.tobytes())

        ids_buf    = pre["ids"]
        pos_buf    = pre["pos"]
        slot_map   = pre["slot_map"]
        bt_buf     = pre["bt"]
        x_buf      = pre["x"]
        norm_out   = pre["norm_out"]
        logits_buf = pre["logits"]
        vocab = self.vocab_size

        # Single outer encoder for the entire forward pass — one queue.submit().
        # Inner _batched_dispatch() calls in layer methods are re-entrant no-ops
        # when profiling=False (default), recording all dispatches here.
        _rms_base = self._rms_consts
        sc = self._sc

        with self._batched_dispatch():
            self._dispatch("embedding_lookup",
                           [self.weights["model.embed_tokens.weight"], ids_buf, x_buf],
                           {"HIDDEN_DIM": hidden}, (num_tokens, 1, 1))

            # Initial pre-norm for layer 0; subsequent pre-norms are fused into each
            # layer's final add_rms_norm dispatch.
            self._dispatch("rms_norm",
                           [x_buf, self.weights["model.layers.0.input_layernorm.weight"],
                            sc["normed"]],
                           _rms_base, (num_tokens, 1, 1))

            normed_x = sc["normed"]
            for i in range(self.num_layers):
                if self._is_full_attn(i):
                    normed_x, x_buf = self._transformer_layer(
                        i, normed_x, x_buf, pos_buf, slot_map, bt_buf, ctx_len, num_tokens)
                else:
                    normed_x, x_buf = self._gdn_layer_gpu(
                        i, normed_x, x_buf, num_tokens)

            self._dispatch("rms_norm",
                           [x_buf, self.weights["model.norm.weight"], norm_out],
                           _rms_base, (num_tokens, 1, 1))

            # vocab_size exceeds the 65535 workgroup-per-dimension limit, so the split-K
            # path is unusable. Force SPLIT_K=0 (row-per-thread) with ceil(vocab/256) WGs.
            self._dispatch("matmul_quant",
                           [norm_out, self._lm_head_weight, norm_out, logits_buf],
                           {"K": hidden, "N": vocab, "USE_QUANT": 0, "SPLIT_K": 0},
                           ((vocab + 255) // 256, 1, 1))
            # GPU argmax inside the same encoder — 4-byte readback.
            self._dispatch("argmax_f16", [logits_buf, self._ensure_sample_buf(vocab)],
                           {"N": vocab}, (1, 1, 1))
            self._copy_sample_to_staging()

        self._last_logit_buf = logits_buf
        self._last_vocab     = vocab
        tok = self._read_sample_tok()
        return np.array([[tok]], dtype=np.int32)

    def _ffn_dispatch(
        self,
        normed_x: "WebGPUBuffer",
        layer_idx: int,
        num_tokens: int,
    ) -> "WebGPUBuffer":
        """Route to MoE or dense FFN based on model config."""
        if self._is_moe:
            self._moe_ffn_layer(normed_x, layer_idx)
            return self._moe_sc["expert_out"]
        return super()._ffn_dispatch(normed_x, layer_idx, num_tokens)

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
        """Standard full-attention transformer layer. Receives pre-normed input.

        Adds Qwen3.5-specific behaviour on top of the Llama scaffold:
          - attn_output_gate: sigmoid-gated attention output before o_proj.
          - GEMMA_NORM, ROTARY_DIM, INTERLEAVED passed to per-head norm+rope shaders.
          - MoE FFN routed through _ffn_dispatch() override.
        """
        sc = self._sc
        hidden = self.hidden_size
        p = f"model.layers.{layer_idx}"
        q_dim = self.num_q_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        # Per-weight quantization detection: Q4_K (type 12) → GPU block decoder.
        _uq = self._uq_for_key

        h_names = ["h0", "h1", "h2"]
        residual = sc[h_names[(self._hstate + 1) % 3]]
        out = sc[h_names[(self._hstate + 2) % 3]]
        add_n = num_tokens * hidden

        _rms_h = self._rms_consts
        k_cache, v_cache = self.kv_pool[layer_idx]

        with self._batched_dispatch():
            # normed_x is already the pre-normalized input from the caller.
            for out_buf, proj, dim in [(sc["q_buf"], "q_proj", q_dim),
                                       (sc["k_buf"], "k_proj", kv_dim),
                                       (sc["v_buf"], "v_proj", kv_dim)]:
                w_key = f"{p}.self_attn.{proj}.weight"
                uq = _uq(w_key)
                qi = self._quant_extra(f"{p}.self_attn.{proj}", uq)
                self._dispatch("matmul_quant",
                               [normed_x, self.weights[w_key],
                                self._scales_buf(w_key, uq, normed_x), out_buf],
                               {"K": hidden, "N": dim, "USE_QUANT": uq,
                                **self._split_k_extra(uq), **qi},
                               _gemv_wg(dim, uq))

            # When attn_output_gate=True, q_proj.weight was split at load time.
            # Compute the gate projection: normed_x → q_gate_buf [q_dim f16].
            # The gate is applied as sigmoid(gate)*attn_out before o_proj (step below).
            if self._attn_output_gate:
                gate_wk = f"{p}.self_attn.q_gate_proj.weight"
                if self.weights.get(gate_wk) is not None:
                    uq_gate = _uq(gate_wk)
                    qi_gate = self._quant_extra(f"{p}.self_attn.q_gate_proj", uq_gate)
                    self._dispatch("matmul_quant",
                                   [normed_x, self.weights[gate_wk],
                                    self._scales_buf(gate_wk, uq_gate, normed_x),
                                    sc["q_gate_buf"]],
                                   {"K": hidden, "N": q_dim, "USE_QUANT": uq_gate,
                                    **self._split_k_extra(uq_gate), **qi_gate},
                                   _gemv_wg(q_dim, uq_gate))

            # Fused per-head RMSNorm + RoPE for Q and K.
            # When both norm weights exist, fuse into one dispatch using K_SEPARATE=1.
            _q_norm_w = self.weights.get(f"{p}.self_attn.q_norm.weight")
            _k_norm_w = self.weights.get(f"{p}.self_attn.k_norm.weight")
            _freq_buf = self._rope_freq_buf
            _rope_base = {"ROPE_BASE": float(self.rope_theta),
                          "LN_ROPE_BASE": self._ln_rope_theta,
                          "USE_FREQ_BUF": int(self._use_freq_buf),
                          "ATTN_SCALE": self._yarn_mscale}
            if _q_norm_w is not None and _k_norm_w is not None:
                # Binding 7 (inv_freq_buf): always provided.
                self._dispatch("fused_qk_norm_rope",
                               [sc["q_buf"], _q_norm_w, _k_norm_w, pos_buf,
                                sc["q_rope"], sc["k_rope"], sc["k_buf"], _freq_buf],
                               {**_rope_base,
                                "HEAD_DIM": self.head_dim,
                                "NUM_Q_HEADS": self.num_q_heads,
                                "NUM_KV_HEADS": self.num_kv_heads,
                                "HAS_WEIGHT": 1,
                                "GEMMA_NORM": self._gemma_norm,
                                "ROTARY_DIM": self._rotary_dim,
                                "INTERLEAVED": self._rope_interleaved,
                                "INPUT_OFFSET_K": 0,
                                "K_SEPARATE": 1},
                               (self.num_q_heads + self.num_kv_heads, num_tokens, 1))
            else:
                for src, dst, n_heads, w_key in [
                    (sc["q_buf"], sc["q_rope"], self.num_q_heads, f"{p}.self_attn.q_norm.weight"),
                    (sc["k_buf"], sc["k_rope"], self.num_kv_heads, f"{p}.self_attn.k_norm.weight"),
                ]:
                    norm_w = self.weights.get(w_key)
                    if norm_w is not None:
                        # Binding 4 (inv_freq_buf): always provided.
                        self._dispatch("fused_per_head_norm_rope",
                                       [src, norm_w, pos_buf, dst, _freq_buf],
                                       {**_rope_base, "HEAD_DIM": self.head_dim,
                                        "NUM_HEADS": n_heads, "HAS_WEIGHT": 1,
                                        "GEMMA_NORM": self._gemma_norm,
                                        "ROTARY_DIM": self._rotary_dim,
                                        "INTERLEAVED": self._rope_interleaved},
                                       (n_heads, num_tokens, 1))
                    else:
                        # Binding 3 (inv_freq_buf): always provided.
                        self._dispatch("rope", [src, pos_buf, dst, _freq_buf],
                                       {**_rope_base, "HEAD_DIM": self.head_dim,
                                        "NUM_HEADS": n_heads},
                                       (num_tokens, n_heads, 1))

            # Fused K+V cache store
            self._dispatch("kv_cache_store_both",
                           [sc["k_rope"], k_cache, sc["v_buf"], v_cache, slot_map],
                           {"BLOCK_SIZE": self.block_size, "NUM_KV_HEADS": self.num_kv_heads,
                            "HEAD_DIM": self.head_dim},
                           (num_tokens, self.num_kv_heads, 1))

            # Always use flash_attn_decode for single-token decode.
            # The 65535 limit applied to attn_score's dispatch dimension; flash_attn_decode
            # loops internally and has no dispatch dimension limit.
            self._dispatch("flash_attn_decode",
                           [sc["q_rope"], k_cache, v_cache, bt_buf, sc["attn_out"]],
                           {"BLOCK_SIZE": self.block_size, "NUM_Q_HEADS": self.num_q_heads,
                            "NUM_KV_HEADS": self.num_kv_heads, "HEAD_DIM": self.head_dim,
                            "CTX_LEN": self._effective_ctx_len(ctx_len)},
                           (self.num_q_heads, 1, 1))

            # Apply attention output gate if enabled: gated = sigmoid(gate) * attn_out.
            # q_buf is free at this point (written in q_proj, last read in RoPE), so
            # reuse it as the output buffer for the gated result.
            if self._attn_output_gate and self.weights.get(f"{p}.self_attn.q_gate_proj.weight") is not None:
                gate_n = num_tokens * q_dim
                self._dispatch("sigmoid_gate",
                               [sc["q_gate_buf"], sc["attn_out"], sc["q_buf"]],
                               {"N": gate_n}, ((gate_n // 4 + 255) // 256, 1, 1))
                o_proj_in = sc["q_buf"]
            else:
                o_proj_in = sc["attn_out"]

            w_key = f"{p}.self_attn.o_proj.weight"
            uq = _uq(w_key)
            qi_o = self._quant_extra(f"{p}.self_attn.o_proj", uq)
            self._dispatch("matmul_quant",
                           [o_proj_in, self.weights[w_key],
                            self._scales_buf(w_key, uq, o_proj_in), sc["o_proj_out"]],
                           {"K": q_dim, "N": hidden, "USE_QUANT": uq,
                            **self._split_k_extra(uq), **qi_o},
                           _gemv_wg(hidden, uq))

            # Fused: add(x, attn_out, residual) + rms_norm(residual, post_attn_w) → ffn_normed
            self._dispatch("add_rms_norm",
                           [x_buf, sc["o_proj_out"],
                            self.weights[f"{p}.post_attention_layernorm.weight"],
                            residual, sc["ffn_normed"]],
                           _rms_h, (num_tokens, 1, 1))

            ffn_out = self._ffn_dispatch(sc["ffn_normed"], layer_idx, num_tokens)

            if layer_idx < self.num_layers - 1:
                next_w = self.weights[f"model.layers.{layer_idx+1}.input_layernorm.weight"]
                self._dispatch("add_rms_norm",
                               [residual, ffn_out, next_w, out, sc["normed"]],
                               _rms_h, (num_tokens, 1, 1))
            else:
                self._dispatch("add", [residual, ffn_out, out],
                               {"N": add_n}, ((add_n // 4 + 255) // 256, 1, 1))

        self._hstate = (self._hstate + 2) % 3
        return sc["normed"], out


