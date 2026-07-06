from __future__ import annotations
import logging
from typing import TYPE_CHECKING

import numpy as np

from vllm_webgpu.models.base import BaseWebGPUModel

if TYPE_CHECKING:
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.device import WebGPUDevice
    from vllm_webgpu.webgpu.pipeline import PipelineCache

logger = logging.getLogger(__name__)


def _gemv_wg(N: int, uq: int) -> tuple:
    """Workgroup count for matmul_quant dispatch.

    SPLIT_K=1 (shader default): one workgroup per output row when USE_QUANT=0.
    Quantized paths (USE_QUANT=1/2) use row-per-thread: ceil(N/256) workgroups.
    """
    if uq == 0:
        return (N, 1, 1)
    return ((N + 255) // 256, 1, 1)


# Qwen3.5-9B fixed architecture constants
_FULL_ATTN_INTERVAL = 4
_LIN_K_HEADS = 16
_LIN_V_HEADS = 32
_LIN_K_DIM = 128
_LIN_V_DIM = 128
_LIN_KEY_DIM = _LIN_K_HEADS * _LIN_K_DIM    # 2048
_LIN_VAL_DIM = _LIN_V_HEADS * _LIN_V_DIM    # 4096
_LIN_CONV_DIM = _LIN_KEY_DIM + _LIN_KEY_DIM + _LIN_VAL_DIM  # 8192 (QKV packed)
_LIN_CONV_KERNEL = 4


def _is_full_attn(layer_idx: int, layer_types: list | None = None) -> bool:
    if layer_types is not None:
        return layer_types[layer_idx] == "full_attention"
    return (layer_idx + 1) % _FULL_ATTN_INTERVAL == 0


class Qwen35WebGPUModel(BaseWebGPUModel):
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
        super().__init__(model_config, wgpu_device, pipeline_cache)
        self.num_layers: int = model_config.num_hidden_layers
        self.num_q_heads: int = model_config.num_attention_heads
        self.num_kv_heads: int = model_config.num_key_value_heads
        self.hidden_size: int = model_config.hidden_size
        self.intermediate_size: int = model_config.intermediate_size
        self.vocab_size: int = model_config.vocab_size
        self.head_dim: int = getattr(model_config, "head_dim", self.hidden_size // self.num_q_heads)
        self.rope_theta: float = getattr(model_config, "rope_theta", 10000.0)
        self._layer_types: list | None = getattr(model_config, "layer_types", None)
        # Partial RoPE: some models only rotate a fraction of head dimensions.
        # partial_rotary_factor=0.25 → rotary_dim = head_dim * 0.25.
        _prf = getattr(model_config, "partial_rotary_factor", 1.0) or 1.0
        self._rotary_dim: int = max(2, int(self.head_dim * _prf))
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
            model_config, "shared_expert_intermediate_size", self.intermediate_size)
        # _is_moe is set from config here and may be overridden in load_weights
        # once we can verify against actual weight keys.
        self._is_moe: bool = self._moe_num_experts > 0 and self._moe_k > 0

        from vllm_webgpu.config import get_config
        self.block_size: int = get_config().block_size

        for name, val in [("hidden_size", self.hidden_size),
                          ("intermediate_size", self.intermediate_size),
                          ("head_dim", self.head_dim)]:
            if val % 2 != 0:
                raise ValueError(f"{name}={val} must be even for f16 GEMV")
        for name, val in [("hidden_size", self.hidden_size),
                          ("intermediate_size", self.intermediate_size)]:
            if val % 4 != 0:
                raise ValueError(f"{name}={val} must be divisible by 4 for vec4<f16> shaders")

        max_ctx = getattr(model_config, "max_position_embeddings", 8192)
        self._init_scratch_buffers(max_ctx)

        # Persistent GPU buffers for recurrent state (allocated after load_weights).
        # SSM state:  [NUM_V_HEADS, K_DIM, V_DIM] f32 = 2MB per linear-attn layer
        # Conv state: [CONV_KERNEL-1, CONV_DIM] f16 = 49KB per linear-attn layer
        self._ssm_gpu: list = []   # one WebGPUBuffer per layer (or None for full-attn)
        self._conv_gpu: list = []  # one WebGPUBuffer per layer
        # NOTE: profiling=True is incompatible with MoE forward (per-layer submit breaks
        # _batched_dispatch encoder management). Set profiling=False before forward().

    def _is_full_attn(self, i: int) -> bool:
        return _is_full_attn(i, self._layer_types)

    def _init_scratch_buffers(self, max_ctx: int) -> None:
        import wgpu as wgpu_lib
        from vllm_webgpu.webgpu.buffer import WebGPUBuffer

        dev = self.wgpu_device.wgpu_device
        rw = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST
        T = 1
        H = self.hidden_size
        I = self.intermediate_size
        Q = self.num_q_heads * self.head_dim
        KV = self.num_kv_heads * self.head_dim
        NQ = self.num_q_heads

        def mk(n: int) -> "WebGPUBuffer":
            return WebGPUBuffer.empty(dev, n, usage=rw)

        # Pre-allocated per-step buffers (reused every decode via write_buffer).
        V = self.vocab_size
        self._pre: dict[str, "WebGPUBuffer"] = {
            "ids":      mk(T * 4),
            "pos":      mk(T * 4),
            "slot_map": mk(T * 4),
            "bt":       mk(512 * 4),
            "x":        mk(T * H * 2),
            "norm_out": mk(T * H * 2),
            "logits":   mk(T * V * 2),
        }

        self._sc: dict[str, "WebGPUBuffer"] = {
            "normed":     mk(T * H * 2),
            "q_buf":      mk(T * Q * 2),
            "k_buf":      mk(T * KV * 2),
            "v_buf":      mk(T * KV * 2),
            "q_rope":     mk(T * Q * 2),
            "k_rope":     mk(T * KV * 2),
            "scores_buf": mk(NQ * max_ctx * 2),
            "sm_buf":     mk(NQ * max_ctx * 2),
            "attn_out":   mk(T * Q * 2),
            "q_gate_buf": mk(T * Q * 2),  # attention output gate (silu(gate)*attn_out)
            "o_proj_out": mk(T * H * 2),
            "ffn_normed": mk(T * H * 2),
            "gate_buf":    mk(T * I * 2),
            "up_buf":   mk(T * I * 2),
            "ffn_act":  mk(T * I * 2),
            "ffn_out":  mk(T * H * 2),
            "h0":         mk(T * H * 2),
            "h1":         mk(T * H * 2),
            "h2":         mk(T * H * 2),
            # GDN linear-attention scratch buffers (sized from config, not hardcoded)
            "qkv_buf":    mk(self._lin_conv_dim * 2),         # in_proj_qkv output
            "qkv_conv":   mk(self._lin_conv_dim * 2),         # post-conv output
            "a_buf":      mk(self._lin_k_heads * 2),          # in_proj_a output [K_HEADS f16]
            "z_buf":      mk(self._lin_val_dim * 2),          # in_proj_z output
            "gdn_out":    mk(self._lin_val_dim * 2),          # GDN attn output
            "gated":      mk(self._lin_val_dim * 2),          # after norm+gate
            "b_buf":      mk(self._lin_k_heads * 2),          # in_proj_b output [K_HEADS f16]
            # Dummy binding-2 scales buffer for USE_QUANT=0 dispatches.
            # Prevents sc["normed"] from being silently aliased as a scales buffer,
            # which would corrupt output if a dispatch is promoted to USE_QUANT=1/2.
            "dummy_scales": mk(8),
        }

        if self._is_moe and self._moe_num_experts > 0 and self._moe_k > 0:
            _moe_act_sz = max(self._moe_inter, self._moe_shared_inter, 1)
            self._sc.update({
                "moe_router_out":  mk(self._moe_num_experts * 2),  # [N_E] f16 router logits
                "moe_topk_idx":    mk(self._moe_k * 4),             # [K] u32 expert indices
                "moe_topk_w":      mk(self._moe_k * 4),             # [K] f32 softmax weights
                "moe_expert_act":  mk(_moe_act_sz * 2),             # [max_inter] f16 gate_act output
                "moe_expert_down": mk(H * 2),                       # [hidden] f16 down_proj output
                "moe_w_buf":       mk(self._moe_k * 4),             # [K] f32 written before each encoder
            })

        self._hstate: int = 0

    def _uq_weight(self, key: str) -> int:
        """Return USE_QUANT value for a weight key (same logic as llama.py)."""
        w = self.weights.get(key)
        if w is not None:
            dtype = getattr(w, "dtype", "f16")
            qmeta = self.weights.get("__quant_meta__", {})
            meta = qmeta.get(key[:-7], {}) if isinstance(qmeta, dict) else {}
            fmt = meta.get("fmt", "")
            if dtype == "i32":
                return 4 if fmt == "awq_sym" else 3
            if dtype == "u8":
                if fmt == "nvfp4_gpu": return 6
                if fmt == "int8_gpu":  return 7
                return 5
        quant_types = self.weights.get("__quant_types__", {})
        _qt = quant_types if isinstance(quant_types, dict) else {}
        if _qt.get(key, 0) == 12:
            return 2
        if self.weights.get(key[:-7] + ".scales") is not None:
            return 1
        return 0

    def _scales_buf(self, w_key: str, uq: int, fallback: "WebGPUBuffer") -> "WebGPUBuffer":
        """Return the scales buffer for any quant format."""
        if uq in (3, 4, 5, 6):
            return self.weights.get(w_key + ".scales", fallback)
        return self.weights.get(w_key[:-7] + ".scales", fallback)

    def _postprocess_weights(self) -> None:
        """Post-load weight transformations for full-attn layers:

        1. Tile q_norm/k_norm from (head_dim,) to (num_heads * head_dim,).
        2. When attn_output_gate=True: split q_proj.weight [2*q_dim, hidden] into
           q_proj.weight [q_dim, hidden] (Q part) and q_gate_proj.weight [q_dim, hidden]
           (gate part). The gate is applied as silu(gate)*attn_out before o_proj.
        """
        import wgpu as wgpu_lib
        from vllm_webgpu.webgpu.buffer import WebGPUBuffer
        dev = self.wgpu_device.wgpu_device
        rw = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST

        for i in range(self.num_layers):
            if not self._is_full_attn(i):
                continue
            p = f"model.layers.{i}"
            for norm_key, num_heads in [
                (f"{p}.self_attn.q_norm.weight", self.num_q_heads),
                (f"{p}.self_attn.k_norm.weight", self.num_kv_heads),
            ]:
                buf = self.weights.get(norm_key)
                if buf is None:
                    continue
                expected = (num_heads * self.head_dim,)
                if buf.shape == expected:
                    continue
                if buf.shape == (self.head_dim,):
                    w_np = buf.to_numpy().view(np.float16)
                    tiled = np.tile(w_np, num_heads)
                    self.weights[norm_key] = WebGPUBuffer.from_numpy(dev, tiled, usage=rw)
                else:
                    raise ValueError(
                        f"{norm_key}: unexpected shape {buf.shape}, "
                        f"expected {expected} or ({self.head_dim},)"
                    )

            # Split fused Q+gate weight when attn_output_gate=True.
            # q_proj.weight has shape [2*q_dim, hidden]: rows 0..q_dim-1 = Q,
            # rows q_dim..2*q_dim-1 = attention output gate.
            if self._attn_output_gate:
                q_dim = self.num_q_heads * self.head_dim
                q_proj_key = f"{p}.self_attn.q_proj.weight"
                buf = self.weights.get(q_proj_key)
                if buf is not None and len(buf.shape) >= 1 and buf.shape[0] == 2 * q_dim:
                    arr = buf.to_numpy().view(np.float16).reshape(buf.shape)
                    q_arr = np.ascontiguousarray(arr[:q_dim, :])
                    gate_arr = np.ascontiguousarray(arr[q_dim:, :])
                    self.weights[q_proj_key] = WebGPUBuffer.from_numpy(dev, q_arr, usage=rw)
                    gate_key = f"{p}.self_attn.q_gate_proj.weight"
                    self.weights[gate_key] = WebGPUBuffer.from_numpy(dev, gate_arr, usage=rw)
                    logger.debug("Split q_proj gate for layer %d: q=[%d,%d] gate=[%d,%d]",
                                 i, q_arr.shape[0], q_arr.shape[1],
                                 gate_arr.shape[0], gate_arr.shape[1])

    def _alloc_lin_states(self) -> None:
        """Allocate GPU buffers for persistent GDN recurrent state.

        Called after load_weights. Each linear-attention layer gets:
          - SSM state:  [NUM_V_HEADS * K_DIM * V_DIM] f32 (zero-initialized)
          - Conv state: [(CONV_KERNEL-1) * CONV_DIM] f16 (zero-initialized)

        conv1d weight from MLX has shape [8192, 4, 1]; reshape the last dim
        so the shader reads [CONV_DIM, KERNEL] = [8192, 4] correctly.
        """
        import wgpu as wgpu_lib
        from vllm_webgpu.webgpu.buffer import WebGPUBuffer

        dev = self.wgpu_device.wgpu_device
        rw = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST

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
        import wgpu as wgpu_lib
        from vllm_webgpu.webgpu.buffer import WebGPUBuffer
        dev = self.wgpu_device.wgpu_device
        rw = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST
        ssm_bytes  = self._lin_v_heads * self._lin_k_dim * self._lin_v_dim * 4
        conv_bytes = (self._lin_conv_kernel - 1) * self._lin_conv_dim * 2
        for i in range(self.num_layers):
            if self._is_full_attn(i):
                continue
            self._ssm_gpu[i]  = WebGPUBuffer.empty(dev, ssm_bytes,  usage=rw)
            self._conv_gpu[i] = WebGPUBuffer.empty(dev, conv_bytes, usage=rw)

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
        import math
        sc = self._sc
        hidden = self.hidden_size
        inter = self.intermediate_size
        p = f"model.layers.{layer_idx}.linear_attn"
        pp = f"model.layers.{layer_idx}"
        add_n = num_tokens * hidden
        gelu_n = num_tokens * inter

        h_names = ["h0", "h1", "h2"]
        residual = sc[h_names[(self._hstate + 1) % 3]]
        out = sc[h_names[(self._hstate + 2) % 3]]

        # GDN linear attention has no KV cache — state is in ssm_gpu/conv_gpu buffers.
        # Offsets into flat QKV buffer (f16 elements)
        q_base = 0                                  # Q: [NUM_K_HEADS × K_DIM]
        k_base = self._lin_k_heads * self._lin_k_dim  # K starts after Q
        v_base = self._lin_key_dim * 2                # V starts after Q+K

        _rms_h = {"HIDDEN_DIM": hidden,
                  "VALS_PER_THREAD": min((hidden + 255) // 256, 16) if hidden <= 4096 else 0}

        with self._batched_dispatch(label=f"L{layer_idx:02d}"):
            # normed_x is already the pre-normalized input from the caller.
            cd = self._lin_conv_dim
            vd = self._lin_val_dim
            kh = self._lin_k_heads
            kd = self._lin_k_dim
            vh = self._lin_v_heads
            vdh = self._lin_v_dim

            # 2. QKV projection: [hidden] → [conv_dim]
            self._dispatch("matmul_quant",
                           [normed_x, self.weights[f"{p}.in_proj_qkv.weight"],
                            sc["dummy_scales"], sc["qkv_buf"]],
                           {"K": hidden, "N": cd, "USE_QUANT": 0},
                           (cd, 1, 1))

            # 3. Causal conv step: updates conv_state in-place, writes qkv_conv
            conv_w = self.weights.get(f"{p}.conv1d.weight", normed_x)
            self._dispatch("causal_conv_step",
                           [sc["qkv_buf"], conv_w, self._conv_gpu[layer_idx], sc["qkv_conv"]],
                           {"CONV_DIM": cd, "KERNEL": self._lin_conv_kernel, "WG_SIZE": 256},
                           ((cd + 255) // 256, 1, 1))

            # 4. a projection: normed → [K_HEADS] (dt for decay)
            self._dispatch("matmul_quant",
                           [normed_x, self.weights[f"{p}.in_proj_a.weight"],
                            sc["dummy_scales"], sc["a_buf"]],
                           {"K": hidden, "N": kh, "USE_QUANT": 0},
                           (kh, 1, 1))

            # 5a. b projection: normed → [K_HEADS] (outer-product gate)
            self._dispatch("matmul_quant",
                           [normed_x, self.weights[f"{p}.in_proj_b.weight"],
                            sc["dummy_scales"], sc["b_buf"]],
                           {"K": hidden, "N": kh, "USE_QUANT": 0},
                           (kh, 1, 1))

            # 5b. z gate projection: normed → [val_dim]
            self._dispatch("matmul_quant",
                           [normed_x, self.weights[f"{p}.in_proj_z.weight"],
                            sc["dummy_scales"], sc["z_buf"]],
                           {"K": hidden, "N": vd, "USE_QUANT": 0},
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
            self._dispatch("matmul_quant",
                           [sc["gated"], self.weights[f"{p}.out_proj.weight"],
                            sc["dummy_scales"], sc["o_proj_out"]],
                           {"K": vd, "N": hidden, "USE_QUANT": 0},
                           (hidden, 1, 1))

            # 9+10 fused: add(x, attn_out, residual) + rms_norm(residual, post_attn_norm) → ffn_normed
            self._dispatch("add_rms_norm",
                           [x_buf, sc["o_proj_out"],
                            self.weights[f"{pp}.post_attention_layernorm.weight"],
                            residual, sc["ffn_normed"]],
                           _rms_h, (num_tokens, 1, 1))

            # 10. FFN (MoE or dense)
            if self._is_moe:
                self._moe_ffn_dispatch(layer_idx, sc["ffn_normed"], sc["ffn_out"],
                                       num_tokens)
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
                                sc["ffn_act"], sc["ffn_out"]],
                               {"K": inter, "N": hidden, "USE_QUANT": 0},
                               (hidden, 1, 1))

            if layer_idx < self.num_layers - 1:
                next_w = self.weights[f"model.layers.{layer_idx+1}.input_layernorm.weight"]
                self._dispatch("add_rms_norm",
                               [residual, sc["ffn_out"], next_w, out, sc["normed"]],
                               _rms_h, (num_tokens, 1, 1))
            else:
                self._dispatch("add", [residual, sc["ffn_out"], out],
                               {"N": add_n}, ((add_n // 4 + 255) // 256, 1, 1))

        self._hstate = (self._hstate + 2) % 3
        return sc["normed"], out

    def _moe_ffn_dispatch(
        self,
        layer_idx: int,
        normed_x: "WebGPUBuffer",
        out_buf: "WebGPUBuffer",
        num_tokens: int,
    ) -> None:
        """MoE FFN: router → top-K selection (with submit/sync) → expert dispatch.

        Two-phase approach required because expert selection is data-dependent:
          Phase A: dispatch router + topk_sort into the current encoder, then flush
                   and sync so the CPU can read the selected expert indices.
          Phase B: create a fresh encoder, dispatch the shared expert and the K
                   selected experts with weighted accumulation into out_buf.

        Caller's self._active_encoder is replaced with the new Phase B encoder on
        return. Subsequent dispatches in the same layer method (add_rms_norm for
        the residual connection) land in that new encoder, which is correct.
        """
        import struct
        import wgpu as wgpu_lib

        dev = self.wgpu_device.wgpu_device
        sc = self._sc
        p = f"model.layers.{layer_idx}.mlp"
        hidden = self.hidden_size
        moe_inter = self._moe_inter
        shared_inter = self._moe_shared_inter
        N_E = self._moe_num_experts
        K = self._moe_k

        # ── Phase A: router + top-K (into current encoder) ───────────────────
        # Router: normed_x [hidden] → moe_router_out [N_E] logits
        rw_k = f"{p}.gate.weight"
        uq_r = self._uq_weight(rw_k)
        qi_r = self._quant_extra(f"{p}.gate", uq_r)
        extra_r: dict = {"SPLIT_K": 0} if uq_r not in (0, 3, 4, 5, 6) else {}
        self._dispatch(
            "matmul_quant",
            [normed_x, self.weights[rw_k],
             self._scales_buf(rw_k, uq_r, sc["dummy_scales"]), sc["moe_router_out"]],
            {"K": hidden, "N": N_E, "USE_QUANT": uq_r, **extra_r, **qi_r},
            _gemv_wg(N_E, uq_r),
        )
        # top-K selection + softmax normalization (single workgroup, thread 0 only)
        self._dispatch(
            "topk_sort",
            [sc["moe_router_out"], sc["moe_topk_idx"], sc["moe_topk_w"]],
            {"N_EXPERTS": N_E, "K": K},
            (1, 1, 1),
        )

        # Flush current encoder and wait for GPU to complete the router + topk.
        dev.queue.submit([self._active_encoder.finish()])
        dev.queue.on_submitted_work_done_sync()

        # Read topk indices and weights from GPU storage buffers.
        # .to_numpy() creates an internal staging buffer, submits a copy, and maps.
        raw_idx = sc["moe_topk_idx"].to_numpy().view(np.uint32)
        raw_w   = sc["moe_topk_w"].to_numpy().view(np.float32)
        expert_indices = [int(raw_idx[k]) for k in range(K)]
        expert_weights = [float(raw_w[k]) for k in range(K)]

        logger.debug("L%02d MoE experts: %s  weights: %s",
                     layer_idx, expert_indices,
                     [f"{w:.3f}" for w in expert_weights])

        # Write all K softmax weights into the combined weight buffer so the
        # moe_accumulate shader can read w_buf[K_IDX] without per-dispatch overhead.
        dev.queue.write_buffer(sc["moe_w_buf"].buf, 0,
                               struct.pack(f"<{K}f", *expert_weights))

        # ── Phase B: expert dispatches (new encoder) ──────────────────────────
        # Subsequent _dispatch() calls (including add_rms_norm after the FFN in
        # the calling layer method) will land in this new encoder.
        self._active_encoder = dev.create_command_encoder()

        # Shared expert — always active, contributes with coefficient 1.0.
        sp = f"{p}.shared_expert"
        sgw_k = f"{sp}.gate_proj.weight"
        if self.weights.get(sgw_k) is not None:
            suw_k = f"{sp}.up_proj.weight"
            sdw_k = f"{sp}.down_proj.weight"
            self._dispatch(
                "fused_gate_act",
                [normed_x, self.weights[sgw_k], self.weights[suw_k],
                 sc["moe_expert_act"]],
                {"K": hidden, "N": shared_inter, "GELU": 0},
                (shared_inter, 1, 1),
            )
            uq_sd = self._uq_weight(sdw_k)
            qi_sd = self._quant_extra(f"{sp}.down_proj", uq_sd)
            extra_sd: dict = {"SPLIT_K": 0} if uq_sd not in (0, 3, 4, 5, 6) else {}
            self._dispatch(
                "matmul_quant",
                [sc["moe_expert_act"], self.weights[sdw_k],
                 self._scales_buf(sdw_k, uq_sd, sc["dummy_scales"]), out_buf],
                {"K": shared_inter, "N": hidden, "USE_QUANT": uq_sd,
                 **extra_sd, **qi_sd},
                _gemv_wg(hidden, uq_sd),
            )
        else:
            # No shared expert weights loaded — zero-initialize out_buf so
            # the first expert's accumulation starts from 0 (not garbage).
            dev.queue.write_buffer(out_buf.buf, 0, b"\x00" * (hidden * 2))

        # Selected experts — each contributes expert_weights[k_idx] * expert_out.
        for k_idx, exp_idx in enumerate(expert_indices):
            if expert_weights[k_idx] == 0.0:
                continue
            ep = f"{p}.experts.{exp_idx}"
            egw_k = f"{ep}.gate_proj.weight"
            if self.weights.get(egw_k) is None:
                logger.debug("L%02d expert %d weights not loaded, skipping",
                             layer_idx, exp_idx)
                continue
            euw_k = f"{ep}.up_proj.weight"
            edw_k = f"{ep}.down_proj.weight"

            # Gate + up projection with SiLU activation → moe_expert_act
            self._dispatch(
                "fused_gate_act",
                [normed_x, self.weights[egw_k], self.weights[euw_k],
                 sc["moe_expert_act"]],
                {"K": hidden, "N": moe_inter, "GELU": 0},
                (moe_inter, 1, 1),
            )
            # Down projection → moe_expert_down
            uq_ed = self._uq_weight(edw_k)
            qi_ed = self._quant_extra(f"{ep}.down_proj", uq_ed)
            extra_ed: dict = {"SPLIT_K": 0} if uq_ed not in (0, 3, 4, 5, 6) else {}
            self._dispatch(
                "matmul_quant",
                [sc["moe_expert_act"], self.weights[edw_k],
                 self._scales_buf(edw_k, uq_ed, sc["dummy_scales"]),
                 sc["moe_expert_down"]],
                {"K": moe_inter, "N": hidden, "USE_QUANT": uq_ed,
                 **extra_ed, **qi_ed},
                _gemv_wg(hidden, uq_ed),
            )
            # Weighted in-place accumulate: out_buf[i] += w_buf[k_idx] * expert_down[i]
            # K_IDX selects the correct weight from moe_w_buf without a runtime buffer
            # read per dispatch — 8 unique pipelines compiled once and cached forever.
            self._dispatch(
                "moe_accumulate",
                [out_buf, sc["moe_expert_down"], sc["moe_w_buf"]],
                {"N": hidden, "K_IDX": k_idx},
                ((hidden + 255) // 256, 1, 1),
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

        _moe_ffn_dispatch() flushes and replaces _active_encoder mid-layer to
        handle the CPU readback required for expert index selection.
        """
        import wgpu as wgpu_lib

        dev = self.wgpu_device.wgpu_device
        num_tokens = len(input_ids)
        hidden = self.hidden_size
        self._hstate = 0
        sc = self._sc

        if hasattr(attn_metadata, "block_tables") and len(attn_metadata.block_tables) > 1:
            raise RuntimeError("multi-sequence batching not supported in this build")

        ctx_len = int(attn_metadata.max_decode_seq_len
                      if attn_metadata.max_decode_seq_len is not None
                      else num_tokens)
        if ctx_len <= 0:
            ctx_len = num_tokens
        if ctx_len > 65535:
            raise RuntimeError(
                f"ctx_len={ctx_len} exceeds WebGPU dispatch limit of 65535.")

        pre = self._pre
        dev.queue.write_buffer(pre["ids"].buf, 0, input_ids.astype(np.uint32).tobytes())
        dev.queue.write_buffer(pre["pos"].buf, 0, positions.astype(np.uint32).tobytes())
        dev.queue.write_buffer(
            pre["slot_map"].buf, 0,
            np.array(attn_metadata.slot_mapping, dtype=np.uint32).tobytes())
        bt_arr = np.array(
            attn_metadata.block_tables[0]
            if hasattr(attn_metadata, "block_tables") else [0],
            dtype=np.uint32)
        dev.queue.write_buffer(pre["bt"].buf, 0, bt_arr.tobytes())

        ids_buf    = pre["ids"]
        pos_buf    = pre["pos"]
        slot_map   = pre["slot_map"]
        bt_buf     = pre["bt"]
        x_buf      = pre["x"]
        norm_out   = pre["norm_out"]
        logits_buf = pre["logits"]
        vocab = self.vocab_size

        _vpt = min((hidden + 255) // 256, 16) if hidden <= 4096 else 0
        _rms_base = {"HIDDEN_DIM": hidden, "VALS_PER_THREAD": _vpt}

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
                normed_x, x_buf = self._full_attn_layer(
                    i, normed_x, x_buf, pos_buf, slot_map, bt_buf, ctx_len, num_tokens)
            else:
                normed_x, x_buf = self._linear_attn_layer(i, normed_x, x_buf, num_tokens)

        self._dispatch(
            "rms_norm",
            [x_buf, self.weights["model.norm.weight"], norm_out],
            _rms_base, (num_tokens, 1, 1))

        lm_head_w = (self.weights.get("lm_head.weight")
                     or self.weights.get("model.lm_head.weight")
                     or self.weights["model.embed_tokens.weight"])
        self._dispatch(
            "matmul_quant",
            [norm_out, lm_head_w, norm_out, logits_buf],
            {"K": hidden, "N": vocab, "USE_QUANT": 0, "SPLIT_K": 0},
            ((vocab + 255) // 256, 1, 1))
        self._dispatch(
            "argmax_f16", [logits_buf, self._ensure_sample_buf(vocab)],
            {"N": vocab}, (1, 1, 1))
        self._copy_sample_to_staging()

        # Submit the final encoder and release it.
        dev.queue.submit([self._active_encoder.finish()])
        self._active_encoder = None

        self._last_logit_buf = logits_buf
        self._last_vocab     = vocab
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

        from vllm_webgpu.webgpu.buffer import WebGPUBuffer
        import wgpu as wgpu_lib

        dev = self.wgpu_device.wgpu_device
        num_tokens = len(input_ids)
        hidden = self.hidden_size
        self._hstate = 0
        rw_usage = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST

        if hasattr(attn_metadata, "block_tables") and len(attn_metadata.block_tables) > 1:
            raise RuntimeError("multi-sequence batching not supported in this build")

        ctx_len = int(attn_metadata.max_decode_seq_len
                      if attn_metadata.max_decode_seq_len is not None
                      else num_tokens)
        if ctx_len <= 0:
            ctx_len = num_tokens
        if ctx_len > 65535:
            raise RuntimeError(f"ctx_len={ctx_len} exceeds WebGPU dispatch limit of 65535.")

        pre = self._pre
        dev.queue.write_buffer(pre["ids"].buf, 0, input_ids.astype(np.uint32).tobytes())
        dev.queue.write_buffer(pre["pos"].buf, 0, positions.astype(np.uint32).tobytes())
        dev.queue.write_buffer(
            pre["slot_map"].buf, 0,
            np.array(attn_metadata.slot_mapping, dtype=np.uint32).tobytes())
        bt_arr = np.array(
            attn_metadata.block_tables[0] if hasattr(attn_metadata, "block_tables") else [0],
            dtype=np.uint32)
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
        _vpt = min((hidden + 255) // 256, 16) if hidden <= 4096 else 0
        _rms_base = {"HIDDEN_DIM": hidden, "VALS_PER_THREAD": _vpt}
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
                    normed_x, x_buf = self._full_attn_layer(
                        i, normed_x, x_buf, pos_buf, slot_map, bt_buf, ctx_len, num_tokens)
                else:
                    normed_x, x_buf = self._linear_attn_layer(
                        i, normed_x, x_buf, num_tokens)

            self._dispatch("rms_norm",
                           [x_buf, self.weights["model.norm.weight"], norm_out],
                           _rms_base, (num_tokens, 1, 1))

            lm_head_w = (self.weights.get("lm_head.weight")
                         or self.weights.get("model.lm_head.weight")
                         or self.weights["model.embed_tokens.weight"])
            # vocab_size exceeds the 65535 workgroup-per-dimension limit, so the split-K
            # path is unusable. Force SPLIT_K=0 (row-per-thread) with ceil(vocab/256) WGs.
            self._dispatch("matmul_quant",
                           [norm_out, lm_head_w, norm_out, logits_buf],
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

    def logit_readback(self) -> "np.ndarray":
        return self._last_logit_buf.to_numpy().view(np.float16).reshape(1, self._last_vocab).astype(np.float32)

    def _ensure_sample_buf(self, vocab: int) -> "WebGPUBuffer":
        self._ensure_gpu_sampler(vocab)
        return self._gpu_sample_tok

    def _full_attn_layer(
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
        """Standard full-attention transformer layer. Receives pre-normed input."""
        import math

        sc = self._sc
        hidden = self.hidden_size
        p = f"model.layers.{layer_idx}"
        q_dim = self.num_q_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        inter = self.intermediate_size
        ln_rope = math.log(self.rope_theta)

        # Per-weight quantization detection: Q4_K (type 12) → GPU block decoder.
        _uq = self._uq_weight

        h_names = ["h0", "h1", "h2"]
        residual = sc[h_names[(self._hstate + 1) % 3]]
        out = sc[h_names[(self._hstate + 2) % 3]]
        add_n = num_tokens * hidden
        gelu_n = num_tokens * inter

        _vpt = min((hidden + 255) // 256, 16) if hidden <= 4096 else 0
        _rms_h = {"HIDDEN_DIM": hidden, "VALS_PER_THREAD": _vpt}
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
                                **({"SPLIT_K": 0} if uq not in (0, 3, 4, 5, 6) else {}), **qi},
                               _gemv_wg(dim, uq))

            # When attn_output_gate=True, q_proj.weight was split at load time.
            # Compute the gate projection: normed_x → q_gate_buf [q_dim f16].
            # The gate is applied as silu(gate)*attn_out before o_proj (step below).
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
                                    **({"SPLIT_K": 0} if uq_gate not in (0, 3, 4, 5, 6) else {}),
                                    **qi_gate},
                                   _gemv_wg(q_dim, uq_gate))

            for src, dst, n_heads, w_key in [
                (sc["q_buf"], sc["q_rope"], self.num_q_heads, f"{p}.self_attn.q_norm.weight"),
                (sc["k_buf"], sc["k_rope"], self.num_kv_heads, f"{p}.self_attn.k_norm.weight"),
            ]:
                norm_w = self.weights.get(w_key)
                if norm_w is not None:
                    self._dispatch("fused_per_head_norm_rope",
                                   [src, norm_w, pos_buf, dst],
                                   {"HEAD_DIM": self.head_dim, "NUM_HEADS": n_heads,
                                    "ROPE_BASE": float(self.rope_theta),
                                    "LN_ROPE_BASE": ln_rope, "HAS_WEIGHT": 1,
                                    "ROTARY_DIM": self._rotary_dim,
                                    "INTERLEAVED": self._rope_interleaved},
                                   (n_heads, num_tokens, 1))
                else:
                    self._dispatch("rope", [src, pos_buf, dst],
                                   {"HEAD_DIM": self.head_dim, "NUM_HEADS": n_heads,
                                    "LN_ROPE_BASE": ln_rope},
                                   (num_tokens, n_heads, 1))

            # Fused K+V cache store
            self._dispatch("kv_cache_store_both",
                           [sc["k_rope"], k_cache, sc["v_buf"], v_cache, slot_map],
                           {"BLOCK_SIZE": self.block_size, "NUM_KV_HEADS": self.num_kv_heads,
                            "HEAD_DIM": self.head_dim},
                           (num_tokens, self.num_kv_heads, 1))

            self._dispatch("attn_score", [sc["q_rope"], k_cache, bt_buf, sc["scores_buf"]],
                           {"BLOCK_SIZE": self.block_size, "NUM_Q_HEADS": self.num_q_heads,
                            "NUM_KV_HEADS": self.num_kv_heads, "HEAD_DIM": self.head_dim,
                            "MAX_SEQ_LEN": ctx_len},
                           (self.num_q_heads, ctx_len, 1))
            self._dispatch("softmax", [sc["scores_buf"], sc["sm_buf"]],
                           {"SEQ_LEN": ctx_len}, (self.num_q_heads, 1, 1))
            self._dispatch("attn_output", [sc["sm_buf"], v_cache, bt_buf, sc["attn_out"]],
                           {"BLOCK_SIZE": self.block_size, "NUM_Q_HEADS": self.num_q_heads,
                            "NUM_KV_HEADS": self.num_kv_heads, "HEAD_DIM": self.head_dim,
                            "CTX_LEN": ctx_len},
                           (self.num_q_heads, 1, 1))

            # Apply attention output gate if enabled: gated = silu(q_gate_buf) * attn_out.
            # q_buf is free at this point (written in q_proj, last read in RoPE), so
            # reuse it as the output buffer for the gated result.
            if self._attn_output_gate and self.weights.get(f"{p}.self_attn.q_gate_proj.weight") is not None:
                gate_n = num_tokens * q_dim
                self._dispatch("gelu_mul",
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
                            **({"SPLIT_K": 0} if uq not in (0, 3, 4, 5, 6) else {}), **qi_o},
                           _gemv_wg(hidden, uq))

            # Fused: add(x, attn_out, residual) + rms_norm(residual, post_attn_w) → ffn_normed
            self._dispatch("add_rms_norm",
                           [x_buf, sc["o_proj_out"],
                            self.weights[f"{p}.post_attention_layernorm.weight"],
                            residual, sc["ffn_normed"]],
                           _rms_h, (num_tokens, 1, 1))

            # FFN (MoE or dense)
            if self._is_moe:
                self._moe_ffn_dispatch(layer_idx, sc["ffn_normed"], sc["ffn_out"],
                                       num_tokens)
            else:
                gw_k2 = f"{p}.mlp.gate_proj.weight"
                uw_k2 = f"{p}.mlp.up_proj.weight"
                uq_g2 = _uq(gw_k2); uq_u2 = _uq(uw_k2)
                if uq_g2 == 0 and uq_u2 == 0:
                    self._dispatch("fused_gate_act",
                                   [sc["ffn_normed"], self.weights[gw_k2], self.weights[uw_k2],
                                    sc["ffn_act"]],
                                   {"K": hidden, "N": inter, "GELU": 0}, (inter, 1, 1))
                else:
                    for out_b, proj2, w_k, uq2 in [
                            (sc["gate_buf"], "gate_proj", gw_k2, uq_g2),
                            (sc["up_buf"],   "up_proj",   uw_k2, uq_u2)]:
                        qi2 = self._quant_extra(f"{p}.mlp.{proj2}", uq2)
                        self._dispatch("matmul_quant",
                                       [sc["ffn_normed"], self.weights[w_k],
                                        self._scales_buf(w_k, uq2, sc["ffn_normed"]), out_b],
                                       {"K": hidden, "N": inter, "USE_QUANT": uq2,
                                        **({"SPLIT_K": 0} if uq2 not in (0, 3, 4, 5, 6) else {}),
                                        **qi2},
                                       _gemv_wg(inter, uq2))
                    self._dispatch("gelu_mul", [sc["gate_buf"], sc["up_buf"], sc["ffn_act"]],
                                   {"N": gelu_n}, ((gelu_n // 4 + 255) // 256, 1, 1))

                w_k = f"{p}.mlp.down_proj.weight"
                uq = _uq(w_k)
                qi_d = self._quant_extra(f"{p}.mlp.down_proj", uq)
                self._dispatch("matmul_quant",
                               [sc["ffn_act"], self.weights[w_k],
                                self._scales_buf(w_k, uq, sc["ffn_act"]), sc["ffn_out"]],
                               {"K": inter, "N": hidden, "USE_QUANT": uq,
                                **({"SPLIT_K": 0} if uq not in (0, 3, 4, 5, 6) else {}), **qi_d},
                               _gemv_wg(hidden, uq))

            if layer_idx < self.num_layers - 1:
                next_w = self.weights[f"model.layers.{layer_idx+1}.input_layernorm.weight"]
                self._dispatch("add_rms_norm",
                               [residual, sc["ffn_out"], next_w, out, sc["normed"]],
                               _rms_h, (num_tokens, 1, 1))
            else:
                self._dispatch("add", [residual, sc["ffn_out"], out],
                               {"N": add_n}, ((add_n // 4 + 255) // 256, 1, 1))

        self._hstate = (self._hstate + 2) % 3
        return sc["normed"], out

    def _linear_attn_layer(
        self,
        layer_idx: int,
        normed_x: "WebGPUBuffer",
        x_buf: "WebGPUBuffer",
        num_tokens: int,
    ) -> "tuple[WebGPUBuffer, WebGPUBuffer]":
        """GDN linear-attention layer — delegates to _gdn_layer_gpu (pure WebGPU)."""
        return self._gdn_layer_gpu(layer_idx, normed_x, x_buf, num_tokens)

