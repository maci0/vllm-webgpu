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
            "o_proj_out": mk(T * H * 2),
            "ffn_normed": mk(T * H * 2),
            "gate_buf":    mk(T * I * 2),
            "up_buf":      mk(T * I * 2),
            "ffn_act":     mk(T * I * 2),
            "ffn_gate_up": mk(T * I * 4),  # [2*inter] for fused gate+up
            "ffn_out":    mk(T * H * 2),
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
        self._hstate: int = 0

    def _postprocess_weights(self) -> None:
        """Tile q_norm/k_norm from (head_dim,) to (num_heads * head_dim,) for full-attn layers."""
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

            # Reshape conv1d weight [CONV_DIM, KERNEL, 1] -> [CONV_DIM, KERNEL] if needed.
            p = f"model.layers.{i}.linear_attn"
            conv_w_key = f"{p}.conv1d.weight"
            w = self.weights.get(conv_w_key)
            if w is not None and len(w.shape) == 3 and w.shape[2] == 1:
                arr = w.to_numpy().view(np.float16).reshape(w.shape[0], w.shape[1])
                self.weights[conv_w_key] = WebGPUBuffer.from_numpy(dev, arr, usage=rw)

    def load_weights(self, path: str) -> None:
        super().load_weights(path)
        self._postprocess_weights()
        self._alloc_lin_states()

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
        x_buf: "WebGPUBuffer",
        num_tokens: int,
    ) -> "WebGPUBuffer":
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

        with self._batched_dispatch(label=f"L{layer_idx:02d}"):
            # 1. Pre-norm
            _rms_h = {"HIDDEN_DIM": hidden,
                      "VALS_PER_THREAD": min((hidden + 255) // 256, 16) if hidden <= 4096 else 0}
            self._dispatch("rms_norm",
                           [x_buf, self.weights[f"{pp}.input_layernorm.weight"], sc["normed"]],
                           _rms_h, (num_tokens, 1, 1))

            cd = self._lin_conv_dim
            vd = self._lin_val_dim
            kh = self._lin_k_heads
            kd = self._lin_k_dim
            vh = self._lin_v_heads
            vdh = self._lin_v_dim

            # 2. QKV projection: [hidden] → [conv_dim]
            self._dispatch("matmul_quant",
                           [sc["normed"], self.weights[f"{p}.in_proj_qkv.weight"],
                            sc["dummy_scales"], sc["qkv_buf"]],
                           {"K": hidden, "N": cd, "USE_QUANT": 0},
                           (cd, 1, 1))

            # 3. Causal conv step: updates conv_state in-place, writes qkv_conv
            conv_w = self.weights.get(f"{p}.conv1d.weight", sc["normed"])
            self._dispatch("causal_conv_step",
                           [sc["qkv_buf"], conv_w, self._conv_gpu[layer_idx], sc["qkv_conv"]],
                           {"CONV_DIM": cd, "KERNEL": self._lin_conv_kernel, "WG_SIZE": 256},
                           ((cd + 255) // 256, 1, 1))

            # 4. a projection: normed → [K_HEADS] (dt for decay)
            # kh rows: must dispatch (kh, 1, 1) with SPLIT_K=1 (one WG per output row).
            self._dispatch("matmul_quant",
                           [sc["normed"], self.weights[f"{p}.in_proj_a.weight"],
                            sc["dummy_scales"], sc["a_buf"]],
                           {"K": hidden, "N": kh, "USE_QUANT": 0},
                           (kh, 1, 1))

            # 5a. b projection: normed → [K_HEADS] (outer-product gate)
            self._dispatch("matmul_quant",
                           [sc["normed"], self.weights[f"{p}.in_proj_b.weight"],
                            sc["dummy_scales"], sc["b_buf"]],
                           {"K": hidden, "N": kh, "USE_QUANT": 0},
                           (kh, 1, 1))

            # 5b. z gate projection: normed → [val_dim]
            self._dispatch("matmul_quant",
                           [sc["normed"], self.weights[f"{p}.in_proj_z.weight"],
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

            # 9. Attention residual add
            self._dispatch("add", [x_buf, sc["o_proj_out"], residual],
                           {"N": add_n}, ((add_n // 4 + 255) // 256, 1, 1))

            # 10. FFN
            _rms_ff = _rms_h
            self._dispatch("rms_norm",
                           [residual, self.weights[f"{pp}.post_attention_layernorm.weight"],
                            sc["ffn_normed"]],
                           _rms_ff, (num_tokens, 1, 1))

            # Fused gate+up: 2 dispatches → 1
            gw_k = f"{pp}.mlp.gate_proj.weight"
            uw_k = f"{pp}.mlp.up_proj.weight"
            self._dispatch("fused_gate_up",
                           [sc["ffn_normed"], self.weights[gw_k], self.weights[uw_k],
                            sc["ffn_gate_up"]],
                           {"K": hidden, "N": inter}, (inter, 1, 1))
            self._dispatch("gelu_mul_fused", [sc["ffn_gate_up"], sc["ffn_act"]],
                           {"N": gelu_n}, ((gelu_n // 4 + 255) // 256, 1, 1))

            w_k = f"{pp}.mlp.down_proj.weight"
            self._dispatch("matmul_quant",
                           [sc["ffn_act"], self.weights[w_k],
                            sc["ffn_act"], sc["ffn_out"]],
                           {"K": inter, "N": hidden, "USE_QUANT": 0},
                           (hidden, 1, 1))

            self._dispatch("add", [residual, sc["ffn_out"], out],
                           {"N": add_n}, ((add_n // 4 + 255) // 256, 1, 1))

        self._hstate = (self._hstate + 2) % 3
        return out

    def forward(
        self,
        input_ids: np.ndarray,
        positions: np.ndarray,
        attn_metadata: object,
    ) -> np.ndarray:
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
        with self._batched_dispatch():
            self._dispatch("embedding_lookup",
                           [self.weights["model.embed_tokens.weight"], ids_buf, x_buf],
                           {"HIDDEN_DIM": hidden}, (num_tokens, 1, 1))

            for i in range(self.num_layers):
                if self._is_full_attn(i):
                    x_buf = self._full_attn_layer(
                        i, x_buf, pos_buf, slot_map, bt_buf, ctx_len, num_tokens)
                else:
                    x_buf = self._linear_attn_layer(i, x_buf, num_tokens)

            _vpt = min((hidden + 255) // 256, 16) if hidden <= 4096 else 0
            self._dispatch("rms_norm",
                           [x_buf, self.weights["model.norm.weight"], norm_out],
                           {"HIDDEN_DIM": hidden, "VALS_PER_THREAD": _vpt},
                           (num_tokens, 1, 1))

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
        x_buf: "WebGPUBuffer",
        pos_buf: "WebGPUBuffer",
        slot_map: "WebGPUBuffer",
        bt_buf: "WebGPUBuffer",
        ctx_len: int,
        num_tokens: int,
    ) -> "WebGPUBuffer":
        """Standard full-attention transformer layer (identical to LlamaWebGPUModel)."""
        import math

        sc = self._sc
        hidden = self.hidden_size
        p = f"model.layers.{layer_idx}"
        q_dim = self.num_q_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        inter = self.intermediate_size
        ln_rope = math.log(self.rope_theta)

        # Per-weight quantization detection: Q4_K (type 12) → GPU block decoder.
        quant_types = self.weights.get("__quant_types__", {})
        _qt = quant_types if isinstance(quant_types, dict) else {}

        def _uq(key: str) -> int:
            tt = _qt.get(key, 0)
            if tt == 12:
                return 2
            if self.weights.get(key[:-7] + ".scales") is not None:
                return 1
            return 0

        h_names = ["h0", "h1", "h2"]
        residual = sc[h_names[(self._hstate + 1) % 3]]
        out = sc[h_names[(self._hstate + 2) % 3]]
        add_n = num_tokens * hidden
        gelu_n = num_tokens * inter

        k_cache, v_cache = self.kv_pool[layer_idx]

        with self._batched_dispatch():
            self._dispatch("rms_norm",
                           [x_buf, self.weights[f"{p}.input_layernorm.weight"], sc["normed"]],
                           {"HIDDEN_DIM": hidden}, (num_tokens, 1, 1))

            for out_buf, proj, dim in [(sc["q_buf"], "q_proj", q_dim),
                                       (sc["k_buf"], "k_proj", kv_dim),
                                       (sc["v_buf"], "v_proj", kv_dim)]:
                w_key = f"{p}.self_attn.{proj}.weight"
                s_key = f"{p}.self_attn.{proj}.scales"
                uq = _uq(w_key)
                self._dispatch("matmul_quant",
                               [sc["normed"], self.weights[w_key],
                                self.weights.get(s_key, sc["normed"]), out_buf],
                               {"K": hidden, "N": dim, "USE_QUANT": uq, **({"SPLIT_K": 0} if uq else {})},
                               _gemv_wg(dim, uq))

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

            w_key = f"{p}.self_attn.o_proj.weight"
            s_key = f"{p}.self_attn.o_proj.scales"
            uq = _uq(w_key)
            self._dispatch("matmul_quant",
                           [sc["attn_out"], self.weights[w_key],
                            self.weights.get(s_key, sc["attn_out"]), sc["o_proj_out"]],
                           {"K": q_dim, "N": hidden, "USE_QUANT": uq, **({"SPLIT_K": 0} if uq else {})},
                           _gemv_wg(hidden, uq))

            self._dispatch("add", [x_buf, sc["o_proj_out"], residual],
                           {"N": add_n}, ((add_n // 4 + 255) // 256, 1, 1))

            self._dispatch("rms_norm",
                           [residual, self.weights[f"{p}.post_attention_layernorm.weight"],
                            sc["ffn_normed"]],
                           {"HIDDEN_DIM": hidden}, (num_tokens, 1, 1))

            # Fused gate+up (f16 only; fallback for quantized)
            gw_k2 = f"{p}.mlp.gate_proj.weight"
            uw_k2 = f"{p}.mlp.up_proj.weight"
            uq_g2 = _uq(gw_k2); uq_u2 = _uq(uw_k2)
            if uq_g2 == 0 and uq_u2 == 0:
                self._dispatch("fused_gate_up",
                               [sc["ffn_normed"], self.weights[gw_k2], self.weights[uw_k2],
                                sc["ffn_gate_up"]],
                               {"K": hidden, "N": inter}, (inter, 1, 1))
                self._dispatch("gelu_mul_fused", [sc["ffn_gate_up"], sc["ffn_act"]],
                               {"N": gelu_n}, ((gelu_n // 4 + 255) // 256, 1, 1))
            else:
                for out_b, proj, w_k, uq in [
                        (sc["gate_buf"], "gate_proj", gw_k2, uq_g2),
                        (sc["up_buf"],   "up_proj",   uw_k2, uq_u2)]:
                    s_k = f"{p}.mlp.{proj}.scales"
                    self._dispatch("matmul_quant",
                                   [sc["ffn_normed"], self.weights[w_k],
                                    self.weights.get(s_k, sc["ffn_normed"]), out_b],
                                   {"K": hidden, "N": inter, "USE_QUANT": uq,
                                    **({"SPLIT_K": 0} if uq else {})},
                                   _gemv_wg(inter, uq))
                self._dispatch("gelu_mul", [sc["gate_buf"], sc["up_buf"], sc["ffn_act"]],
                               {"N": gelu_n}, ((gelu_n // 4 + 255) // 256, 1, 1))

            w_k = f"{p}.mlp.down_proj.weight"
            s_k = f"{p}.mlp.down_proj.scales"
            uq = _uq(w_k)
            self._dispatch("matmul_quant",
                           [sc["ffn_act"], self.weights[w_k],
                            self.weights.get(s_k, sc["ffn_act"]), sc["ffn_out"]],
                           {"K": inter, "N": hidden, "USE_QUANT": uq, **({"SPLIT_K": 0} if uq else {})},
                           _gemv_wg(hidden, uq))

            self._dispatch("add", [residual, sc["ffn_out"], out],
                           {"N": add_n}, ((add_n // 4 + 255) // 256, 1, 1))

        self._hstate = (self._hstate + 2) % 3
        return out

    def _linear_attn_layer(
        self,
        layer_idx: int,
        x_buf: "WebGPUBuffer",
        num_tokens: int,
    ) -> "WebGPUBuffer":
        """GDN linear-attention layer — delegates to _gdn_layer_gpu (pure WebGPU)."""
        return self._gdn_layer_gpu(layer_idx, x_buf, num_tokens)

