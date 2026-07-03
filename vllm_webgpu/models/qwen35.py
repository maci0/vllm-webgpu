from __future__ import annotations
import logging
from typing import TYPE_CHECKING

import numpy as np

from vllm_webgpu.models.base import BaseWebGPUModel

# Lazy singleton cache for vLLM GDN ops.
# Imported on first call to _get_vllm_gdn_ops() to avoid module-load-time
# failures, but cached so that repeated calls are free.
_vllm_gdn_ops_cache: dict | None = None


def _get_vllm_gdn_ops() -> dict:
    """Return the two vLLM CPU GDN ops, registering and caching them on first call."""
    global _vllm_gdn_ops_cache
    if _vllm_gdn_ops_cache is not None:
        return _vllm_gdn_ops_cache
    try:
        from vllm.model_executor.layers.mamba.ops.cpu.gdn_attention import (
            register_cpu_gdn_attention_ops,
        )
        register_cpu_gdn_attention_ops()
        from vllm.model_executor.layers.mamba.ops.cpu.causal_conv1d import (
            causal_conv1d_update_torch,
        )
        import vllm._custom_ops as vllm_ops
        _vllm_gdn_ops_cache = {
            "causal_conv1d_update_torch": causal_conv1d_update_torch,
            "fused_sigmoid_gating_delta_rule_update_cpu":
                vllm_ops.fused_sigmoid_gating_delta_rule_update_cpu,
        }
        return _vllm_gdn_ops_cache
    except Exception as exc:
        raise RuntimeError(
            "vLLM CPU GDN ops unavailable. Install vllm with CPU support."
        ) from exc

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
    Qwen3.5-9B hybrid inference model.

    Full-attention layers (every 4th, indices 3/7/11/.../31): standard GQA on
    GPU using existing WebGPU kernels, same path as LlamaWebGPUModel.

    Linear-attention layers (all others): GDN (Gated Delta Networks) on CPU
    using vLLM's compiled ops, with FFN on GPU.

    GDN CPU path per layer:
      1. GPU readback of normed x
      2. in_proj_qkv on CPU (torch linear, bf16)
      3. causal depthwise conv1d update (vLLM causal_conv1d_update_torch)
      4. in_proj_a / in_proj_b on CPU
      5. fused_sigmoid_gating_delta_rule_update_cpu (vLLM op, mutates SSM state)
      6. RMS norm + sigmoid gate + out_proj on CPU
      7. Upload result to GPU, continue FFN on WebGPU
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

        from vllm_webgpu.config import get_config
        self.block_size: int = get_config().block_size

        # Validate dimensions for GPU shaders
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

        # CPU torch tensors for linear-attention layers (populated in load_weights)
        self._lin_cpu: list[dict | None] = [None] * self.num_layers

        # Per-layer recurrent states (SSM + conv)
        # SSM:  torch.Tensor [1, num_v_heads, k_dim, v_dim] float32
        # Conv: torch.Tensor [1, conv_dim, kernel-1] bfloat16 (mutated in-place)
        self._ssm_states: list = [None] * self.num_layers
        self._conv_states: list = [None] * self.num_layers

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
            "gate_buf":   mk(T * I * 2),
            "up_buf":     mk(T * I * 2),
            "ffn_act":    mk(T * I * 2),
            "ffn_out":    mk(T * H * 2),
            "h0":         mk(T * H * 2),
            "h1":         mk(T * H * 2),
            "h2":         mk(T * H * 2),
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

    def _extract_lin_weights(self) -> None:
        """Pull linear-attn weights from GPU buffers into CPU torch BF16 tensors.

        Also pre-warms the vLLM op cache so the real ops are resolved now,
        before any test mock can shadow sys.modules["vllm"].

        vLLM's GDN ops require bfloat16 torch tensors on CPU. The MLX loader
        dequantizes everything to f16 and uploads to GPU; here we download and
        reinterpret as bfloat16 via float32.

        conv1d weight from MLX is [8192, 4, 1]; we squeeze the trailing 1 to
        get [8192, 4], which matches causal_conv1d_update_torch's expected
        [dim, kernel] shape.

        A_log was originally float32 in MLX but is stored as f16 in the GPU
        buffer; we promote it back to float32 before handing to the GDN op.
        """
        import torch

        # Pre-warm the op cache while the real vllm is still accessible.
        try:
            _get_vllm_gdn_ops()
        except RuntimeError:
            logger.warning("vLLM GDN ops unavailable; linear-attention layers will error at inference.")

        def _to_bf16(key: str) -> "torch.Tensor | None":
            buf = self.weights.get(key)
            if buf is None:
                return None
            arr = buf.to_numpy().view(np.float16).reshape(buf.shape)
            return torch.from_numpy(arr.astype(np.float32)).to(torch.bfloat16)

        def _to_f32(key: str) -> "torch.Tensor | None":
            buf = self.weights.get(key)
            if buf is None:
                return None
            arr = buf.to_numpy().view(np.float16).reshape(buf.shape)
            return torch.from_numpy(arr.astype(np.float32))

        for i in range(self.num_layers):
            if self._is_full_attn(i):
                continue

            p = f"model.layers.{i}.linear_attn"

            conv_w = _to_bf16(f"{p}.conv1d.weight")
            if conv_w is not None and conv_w.ndim == 3:
                conv_w = conv_w.squeeze(-1)  # [8192, 4, 1] -> [8192, 4]

            self._lin_cpu[i] = {
                "in_proj_qkv": _to_bf16(f"{p}.in_proj_qkv.weight"),
                "in_proj_z":   _to_bf16(f"{p}.in_proj_z.weight"),
                "in_proj_a":   _to_bf16(f"{p}.in_proj_a.weight"),
                "in_proj_b":   _to_bf16(f"{p}.in_proj_b.weight"),
                "conv1d":      conv_w,
                "A_log":       _to_f32(f"{p}.A_log"),
                "dt_bias":     _to_bf16(f"{p}.dt_bias"),
                "norm_weight": _to_bf16(f"{p}.norm.weight"),
                "out_proj":    _to_bf16(f"{p}.out_proj.weight"),
            }

            self._ssm_states[i] = torch.zeros(
                1, _LIN_V_HEADS, _LIN_K_DIM, _LIN_V_DIM, dtype=torch.float32
            )
            self._conv_states[i] = torch.zeros(
                1, _LIN_CONV_DIM, _LIN_CONV_KERNEL - 1, dtype=torch.bfloat16
            )

    def load_weights(self, path: str) -> None:
        super().load_weights(path)
        self._postprocess_weights()
        self._extract_lin_weights()

    def _reset_recurrent_states(self) -> None:
        """Zero out GDN recurrent states (SSM + conv) for all linear-attention layers."""
        import torch
        for i in range(self.num_layers):
            if self._is_full_attn(i):
                continue
            self._ssm_states[i] = torch.zeros(
                1, _LIN_V_HEADS, _LIN_K_DIM, _LIN_V_DIM, dtype=torch.float32
            )
            self._conv_states[i] = torch.zeros(
                1, _LIN_CONV_DIM, _LIN_CONV_KERNEL - 1, dtype=torch.bfloat16
            )

    def _gdn_decode(self, layer_idx: int, x_bf16: "torch.Tensor") -> "torch.Tensor":
        """Single-token GDN decode step using vLLM's compiled CPU ops.

        x_bf16: [hidden_size] bfloat16 torch tensor on CPU
        Returns: [hidden_size] bfloat16 torch tensor on CPU
        """
        import torch
        import torch.nn.functional as F

        vllm_ops = _get_vllm_gdn_ops()
        causal_conv1d_update_torch = vllm_ops["causal_conv1d_update_torch"]
        fused_gdn_update = vllm_ops["fused_sigmoid_gating_delta_rule_update_cpu"]

        lw = self._lin_cpu[layer_idx]
        if lw is None:
            raise RuntimeError(f"Linear-attn weights not loaded for layer {layer_idx}")

        # 1. QKV projection: [hidden] -> [8192]
        qkv = F.linear(x_bf16.unsqueeze(0), lw["in_proj_qkv"])  # [1, 8192]

        # 2. Causal depthwise conv1d (mutates conv_state in-place)
        # expects x: [B, dim, 1], conv_state: [B, dim, kernel-1], weight: [dim, kernel]
        qkv_conv = causal_conv1d_update_torch(
            x=qkv.unsqueeze(-1),
            conv_state=self._conv_states[layer_idx],
            weight=lw["conv1d"],
            bias=None,
            activation="silu",
        )  # [1, 8192, 1]
        qkv_conv = qkv_conv.squeeze(-1).squeeze(0)  # [8192]

        # 3. Split Q=[2048], K=[2048], V=[4096]
        q_flat = qkv_conv[:_LIN_KEY_DIM]
        k_flat = qkv_conv[_LIN_KEY_DIM:_LIN_KEY_DIM * 2]
        v_flat = qkv_conv[_LIN_KEY_DIM * 2:]

        # 4. Reshape to 4D as required by the fused GDN op: [B, T, heads, dim]
        q = q_flat.view(1, 1, _LIN_K_HEADS, _LIN_K_DIM)
        k = k_flat.view(1, 1, _LIN_K_HEADS, _LIN_K_DIM)
        v = v_flat.view(1, 1, _LIN_V_HEADS, _LIN_V_DIM)

        # 5. a and b projections: [hidden] -> [num_v_heads=32]
        a = F.linear(x_bf16.unsqueeze(0), lw["in_proj_a"])  # [1, 32]
        b = F.linear(x_bf16.unsqueeze(0), lw["in_proj_b"])  # [1, 32]

        # 6. Fused GDN state update (mutates ssm_state in-place via state_indices)
        state_indices = torch.zeros(1, dtype=torch.int32)
        cu_seqlens = torch.tensor([0, 1], dtype=torch.int32)

        gdn_out = fused_gdn_update(
            A_log=lw["A_log"],
            dt_bias=lw["dt_bias"],
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=self._ssm_states[layer_idx],
            initial_state_indices=state_indices,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )  # [1, 1, num_v_heads, v_head_dim]

        # 7. Flatten: [1, 1, 32, 128] -> [4096]
        gdn_flat = gdn_out.reshape(-1)  # [4096]

        # 8. RMS norm per v_head using the shared norm weight [128]
        gdn_heads = gdn_flat.view(_LIN_V_HEADS, _LIN_V_DIM)
        rms = gdn_heads.float().pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        gdn_normed = (gdn_heads.float() / rms * lw["norm_weight"].float()).to(torch.bfloat16)
        gdn_normed_flat = gdn_normed.reshape(-1)  # [4096]

        # 9. Sigmoid gate: z = in_proj_z(x_original), gate = sigmoid(z)
        z = F.linear(x_bf16.unsqueeze(0), lw["in_proj_z"]).squeeze(0)  # [4096]
        gated = gdn_normed_flat * torch.sigmoid(z)  # [4096]

        # 10. Output projection: [4096] -> [hidden_size]
        return F.linear(gated.unsqueeze(0), lw["out_proj"]).squeeze(0)

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

        ids_buf = WebGPUBuffer.from_numpy(dev, input_ids.astype(np.uint32))
        x_buf = WebGPUBuffer.empty(dev, num_tokens * hidden * 2, usage=rw_usage)
        self._dispatch(
            "embedding_lookup",
            [self.weights["model.embed_tokens.weight"], ids_buf, x_buf],
            {"HIDDEN_DIM": hidden},
            (num_tokens, 1, 1),
        )

        if hasattr(attn_metadata, "block_tables") and len(attn_metadata.block_tables) > 1:
            raise RuntimeError("multi-sequence batching not supported in this build")

        ctx_len = int(attn_metadata.max_decode_seq_len
                      if attn_metadata.max_decode_seq_len is not None
                      else num_tokens)
        if ctx_len <= 0:
            ctx_len = num_tokens
        if ctx_len > 65535:
            raise RuntimeError(f"ctx_len={ctx_len} exceeds WebGPU dispatch limit of 65535.")

        pos_buf = WebGPUBuffer.from_numpy(dev, positions.astype(np.uint32))
        slot_map = WebGPUBuffer.from_numpy(
            dev, np.array(attn_metadata.slot_mapping, dtype=np.uint32))
        bt_arr = np.array(
            attn_metadata.block_tables[0] if hasattr(attn_metadata, "block_tables") else [0],
            dtype=np.uint32)
        bt_buf = WebGPUBuffer.from_numpy(dev, bt_arr)

        for i in range(self.num_layers):
            if self._is_full_attn(i):
                x_buf = self._full_attn_layer(
                    i, x_buf, pos_buf, slot_map, bt_buf, ctx_len, num_tokens)
            else:
                x_buf = self._linear_attn_layer(i, x_buf, num_tokens)

        # Final norm
        norm_out = WebGPUBuffer.empty(dev, num_tokens * hidden * 2, usage=rw_usage)
        self._dispatch(
            "rms_norm",
            [x_buf, self.weights["model.norm.weight"], norm_out],
            {"HIDDEN_DIM": hidden},
            (num_tokens, 1, 1),
        )

        # LM head: 'lm_head.weight' (MLX) or tied embeddings
        vocab = self.vocab_size
        logits_buf = WebGPUBuffer.empty(dev, num_tokens * vocab * 2, usage=rw_usage)
        lm_head_w = (self.weights.get("lm_head.weight")
                     or self.weights.get("model.lm_head.weight")
                     or self.weights["model.embed_tokens.weight"])
        self._dispatch(
            "matmul_quant",
            [norm_out, lm_head_w, norm_out, logits_buf],
            {"K": hidden, "N": vocab, "USE_QUANT": 0},
            ((vocab + 255) // 256, 1, 1),
        )

        return logits_buf.to_numpy().view(np.float16).reshape(num_tokens, vocab).astype(np.float32)

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
        use_quant = 0

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
                self._dispatch("matmul_quant",
                               [sc["normed"], self.weights[w_key],
                                self.weights.get(s_key, sc["normed"]), out_buf],
                               {"K": hidden, "N": dim, "USE_QUANT": use_quant},
                               ((dim + 255) // 256, 1, 1))

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
                                    "LN_ROPE_BASE": ln_rope, "HAS_WEIGHT": 1},
                                   (n_heads, num_tokens, 1))
                else:
                    self._dispatch("rope", [src, pos_buf, dst],
                                   {"HEAD_DIM": self.head_dim, "NUM_HEADS": n_heads,
                                    "LN_ROPE_BASE": ln_rope},
                                   (num_tokens, n_heads, 1))

            self._dispatch("kv_cache_store", [sc["k_rope"], k_cache, slot_map],
                           {"BLOCK_SIZE": self.block_size, "NUM_KV_HEADS": self.num_kv_heads,
                            "HEAD_DIM": self.head_dim},
                           (num_tokens, self.num_kv_heads, 1))
            self._dispatch("kv_cache_store", [sc["v_buf"], v_cache, slot_map],
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
            self._dispatch("matmul_quant",
                           [sc["attn_out"], self.weights[w_key],
                            self.weights.get(s_key, sc["attn_out"]), sc["o_proj_out"]],
                           {"K": q_dim, "N": hidden, "USE_QUANT": use_quant},
                           ((hidden + 255) // 256, 1, 1))

            self._dispatch("add", [x_buf, sc["o_proj_out"], residual],
                           {"N": add_n}, ((add_n // 4 + 255) // 256, 1, 1))

            self._dispatch("rms_norm",
                           [residual, self.weights[f"{p}.post_attention_layernorm.weight"],
                            sc["ffn_normed"]],
                           {"HIDDEN_DIM": hidden}, (num_tokens, 1, 1))

            for out_b, proj in [(sc["gate_buf"], "gate_proj"), (sc["up_buf"], "up_proj")]:
                w_k = f"{p}.mlp.{proj}.weight"
                s_k = f"{p}.mlp.{proj}.scales"
                self._dispatch("matmul_quant",
                               [sc["ffn_normed"], self.weights[w_k],
                                self.weights.get(s_k, sc["ffn_normed"]), out_b],
                               {"K": hidden, "N": inter, "USE_QUANT": use_quant},
                               ((inter + 255) // 256, 1, 1))

            self._dispatch("gelu_mul", [sc["gate_buf"], sc["up_buf"], sc["ffn_act"]],
                           {"N": gelu_n}, ((gelu_n // 4 + 255) // 256, 1, 1))

            w_k = f"{p}.mlp.down_proj.weight"
            s_k = f"{p}.mlp.down_proj.scales"
            self._dispatch("matmul_quant",
                           [sc["ffn_act"], self.weights[w_k],
                            self.weights.get(s_k, sc["ffn_act"]), sc["ffn_out"]],
                           {"K": inter, "N": hidden, "USE_QUANT": use_quant},
                           ((hidden + 255) // 256, 1, 1))

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
        """GDN linear-attention layer: CPU GDN via vLLM ops, FFN on GPU."""
        import torch
        import wgpu as wgpu_lib
        from vllm_webgpu.webgpu.buffer import WebGPUBuffer

        dev = self.wgpu_device.wgpu_device
        rw = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST

        sc = self._sc
        hidden = self.hidden_size
        inter = self.intermediate_size
        p = f"model.layers.{layer_idx}"

        h_names = ["h0", "h1", "h2"]
        residual = sc[h_names[(self._hstate + 1) % 3]]
        out = sc[h_names[(self._hstate + 2) % 3]]
        add_n = num_tokens * hidden
        gelu_n = num_tokens * inter
        use_quant = 0

        # 1. Pre-norm on GPU
        with self._batched_dispatch():
            self._dispatch("rms_norm",
                           [x_buf, self.weights[f"{p}.input_layernorm.weight"], sc["normed"]],
                           {"HIDDEN_DIM": hidden}, (num_tokens, 1, 1))

        # 2. Readback normed hidden state to CPU as bfloat16
        normed_np = sc["normed"].to_numpy().view(np.float16).reshape(num_tokens, hidden)
        normed_bf16 = torch.from_numpy(normed_np.astype(np.float32)).to(torch.bfloat16)

        # 3. GDN decode on CPU using vLLM ops (one token at a time)
        gdn_outs = [self._gdn_decode(layer_idx, normed_bf16[t]) for t in range(num_tokens)]
        gdn_bf16 = torch.stack(gdn_outs, dim=0)  # [num_tokens, hidden]

        # 4. Upload GDN output to GPU via o_proj_out scratch buffer
        gdn_f16 = np.clip(gdn_bf16.float().numpy().astype(np.float16), -65504.0, 65504.0)
        dev.queue.write_buffer(sc["o_proj_out"].buf, 0,
                               np.ascontiguousarray(gdn_f16).tobytes())

        # 5. Residual add + FFN on GPU
        with self._batched_dispatch():
            self._dispatch("add", [x_buf, sc["o_proj_out"], residual],
                           {"N": add_n}, ((add_n // 4 + 255) // 256, 1, 1))

            self._dispatch("rms_norm",
                           [residual, self.weights[f"{p}.post_attention_layernorm.weight"],
                            sc["ffn_normed"]],
                           {"HIDDEN_DIM": hidden}, (num_tokens, 1, 1))

            for out_b, proj in [(sc["gate_buf"], "gate_proj"), (sc["up_buf"], "up_proj")]:
                w_k = f"{p}.mlp.{proj}.weight"
                s_k = f"{p}.mlp.{proj}.scales"
                self._dispatch("matmul_quant",
                               [sc["ffn_normed"], self.weights[w_k],
                                self.weights.get(s_k, sc["ffn_normed"]), out_b],
                               {"K": hidden, "N": inter, "USE_QUANT": use_quant},
                               ((inter + 255) // 256, 1, 1))

            self._dispatch("gelu_mul", [sc["gate_buf"], sc["up_buf"], sc["ffn_act"]],
                           {"N": gelu_n}, ((gelu_n // 4 + 255) // 256, 1, 1))

            w_k = f"{p}.mlp.down_proj.weight"
            s_k = f"{p}.mlp.down_proj.scales"
            self._dispatch("matmul_quant",
                           [sc["ffn_act"], self.weights[w_k],
                            self.weights.get(s_k, sc["ffn_act"]), sc["ffn_out"]],
                           {"K": inter, "N": hidden, "USE_QUANT": use_quant},
                           ((hidden + 255) // 256, 1, 1))

            self._dispatch("add", [residual, sc["ffn_out"], out],
                           {"N": add_n}, ((add_n // 4 + 255) // 256, 1, 1))

        self._hstate = (self._hstate + 2) % 3
        return out
