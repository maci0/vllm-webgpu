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


class Gemma4WebGPUModel(BaseWebGPUModel):
    """
    Gemma 4 adds on top of the Llama-style transformer:
    - Per-head RMSNorm (no weight) on V heads
    - PLE (per-layer embedding) pipeline between transformer blocks at configured indices
    - Logit softcap before sampling
    """

    def __init__(self, model_config, wgpu_device: "WebGPUDevice", pipeline_cache: "PipelineCache") -> None:
        super().__init__(model_config, wgpu_device, pipeline_cache)
        self.num_layers: int = model_config.num_hidden_layers
        self.num_q_heads: int = model_config.num_attention_heads
        self.num_kv_heads: int = model_config.num_key_value_heads
        self.hidden_size: int = model_config.hidden_size
        self.intermediate_size: int = model_config.intermediate_size
        self.vocab_size: int = model_config.vocab_size
        self.head_dim: int = getattr(model_config, "head_dim", model_config.hidden_size // model_config.num_attention_heads)
        self.softcap: float = getattr(model_config, "final_logit_softcapping", 30.0)
        self.ple_layer_indices: set[int] = set(getattr(model_config, "ple_layer_indices", []))
        self.rope_theta: float = getattr(model_config, "rope_theta", 10000.0)
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

    def _init_scratch_buffers(self, max_ctx: int) -> None:
        """Pre-allocate all intermediate scratch buffers used in _transformer_layer.

        Eliminates 18 GPU buffer allocations per layer per decode token.
        Gemma4 has one extra buffer (v_normed) vs Llama.
        """
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
            "v_normed":   mk(T * KV * 2),  # Gemma4-specific: per-head RMSNorm on V
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
        vocab = self.vocab_size
        rw = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST

        # Reset hidden-state rotation at the start of each forward pass
        self._hstate = 0

        ids_buf = WebGPUBuffer.from_numpy(dev, input_ids.astype(np.uint32))
        x_buf = WebGPUBuffer.empty(dev, num_tokens * hidden * 2, usage=rw)
        self._dispatch("embedding_lookup",
                       [self.weights["model.embed_tokens.weight"], ids_buf, x_buf],
                       {"HIDDEN_DIM": hidden},
                       (num_tokens, 1, 1))

        if hasattr(attn_metadata, "block_tables") and len(attn_metadata.block_tables) > 1:
            raise RuntimeError("multi-sequence batching not supported in this build")

        ctx_len = int(attn_metadata.max_decode_seq_len
                      if attn_metadata.max_decode_seq_len is not None
                      else num_tokens)
        if ctx_len <= 0:
            ctx_len = num_tokens
        if ctx_len > 65535:
            raise RuntimeError(
                f"ctx_len={ctx_len} exceeds WebGPU dispatch limit of 65535."
            )

        pos_buf = WebGPUBuffer.from_numpy(dev, positions.astype(np.uint32))
        slot_map = WebGPUBuffer.from_numpy(
            dev, np.array(attn_metadata.slot_mapping, dtype=np.uint32))
        bt_arr = np.array(attn_metadata.block_tables[0] if hasattr(attn_metadata, "block_tables") else [0],
                          dtype=np.uint32)
        bt_buf = WebGPUBuffer.from_numpy(dev, bt_arr)

        for i in range(self.num_layers):
            x_buf = self._transformer_layer(
                i, x_buf, pos_buf, slot_map, bt_buf, ctx_len, num_tokens)
            if i in self.ple_layer_indices:
                x_buf = self._ple_block(i, x_buf, ids_buf, num_tokens)

        norm_out = WebGPUBuffer.empty(dev, num_tokens * hidden * 2, usage=rw)
        self._dispatch("rms_norm",
                       [x_buf, self.weights["model.norm.weight"], norm_out],
                       {"HIDDEN_DIM": hidden}, (num_tokens, 1, 1))

        logits_buf = WebGPUBuffer.empty(dev, num_tokens * vocab * 2, usage=rw)
        lm_head_w = self.weights.get("lm_head.weight", self.weights["model.embed_tokens.weight"])
        self._dispatch("matmul_quant_mr4",
                       [norm_out, lm_head_w, self.weights.get("lm_head.scales", norm_out), logits_buf],
                       {"K": hidden, "N": vocab, "M": num_tokens, "USE_QUANT": 0},
                       ((num_tokens + 3) // 4, vocab, 1))

        # Apply Gemma logit softcap; CAP is f32 in WGSL, must pass as float
        capped = WebGPUBuffer.empty(dev, logits_buf.nbytes, usage=rw)
        self._dispatch("logit_softcap", [logits_buf, capped],
                       {"N": num_tokens * vocab, "CAP": float(self.softcap)},
                       ((num_tokens * vocab + 255) // 256, 1, 1),
                       shader_subdir="gemma")

        return capped.to_numpy().view(np.float16).reshape(num_tokens, vocab).astype(np.float32)

    def _transformer_layer(
        self,
        layer_idx: int,
        x_buf: "WebGPUBuffer",
        pos_buf: "WebGPUBuffer",
        slot_map: "WebGPUBuffer",
        bt_buf: "WebGPUBuffer",
        ctx_len: int,
        num_tokens: int,
    ) -> "WebGPUBuffer":
        """Gemma4 transformer layer using pre-allocated scratch buffers and batched dispatch."""
        import math

        sc = self._sc
        hidden = self.hidden_size
        p = f"model.layers.{layer_idx}"
        q_dim = self.num_q_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        inter = self.intermediate_size
        ln_rope = math.log(self.rope_theta)
        use_quant = 0 if self.weights.get(f"{p}.self_attn.q_proj.weight") is not None else 1

        h_names = ["h0", "h1", "h2"]
        residual = sc[h_names[(self._hstate + 1) % 3]]
        out = sc[h_names[(self._hstate + 2) % 3]]
        add_n = num_tokens * hidden
        gelu_n = num_tokens * inter

        k_cache, v_cache = self.kv_pool[layer_idx]

        with self._batched_dispatch():
            # Pre-norm
            self._dispatch("rms_norm", [x_buf, self.weights[f"{p}.input_layernorm.weight"], sc["normed"]],
                           {"HIDDEN_DIM": hidden}, (num_tokens, 1, 1))

            # QKV projection
            for out_buf, proj, dim in [(sc["q_buf"], "q_proj", q_dim),
                                       (sc["k_buf"], "k_proj", kv_dim),
                                       (sc["v_buf"], "v_proj", kv_dim)]:
                w_key = f"{p}.self_attn.{proj}.weight"
                s_key = f"{p}.self_attn.{proj}.scales"
                self._dispatch("matmul_quant",
                               [sc["normed"], self.weights[w_key], self.weights.get(s_key, sc["normed"]), out_buf],
                               {"K": hidden, "N": dim, "USE_QUANT": use_quant}, ((dim + 255) // 256, 1, 1))

            # Fused per-head norm + RoPE for Q and K
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
                                    "LN_ROPE_BASE": ln_rope,
                                    "HAS_WEIGHT": 1},
                                   (n_heads, num_tokens, 1))
                else:
                    self._dispatch("rope", [src, pos_buf, dst],
                                   {"HEAD_DIM": self.head_dim, "NUM_HEADS": n_heads,
                                    "LN_ROPE_BASE": ln_rope},
                                   (num_tokens, n_heads, 1))

            # Gemma4 per-head RMSNorm (no weight) on V before caching
            self._dispatch("per_head_rms_norm_no_weight", [sc["v_buf"], sc["v_normed"]],
                           {"HEAD_DIM": self.head_dim, "NUM_HEADS": self.num_kv_heads, "WG_SIZE": 128},
                           (self.num_kv_heads, num_tokens, 1), shader_subdir="gemma")

            # KV cache store
            self._dispatch("kv_cache_store", [sc["k_rope"], k_cache, slot_map],
                           {"BLOCK_SIZE": self.block_size, "NUM_KV_HEADS": self.num_kv_heads, "HEAD_DIM": self.head_dim},
                           (num_tokens, self.num_kv_heads, 1))
            self._dispatch("kv_cache_store", [sc["v_normed"], v_cache, slot_map],
                           {"BLOCK_SIZE": self.block_size, "NUM_KV_HEADS": self.num_kv_heads, "HEAD_DIM": self.head_dim},
                           (num_tokens, self.num_kv_heads, 1))

            # Attention scores + output
            self._dispatch("attn_score", [sc["q_rope"], k_cache, bt_buf, sc["scores_buf"]],
                           {"BLOCK_SIZE": self.block_size, "NUM_Q_HEADS": self.num_q_heads,
                            "NUM_KV_HEADS": self.num_kv_heads, "HEAD_DIM": self.head_dim,
                            "MAX_SEQ_LEN": ctx_len}, (self.num_q_heads, ctx_len, 1))

            self._dispatch("softmax", [sc["scores_buf"], sc["sm_buf"]],
                           {"SEQ_LEN": ctx_len}, (self.num_q_heads, 1, 1))

            self._dispatch("attn_output", [sc["sm_buf"], v_cache, bt_buf, sc["attn_out"]],
                           {"BLOCK_SIZE": self.block_size, "NUM_Q_HEADS": self.num_q_heads,
                            "NUM_KV_HEADS": self.num_kv_heads, "HEAD_DIM": self.head_dim,
                            "CTX_LEN": ctx_len}, (self.num_q_heads, 1, 1))

            # Output projection
            w_key = f"{p}.self_attn.o_proj.weight"
            s_key = f"{p}.self_attn.o_proj.scales"
            self._dispatch("matmul_quant", [sc["attn_out"], self.weights[w_key],
                                            self.weights.get(s_key, sc["attn_out"]), sc["o_proj_out"]],
                           {"K": q_dim, "N": hidden, "USE_QUANT": use_quant}, ((hidden + 255) // 256, 1, 1))

            # Residual add (vec4 path)
            self._dispatch("add", [x_buf, sc["o_proj_out"], residual],
                           {"N": add_n}, ((add_n // 4 + 255) // 256, 1, 1))

            # FFN pre-norm
            self._dispatch("rms_norm", [residual, self.weights[f"{p}.post_attention_layernorm.weight"], sc["ffn_normed"]],
                           {"HIDDEN_DIM": hidden}, (num_tokens, 1, 1))

            # Gate + up projection
            for out_b, proj in [(sc["gate_buf"], "gate_proj"), (sc["up_buf"], "up_proj")]:
                w_k = f"{p}.mlp.{proj}.weight"
                s_k = f"{p}.mlp.{proj}.scales"
                self._dispatch("matmul_quant", [sc["ffn_normed"], self.weights[w_k],
                                                self.weights.get(s_k, sc["ffn_normed"]), out_b],
                               {"K": hidden, "N": inter, "USE_QUANT": use_quant}, ((inter + 255) // 256, 1, 1))

            # SwiGLU (vec4 path)
            self._dispatch("gelu_mul", [sc["gate_buf"], sc["up_buf"], sc["ffn_act"]],
                           {"N": gelu_n}, ((gelu_n // 4 + 255) // 256, 1, 1))

            # Down projection
            w_k = f"{p}.mlp.down_proj.weight"
            s_k = f"{p}.mlp.down_proj.scales"
            self._dispatch("matmul_quant", [sc["ffn_act"], self.weights[w_k],
                                            self.weights.get(s_k, sc["ffn_act"]), sc["ffn_out"]],
                           {"K": inter, "N": hidden, "USE_QUANT": use_quant}, ((hidden + 255) // 256, 1, 1))

            # Final residual (vec4 path)
            self._dispatch("add", [residual, sc["ffn_out"], out],
                           {"N": add_n}, ((add_n // 4 + 255) // 256, 1, 1))

        self._hstate = (self._hstate + 2) % 3
        return out

    def _ple_block(self, layer_idx: int, x_buf: "WebGPUBuffer", ids_buf: "WebGPUBuffer", num_tokens: int) -> "WebGPUBuffer":
        """PLE pipeline: stage1_fuse -> gelu_mul -> skip_scale_add. Not yet implemented."""
        raise NotImplementedError("PLE block requires loaded PLE weights; not supported in this build")
