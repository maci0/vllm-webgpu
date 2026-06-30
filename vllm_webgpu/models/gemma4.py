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

        ids_buf = WebGPUBuffer.from_numpy(dev, input_ids.astype(np.uint32))
        x_buf = WebGPUBuffer.empty(dev, num_tokens * hidden * 2, usage=rw)
        self._dispatch("embedding_lookup",
                       [self.weights["model.embed_tokens.weight"], ids_buf, x_buf],
                       {"HIDDEN_DIM": hidden, "NUM_TOKENS": num_tokens},
                       (num_tokens, 1, 1))

        for i in range(self.num_layers):
            x_buf = self._transformer_layer(i, x_buf, positions, attn_metadata, num_tokens)
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

        # Apply Gemma logit softcap
        capped = WebGPUBuffer.empty(dev, logits_buf.nbytes, usage=rw)
        self._dispatch("logit_softcap", [logits_buf, capped],
                       {"N": num_tokens * vocab, "CAP": int(self.softcap)},
                       ((num_tokens * vocab + 255) // 256, 1, 1),
                       shader_subdir="gemma")

        return capped.to_numpy().view(np.float16).reshape(num_tokens, vocab).astype(np.float32)

    def _transformer_layer(
        self,
        layer_idx: int,
        x_buf: "WebGPUBuffer",
        positions: np.ndarray,
        attn_metadata: object,
        num_tokens: int,
    ) -> "WebGPUBuffer":
        """Llama-style transformer layer. Gemma4 differs from Llama in V-norm and PLE injection."""
        import wgpu as wgpu_lib
        from vllm_webgpu.webgpu.buffer import WebGPUBuffer

        dev = self.wgpu_device.wgpu_device
        rw = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST
        hidden = self.hidden_size
        p = f"model.layers.{layer_idx}"

        # Pre-norm
        normed = WebGPUBuffer.empty(dev, num_tokens * hidden * 2, usage=rw)
        self._dispatch("rms_norm", [x_buf, self.weights[f"{p}.input_layernorm.weight"], normed],
                       {"HIDDEN_DIM": hidden}, (num_tokens, 1, 1))

        # QKV projection
        q_dim = self.num_q_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        q_buf = WebGPUBuffer.empty(dev, num_tokens * q_dim * 2, usage=rw)
        k_buf = WebGPUBuffer.empty(dev, num_tokens * kv_dim * 2, usage=rw)
        v_buf = WebGPUBuffer.empty(dev, num_tokens * kv_dim * 2, usage=rw)

        use_quant = 0 if self.weights.get(f"{p}.self_attn.q_proj.weight") is not None else 1
        for out_buf, proj, dim in [(q_buf, "q_proj", q_dim), (k_buf, "k_proj", kv_dim), (v_buf, "v_proj", kv_dim)]:
            w_key = f"{p}.self_attn.{proj}.weight"
            s_key = f"{p}.self_attn.{proj}.scales"
            self._dispatch("matmul_quant",
                           [normed, self.weights[w_key], self.weights.get(s_key, normed), out_buf],
                           {"K": hidden, "N": dim, "USE_QUANT": use_quant}, (dim, 1, 1))

        # Fused per-head norm + RoPE for Q and K
        pos_buf = WebGPUBuffer.from_numpy(dev, positions.astype(np.uint32))
        q_rope = WebGPUBuffer.empty(dev, q_buf.nbytes, usage=rw)
        k_rope = WebGPUBuffer.empty(dev, k_buf.nbytes, usage=rw)
        for src, dst, n_heads, w_key in [
            (q_buf, q_rope, self.num_q_heads, f"{p}.self_attn.q_norm.weight"),
            (k_buf, k_rope, self.num_kv_heads, f"{p}.self_attn.k_norm.weight"),
        ]:
            norm_w = self.weights.get(w_key)
            if norm_w is not None:
                self._dispatch("fused_per_head_norm_rope",
                               [src, norm_w, pos_buf, dst],
                               {"HEAD_DIM": self.head_dim, "NUM_HEADS": n_heads,
                                "ROPE_BASE": int(self.rope_theta), "HAS_WEIGHT": 1},
                               (n_heads, num_tokens, 1))
            else:
                self._dispatch("rope", [src, pos_buf, dst],
                               {"HEAD_DIM": self.head_dim, "NUM_HEADS": n_heads},
                               (num_tokens, n_heads, 1))

        # KV cache store
        slot_map = WebGPUBuffer.from_numpy(
            dev, np.array(attn_metadata.slot_mapping, dtype=np.uint32))
        k_cache, v_cache = self.kv_pool[layer_idx]
        self._dispatch("kv_cache_store", [k_rope, k_cache, slot_map],
                       {"BLOCK_SIZE": 16, "NUM_KV_HEADS": self.num_kv_heads, "HEAD_DIM": self.head_dim},
                       (num_tokens, self.num_kv_heads, 1))
        self._dispatch("kv_cache_store", [v_buf, v_cache, slot_map],
                       {"BLOCK_SIZE": 16, "NUM_KV_HEADS": self.num_kv_heads, "HEAD_DIM": self.head_dim},
                       (num_tokens, self.num_kv_heads, 1))

        # Attention scores + output
        ctx_len = int(attn_metadata.max_decode_seq_len or num_tokens)
        bt_arr = np.array(attn_metadata.block_tables[0] if hasattr(attn_metadata, "block_tables") else [0],
                          dtype=np.uint32)
        bt_buf = WebGPUBuffer.from_numpy(dev, bt_arr)

        scores_buf = WebGPUBuffer.empty(dev, self.num_q_heads * ctx_len * 4, usage=rw)
        self._dispatch("attn_score", [q_rope, k_cache, bt_buf, scores_buf],
                       {"BLOCK_SIZE": 16, "NUM_Q_HEADS": self.num_q_heads,
                        "NUM_KV_HEADS": self.num_kv_heads, "HEAD_DIM": self.head_dim,
                        "MAX_SEQ_LEN": ctx_len}, (self.num_q_heads, ctx_len, 1))

        sm_buf = WebGPUBuffer.empty(dev, scores_buf.nbytes, usage=rw)
        self._dispatch("softmax", [scores_buf, sm_buf],
                       {"SEQ_LEN": ctx_len, "BATCH": self.num_q_heads}, (self.num_q_heads, 1, 1))

        attn_out = WebGPUBuffer.empty(dev, self.num_q_heads * self.head_dim * 2, usage=rw)
        self._dispatch("attn_output", [sm_buf, v_cache, bt_buf, attn_out],
                       {"BLOCK_SIZE": 16, "NUM_Q_HEADS": self.num_q_heads,
                        "NUM_KV_HEADS": self.num_kv_heads, "HEAD_DIM": self.head_dim,
                        "CTX_LEN": ctx_len}, (self.num_q_heads, 1, 1))

        # Output projection
        o_proj_out = WebGPUBuffer.empty(dev, num_tokens * hidden * 2, usage=rw)
        w_key = f"{p}.self_attn.o_proj.weight"
        s_key = f"{p}.self_attn.o_proj.scales"
        self._dispatch("matmul_quant", [attn_out, self.weights[w_key],
                                        self.weights.get(s_key, attn_out), o_proj_out],
                       {"K": q_dim, "N": hidden, "USE_QUANT": use_quant}, (hidden, 1, 1))

        # Residual add
        residual = WebGPUBuffer.empty(dev, x_buf.nbytes, usage=rw)
        self._dispatch("add", [x_buf, o_proj_out, residual],
                       {"N": num_tokens * hidden}, ((num_tokens * hidden + 255) // 256, 1, 1))

        # FFN pre-norm
        ffn_normed = WebGPUBuffer.empty(dev, num_tokens * hidden * 2, usage=rw)
        self._dispatch("rms_norm", [residual, self.weights[f"{p}.post_attention_layernorm.weight"], ffn_normed],
                       {"HIDDEN_DIM": hidden}, (num_tokens, 1, 1))

        # Gate + up projection
        inter = self.intermediate_size
        gate_buf = WebGPUBuffer.empty(dev, num_tokens * inter * 2, usage=rw)
        up_buf = WebGPUBuffer.empty(dev, num_tokens * inter * 2, usage=rw)
        for out_b, proj in [(gate_buf, "gate_proj"), (up_buf, "up_proj")]:
            w_k = f"{p}.mlp.{proj}.weight"
            s_k = f"{p}.mlp.{proj}.scales"
            self._dispatch("matmul_quant", [ffn_normed, self.weights[w_k],
                                            self.weights.get(s_k, ffn_normed), out_b],
                           {"K": hidden, "N": inter, "USE_QUANT": use_quant}, (inter, 1, 1))

        # SwiGLU
        ffn_act = WebGPUBuffer.empty(dev, num_tokens * inter * 2, usage=rw)
        self._dispatch("gelu_mul", [gate_buf, up_buf, ffn_act],
                       {"N": num_tokens * inter}, ((num_tokens * inter + 255) // 256, 1, 1))

        # Down projection
        ffn_out = WebGPUBuffer.empty(dev, num_tokens * hidden * 2, usage=rw)
        w_k = f"{p}.mlp.down_proj.weight"
        s_k = f"{p}.mlp.down_proj.scales"
        self._dispatch("matmul_quant", [ffn_act, self.weights[w_k],
                                        self.weights.get(s_k, ffn_act), ffn_out],
                       {"K": inter, "N": hidden, "USE_QUANT": use_quant}, (hidden, 1, 1))

        # Final residual
        out = WebGPUBuffer.empty(dev, residual.nbytes, usage=rw)
        self._dispatch("add", [residual, ffn_out, out],
                       {"N": num_tokens * hidden}, ((num_tokens * hidden + 255) // 256, 1, 1))

        return out

    def _ple_block(self, layer_idx: int, x_buf: "WebGPUBuffer", ids_buf: "WebGPUBuffer", num_tokens: int) -> "WebGPUBuffer":
        """PLE pipeline: stage1_fuse -> gelu_mul -> skip_scale_add. Not yet implemented."""
        raise NotImplementedError("PLE block requires loaded PLE weights; not supported in this build")
