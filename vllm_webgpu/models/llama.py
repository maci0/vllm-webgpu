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


class LlamaWebGPUModel(BaseWebGPUModel):
    """
    Handles Llama 3.x and Qwen 2.5/3.x (architecturally identical).
    Layer order per token:
      embedding_lookup
      -> N x (rms_norm -> qkv_proj -> fused_per_head_norm_rope ->
              kv_cache_store -> attn_score -> softmax -> attn_output ->
              o_proj -> add -> rms_norm -> gate_proj + up_proj ->
              gelu_mul -> down_proj -> add)
      -> rms_norm -> lm_head -> logits
    """

    def __init__(self, model_config, wgpu_device: "WebGPUDevice", pipeline_cache: "PipelineCache") -> None:
        super().__init__(model_config, wgpu_device, pipeline_cache)
        self.num_layers: int = model_config.num_hidden_layers
        self.num_q_heads: int = model_config.num_attention_heads
        self.num_kv_heads: int = model_config.num_key_value_heads
        self.hidden_size: int = model_config.hidden_size
        self.intermediate_size: int = model_config.intermediate_size
        self.vocab_size: int = model_config.vocab_size
        self.head_dim: int = self.hidden_size // self.num_q_heads
        self.rope_theta: float = getattr(model_config, "rope_theta", 10000.0)

    def forward(
        self,
        input_ids: np.ndarray,
        positions: np.ndarray,
        attn_metadata: object,
    ) -> np.ndarray:
        """
        Args:
            input_ids:    [num_tokens]  uint32
            positions:    [num_tokens]  uint32
            attn_metadata: carries slot_mapping and block_table

        Returns:
            logits: [num_tokens, vocab_size]  float32
        """
        from vllm_webgpu.webgpu.buffer import WebGPUBuffer
        import wgpu as wgpu_lib

        dev = self.wgpu_device.wgpu_device
        num_tokens = len(input_ids)
        hidden = self.hidden_size
        rw_usage = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST

        # Embedding lookup
        ids_buf = WebGPUBuffer.from_numpy(dev, input_ids.astype(np.uint32))
        x_buf = WebGPUBuffer.empty(dev, num_tokens * hidden * 2, usage=rw_usage)
        self._dispatch(
            "embedding_lookup",
            [self.weights["model.embed_tokens.weight"], ids_buf, x_buf],
            {"HIDDEN_DIM": hidden, "NUM_TOKENS": num_tokens},
            (num_tokens, 1, 1),
        )

        # Transformer layers
        for i in range(self.num_layers):
            x_buf = self._transformer_layer(i, x_buf, positions, attn_metadata, num_tokens)

        # Final norm
        norm_out = WebGPUBuffer.empty(dev, num_tokens * hidden * 2, usage=rw_usage)
        self._dispatch(
            "rms_norm",
            [x_buf, self.weights["model.norm.weight"], norm_out],
            {"HIDDEN_DIM": hidden},
            (num_tokens, 1, 1),
        )

        # LM head projection: [num_tokens, hidden] x [vocab, hidden]^T -> [num_tokens, vocab]
        vocab = self.vocab_size
        logits_buf = WebGPUBuffer.empty(dev, num_tokens * vocab * 2, usage=rw_usage)
        self._dispatch(
            "matmul_quant_mr4",
            [norm_out, self.weights.get("lm_head.weight", self.weights["model.embed_tokens.weight"]),
             self.weights.get("lm_head.scales", norm_out),  # unused for f16
             logits_buf],
            {"K": hidden, "N": vocab, "M": num_tokens, "USE_QUANT": 0},
            ((num_tokens + 3) // 4, vocab, 1),
        )

        return logits_buf.to_numpy().view(np.float16).reshape(num_tokens, vocab).astype(np.float32)

    def _transformer_layer(
        self,
        layer_idx: int,
        x_buf: "WebGPUBuffer",
        positions: np.ndarray,
        attn_metadata: object,
        num_tokens: int,
    ) -> "WebGPUBuffer":
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

        # QKV projection (combined or separate depending on checkpoint)
        q_dim = self.num_q_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        q_buf = WebGPUBuffer.empty(dev, num_tokens * q_dim * 2, usage=rw)
        k_buf = WebGPUBuffer.empty(dev, num_tokens * kv_dim * 2, usage=rw)
        v_buf = WebGPUBuffer.empty(dev, num_tokens * kv_dim * 2, usage=rw)

        use_quant = 0 if self.weights.get(f"{p}.self_attn.q_proj.weight", None) is not None else 1
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

        scores_buf = WebGPUBuffer.empty(dev, self.num_q_heads * ctx_len * 4, usage=rw)  # f32
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
        up_buf2 = WebGPUBuffer.empty(dev, num_tokens * inter * 2, usage=rw)
        for out_b, proj in [(gate_buf, "gate_proj"), (up_buf2, "up_proj")]:
            w_k = f"{p}.mlp.{proj}.weight"
            s_k = f"{p}.mlp.{proj}.scales"
            self._dispatch("matmul_quant", [ffn_normed, self.weights[w_k],
                                            self.weights.get(s_k, ffn_normed), out_b],
                           {"K": hidden, "N": inter, "USE_QUANT": use_quant}, (inter, 1, 1))

        # SwiGLU
        ffn_act = WebGPUBuffer.empty(dev, num_tokens * inter * 2, usage=rw)
        self._dispatch("gelu_mul", [gate_buf, up_buf2, ffn_act],
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
