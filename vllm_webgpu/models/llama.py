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

    SPLIT_K=1 (one workgroup per output row): USE_QUANT in (0,3,4,5,6).
    Row-per-thread: USE_QUANT in (1,2).
    """
    if uq in (0, 3, 4, 5, 6):
        return (N, 1, 1)
    return ((N + 255) // 256, 1, 1)


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
        # Use explicit head_dim when present (e.g. Qwen3: head_dim=128, hidden=2560, heads=32,
        # so hidden//heads=80 but actual Q dim per head is 128).
        self.head_dim: int = getattr(model_config, "head_dim", self.hidden_size // self.num_q_heads)
        self.rope_theta: float = getattr(model_config, "rope_theta", 10000.0)
        from vllm_webgpu.config import get_config
        self.block_size: int = get_config().block_size
        # matmul_quant f16 path packs two f16 values per u32. Row boundaries only
        # align to u32 boundaries when K is even; odd K silently produces wrong results.
        for name, val in [("hidden_size", self.hidden_size),
                          ("intermediate_size", self.intermediate_size),
                          ("head_dim", self.head_dim)]:
            if val % 2 != 0:
                raise ValueError(f"{name}={val} must be even for f16 GEMV")
        # add.wgsl and gelu_mul.wgsl use vec4<f16>: dimensions must be divisible by 4.
        for name, val in [("hidden_size", self.hidden_size),
                          ("intermediate_size", self.intermediate_size)]:
            if val % 4 != 0:
                raise ValueError(f"{name}={val} must be divisible by 4 for vec4<f16> shaders")
        max_ctx = getattr(model_config, "max_position_embeddings", 8192)
        self._init_scratch_buffers(max_ctx)

    def _init_scratch_buffers(self, max_ctx: int) -> None:
        """Pre-allocate all intermediate scratch buffers used in _transformer_layer.

        Eliminates 17 GPU buffer allocations per layer per decode token.
        Decode path only (num_tokens=1). Sizes are fixed by model dimensions.
        """
        import wgpu as wgpu_lib
        from vllm_webgpu.webgpu.buffer import WebGPUBuffer

        dev = self.wgpu_device.wgpu_device
        rw = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST
        T = 1  # decode: num_tokens == 1
        H = self.hidden_size
        I = self.intermediate_size
        Q = self.num_q_heads * self.head_dim
        KV = self.num_kv_heads * self.head_dim
        NQ = self.num_q_heads

        def mk(n: int) -> "WebGPUBuffer":
            return WebGPUBuffer.empty(dev, n, usage=rw)

        # Pre-allocated per-step buffers: reused every decode call via write_buffer.
        # Eliminates GPU allocation overhead (~5-10ms per token on Metal).
        max_bt_blocks = 512  # max block table entries; enough for 512 * block_size ctx
        self._pre: dict[str, "WebGPUBuffer"] = {
            "ids":      mk(T * 4),              # [1] uint32 token id
            "pos":      mk(T * 4),              # [1] uint32 position
            "slot_map": mk(T * 4),              # [1] uint32 physical slot
            "bt":       mk(max_bt_blocks * 4),  # [max_blocks] uint32 block table
            "x":        mk(T * H * 2),          # [1, H] f16 residual / embedding
            "norm_out": mk(T * H * 2),          # [1, H] f16 final norm output
            "logits":   mk(T * self.vocab_size * 2),  # [1, vocab] f16 logits
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
            "gate_buf":   mk(T * I * 2),
            "up_buf":     mk(T * I * 2),
            "ffn_act":    mk(T * I * 2),
            "ffn_out":    mk(T * H * 2),
            # Three hidden-state buffers: ping-pong between h0/h1/h2 so that
            # x_buf, residual, and out are always distinct within a single layer.
            "h0":         mk(T * H * 2),
            "h1":         mk(T * H * 2),
            "h2":         mk(T * H * 2),
        }
        # Index into hidden-state rotation: the layer output cycles h0 -> h1 -> h2 -> h0 ...
        self._hstate: int = 0

    def _postprocess_weights(self) -> None:
        """Fix weight shapes that differ between model variants.

        Qwen3's q_norm/k_norm weights are shared across all heads: shape (head_dim,).
        Our fused_per_head_norm_rope shader indexes weight[head_idx * HEAD_DIM + i],
        expecting shape (num_heads * head_dim,). Tile if the loaded shape is just (head_dim,).
        """
        import numpy as np
        import wgpu as wgpu_lib
        from vllm_webgpu.webgpu.buffer import WebGPUBuffer
        dev = self.wgpu_device.wgpu_device
        rw = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST

        for i in range(self.num_layers):
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
                    # Shared norm: tile to (num_heads * head_dim,) so each head uses same weights.
                    tiled = np.tile(buf.to_numpy().view(np.float16), num_heads)
                    self.weights[norm_key] = WebGPUBuffer.from_numpy(dev, tiled, usage=rw)
                else:
                    raise ValueError(
                        f"{norm_key}: unexpected shape {buf.shape}, "
                        f"expected {expected} or ({self.head_dim},)"
                    )

    def load_weights(self, path: str) -> None:
        super().load_weights(path)
        self._postprocess_weights()

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
        if ctx_len > 65535:
            raise RuntimeError(
                f"ctx_len={ctx_len} exceeds WebGPU dispatch limit of 65535. "
                "Long-context support requires splitting the attention computation."
            )

        # Update pre-allocated buffers via write_buffer — no GPU allocation per step.
        pre = self._pre
        dev.queue.write_buffer(pre["ids"].buf, 0, input_ids.astype(np.uint32).tobytes())
        dev.queue.write_buffer(pre["pos"].buf, 0, positions.astype(np.uint32).tobytes())
        dev.queue.write_buffer(
            pre["slot_map"].buf, 0,
            np.array(attn_metadata.slot_mapping, dtype=np.uint32).tobytes())
        bt_arr = np.array(
            attn_metadata.block_tables[0] if hasattr(attn_metadata, "block_tables") else [0],
            dtype=np.uint32)
        # bt_buf is pre-allocated for up to 512 blocks; write only what's needed.
        dev.queue.write_buffer(pre["bt"].buf, 0, bt_arr.tobytes())

        ids_buf   = pre["ids"]
        pos_buf   = pre["pos"]
        slot_map  = pre["slot_map"]
        bt_buf    = pre["bt"]
        x_buf     = pre["x"]
        norm_out  = pre["norm_out"]
        logits_buf = pre["logits"]

        vocab = self.vocab_size

        with self._batched_dispatch():
            # Embed (single dispatch; removed the duplicate standalone dispatch)
            self._dispatch(
                "embedding_lookup",
                [self.weights["model.embed_tokens.weight"], ids_buf, x_buf],
                {"HIDDEN_DIM": hidden},
                (num_tokens, 1, 1),
            )

            for i in range(self.num_layers):
                x_buf = self._transformer_layer(
                    i, x_buf, pos_buf, slot_map, bt_buf, ctx_len, num_tokens)

            # Final norm
            _vpt = min((hidden + 255) // 256, 16) if hidden <= 4096 else 0
            self._dispatch(
                "rms_norm",
                [x_buf, self.weights["model.norm.weight"], norm_out],
                {"HIDDEN_DIM": hidden, "VALS_PER_THREAD": _vpt},
                (num_tokens, 1, 1),
            )

            # LM head: vocab_size (e.g. 151936) exceeds the WebGPU
            # maxComputeWorkgroupsPerDimension limit of 65535, so the split-K path
            # (one workgroup per output row) is unusable. Force SPLIT_K=0 to use the
            # row-per-thread path, which dispatches ceil(vocab/256) workgroups instead.
            self._dispatch(
                "matmul_quant",
                [norm_out, self.weights.get("lm_head.weight", self.weights["model.embed_tokens.weight"]),
                 self.weights.get("lm_head.scales", norm_out),  # unused for f16
                 logits_buf],
                {"K": hidden, "N": vocab, "USE_QUANT": 0, "SPLIT_K": 0},
                ((vocab + 255) // 256, 1, 1),
            )
        # GPU readback after the single submit has completed.
        return logits_buf.to_numpy().view(np.float16).reshape(num_tokens, vocab).astype(np.float32)

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
        import math

        sc = self._sc
        hidden = self.hidden_size
        p = f"model.layers.{layer_idx}"
        q_dim = self.num_q_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        inter = self.intermediate_size
        ln_rope = math.log(self.rope_theta)
        # Per-weight quant detection: Q4_K (type 12) → GPU block decoder (USE_QUANT=2).
        # Eagerly-dequantized Q6_K and all f16 weights → USE_QUANT=0.
        quant_types = self.weights.get("__quant_types__", {})
        _qt = quant_types if isinstance(quant_types, dict) else {}

        def _uq(key: str) -> int:
            w = self.weights.get(key)
            if w is not None:
                dtype = getattr(w, "dtype", "f16")
                base = key[:-7]  # strip ".weight"
                qmeta = self.weights.get("__quant_meta__", {})
                meta = qmeta.get(base, {}) if isinstance(qmeta, dict) else {}
                fmt = meta.get("fmt", "")
                if dtype == "i32":
                    if fmt == "awq_sym":
                        return 4  # GPU AWQ
                    return 3  # GPU GPTQ
                if dtype == "u8":
                    if fmt == "nvfp4_gpu":
                        return 6  # GPU NVFP4
                    if fmt == "fp8_gpu":
                        return 5  # GPU FP8
            tt = _qt.get(key, 0)
            if tt == 12:  # Q4_K — use GPU block decoder
                return 2
            if self.weights.get(key[:-7] + ".scales") is not None:
                return 1  # simple custom Q4 with separate scales
            return 0      # f16 (including CPU-dequantized tensors)
        # Register-tile depth for rms_norm: HIDDEN_DIM / WG_SIZE (max 16 for HIDDEN_DIM≤4096).
        # Set to 0 for large models (HIDDEN_DIM > 4096) to use global re-read fallback.
        _wg_size = 256
        _vals_per_thread = min((hidden + _wg_size - 1) // _wg_size, 16) if hidden <= _wg_size * 16 else 0

        # Hidden-state rotation: h0/h1/h2 cycle so x_buf, residual, out are always distinct.
        h_names = ["h0", "h1", "h2"]
        residual = sc[h_names[(self._hstate + 1) % 3]]
        out = sc[h_names[(self._hstate + 2) % 3]]
        add_n = num_tokens * hidden
        gelu_n = num_tokens * inter

        k_cache, v_cache = self.kv_pool[layer_idx]

        _rms_consts = {"HIDDEN_DIM": hidden, "VALS_PER_THREAD": _vals_per_thread}

        with self._batched_dispatch(label=f"L{layer_idx:02d}"):
            # Pre-norm
            self._dispatch("rms_norm", [x_buf, self.weights[f"{p}.input_layernorm.weight"], sc["normed"]],
                           _rms_consts, (num_tokens, 1, 1))

            def _scales(w_key: str, uq: int, fallback) -> "WebGPUBuffer":
                """Return the scales buffer for this weight (any quant format)."""
                if uq in (3, 4, 5, 6):
                    # GPU INT4/FP8/NVFP4: companion scales stored as w_key + ".scales"
                    return self.weights.get(w_key + ".scales", fallback)
                return self.weights.get(w_key[:-7] + ".scales", fallback)

            # QKV projection
            for out_buf, proj, dim in [(sc["q_buf"], "q_proj", q_dim),
                                       (sc["k_buf"], "k_proj", kv_dim),
                                       (sc["v_buf"], "v_proj", kv_dim)]:
                w_key = f"{p}.self_attn.{proj}.weight"
                uq = _uq(w_key)
                qi = self._quant_extra(f"{p}.self_attn.{proj}", uq)
                self._dispatch("matmul_quant",
                               [sc["normed"], self.weights[w_key], _scales(w_key, uq, sc["normed"]), out_buf],
                               {"K": hidden, "N": dim, "USE_QUANT": uq, **({"SPLIT_K": 0} if uq not in (0, 3, 4, 5, 6) else {}), **qi},
                               _gemv_wg(dim, uq))

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

            # KV cache store
            self._dispatch("kv_cache_store", [sc["k_rope"], k_cache, slot_map],
                           {"BLOCK_SIZE": self.block_size, "NUM_KV_HEADS": self.num_kv_heads, "HEAD_DIM": self.head_dim},
                           (num_tokens, self.num_kv_heads, 1))
            self._dispatch("kv_cache_store", [sc["v_buf"], v_cache, slot_map],
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
            uq = _uq(w_key)
            qi = self._quant_extra(f"{p}.self_attn.o_proj", uq)
            self._dispatch("matmul_quant", [sc["attn_out"], self.weights[w_key],
                                            _scales(w_key, uq, sc["attn_out"]), sc["o_proj_out"]],
                           {"K": q_dim, "N": hidden, "USE_QUANT": uq, **({"SPLIT_K": 0} if uq not in (0, 3, 4, 5, 6) else {}), **qi},
                           _gemv_wg(hidden, uq))

            # Residual add (vec4 path: dispatch N/4 threads)
            self._dispatch("add", [x_buf, sc["o_proj_out"], residual],
                           {"N": add_n}, ((add_n // 4 + 255) // 256, 1, 1))

            # FFN pre-norm
            self._dispatch("rms_norm", [residual, self.weights[f"{p}.post_attention_layernorm.weight"], sc["ffn_normed"]],
                           _rms_consts, (num_tokens, 1, 1))

            # Gate + up projection
            for out_b, mlp_proj in [(sc["gate_buf"], "gate_proj"), (sc["up_buf"], "up_proj")]:
                w_k = f"{p}.mlp.{mlp_proj}.weight"
                uq = _uq(w_k)
                qi2 = self._quant_extra(f"{p}.mlp.{mlp_proj}", uq)
                self._dispatch("matmul_quant", [sc["ffn_normed"], self.weights[w_k],
                                                _scales(w_k, uq, sc["ffn_normed"]), out_b],
                               {"K": hidden, "N": inter, "USE_QUANT": uq, **({"SPLIT_K": 0} if uq not in (0, 3, 4, 5, 6) else {}), **qi2},
                               _gemv_wg(inter, uq))

            # SwiGLU (vec4 path: dispatch N/4 threads)
            self._dispatch("gelu_mul", [sc["gate_buf"], sc["up_buf"], sc["ffn_act"]],
                           {"N": gelu_n}, ((gelu_n // 4 + 255) // 256, 1, 1))

            # Down projection
            w_k = f"{p}.mlp.down_proj.weight"
            uq = _uq(w_k)
            qi3 = self._quant_extra(f"{p}.mlp.down_proj", uq)
            self._dispatch("matmul_quant", [sc["ffn_act"], self.weights[w_k],
                                            _scales(w_k, uq, sc["ffn_act"]), sc["ffn_out"]],
                           {"K": inter, "N": hidden, "USE_QUANT": uq, **({"SPLIT_K": 0} if uq not in (0, 3, 4, 5, 6) else {}), **qi3},
                           _gemv_wg(hidden, uq))

            # Final residual (vec4 path)
            self._dispatch("add", [residual, sc["ffn_out"], out],
                           {"N": add_n}, ((add_n // 4 + 255) // 256, 1, 1))

        # Advance rotation: next layer's x_buf = out = h[(hstate+2)%3]
        self._hstate = (self._hstate + 2) % 3
        return out
