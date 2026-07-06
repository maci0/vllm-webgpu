from __future__ import annotations
import logging
import math
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


class Gemma4WebGPUModel(BaseWebGPUModel):
    """
    Gemma 4 transformer with heterogeneous per-layer attention.

    Gemma4-12B mixes two attention types:
    - Local layers (head_dim=256, 8 KV heads): standard GQA with per-head RMSNorm on V
    - Global layers (head_dim=512, 1 KV head): MQA, no separate V projection (V=K)

    Every 6th layer (indices 5, 11, 17, ...) is a global attention layer.
    Scratch buffers are allocated at maximum dimensions to handle both types.
    """

    def __init__(self, model_config, wgpu_device: "WebGPUDevice", pipeline_cache: "PipelineCache") -> None:
        super().__init__(model_config, wgpu_device, pipeline_cache)
        self.num_layers: int = model_config.num_hidden_layers
        self.num_q_heads: int = model_config.num_attention_heads
        self.hidden_size: int = model_config.hidden_size
        self.intermediate_size: int = model_config.intermediate_size
        self.vocab_size: int = model_config.vocab_size
        # Softcap is optional — Gemma4 uses 30.0, Gemma3 uses None (no cap)
        self.softcap: float | None = getattr(model_config, "final_logit_softcapping", None)
        self.rope_theta: float = getattr(model_config, "rope_theta", 10000.0)
        from vllm_webgpu.config import get_config
        self.block_size: int = get_config().block_size

        # Gemma3 vs Gemma4 capability flags:
        # - GEMMA_NORM=1: all Gemma models use (1+w) RMSNorm (weights trained as deviations from 0)
        # - _apply_v_norm: only Gemma4 applies per-head RMS norm to V before caching
        archs = getattr(model_config, "architectures", [])
        self._is_gemma4 = any("Gemma4" in a for a in archs)
        self._apply_v_norm = self._is_gemma4  # Gemma3 does NOT normalize V
        self._gemma_norm_const = 1             # (1+w) RMSNorm for all Gemma models

        # Per-layer attention parameters (head_dim, num_kv_heads, q_dim, kv_dim, has_v_proj).
        # Set from _layer_attention_params if available (parsed from GGUF), otherwise derive
        # using the heuristic that every 6th layer (idx%6==5) is global attention.
        raw_lp = getattr(model_config, "_layer_attention_params", None)
        default_hd = getattr(model_config, "head_dim",
                             self.hidden_size // self.num_q_heads)
        default_kv = getattr(model_config, "num_key_value_heads", 1)

        # Gemma4 safetensors: derive per-layer params from layer_types + global_head_dim.
        layer_types = getattr(model_config, "layer_types", None)
        global_hd   = getattr(model_config, "global_head_dim", default_hd)
        global_kv   = getattr(model_config, "global_kv_heads",
                              getattr(model_config, "num_global_key_value_heads", 1))

        if raw_lp and len(raw_lp) == self.num_layers:
            self._lp: list[dict] = raw_lp
        elif layer_types and len(layer_types) == self.num_layers:
            # Build per-layer params from layer_types list (Gemma4 safetensors config).
            # sliding_attention: local GQA, head_dim=default_hd, has_v_proj=True
            # full_attention:    global MQA, head_dim=global_hd, num_kv=1, has_v_proj=False
            self._lp = []
            for lt in layer_types:
                if lt == "full_attention":
                    hd_l = global_hd
                    nkv_l = global_kv
                    hv = False  # global: V = K, no separate v_proj
                else:
                    hd_l = default_hd
                    nkv_l = default_kv
                    hv = True
                self._lp.append({
                    "head_dim":    hd_l,
                    "num_q_heads": self.num_q_heads,
                    "num_kv_heads": nkv_l,
                    "q_dim":       self.num_q_heads * hd_l,
                    "kv_dim":      nkv_l * hd_l,
                    "has_v_proj":  hv,
                })
        else:
            # Uniform fallback: all layers use the config defaults.
            # For Gemma3 safetensors (uniform attention) this is correct.
            hd = default_hd
            nkv = default_kv
            uniform_lp = {
                "head_dim": hd,
                "num_q_heads": self.num_q_heads,
                "num_kv_heads": nkv,
                "q_dim": self.num_q_heads * hd,
                "kv_dim": nkv * hd,
                "has_v_proj": True,
            }
            self._lp = [dict(uniform_lp) for _ in range(self.num_layers)]

        # Validate even dimensions required by WGSL shaders
        for name, val in [("hidden_size", self.hidden_size),
                          ("intermediate_size", self.intermediate_size)]:
            if val % 4 != 0:
                raise ValueError(f"{name}={val} must be divisible by 4 for vec4<f16> shaders")

        # Expose a representative num_kv_heads/head_dim for compatibility (uses config defaults).
        self.num_kv_heads: int = getattr(model_config, "num_key_value_heads", 1)
        self.head_dim: int = getattr(model_config, "head_dim",
                                     self.hidden_size // self.num_q_heads)

        # Compute max dimensions across all layers for scratch buffer sizing
        max_q_dim = max(lp["q_dim"] for lp in self._lp)
        max_kv_dim = max(lp["kv_dim"] for lp in self._lp)
        max_ctx = getattr(model_config, "max_position_embeddings", 8192)
        self._max_q_dim = max_q_dim
        self._max_kv_dim = max_kv_dim

        self._init_scratch_buffers(max_ctx, max_q_dim, max_kv_dim)

    def _init_scratch_buffers(self, max_ctx: int, max_q_dim: int, max_kv_dim: int) -> None:
        """Pre-allocate scratch buffers at maximum layer dimensions."""
        import wgpu as wgpu_lib
        from vllm_webgpu.webgpu.buffer import WebGPUBuffer

        dev = self.wgpu_device.wgpu_device
        rw = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST
        T = 1
        H = self.hidden_size
        I = self.intermediate_size
        NQ = self.num_q_heads

        def mk(n: int) -> "WebGPUBuffer":
            return WebGPUBuffer.empty(dev, n, usage=rw)

        # Pre-allocated per-step buffers (reused every decode via write_buffer).
        V = self.vocab_size
        self._pre: dict[str, "WebGPUBuffer"] = {
            "ids":      mk(T * 4),         # [1] uint32 token id
            "pos":      mk(T * 4),         # [1] uint32 position
            "slot_map": mk(T * 4),         # [1] uint32 physical slot
            "bt":       mk(512 * 4),       # [512] uint32 block table
            "x":        mk(T * H * 4),     # [1, H] f32 residual
            "norm_out": mk(T * H * 2),     # [1, H] f16 final norm
            "logits":   mk(T * V * 2),     # [1, V] f16 logits
            "capped":   mk(T * V * 2),     # [1, V] f16 softcapped logits (Gemma4)
        }

        self._sc: dict[str, "WebGPUBuffer"] = {
            "normed":     mk(T * H * 2),          # f16
            "q_buf":      mk(T * max_q_dim * 2),  # f16
            "k_buf":      mk(T * max_kv_dim * 2), # f16
            "v_buf":      mk(T * max_kv_dim * 2), # f16
            "v_normed":   mk(T * max_kv_dim * 2), # f16
            "q_rope":     mk(T * max_q_dim * 2),  # f16
            "k_rope":     mk(T * max_kv_dim * 2), # f16
            "scores_buf": mk(NQ * max_ctx * 2),   # f16
            "sm_buf":     mk(NQ * max_ctx * 2),   # f16
            "attn_out":   mk(T * max_q_dim * 2),  # f16
            "o_proj_out": mk(T * H * 2),           # f16
            "ffn_normed": mk(T * H * 2),           # f16
            "gate_buf":   mk(T * I * 2),           # f16
            "up_buf":     mk(T * I * 2),           # f16
            "ffn_act":    mk(T * I * 2),           # f16
            "ffn_out":    mk(T * H * 2),           # f16
            "ffn_gate_up": mk(T * I * 4),          # [2*inter] f16 — fused gate+up
            # Residual buffers stored in f32 for precision.
            # Gemma4 has output_norm weights up to 600 which cause f16 saturation
            # when accumulated across 48 layers — f32 residuals prevent this.
            "h0":         mk(T * H * 4),           # f32 (4 bytes)
            "h1":         mk(T * H * 4),           # f32
            "h2":         mk(T * H * 4),           # f32
        }
        self._hstate: int = 0

    def _postprocess_weights(self) -> None:
        """Tile per-head norm weights that have shape (head_dim,) to (num_heads * head_dim,).

        Gemma4 q_norm/k_norm weights are per-head (shape = head_dim) but the
        fused_per_head_norm_rope shader indexes weight[head_idx * HEAD_DIM + i],
        requiring shape (num_heads * head_dim). Tile to match, using per-layer params.
        """
        import numpy as np
        import wgpu as wgpu_lib
        from vllm_webgpu.webgpu.buffer import WebGPUBuffer
        dev = self.wgpu_device.wgpu_device
        rw = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST

        for i, lp in enumerate(self._lp):
            p = f"model.layers.{i}"
            hd = lp["head_dim"]
            for norm_key, num_heads in [
                (f"{p}.self_attn.q_norm.weight", lp["num_q_heads"]),
                (f"{p}.self_attn.k_norm.weight", lp["num_kv_heads"]),
            ]:
                buf = self.weights.get(norm_key)
                if buf is None:
                    continue
                expected_len = num_heads * hd
                raw = buf.to_numpy().view(np.float16)
                if len(raw) == expected_len:
                    continue
                if len(raw) == hd:
                    tiled = np.tile(raw, num_heads)
                    self.weights[norm_key] = WebGPUBuffer.from_numpy(dev, tiled, usage=rw)
                else:
                    logger.warning("Unexpected q/k_norm shape for %s: got %d, expected %d or %d",
                                   norm_key, len(raw), expected_len, hd)

    def load_weights(self, path: str) -> None:
        super().load_weights(path)
        self._postprocess_weights()
        # Cache layer_scalar values on CPU at load time — avoids 48 GPU→CPU
        # readbacks per token (each to_numpy() is a blocking ~100µs sync).
        self._layer_scales: list[float] = []
        for i in range(self.num_layers):
            p = f"model.layers.{i}"
            ls_buf = (self.weights.get(f"{p}.self_attn.layer_scale") or
                      self.weights.get(f"{p}.layer_scalar"))
            if ls_buf is not None:
                import numpy as _np
                self._layer_scales.append(float(ls_buf.to_numpy().view(_np.float16)[0]))
            else:
                self._layer_scales.append(1.0)
        logger.info("Loaded %d weight tensors", len(self.weights))

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

        self._hstate = 0

        if hasattr(attn_metadata, "block_tables") and len(attn_metadata.block_tables) > 1:
            raise RuntimeError("multi-sequence batching not supported in this build")

        ctx_len = int(attn_metadata.max_decode_seq_len
                      if attn_metadata.max_decode_seq_len is not None
                      else num_tokens)
        if ctx_len <= 0:
            ctx_len = num_tokens
        if ctx_len > 65535:
            raise RuntimeError(f"ctx_len={ctx_len} exceeds WebGPU dispatch limit of 65535")

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
        dev.queue.write_buffer(pre["bt"].buf, 0, bt_arr.tobytes())

        ids_buf    = pre["ids"]
        pos_buf    = pre["pos"]
        slot_map   = pre["slot_map"]
        bt_buf     = pre["bt"]
        x_buf      = pre["x"]
        norm_out   = pre["norm_out"]
        logits_buf = pre["logits"]
        capped_buf = pre["capped"]

        _vpt = min((hidden + 255) // 256, 16) if hidden <= 4096 else 0
        _rms_base = {"HIDDEN_DIM": hidden, "VALS_PER_THREAD": _vpt, "GEMMA_NORM": self._gemma_norm_const}
        sc = self._sc

        with self._batched_dispatch():
            # Embedding lookup → f32 output for f32 residual pipeline
            self._dispatch("embedding_lookup_f32",
                           [self.weights["model.embed_tokens.weight"], ids_buf, x_buf],
                           {"HIDDEN_DIM": hidden}, (num_tokens, 1, 1))

            # Initial pre-norm for layer 0 (subsequent pre-norms are fused into each
            # layer's final add_f32_rms_norm dispatch).
            self._dispatch("rms_norm_f32in",
                           [x_buf, self.weights["model.layers.0.input_layernorm.weight"], sc["normed"]],
                           _rms_base, (num_tokens, 1, 1))

            normed_x = sc["normed"]
            for i in range(self.num_layers):
                normed_x, x_buf = self._transformer_layer(
                    i, normed_x, x_buf, pos_buf, slot_map, bt_buf, ctx_len, num_tokens)

            # Final norm: reads f32 residual, writes f16 norm_out
            self._dispatch("rms_norm_f32in",
                           [x_buf, self.weights["model.norm.weight"], norm_out],
                           _rms_base, (num_tokens, 1, 1))

            lm_head_w = self.weights.get("lm_head.weight",
                                         self.weights["model.embed_tokens.weight"])
            # vocab_size exceeds the 65535 workgroup-per-dimension limit, so the split-K
            # path is unusable. Force SPLIT_K=0 (row-per-thread) with ceil(vocab/256) WGs.
            self._dispatch("matmul_quant",
                           [norm_out, lm_head_w,
                            self.weights.get("lm_head.scales", norm_out), logits_buf],
                           {"K": hidden, "N": vocab, "USE_QUANT": 0, "SPLIT_K": 0},
                           ((vocab + 255) // 256, 1, 1))

            if self.softcap is not None and self.softcap > 0:
                self._dispatch("logit_softcap", [logits_buf, capped_buf],
                               {"N": num_tokens * vocab, "CAP": float(self.softcap)},
                               ((num_tokens * vocab + 255) // 256, 1, 1),
                               shader_subdir="gemma")
                result_buf = capped_buf
            else:
                result_buf = logits_buf

            self._dispatch("argmax_f16", [result_buf, self._ensure_sample_buf(vocab)],
                           {"N": vocab}, (1, 1, 1))
            self._copy_sample_to_staging()

        self._last_logit_buf = result_buf
        self._last_vocab     = vocab
        tok = self._read_sample_tok()
        return np.array([[tok]], dtype=np.int32)

    def logit_readback(self) -> "np.ndarray":
        return self._last_logit_buf.to_numpy().view(np.float16).reshape(1, self._last_vocab).astype(np.float32)

    def _ensure_sample_buf(self, vocab: int) -> "WebGPUBuffer":
        self._ensure_gpu_sampler(vocab)
        return self._gpu_sample_tok

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
        """Returns (normed_out, raw_out) — normed_out is sc['normed'] for next layer."""
        sc = self._sc
        lp = self._lp[layer_idx]
        hidden = self.hidden_size
        p = f"model.layers.{layer_idx}"
        q_dim = lp["q_dim"]
        kv_dim = lp["kv_dim"]
        head_dim = lp["head_dim"]
        num_kv_heads = lp["num_kv_heads"]
        has_v = lp["has_v_proj"]
        inter = self.intermediate_size
        ln_rope = math.log(self.rope_theta)
        # Per-weight quant detection helper: returns USE_QUANT value for a given weight key.
        # Q4_K (type 12) → use GPU Q4_K block decoder (USE_QUANT=2).
        # Everything else (F16, Q6_K after eager-dequant, etc.) → plain f16 (USE_QUANT=0).
        quant_types = self.weights.get("__quant_types__", {})
        _qt = quant_types if isinstance(quant_types, dict) else {}

        # Per-layer output scale from GGUF (layer_output_scale.weight ≈ 0.053).
        # Applied to sublayer contributions before residual add.
        # Layer scale cached at load_weights() — no GPU→CPU readback per token.
        _ls = self._layer_scales[layer_idx]

        def _uq(key: str) -> int:
            tt = _qt.get(key, 0)
            if tt == 12:  # Q4_K — GPU block decoder
                return 2
            # key ends with ".weight" (7 chars); trim to get the base, then append ".scales"
            base = key[:-7]  # e.g. "model.layers.0.self_attn.q_proj"
            if self.weights.get(base + ".scales") is not None:
                return 1  # simple custom Q4 with scales
            return 0      # f16 (including eagerly dequantized Q6_K)

        h_names = ["h0", "h1", "h2"]
        residual = sc[h_names[(self._hstate + 1) % 3]]
        out = sc[h_names[(self._hstate + 2) % 3]]
        add_n = num_tokens * hidden
        gelu_n = num_tokens * inter
        _vpt = min((hidden + 255) // 256, 16) if hidden <= 4096 else 0
        _rms_consts = {"HIDDEN_DIM": hidden, "VALS_PER_THREAD": _vpt, "GEMMA_NORM": self._gemma_norm_const}

        k_cache, v_cache = self.kv_pool[layer_idx]

        with self._batched_dispatch(label=f"L{layer_idx:02d}"):
            # normed_x already pre-normalized by caller (or previous layer's fused add_f32_rms_norm).

            # Q projection
            qw = f"{p}.self_attn.q_proj.weight"
            uq = _uq(qw)
            self._dispatch("matmul_quant",
                           [normed_x, self.weights[qw],
                            self.weights.get(f"{p}.self_attn.q_proj.scales", normed_x), sc["q_buf"]],
                           {"K": hidden, "N": q_dim, "USE_QUANT": uq, **({"SPLIT_K": 0} if uq else {})},
                           _gemv_wg(q_dim, uq))

            # K projection
            kw = f"{p}.self_attn.k_proj.weight"
            uq = _uq(kw)
            self._dispatch("matmul_quant",
                           [normed_x, self.weights[kw],
                            self.weights.get(f"{p}.self_attn.k_proj.scales", normed_x), sc["k_buf"]],
                           {"K": hidden, "N": kv_dim, "USE_QUANT": uq, **({"SPLIT_K": 0} if uq else {})},
                           _gemv_wg(kv_dim, uq))

            # V projection: for global layers V=K (no separate weight), reuse k_buf → v_buf
            if has_v:
                vw = f"{p}.self_attn.v_proj.weight"
                uq = _uq(vw)
                self._dispatch("matmul_quant",
                               [normed_x, self.weights[vw],
                                self.weights.get(f"{p}.self_attn.v_proj.scales", normed_x), sc["v_buf"]],
                               {"K": hidden, "N": kv_dim, "USE_QUANT": uq, **({"SPLIT_K": 0} if uq else {})},
                               _gemv_wg(kv_dim, uq))
                v_src = sc["v_buf"]
            else:
                v_src = sc["k_buf"]

            # Fused per-head RMSNorm + RoPE for Q and K
            for src, dst, n_heads, w_key in [
                (sc["q_buf"], sc["q_rope"], self.num_q_heads, f"{p}.self_attn.q_norm.weight"),
                (sc["k_buf"], sc["k_rope"], num_kv_heads, f"{p}.self_attn.k_norm.weight"),
            ]:
                norm_w = self.weights.get(w_key)
                if norm_w is not None:
                    self._dispatch("fused_per_head_norm_rope",
                                   [src, norm_w, pos_buf, dst],
                                   {"HEAD_DIM": head_dim, "NUM_HEADS": n_heads,
                                    "ROPE_BASE": float(self.rope_theta),
                                    "LN_ROPE_BASE": ln_rope, "HAS_WEIGHT": 1,
                                    "GEMMA_NORM": self._gemma_norm_const},
                                   (n_heads, num_tokens, 1))
                else:
                    self._dispatch("rope", [src, pos_buf, dst],
                                   {"HEAD_DIM": head_dim, "NUM_HEADS": n_heads,
                                    "LN_ROPE_BASE": ln_rope},
                                   (num_tokens, n_heads, 1))

            # Per-head RMSNorm (no weight) on V before caching — Gemma4 only.
            # Gemma3 does NOT apply V normalization (no v_norm weight in the model).
            if self._apply_v_norm:
                self._dispatch("per_head_rms_norm_no_weight", [v_src, sc["v_normed"]],
                               {"HEAD_DIM": head_dim, "NUM_HEADS": num_kv_heads, "WG_SIZE": min(head_dim, 128)},
                               (num_kv_heads, num_tokens, 1), shader_subdir="gemma")
                v_to_cache = sc["v_normed"]
            else:
                v_to_cache = v_src  # Gemma3: use V directly without normalization

            # Fused K+V cache store — single dispatch saves 1 overhead per layer.
            self._dispatch("kv_cache_store_both",
                           [sc["k_rope"], k_cache, v_to_cache, v_cache, slot_map],
                           {"BLOCK_SIZE": self.block_size, "NUM_KV_HEADS": num_kv_heads,
                            "HEAD_DIM": head_dim},
                           (num_tokens, num_kv_heads, 1))

            # Attention scores, softmax, weighted V sum
            self._dispatch("attn_score",
                           [sc["q_rope"], k_cache, bt_buf, sc["scores_buf"]],
                           {"BLOCK_SIZE": self.block_size, "NUM_Q_HEADS": self.num_q_heads,
                            "NUM_KV_HEADS": num_kv_heads, "HEAD_DIM": head_dim,
                            "MAX_SEQ_LEN": ctx_len},
                           (self.num_q_heads, ctx_len, 1))

            self._dispatch("softmax", [sc["scores_buf"], sc["sm_buf"]],
                           {"SEQ_LEN": ctx_len}, (self.num_q_heads, 1, 1))

            self._dispatch("attn_output",
                           [sc["sm_buf"], v_cache, bt_buf, sc["attn_out"]],
                           {"BLOCK_SIZE": self.block_size, "NUM_Q_HEADS": self.num_q_heads,
                            "NUM_KV_HEADS": num_kv_heads, "HEAD_DIM": head_dim,
                            "CTX_LEN": ctx_len},
                           (self.num_q_heads, 1, 1))

            # Output projection → sc["o_proj_out"]
            ow = f"{p}.self_attn.o_proj.weight"
            uq = _uq(ow)
            self._dispatch("matmul_quant",
                           [sc["attn_out"], self.weights[ow],
                            self.weights.get(f"{p}.self_attn.o_proj.scales", sc["attn_out"]), sc["o_proj_out"]],
                           {"K": q_dim, "N": hidden, "USE_QUANT": uq, **({"SPLIT_K": 0} if uq else {})},
                           _gemv_wg(hidden, uq))

            # Correct Gemma4 attention sublayer (matches HF Gemma3DecoderLayer.forward):
            #   residual = x
            #   hidden = input_layernorm(x)     → attn → o_proj
            #   hidden = post_attention_layernorm(hidden)   ← norm on ATTN OUTPUT (before residual)
            #   residual = residual + hidden                ← residual add AFTER norm
            post_attn_norm_w = self.weights.get(f"{p}.post_attention_layernorm.weight")
            if post_attn_norm_w is not None:
                self._dispatch("rms_norm", [sc["o_proj_out"], post_attn_norm_w, sc["ffn_normed"]],
                               _rms_consts, (num_tokens, 1, 1))
                attn_delta = sc["ffn_normed"]
            else:
                attn_delta = sc["o_proj_out"]

            # Fused: add_f32(x_buf, attn_delta, residual) + rms_norm_f32in(residual, pre_ffn_norm, normed)
            # saves 1 dispatch vs the separate add_f32 → rms_norm_f32in sequence.
            pre_ffn_norm_w = self.weights.get(f"{p}.pre_feedforward_layernorm.weight")
            if pre_ffn_norm_w is not None:
                self._dispatch("add_f32_rms_norm",
                               [x_buf, attn_delta, pre_ffn_norm_w, residual, sc["normed"]],
                               {**_rms_consts, "SCALE": _ls}, (num_tokens, 1, 1))
                ffn_normed = sc["normed"]
            else:
                self._dispatch("add_f32", [x_buf, attn_delta, residual],
                               {"N": add_n, "SCALE": _ls}, ((add_n // 4 + 255) // 256, 1, 1))
                ffn_normed = residual

            # Gate + up projection
            # Fused gate+up (f16 only); Gemma uses tanh-GELU.
            gw_k = f"{p}.mlp.gate_proj.weight"
            uw_k = f"{p}.mlp.up_proj.weight"
            uq_g = _uq(gw_k); uq_u = _uq(uw_k)
            if uq_g == 0 and uq_u == 0:
                self._dispatch("fused_gate_up",
                               [ffn_normed, self.weights[gw_k], self.weights[uw_k],
                                sc["ffn_gate_up"]],
                               {"K": hidden, "N": inter}, (inter, 1, 1))
                self._dispatch("gelu_mul_fused", [sc["ffn_gate_up"], sc["ffn_act"]],
                               {"N": gelu_n, "GELU": 1},
                               ((gelu_n // 4 + 255) // 256, 1, 1))
            else:
                for out_b, proj, w_k, uq in [
                        (sc["gate_buf"], "gate_proj", gw_k, uq_g),
                        (sc["up_buf"],   "up_proj",   uw_k, uq_u)]:
                    self._dispatch("matmul_quant",
                                   [ffn_normed, self.weights[w_k],
                                    self.weights.get(w_k[:-7]+".scales", ffn_normed), out_b],
                                   {"K": hidden, "N": inter, "USE_QUANT": uq,
                                    **({"SPLIT_K": 0} if uq else {})},
                                   _gemv_wg(inter, uq))
                self._dispatch("gelu_mul", [sc["gate_buf"], sc["up_buf"], sc["ffn_act"]],
                               {"N": gelu_n}, ((gelu_n // 4 + 255) // 256, 1, 1),
                               shader_subdir="gemma")

            # Down projection → sc["ffn_out"]
            w_k = f"{p}.mlp.down_proj.weight"
            s_k = f"{p}.mlp.down_proj.scales"
            uq = _uq(w_k)
            self._dispatch("matmul_quant",
                           [sc["ffn_act"], self.weights[w_k],
                            self.weights.get(s_k, sc["ffn_act"]), sc["ffn_out"]],
                           {"K": inter, "N": hidden, "USE_QUANT": uq, **({"SPLIT_K": 0} if uq else {})},
                           _gemv_wg(hidden, uq))

            # Post-FFN norm on FFN output (before residual add), then fused residual + next pre-norm.
            post_ffw_w = self.weights.get(f"{p}.post_feedforward_layernorm.weight")
            if post_ffw_w is not None:
                self._dispatch("rms_norm", [sc["ffn_out"], post_ffw_w, sc["o_proj_out"]],
                               _rms_consts, (num_tokens, 1, 1))
                ffn_delta = sc["o_proj_out"]
            else:
                ffn_delta = sc["ffn_out"]

            # Fused final add_f32 + next layer's input_layernorm for non-last layers.
            # Last layer: plain add_f32 (final norm is separate).
            if layer_idx < self.num_layers - 1:
                next_w = self.weights[f"model.layers.{layer_idx+1}.input_layernorm.weight"]
                self._dispatch("add_f32_rms_norm",
                               [residual, ffn_delta, next_w, out, sc["normed"]],
                               {**_rms_consts, "SCALE": _ls}, (num_tokens, 1, 1))
            else:
                self._dispatch("add_f32", [residual, ffn_delta, out],
                               {"N": add_n, "SCALE": _ls}, ((add_n // 4 + 255) // 256, 1, 1))

        self._hstate = (self._hstate + 2) % 3
        return sc["normed"], out
