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
            "normed":  mk(T * H * 2),
            "qkv_buf": mk(T * (Q + 2 * KV) * 2),  # [Q|K|V] f16 — fused QKV output
            "q_buf":       mk(T * Q * 2),
            "k_buf":       mk(T * KV * 2),
            "v_buf":       mk(T * KV * 2),
            "q_rope":      mk(T * Q * 2),
            "k_rope":      mk(T * KV * 2),
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

        _vpt = min((hidden + 255) // 256, 16) if hidden <= 4096 else 0
        _rms_base = {"HIDDEN_DIM": hidden, "VALS_PER_THREAD": _vpt}
        sc = self._sc

        with self._batched_dispatch():
            # Embed (single dispatch; removed the duplicate standalone dispatch)
            self._dispatch(
                "embedding_lookup",
                [self.weights["model.embed_tokens.weight"], ids_buf, x_buf],
                {"HIDDEN_DIM": hidden},
                (num_tokens, 1, 1),
            )

            # Pre-norm for layer 0 — subsequent layers' pre-norms are fused into
            # the previous layer's final add_rms_norm dispatch.
            self._dispatch(
                "rms_norm",
                [x_buf, self.weights["model.layers.0.input_layernorm.weight"], sc["normed"]],
                _rms_base, (num_tokens, 1, 1),
            )

            normed_x = sc["normed"]
            for i in range(self.num_layers):
                normed_x, x_buf = self._transformer_layer(
                    i, normed_x, x_buf, pos_buf, slot_map, bt_buf, ctx_len, num_tokens)

            # Final norm
            self._dispatch(
                "rms_norm",
                [x_buf, self.weights["model.norm.weight"], norm_out],
                _rms_base,
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
            # GPU argmax + staging copy — all inside the same command encoder.
            # After the single main sync, map the staging buffer directly (no 2nd sync).
            self._dispatch("argmax_f16", [logits_buf, self._ensure_sample_buf(vocab)],
                           {"N": vocab}, (1, 1, 1))
            self._copy_sample_to_staging()

        # 4-byte readback (argmax index) as the primary return;
        # also expose full logits lazily for callers that need them.
        self._last_logit_buf = logits_buf
        self._last_vocab     = vocab
        # Map staging buffer (already copied during main sync — no extra submit).
        tok = self._read_sample_tok()
        return np.array([[tok]], dtype=np.int32)  # shape (1, 1), 4 bytes

    def _ensure_sample_buf(self, vocab: int) -> "WebGPUBuffer":
        self._ensure_gpu_sampler(vocab)
        return self._gpu_sample_tok

    def logit_readback(self) -> "np.ndarray":
        """Full vocab logits GPU→CPU (only for temperature sampling or analysis)."""
        vocab = self._last_vocab
        return self._last_logit_buf.to_numpy().view(np.float16).reshape(1, vocab).astype(np.float32)

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
        """Returns (normed_out, raw_out).

        normed_out: sc['normed'] — pre-normalized for next layer's QKV input.
        raw_out: the updated hidden state (raw residual) for the next layer.

        The initial rms_norm is handled by the CALLER before the loop. This
        allows fusing the final residual-add with the next layer's pre-norm into
        a single add_rms_norm dispatch, saving 2 dispatches per non-last layer.
        """
        import math

        sc = self._sc
        hidden = self.hidden_size
        p = f"model.layers.{layer_idx}"
        q_dim = self.num_q_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        inter = self.intermediate_size
        ln_rope = math.log(self.rope_theta)
        quant_types = self.weights.get("__quant_types__", {})
        _qt = quant_types if isinstance(quant_types, dict) else {}

        def _uq(key: str) -> int:
            w = self.weights.get(key)
            if w is not None:
                dtype = getattr(w, "dtype", "f16")
                base = key[:-7]
                qmeta = self.weights.get("__quant_meta__", {})
                meta = qmeta.get(base, {}) if isinstance(qmeta, dict) else {}
                fmt = meta.get("fmt", "")
                if dtype == "i32":
                    return 4 if fmt == "awq_sym" else 3
                if dtype == "u8":
                    if fmt == "nvfp4_gpu":
                        return 6
                    if fmt == "int8_gpu":
                        return 7
                    return 5  # fp8_gpu
            tt = _qt.get(key, 0)
            if tt == 12:
                return 2
            if self.weights.get(key[:-7] + ".scales") is not None:
                return 1
            return 0

        _wg_size = 256
        _vpt = min((hidden + _wg_size - 1) // _wg_size, 16) if hidden <= _wg_size * 16 else 0
        _rms_c = {"HIDDEN_DIM": hidden, "VALS_PER_THREAD": _vpt}

        h_names = ["h0", "h1", "h2"]
        residual = sc[h_names[(self._hstate + 1) % 3]]
        out = sc[h_names[(self._hstate + 2) % 3]]
        add_n = num_tokens * hidden
        gelu_n = num_tokens * inter

        k_cache, v_cache = self.kv_pool[layer_idx]

        with self._batched_dispatch(label=f"L{layer_idx:02d}"):
            # normed_x is already the pre-normed input (no rms_norm dispatch here).

            def _scales(w_key: str, uq: int, fallback) -> "WebGPUBuffer":
                if uq in (3, 4, 5, 6):
                    return self.weights.get(w_key + ".scales", fallback)
                return self.weights.get(w_key[:-7] + ".scales", fallback)

            # QKV projections: fused for f16 with per-head norm weights; separate otherwise.
            q_wk = f"{p}.self_attn.q_proj.weight"
            k_wk = f"{p}.self_attn.k_proj.weight"
            v_wk = f"{p}.self_attn.v_proj.weight"
            uq_q, uq_k, uq_v = _uq(q_wk), _uq(k_wk), _uq(v_wk)
            _has_qnorm = self.weights.get(f"{p}.self_attn.q_norm.weight") is not None

            _use_fused_qkv = uq_q == 0 and uq_k == 0 and uq_v == 0 and _has_qnorm

            if _use_fused_qkv:
                # All f16 + per-head norms: single fused_qkv → qkv_buf[Q|K|V].
                self._dispatch("fused_qkv",
                               [normed_x, self.weights[q_wk], self.weights[k_wk], self.weights[v_wk],
                                sc["qkv_buf"]],
                               {"K": hidden, "Q_DIM": q_dim, "KV_DIM": kv_dim},
                               (q_dim + 2 * kv_dim, 1, 1))
                _q_src = sc["qkv_buf"]
                _k_src = sc["qkv_buf"]
                _v_src = sc["qkv_buf"]
                _v_offset = q_dim + kv_dim  # f16 elements before V section
            else:
                for out_buf, proj, dim, uq in [(sc["q_buf"], "q_proj", q_dim, uq_q),
                                               (sc["k_buf"], "k_proj", kv_dim, uq_k),
                                               (sc["v_buf"], "v_proj", kv_dim, uq_v)]:
                    w_key = f"{p}.self_attn.{proj}.weight"
                    qi = self._quant_extra(f"{p}.self_attn.{proj}", uq)
                    self._dispatch("matmul_quant",
                                   [normed_x, self.weights[w_key], _scales(w_key, uq, normed_x), out_buf],
                                   {"K": hidden, "N": dim, "USE_QUANT": uq,
                                    **({"SPLIT_K": 0} if uq not in (0, 3, 4, 5, 6) else {}), **qi},
                                   _gemv_wg(dim, uq))
                _q_src = sc["q_buf"]
                _k_src = sc["k_buf"]
                _v_src = sc["v_buf"]
                _v_offset = 0

            # Per-head norm + RoPE for Q and K.
            # When using fused QKV (f16 + per-head norms): single fused_qk_norm_rope dispatch.
            # Otherwise: two separate fused_per_head_norm_rope (or plain rope) calls.
            q_norm_w = self.weights.get(f"{p}.self_attn.q_norm.weight")
            k_norm_w = self.weights.get(f"{p}.self_attn.k_norm.weight")
            _rope_consts = {"HEAD_DIM": self.head_dim,
                            "ROPE_BASE": float(self.rope_theta),
                            "LN_ROPE_BASE": ln_rope}

            if _use_fused_qkv and q_norm_w is not None:
                # fused_qk_norm_rope: Q+K norm+rope in one dispatch.
                self._dispatch("fused_qk_norm_rope",
                               [sc["qkv_buf"], q_norm_w, k_norm_w, pos_buf,
                                sc["q_rope"], sc["k_rope"]],
                               {**_rope_consts,
                                "NUM_Q_HEADS": self.num_q_heads,
                                "NUM_KV_HEADS": self.num_kv_heads,
                                "HAS_WEIGHT": 1,
                                "INPUT_OFFSET_K": q_dim},
                               (self.num_q_heads + self.num_kv_heads, num_tokens, 1))
            else:
                for src, dst, n_heads, norm_w, in_off in [
                    (_q_src, sc["q_rope"], self.num_q_heads,  q_norm_w, 0),
                    (_k_src, sc["k_rope"], self.num_kv_heads, k_norm_w, q_dim if _use_fused_qkv else 0),
                ]:
                    if norm_w is not None:
                        self._dispatch("fused_per_head_norm_rope",
                                       [src, norm_w, pos_buf, dst],
                                       {**_rope_consts, "NUM_HEADS": n_heads,
                                        "HAS_WEIGHT": 1, "INPUT_OFFSET": in_off},
                                       (n_heads, num_tokens, 1))
                    else:
                        self._dispatch("rope", [src, pos_buf, dst],
                                       {**_rope_consts, "NUM_HEADS": n_heads},
                                       (num_tokens, n_heads, 1))

            # Fused K+V cache store.
            # When using fused QKV, V lives in qkv_buf starting at element (q_dim+kv_dim).
            self._dispatch("kv_cache_store_both",
                           [sc["k_rope"], k_cache, _v_src, v_cache, slot_map],
                           {"BLOCK_SIZE": self.block_size, "NUM_KV_HEADS": self.num_kv_heads,
                            "HEAD_DIM": self.head_dim, "V_IN_OFFSET": _v_offset},
                           (num_tokens, self.num_kv_heads, 1))

            # Attention scores + softmax + weighted V sum.
            # Three-pass approach: (num_q_heads × ctx_len) WGs for attn_score gives
            # full GPU utilization. flash_attn_decode exists but uses only num_q_heads
            # WGs (poor utilization for short ctx on integrated GPUs).
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
                           {"K": q_dim, "N": hidden, "USE_QUANT": uq,
                            **({"SPLIT_K": 0} if uq not in (0, 3, 4, 5, 6) else {}), **qi},
                           _gemv_wg(hidden, uq))

            # Fused post-attn residual-add + FFN pre-norm: saves 1 dispatch/layer.
            # residual = x_buf + o_proj_out; ffn_normed = rms_norm(residual, weight)
            self._dispatch("add_rms_norm",
                           [x_buf, sc["o_proj_out"],
                            self.weights[f"{p}.post_attention_layernorm.weight"],
                            residual, sc["ffn_normed"]],
                           _rms_c, (num_tokens, 1, 1))

            # FFN: fused_gate_act (f16) or separate matmul_quant (quantized).
            gw_k = f"{p}.mlp.gate_proj.weight"
            uw_k = f"{p}.mlp.up_proj.weight"
            uq_g = _uq(gw_k); uq_u = _uq(uw_k)
            if uq_g == 0 and uq_u == 0:
                # Single dispatch: GEMV for gate+up with inline SiLU → ffn_act.
                self._dispatch("fused_gate_act",
                               [sc["ffn_normed"], self.weights[gw_k], self.weights[uw_k],
                                sc["ffn_act"]],
                               {"K": hidden, "N": inter, "GELU": 0}, (inter, 1, 1))
            else:
                for out_b, w_k, uq2, mlp_proj in [
                        (sc["gate_buf"], gw_k, uq_g, "gate_proj"),
                        (sc["up_buf"],  uw_k, uq_u, "up_proj")]:
                    qi2 = self._quant_extra(f"{p}.mlp.{mlp_proj}", uq2)
                    self._dispatch("matmul_quant", [sc["ffn_normed"], self.weights[w_k],
                                                    _scales(w_k, uq2, sc["ffn_normed"]), out_b],
                                   {"K": hidden, "N": inter, "USE_QUANT": uq2,
                                    **({"SPLIT_K": 0} if uq2 not in (0, 3, 4, 5, 6) else {}), **qi2},
                                   _gemv_wg(inter, uq2))
                self._dispatch("gelu_mul", [sc["gate_buf"], sc["up_buf"], sc["ffn_act"]],
                               {"N": gelu_n}, ((gelu_n // 4 + 255) // 256, 1, 1))

            # Down projection
            w_k = f"{p}.mlp.down_proj.weight"
            uq = _uq(w_k)
            qi3 = self._quant_extra(f"{p}.mlp.down_proj", uq)
            self._dispatch("matmul_quant", [sc["ffn_act"], self.weights[w_k],
                                            _scales(w_k, uq, sc["ffn_act"]), sc["ffn_out"]],
                           {"K": inter, "N": hidden, "USE_QUANT": uq,
                            **({"SPLIT_K": 0} if uq not in (0, 3, 4, 5, 6) else {}), **qi3},
                           _gemv_wg(hidden, uq))

            # Final residual add: fuse with next layer's pre-norm when possible.
            # Last layer: plain add; intermediate layers: add_rms_norm saves 1 dispatch.
            if layer_idx < self.num_layers - 1:
                next_w = self.weights[f"model.layers.{layer_idx+1}.input_layernorm.weight"]
                self._dispatch("add_rms_norm",
                               [residual, sc["ffn_out"], next_w, out, sc["normed"]],
                               _rms_c, (num_tokens, 1, 1))
                normed_out = sc["normed"]
            else:
                self._dispatch("add", [residual, sc["ffn_out"], out],
                               {"N": add_n}, ((add_n // 4 + 255) // 256, 1, 1))
                normed_out = sc["normed"]  # stale; unused after last layer

        self._hstate = (self._hstate + 2) % 3
        return normed_out, out
