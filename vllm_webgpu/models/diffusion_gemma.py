from __future__ import annotations
import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from vllm_webgpu.models.base import _gemv_wg, _H_NAMES
from vllm_webgpu.models.gemma4 import Gemma4WebGPUModel

if TYPE_CHECKING:
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.device import WebGPUDevice
    from vllm_webgpu.webgpu.pipeline import PipelineCache

logger = logging.getLogger(__name__)


class DiffusionGemmaWebGPUModel(Gemma4WebGPUModel):
    """DiffusionGemma: Gemma4 backbone with Shared Expert + MoE FFN.

    Architecture (per layer, model.decoder.layers.N.*):
      - Attention: standard GQA with sliding window (same as Gemma3/4)
      - Shared expert FFN: mlp.gate/up/down (intermediate_size=2112, BF16, always active)
      - MoE FFN: 128 routed experts, top-8 active (moe_intermediate_size=704, NVFP4)
      - Router: router.proj.weight [128, hidden] BF16 + router.scale [hidden] BF16
      - Multiple norm layers per sublayer

    Weight key prefix: model.decoder.layers.N.* (NOT model.layers.N.*)

    Quantization (NVIDIA ModelOpt NVFP4, group_size=16):
      Expert weights: *.weight (U8) + *.weight_scale (F8_E4M3) + *.weight_scale_2 (F32)
      All other weights: BF16 (attention, router, shared mlp, norms)

    Diffusion-specific features (future work):
      - Canvas-based block generation (canvas_length=256)
      - Bidirectional attention during denoising
      - Self-conditioning (self_conditioning.* weights)
    """

    # forward() returns full float32 logits [num_tokens, vocab], not a (1,1) token ID.
    logit_returns_token_id: bool = False

    def __init__(self, model_config, wgpu_device: "WebGPUDevice",
                 pipeline_cache: "PipelineCache") -> None:
        super().__init__(model_config, wgpu_device, pipeline_cache)

        # MoE configuration
        self.num_experts: int = getattr(model_config, "num_experts", 0)
        self.top_k_experts: int = getattr(model_config, "top_k_experts", 8)
        self.moe_intermediate_size: int = getattr(model_config, "moe_intermediate_size",
                                                   self.intermediate_size)
        self.is_moe: bool = self.num_experts > 0

        if self.is_moe:
            # Initialized here for the case where forward() is called without load_weights().
            # load_weights() resets and rebuilds this list with real per-expert scale arrays.
            self._pes_cache: list[np.ndarray | None] = [None] * self.num_layers
            logger.info("DiffusionGemma MoE: %d experts, top-%d, moe_inter=%d",
                        self.num_experts, self.top_k_experts, self.moe_intermediate_size)
            # Extra scratch buffer: shared-expert residual (F32 like h0/h1/h2).
            # Needed because the 3-buffer h-rotation doesn't accommodate 4 distinct
            # tensor states (x_buf, post-attn, post-shared-expert, post-moe).
            from vllm_webgpu.webgpu.buffer import WebGPUBuffer as _WB
            _dev = wgpu_device.wgpu_device
            # canvas_length is the max batch size during diffusion inference (default 256).
            # All per-token scratch buffers must be sized for the full canvas to avoid
            # out-of-bounds writes when num_tokens > 1.
            max_canvas_len = getattr(model_config, "canvas_length", 256)
            self._shared_res_buf = _WB.empty(_dev, max_canvas_len * self.hidden_size * 2)  # F16
            # Pre-allocated GPU top-K buffers — eliminates GPU→CPU router readback.
            self._topk_idx_buf     = _WB.empty(_dev, max_canvas_len * self.top_k_experts * 4)  # [T, K] u32
            self._topk_weight_buf  = _WB.empty(_dev, max_canvas_len * self.top_k_experts * 4)  # [T, K] f32
            self._router_logit_buf = _WB.empty(_dev, max_canvas_len * self.num_experts * 2)    # [T, E] f16
            self._moe_acc_buf      = _WB.empty(_dev, max_canvas_len * self.hidden_size * 2)    # [T, H] f16
            # Per-expert scratch: [T] f32 routing weight for one expert across all tokens.
            self._moe_per_expert_weight_buf = _WB.empty(_dev, max_canvas_len * 4)

    # ── Scratch buffer sizing ────────────────────────────────────────────────

    def _scratch_token_count(self) -> int:
        return getattr(self.model_config, "canvas_length", 256)

    def _scratch_inter_size(self) -> int:
        moe_inter = getattr(self.model_config, "moe_intermediate_size", self.intermediate_size)
        return max(self.intermediate_size, moe_inter)

    # ── Weight key helpers ───────────────────────────────────────────────────

    def _layer_key_prefix(self, layer_idx: int) -> str:
        return f"model.decoder.layers.{layer_idx}"

    def _embed_key(self) -> str:
        """Embedding weight key (DiffusionGemma uses model.decoder.embed_tokens)."""
        return next(
            (k for k in ("model.decoder.embed_tokens.weight", "model.embed_tokens.weight")
             if k in self.weights),
            "model.embed_tokens.weight",
        )

    def _norm_key(self) -> str:
        return next(
            (k for k in ("model.decoder.norm.weight", "model.norm.weight")
             if k in self.weights),
            "model.norm.weight",
        )

    def _lm_head_key(self) -> str:
        return next(
            (k for k in ("lm_head.weight", "model.decoder.lm_head.weight", "model.lm_head.weight")
             if k in self.weights),
            self._embed_key(),  # tied weights fallback
        )

    # ── Weight loading ───────────────────────────────────────────────────────

    def load_weights(self, path: str) -> None:
        """Load weights and cache per_expert_scale arrays to avoid per-step GPU readbacks."""
        super().load_weights(path)
        # Cache per_expert_scale for each MoE layer. Each to_numpy() is a blocking
        # GPU-CPU sync (~100 µs); caching once at load time avoids N syncs per step.
        self._pes_cache: list[np.ndarray | None] = []
        if self.is_moe:
            for i in range(self.num_layers):
                p = self._layer_key_prefix(i)
                pes_w = self.weights.get(f"{p}.router.per_expert_scale")
                if pes_w is not None:
                    self._pes_cache.append(pes_w.to_numpy().astype(np.float32))
                else:
                    self._pes_cache.append(None)
        else:
            self._pes_cache = [None] * self.num_layers

    # ── Override forward() for decoder-prefixed keys ─────────────────────────

    def forward(self, input_ids, positions, attn_metadata) -> "np.ndarray":
        """Forward pass using model.decoder.* weight keys."""
        dev = self.wgpu_device.wgpu_device
        num_tokens = len(input_ids)
        hidden = self.hidden_size
        vocab = self.vocab_size
        self._hstate = 0

        if hasattr(attn_metadata, "block_tables") and len(attn_metadata.block_tables) > 1:
            raise RuntimeError("multi-sequence batching not supported in this build")

        ctx_len = int(attn_metadata.max_decode_seq_len
                      if attn_metadata.max_decode_seq_len is not None else int(positions[-1]) + 1)
        if ctx_len <= 0:
            ctx_len = int(positions[-1]) + 1
        if ctx_len > 65535:
            raise RuntimeError(f"ctx_len={ctx_len} exceeds 65535")

        pre = self._pre
        dev.queue.write_buffer(pre["ids"].buf, 0, input_ids.astype(np.uint32).tobytes())
        dev.queue.write_buffer(pre["pos"].buf, 0, positions.astype(np.uint32).tobytes())
        dev.queue.write_buffer(
            pre["slot_map"].buf, 0,
            np.array(attn_metadata.slot_mapping, dtype=np.uint32).tobytes())
        bt_arr = self._bt_arr(attn_metadata)
        dev.queue.write_buffer(pre["bt"].buf, 0, bt_arr.tobytes())

        ids_buf = pre["ids"]; pos_buf = pre["pos"]
        slot_map = pre["slot_map"]; bt_buf = pre["bt"]
        x_buf = pre["x"]; norm_out = pre["norm_out"]; logits_buf = pre["logits"]

        # Manage the command encoder manually so that _decoder_layer can flush
        # and sync mid-layer before reading back MoE router indices. An outer
        # _batched_dispatch() context would make every inner context re-entrant,
        # preventing the mid-layer flush that topk readback requires.
        self._active_encoder = dev.create_command_encoder()
        try:
            self._dispatch("embedding_lookup_f32",
                           [self.weights[self._embed_key()], ids_buf, x_buf],
                           {"HIDDEN_DIM": hidden}, (num_tokens, 1, 1))

            for i in range(self.num_layers):
                x_buf = self._decoder_layer(
                    i, x_buf, pos_buf, slot_map, bt_buf, ctx_len, num_tokens)

            self._dispatch("rms_norm_f32in",
                           [x_buf, self.weights[self._norm_key()], norm_out],
                           self._rms_consts,
                           (num_tokens, 1, 1))

            lm_head_w = self.weights[self._lm_head_key()]
            if num_tokens > 1:
                # Batched path: matmul_quant_mr4 reads all M rows of norm_out.
                # Dispatch (vocab, num_tokens, 1) so every token gets its logits.
                self._dispatch("matmul_quant_mr4",
                               [norm_out, lm_head_w,
                                self.weights.get("lm_head.scales", norm_out),
                                logits_buf],
                               {"K": hidden, "N": vocab, "M": num_tokens, "USE_QUANT": 0},
                               (vocab, num_tokens, 1))
            else:
                # Single-token decode path.
                self._dispatch("matmul_quant",
                               [norm_out, lm_head_w,
                                self.weights.get("lm_head.scales", norm_out),
                                logits_buf],
                               {"K": hidden, "N": vocab, "USE_QUANT": 0, "SPLIT_K": 0},
                               ((vocab + 255) // 256, 1, 1))

            if self.softcap is not None and self.softcap > 0:
                capped = self._pre["capped"]
                self._dispatch("logit_softcap", [logits_buf, capped],
                               {"N": num_tokens * vocab, "CAP": float(self.softcap)},
                               ((num_tokens * vocab + 255) // 256, 1, 1),
                               shader_subdir="gemma")
                result = capped
            else:
                result = logits_buf

            dev.queue.submit([self._active_encoder.finish()])
        finally:
            self._active_encoder = None

        return result.to_numpy().view(np.float16)[:num_tokens * vocab].reshape(num_tokens, vocab).astype(np.float32)

    # ── Decoder layer (intentionally different signature from parent _transformer_layer) ──
    # Parent Gemma4WebGPUModel._transformer_layer takes normed_x and returns (WebGPUBuffer, WebGPUBuffer).
    # This class uses a fully overridden forward(), so the parent forward() is never called here.
    # Named _decoder_layer to avoid the implicit contract violation.

    def _decoder_layer(
        self,
        layer_idx: int,
        x_buf: "WebGPUBuffer",
        pos_buf: "WebGPUBuffer",
        slot_map: "WebGPUBuffer",
        bt_buf: "WebGPUBuffer",
        ctx_len: int,
        num_tokens: int,
    ) -> "WebGPUBuffer":
        """DiffusionGemma transformer layer with shared + MoE FFN."""
        sc = self._sc
        lp = self._lp[layer_idx]
        hidden = self.hidden_size
        inter_shared = self.intermediate_size      # shared expert intermediate size
        inter_moe = self.moe_intermediate_size     # MoE expert intermediate size
        head_dim = lp["head_dim"]
        q_dim = lp["q_dim"]
        kv_dim = lp["kv_dim"]
        num_kv_heads = lp["num_kv_heads"]
        ln_rope = self._ln_rope_theta
        p = self._layer_key_prefix(layer_idx)

        residual = sc[_H_NAMES[(self._hstate + 1) % 3]]
        out      = sc[_H_NAMES[(self._hstate + 2) % 3]]
        add_n    = num_tokens * hidden
        _rms = self._rms_consts

        k_cache, v_cache = self.kv_pool[layer_idx]

        with self._batched_dispatch(label=f"L{layer_idx:02d}"):
            # ── Attention sublayer ────────────────────────────────────────────
            self._dispatch("rms_norm_f32in",
                           [x_buf, self.weights[f"{p}.input_layernorm.weight"], sc["normed"]],
                           _rms, (num_tokens, 1, 1))

            for out_buf, proj, dim in [(sc["q_buf"], "q_proj", q_dim),
                                       (sc["k_buf"], "k_proj", kv_dim)]:
                wk = f"{p}.self_attn.{proj}.weight"
                uq = self._uq_for_key(wk)
                if num_tokens > 1 and uq in (0, 3):
                    _ex: dict = {"K": hidden, "N": dim, "M": num_tokens, "USE_QUANT": uq}
                    if uq == 3:
                        _ex["GROUP_K"] = self._quant_extra(wk[:-7], uq).get("GROUP_K", 128)
                    self._dispatch("matmul_quant_mr4",
                                   [sc["normed"], self.weights[wk],
                                    self._scales_buf(wk, uq, sc["normed"]), out_buf],
                                   _ex, (dim, num_tokens, 1))
                else:
                    self._dispatch("matmul_quant",
                                   [sc["normed"], self.weights[wk],
                                    self._scales_buf(wk, uq, sc["normed"]), out_buf],
                                   {"K": hidden, "N": dim, "USE_QUANT": uq,
                                    **self._split_k_extra(uq),
                                    **self._quant_extra(wk[:-7], uq)},
                                   _gemv_wg(dim, uq))
            # v_proj: global attention layers (no separate V; V=K) have no v_proj weight
            vw_key = f"{p}.self_attn.v_proj.weight"
            has_v_proj = vw_key in self.weights
            if has_v_proj:
                uq = self._uq_for_key(vw_key)
                if num_tokens > 1 and uq in (0, 3):
                    _ex_v: dict = {"K": hidden, "N": kv_dim, "M": num_tokens, "USE_QUANT": uq}
                    if uq == 3:
                        _ex_v["GROUP_K"] = self._quant_extra(vw_key[:-7], uq).get("GROUP_K", 128)
                    self._dispatch("matmul_quant_mr4",
                                   [sc["normed"], self.weights[vw_key],
                                    self._scales_buf(vw_key, uq, sc["normed"]),
                                    sc["v_buf"]],
                                   _ex_v, (kv_dim, num_tokens, 1))
                else:
                    self._dispatch("matmul_quant",
                                   [sc["normed"], self.weights[vw_key],
                                    self._scales_buf(vw_key, uq, sc["normed"]),
                                    sc["v_buf"]],
                                   {"K": hidden, "N": kv_dim, "USE_QUANT": uq,
                                    **self._split_k_extra(uq),
                                    **self._quant_extra(vw_key[:-7], uq)},
                                   _gemv_wg(kv_dim, uq))
                v_src = sc["v_buf"]
            else:
                v_src = sc["k_buf"]  # global attention: V = K

            _freq_buf = self._rope_freq_buf
            _dg_rope_base = {"ROPE_BASE": float(self.rope_theta), "LN_ROPE_BASE": ln_rope,
                             "USE_FREQ_BUF": int(self._use_freq_buf)}
            for src, dst, n_heads, wk in [
                (sc["q_buf"], sc["q_rope"], self.num_q_heads, f"{p}.self_attn.q_norm.weight"),
                (sc["k_buf"], sc["k_rope"], num_kv_heads, f"{p}.self_attn.k_norm.weight"),
            ]:
                nw = self.weights.get(wk)
                if nw is not None:
                    # Binding 4 (inv_freq_buf): always provided.
                    self._dispatch("fused_per_head_norm_rope",
                                   [src, nw, pos_buf, dst, _freq_buf],
                                   {**_dg_rope_base, "HEAD_DIM": head_dim, "NUM_HEADS": n_heads,
                                    "HAS_WEIGHT": 1, "GEMMA_NORM": self._GEMMA_NORM},
                                   (n_heads, num_tokens, 1))
                else:
                    # Binding 3 (inv_freq_buf): always provided.
                    self._dispatch("rope", [src, pos_buf, dst, _freq_buf],
                                   {**_dg_rope_base, "HEAD_DIM": head_dim, "NUM_HEADS": n_heads},
                                   (num_tokens, n_heads, 1))

            v_to_cache = v_src
            # Write all T tokens' KV to cache before the attention loop.
            # Each query token then attends to the full ctx_len cache (all T tokens),
            # which is non-causal (bidirectional). For the diffusion denoising use-case
            # this is intentional: the denoising process allows each token to attend
            # to all other tokens in the canvas. If causal attention is ever needed
            # (e.g., for an encoder-only pass), store and attend one token at a time
            # (like _prefill_sequential_fallback) or port flash_attn_prefill here.
            self._dispatch("kv_cache_store_both",
                           [sc["k_rope"], k_cache, v_to_cache, v_cache, slot_map],
                           {"BLOCK_SIZE": self.block_size, "NUM_KV_HEADS": num_kv_heads,
                            "HEAD_DIM": head_dim, "V_IN_OFFSET": 0},
                           (num_tokens, num_kv_heads, 1))
            # Multi-token path: loop over tokens using Q_TOKEN_OFFSET / ATTN_TOKEN_OFFSET.
            # Each iteration reuses scores_buf and sm_buf (sized NQ * max_ctx for one token);
            # this is safe because every _dispatch ends its compute pass before the next begins,
            # so GPU memory writes from attn_score are visible to the subsequent softmax.
            # For num_tokens == 1 the loop runs once with offset 0, matching the old dispatch.
            for _t in range(num_tokens):
                _t_q_off = _t * q_dim
                self._dispatch("attn_score",
                               [sc["q_rope"], k_cache, bt_buf, sc["scores_buf"]],
                               {"BLOCK_SIZE": self.block_size, "NUM_Q_HEADS": self.num_q_heads,
                                "NUM_KV_HEADS": num_kv_heads, "HEAD_DIM": head_dim,
                                "MAX_SEQ_LEN": ctx_len, "Q_TOKEN_OFFSET": _t_q_off},
                               (self.num_q_heads, ctx_len, 1))
                self._dispatch("softmax", [sc["scores_buf"], sc["sm_buf"]],
                               {"SEQ_LEN": ctx_len}, (self.num_q_heads, 1, 1))
                self._dispatch("attn_output",
                               [sc["sm_buf"], v_cache, bt_buf, sc["attn_out"]],
                               {"BLOCK_SIZE": self.block_size, "NUM_Q_HEADS": self.num_q_heads,
                                "NUM_KV_HEADS": num_kv_heads, "HEAD_DIM": head_dim,
                                "CTX_LEN": ctx_len, "ATTN_TOKEN_OFFSET": _t_q_off},
                               (self.num_q_heads, 1, 1))

            ow = f"{p}.self_attn.o_proj.weight"
            uq_ow = self._uq_for_key(ow)
            if num_tokens > 1 and uq_ow in (0, 3):
                _ex_ow: dict = {"K": q_dim, "N": hidden, "M": num_tokens, "USE_QUANT": uq_ow}
                if uq_ow == 3:
                    _ex_ow["GROUP_K"] = self._quant_extra(ow[:-7], uq_ow).get("GROUP_K", 128)
                self._dispatch("matmul_quant_mr4",
                               [sc["attn_out"], self.weights[ow],
                                self._scales_buf(ow, uq_ow, sc["attn_out"]),
                                sc["o_proj_out"]],
                               _ex_ow, (hidden, num_tokens, 1))
            else:
                self._dispatch("matmul_quant",
                               [sc["attn_out"], self.weights[ow],
                                self._scales_buf(ow, uq_ow, sc["attn_out"]),
                                sc["o_proj_out"]],
                               {"K": q_dim, "N": hidden, "USE_QUANT": uq_ow,
                                **self._split_k_extra(uq_ow),
                                **self._quant_extra(ow[:-7], uq_ow)},
                               _gemv_wg(hidden, uq_ow))

            # post_attention norm + residual add
            pan_w = self.weights.get(f"{p}.post_attention_layernorm.weight")
            if pan_w is not None:
                self._dispatch("rms_norm", [sc["o_proj_out"], pan_w, sc["ffn_normed"]], _rms,
                               (num_tokens, 1, 1))
                self._dispatch("add_f32", [x_buf, sc["ffn_normed"], residual],
                               {"N": add_n}, ((add_n // 4 + 255) // 256, 1, 1))
            else:
                self._dispatch("add_f32", [x_buf, sc["o_proj_out"], residual],
                               {"N": add_n}, ((add_n // 4 + 255) // 256, 1, 1))

            # ── Shared expert FFN ─────────────────────────────────────────────
            gelu_n_shared = num_tokens * inter_shared
            pfn_w = self.weights.get(f"{p}.pre_feedforward_layernorm.weight")
            if pfn_w is None:
                raise ValueError(
                    f"Layer {layer_idx} missing pre_feedforward_layernorm.weight "
                    "— f32 residual cannot be fed to f16 FFN projection"
                )
            self._dispatch("rms_norm_f32in", [residual, pfn_w, sc["normed"]],
                           _rms, (num_tokens, 1, 1))
            ffn_in = sc["normed"]

            # Shared expert gate + up → tanh-GELU activation
            gw_k = f"{p}.mlp.gate_proj.weight"
            uw_k = f"{p}.mlp.up_proj.weight"
            uq_g = self._uq_for_key(gw_k)
            uq_u = self._uq_for_key(uw_k)
            if num_tokens > 1 and uq_g in (0, 3) and uq_u in (0, 3):
                # Batch path: one matmul_quant_mr4 per projection covers all T tokens.
                for out_b, wk, uq in [
                        (sc["gate_buf"], gw_k, uq_g),
                        (sc["up_buf"],   uw_k, uq_u)]:
                    _ex_mr4: dict = {"K": hidden, "N": inter_shared,
                                     "M": num_tokens, "USE_QUANT": uq}
                    if uq == 3:
                        _ex_mr4["GROUP_K"] = self._quant_extra(wk[:-7], uq).get("GROUP_K", 128)
                    self._dispatch("matmul_quant_mr4",
                                   [ffn_in, self.weights[wk],
                                    self._scales_buf(wk, uq, ffn_in), out_b],
                                   _ex_mr4, (inter_shared, num_tokens, 1))
                self._dispatch("gelu_mul", [sc["gate_buf"], sc["up_buf"], sc["ffn_act"]],
                               {"N": gelu_n_shared}, ((gelu_n_shared // 4 + 255) // 256, 1, 1),
                               shader_subdir="gemma")
            elif uq_g == 0 and uq_u == 0:
                self._dispatch("fused_gate_act",
                               [ffn_in, self.weights[gw_k], self.weights[uw_k], sc["ffn_act"]],
                               {"K": hidden, "N": inter_shared, "GELU": 1}, (inter_shared, 1, 1))
            else:
                for out_b, proj, wk, uq in [
                        (sc["gate_buf"], "gate_proj", gw_k, uq_g),
                        (sc["up_buf"],   "up_proj",   uw_k, uq_u)]:
                    self._dispatch("matmul_quant",
                                   [ffn_in, self.weights[wk],
                                    self._scales_buf(wk, uq, ffn_in), out_b],
                                   {"K": hidden, "N": inter_shared, "USE_QUANT": uq,
                                    **self._split_k_extra(uq),
                                    **self._quant_extra(wk[:-7], uq)},
                                   _gemv_wg(inter_shared, uq))
                self._dispatch("gelu_mul", [sc["gate_buf"], sc["up_buf"], sc["ffn_act"]],
                               {"N": gelu_n_shared}, ((gelu_n_shared // 4 + 255) // 256, 1, 1),
                               shader_subdir="gemma")

            dw = f"{p}.mlp.down_proj.weight"
            uq_dw = self._uq_for_key(dw)
            if num_tokens > 1 and uq_dw in (0, 3):
                _ex_dw: dict = {"K": inter_shared, "N": hidden,
                                "M": num_tokens, "USE_QUANT": uq_dw}
                if uq_dw == 3:
                    _ex_dw["GROUP_K"] = self._quant_extra(dw[:-7], uq_dw).get("GROUP_K", 128)
                self._dispatch("matmul_quant_mr4",
                               [sc["ffn_act"], self.weights[dw],
                                self._scales_buf(dw, uq_dw, sc["ffn_act"]), sc["ffn_out"]],
                               _ex_dw, (hidden, num_tokens, 1))
            else:
                self._dispatch("matmul_quant",
                               [sc["ffn_act"], self.weights[dw],
                                self._scales_buf(dw, uq_dw, sc["ffn_act"]), sc["ffn_out"]],
                               {"K": inter_shared, "N": hidden, "USE_QUANT": uq_dw,
                                **self._split_k_extra(uq_dw),
                                **self._quant_extra(dw[:-7], uq_dw)},
                               _gemv_wg(hidden, uq_dw))

            # MoE layers use post_feedforward_layernorm_1 for the shared MLP stream;
            # non-MoE layers only have the no-suffix key.
            _pfn1_key_1 = f"{p}.post_feedforward_layernorm_1.weight"
            pfn1_w = (self.weights.get(_pfn1_key_1) or
                      self.weights.get(f"{p}.post_feedforward_layernorm.weight"))
            if self.is_moe and pfn1_w is not None:
                self._dispatch("rms_norm", [sc["ffn_out"], pfn1_w, self._shared_res_buf], _rms,
                               (num_tokens, 1, 1))
                hidden_states_1 = self._shared_res_buf
                # Track whether the no-suffix fallback was used: if so the else-branch
                # below must not norm again with the same key.
                _pfn1_used_fallback = self.weights.get(_pfn1_key_1) is None
            else:
                hidden_states_1 = sc["ffn_out"]
                _pfn1_used_fallback = False

        # ── MoE expert FFN (all-GPU: router + top-K selection + expert FFNs) ───
        if self.is_moe and f"{p}.router.proj.weight" in self.weights:
            dev = self.wgpu_device.wgpu_device
            router_logits_buf = self._router_logit_buf
            pfn2_w = self.weights.get(f"{p}.pre_feedforward_layernorm_2.weight")

            with self._batched_dispatch(label=f"L{layer_idx:02d}R"):
                if pfn2_w is not None:
                    self._dispatch("rms_norm_f32in", [residual, pfn2_w, sc["normed"]],
                                   _rms, (num_tokens, 1, 1))
                    moe_in = sc["normed"]
                else:
                    raise ValueError(
                        f"Layer {layer_idx} missing pre_feedforward_layernorm_2.weight. "
                        "The f32 residual cannot be passed directly to f16 MoE projections: "
                        "the shader would silently misinterpret f32 bytes as f16 values. "
                        "A correctly loaded DiffusionGemma checkpoint always has this weight."
                    )

                # Gemma4Router preprocessing (vLLM Gemma4Router.forward, line 292-296):
                #   x = norm(x)          — no-weight RMSNorm on the raw residual
                #   x = x * root_size    — 1/sqrt(hidden_size) scalar
                #   x = x * router.scale — learned per-dimension scale
                # Input is residual (post-attention, pre-MLP accumulation), not moe_in.
                router_scale_w = self.weights.get(f"{p}.router.scale")
                router_in = sc["o_proj_out"]   # reuse free scratch (hidden, f16)
                if router_scale_w is not None:
                    root_size = hidden ** -0.5
                    self._dispatch("router_norm_f32in",
                                   [residual, router_scale_w, router_in],
                                   {"HIDDEN_DIM": hidden, "VALS_PER_THREAD": self._rms_consts["VALS_PER_THREAD"],
                                    "ROOT_SIZE": root_size},
                                   (num_tokens, 1, 1))
                else:
                    logger.warning("L%d: router.scale missing, routing will be incorrect", layer_idx)
                    router_in = moe_in

                rw_ = f"{p}.router.proj.weight"
                uq_rw = self._uq_for_key(rw_)
                _rw_sc = self._scales_buf(rw_, uq_rw, router_in)
                if num_tokens > 1 and uq_rw in (0, 3):
                    # Batched router projection: [T, hidden] x [num_experts, hidden]^T -> [T, E]
                    _rw_extra: dict = {"K": hidden, "N": self.num_experts,
                                       "M": num_tokens, "USE_QUANT": uq_rw}
                    if uq_rw == 3:
                        _rw_extra["GROUP_K"] = self._quant_extra(rw_[:-7], uq_rw).get("GROUP_K", 128)
                    self._dispatch("matmul_quant_mr4",
                                   [router_in, self.weights[rw_], _rw_sc, router_logits_buf],
                                   _rw_extra,
                                   (self.num_experts, num_tokens, 1))
                else:
                    if num_tokens > 1:
                        logger.warning(
                            "L%d: router uq=%d not supported for batched routing; "
                            "token-0 routing applied to all tokens", layer_idx, uq_rw)
                    self._dispatch("matmul_quant",
                                   [router_in, self.weights[rw_], _rw_sc, router_logits_buf],
                                   {"K": hidden, "N": self.num_experts,
                                    "USE_QUANT": uq_rw, "SPLIT_K": 0,
                                    **self._quant_extra(rw_[:-7], uq_rw)},
                                   ((self.num_experts + 255) // 256, 1, 1))
                # GPU top-K: per-token top-K selection from [T, N_EXPERTS] logits.
                # Dispatch (num_tokens, 1, 1): each workgroup handles one token's logits.
                self._dispatch("topk_sort",
                               [router_logits_buf, self._topk_idx_buf, self._topk_weight_buf],
                               {"N_EXPERTS": self.num_experts, "K": self.top_k_experts},
                               (num_tokens, 1, 1))

            # Flush the router + topk dispatches and wait for GPU completion
            # before reading back expert indices. The dispatches recorded into
            # _active_encoder (re-entrant inside the L{i}R block above) are not
            # yet submitted; to_numpy() creates its own copy encoder and would
            # read stale pre-topk_sort data without this explicit flush.
            dev.queue.submit([self._active_encoder.finish()])
            dev.queue.on_submitted_work_done_sync()

            # Read back per-token routing: [T, K] arrays of expert indices and softmax weights.
            n_tk = num_tokens * self.top_k_experts
            top_k_idx = (self._topk_idx_buf.to_numpy().view(np.uint32)[:n_tk]
                         .reshape(num_tokens, self.top_k_experts))
            rw_vals   = (self._topk_weight_buf.to_numpy().view(np.float32)[:n_tk]
                         .reshape(num_tokens, self.top_k_experts))

            # Fresh encoder for the zero-init write, expert FFN dispatches, and
            # post-MoE norm. All subsequent _batched_dispatch calls in this layer
            # are re-entrant and record into this encoder.
            self._active_encoder = dev.create_command_encoder()

            # Apply per-expert learned scale (vLLM gemma4_routing_function_torch:
            # topk_weights *= per_expert_scale[topk_ids]).  The HF checkpoint key is
            # {layer}.router.per_expert_scale; shape [num_experts], dtype bfloat16/float32.
            # Use the cached numpy array (populated at load time) to avoid a
            # blocking GPU-CPU sync per inference step per MoE layer.
            pes = self._pes_cache[layer_idx]
            if pes is not None:
                rw_vals = rw_vals * pes[top_k_idx]  # [T, K] broadcast via advanced indexing

            # Build unique-expert -> per-token-weight mapping.
            # expert_token_weights[eid][t] = routing weight for token t to expert eid
            # (0.0 for tokens that do not route to eid).
            expert_token_weights: dict = defaultdict(lambda: np.zeros(num_tokens, dtype=np.float32))
            for t in range(num_tokens):
                for k in range(self.top_k_experts):
                    eid = int(top_k_idx[t, k])
                    expert_token_weights[eid][t] += float(rw_vals[t, k])

            # GPU: run selected expert FFNs
            gelu_n_moe = num_tokens * inter_moe
            moe_acc = self._moe_acc_buf
            # Zero-initialize the accumulation buffer before the expert loop so
            # moe_accumulate_batched can do in-place += without a ping-pong buffer.
            dev.queue.write_buffer(moe_acc.buf, 0, b"\x00" * (add_n * 2))

            for eid, w_per_token in expert_token_weights.items():
                ep = f"{p}.experts.{eid}"
                g_w = self.weights.get(f"{ep}.gate_proj.weight")
                u_w = self.weights.get(f"{ep}.up_proj.weight")
                d_w = self.weights.get(f"{ep}.down_proj.weight")
                if g_w is None or u_w is None or d_w is None:
                    logger.warning("Missing expert %d for L%d", eid, layer_idx)
                    continue

                uq_g  = self._uq_for_key(f"{ep}.gate_proj.weight")
                uq_u  = self._uq_for_key(f"{ep}.up_proj.weight")
                dk    = f"{ep}.down_proj.weight"
                uq_dk = self._uq_for_key(dk)
                use_mr4 = uq_g in (0, 3) and uq_u in (0, 3) and uq_dk in (0, 3)

                if num_tokens > 1 and not use_mr4:
                    raise RuntimeError(
                        f"MoE multi-token FFN requires f16 (uq=0) or GPTQ (uq=3) weights; "
                        f"expert {eid} at L{layer_idx} has uq=({uq_g},{uq_u},{uq_dk})"
                    )

                # Write per-token routing weights for this expert to GPU scratch buffer.
                dev.queue.write_buffer(self._moe_per_expert_weight_buf.buf, 0,
                                       w_per_token.tobytes())

                with self._batched_dispatch(label=f"L{layer_idx:02d}E{eid}"):
                    if use_mr4 and num_tokens > 1:
                        # Batched GEMM path: process all T tokens through this expert.
                        # gate/up: [T, hidden] x [inter_moe, hidden]^T -> [T, inter_moe]
                        for ob, ew_key, uq in [
                            (sc["gate_buf"], f"{ep}.gate_proj.weight", uq_g),
                            (sc["up_buf"],   f"{ep}.up_proj.weight",   uq_u),
                        ]:
                            _sc_e = self._scales_buf(ew_key, uq, moe_in)
                            _ex_e: dict = {"K": hidden, "N": inter_moe,
                                           "M": num_tokens, "USE_QUANT": uq}
                            if uq == 3:
                                _ex_e["GROUP_K"] = (
                                    self._quant_extra(ew_key[:-7], uq).get("GROUP_K", 128))
                            self._dispatch("matmul_quant_mr4",
                                           [moe_in, self.weights[ew_key], _sc_e, ob],
                                           _ex_e,
                                           (inter_moe, num_tokens, 1))
                        self._dispatch("gelu_mul",
                                       [sc["gate_buf"], sc["up_buf"], sc["ffn_act"]],
                                       {"N": gelu_n_moe},
                                       ((gelu_n_moe // 4 + 255) // 256, 1, 1),
                                       shader_subdir="gemma")
                        # down: [T, inter_moe] x [hidden, inter_moe]^T -> [T, hidden]
                        _sc_dk = self._scales_buf(dk, uq_dk, sc["ffn_act"])
                        _ex_dk: dict = {"K": inter_moe, "N": hidden,
                                        "M": num_tokens, "USE_QUANT": uq_dk}
                        if uq_dk == 3:
                            _ex_dk["GROUP_K"] = (
                                self._quant_extra(dk[:-7], uq_dk).get("GROUP_K", 128))
                        self._dispatch("matmul_quant_mr4",
                                       [sc["ffn_act"], self.weights[dk], _sc_dk, sc["ffn_out"]],
                                       _ex_dk,
                                       (hidden, num_tokens, 1))
                        # Per-token weighted accumulate: moe_acc[t*H+j] += w[t] * ffn_out[t*H+j]
                        self._dispatch("moe_accumulate_batched",
                                       [moe_acc, sc["ffn_out"],
                                        self._moe_per_expert_weight_buf],
                                       {"N": add_n, "H": hidden},
                                       ((add_n + 255) // 256, 1, 1))
                    else:
                        # Single-token GEMV path (num_tokens==1).
                        for ob, ew_key in [(sc["gate_buf"], f"{ep}.gate_proj.weight"),
                                           (sc["up_buf"],   f"{ep}.up_proj.weight")]:
                            uq = self._uq_for_key(ew_key)
                            self._dispatch("matmul_quant",
                                           [moe_in, self.weights[ew_key],
                                            self._scales_buf(ew_key, uq, moe_in), ob],
                                           {"K": hidden, "N": inter_moe,
                                            "USE_QUANT": uq,
                                            **self._split_k_extra(uq),
                                            **self._quant_extra(ew_key[:-7], uq)},
                                           _gemv_wg(inter_moe, uq))
                        self._dispatch("gelu_mul",
                                       [sc["gate_buf"], sc["up_buf"], sc["ffn_act"]],
                                       {"N": gelu_n_moe},
                                       ((gelu_n_moe // 4 + 255) // 256, 1, 1),
                                       shader_subdir="gemma")
                        self._dispatch("matmul_quant",
                                       [sc["ffn_act"], self.weights[dk],
                                        self._scales_buf(dk, uq_dk, sc["ffn_act"]),
                                        sc["ffn_out"]],
                                       {"K": inter_moe, "N": hidden,
                                        "USE_QUANT": uq_dk,
                                        **self._split_k_extra(uq_dk),
                                        **self._quant_extra(dk[:-7], uq_dk)},
                                       _gemv_wg(hidden, uq_dk))
                        # K_IDX=0: w_per_token[0] is the scalar weight for this expert.
                        self._dispatch("moe_accumulate",
                                       [moe_acc, sc["ffn_out"],
                                        self._moe_per_expert_weight_buf],
                                       {"N": add_n, "K_IDX": 0},
                                       ((add_n + 255) // 256, 1, 1))

            # Post-MoE norm + single residual add (vLLM Gemma4 pattern)
            with self._batched_dispatch(label=f"L{layer_idx:02d}P"):
                pfn2_out_w = self.weights.get(f"{p}.post_feedforward_layernorm_2.weight")
                if pfn2_out_w is not None:
                    self._dispatch("rms_norm", [moe_acc, pfn2_out_w, sc["o_proj_out"]], _rms,
                                   (num_tokens, 1, 1))
                    hidden_states_2 = sc["o_proj_out"]
                else:
                    hidden_states_2 = moe_acc

                # Combine shared-MLP and MoE streams (f16 + f16 -> f16)
                self._dispatch("add", [hidden_states_1, hidden_states_2, sc["normed"]],
                               {"N": add_n}, ((add_n // 4 + 255) // 256, 1, 1))

                # Combined post-FFN norm before residual add
                pfn_w = self.weights.get(f"{p}.post_feedforward_layernorm.weight")
                if pfn_w is not None:
                    self._dispatch("rms_norm", [sc["normed"], pfn_w, sc["ffn_out"]], _rms,
                                   (num_tokens, 1, 1))
                    combined_normed = sc["ffn_out"]
                else:
                    combined_normed = sc["normed"]

                layer_scalar = self._layer_scales[layer_idx]
                self._dispatch("add_f32", [residual, combined_normed, out],
                               {"N": add_n},
                               ((add_n // 4 + 255) // 256, 1, 1))
        else:
            # Apply post_feedforward_layernorm before residual add, matching vLLM's
            # unconditional application in Gemma4DecoderLayer.forward for all layers.
            layer_scalar = self._layer_scales[layer_idx]
            pfn_w = self.weights.get(f"{p}.post_feedforward_layernorm.weight")
            if pfn_w is not None and not _pfn1_used_fallback:
                self._dispatch("rms_norm", [hidden_states_1, pfn_w, sc["normed"]], _rms,
                               (num_tokens, 1, 1))
                hidden_states_1 = sc["normed"]
            self._dispatch("add_f32", [residual, hidden_states_1, out],
                           {"N": add_n}, ((add_n // 4 + 255) // 256, 1, 1))

        if abs(layer_scalar - 1.0) > 1e-6:
            self._dispatch("f32_scale_inplace", [out],
                           {"N": add_n, "SCALE": layer_scalar},
                           ((add_n + 255) // 256, 1, 1))

        self._hstate = (self._hstate + 2) % 3
        return out
