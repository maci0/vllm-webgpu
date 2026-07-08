from __future__ import annotations
import logging
import math
from typing import TYPE_CHECKING

import numpy as np

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
            logger.info("DiffusionGemma MoE: %d experts, top-%d, moe_inter=%d",
                        self.num_experts, self.top_k_experts, self.moe_intermediate_size)
            # Extra scratch buffer: shared-expert residual (F32 like h0/h1/h2).
            # Needed because the 3-buffer h-rotation doesn't accommodate 4 distinct
            # tensor states (x_buf, post-attn, post-shared-expert, post-moe).
            import wgpu as _wgpu
            from vllm_webgpu.webgpu.buffer import WebGPUBuffer as _WB
            _dev = wgpu_device.wgpu_device
            _rw = _wgpu.BufferUsage.STORAGE | _wgpu.BufferUsage.COPY_SRC | _wgpu.BufferUsage.COPY_DST
            self._shared_res_buf = _WB.empty(_dev, 1 * self.hidden_size * 4, usage=_rw)
            # Pre-allocated GPU top-K buffers — eliminates GPU→CPU router readback.
            self._topk_idx_buf     = _WB.empty(_dev, self.top_k_experts * 4, usage=_rw)  # [K] u32
            self._topk_weight_buf  = _WB.empty(_dev, self.top_k_experts * 4, usage=_rw)  # [K] f32
            self._router_logit_buf = _WB.empty(_dev, self.num_experts * 2, usage=_rw)     # [E] f16
            self._moe_acc_buf      = _WB.empty(_dev, self.hidden_size * 2, usage=_rw)     # [H] f16 accumulator

    # ── Weight key helpers ───────────────────────────────────────────────────

    def _pk(self, layer_idx: int) -> str:
        return f"model.decoder.layers.{layer_idx}"

    def _layer_key_prefix(self, layer_idx: int) -> str:
        return self._pk(layer_idx)

    def load_weights(self, path: str) -> None:
        super().load_weights(path)

    def _embed_key(self) -> str:
        """Embedding weight key (DiffusionGemma uses model.decoder.embed_tokens)."""
        # Try decoder prefix first, fall back to standard
        for k in ("model.decoder.embed_tokens.weight", "model.embed_tokens.weight"):
            if k in self.weights:
                return k
        return "model.embed_tokens.weight"

    def _norm_key(self) -> str:
        for k in ("model.decoder.norm.weight", "model.norm.weight"):
            if k in self.weights:
                return k
        return "model.norm.weight"

    def _lm_head_key(self) -> str:
        for k in ("lm_head.weight", "model.decoder.lm_head.weight",
                  "model.lm_head.weight"):
            if k in self.weights:
                return k
        return self._embed_key()  # tied weights fallback

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
                      if attn_metadata.max_decode_seq_len is not None else num_tokens)
        if ctx_len <= 0:
            ctx_len = num_tokens
        if ctx_len > 65535:
            raise RuntimeError(f"ctx_len={ctx_len} exceeds 65535")

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

        ids_buf = pre["ids"]; pos_buf = pre["pos"]
        slot_map = pre["slot_map"]; bt_buf = pre["bt"]
        x_buf = pre["x"]; norm_out = pre["norm_out"]; logits_buf = pre["logits"]

        _vpt = min((hidden + 255) // 256, 16) if hidden <= 4096 else 0

        with self._batched_dispatch():
            self._dispatch("embedding_lookup_f32",
                           [self.weights[self._embed_key()], ids_buf, x_buf],
                           {"HIDDEN_DIM": hidden}, (num_tokens, 1, 1))

            for i in range(self.num_layers):
                x_buf = self._decoder_layer(
                    i, x_buf, pos_buf, slot_map, bt_buf, ctx_len, num_tokens)

            self._dispatch("rms_norm_f32in",
                           [x_buf, self.weights[self._norm_key()], norm_out],
                           {"HIDDEN_DIM": hidden, "VALS_PER_THREAD": _vpt,
                            "GEMMA_NORM": self._gemma_norm_const},
                           (num_tokens, 1, 1))

            lm_head_w = self.weights.get(self._lm_head_key(),
                                         self.weights[self._embed_key()])
            self._dispatch("matmul_quant",
                           [norm_out, lm_head_w, self.weights.get("lm_head.scales", norm_out),
                            logits_buf],
                           {"K": hidden, "N": vocab, "USE_QUANT": 0, "SPLIT_K": 0},
                           ((vocab + 255) // 256, 1, 1))

            if self.softcap is not None and self.softcap > 0:
                capped = self._pre.get("capped", logits_buf)
                self._dispatch("logit_softcap", [logits_buf, capped],
                               {"N": num_tokens * vocab, "CAP": float(self.softcap)},
                               ((num_tokens * vocab + 255) // 256, 1, 1),
                               shader_subdir="gemma")
                result = capped
            else:
                result = logits_buf

        return result.to_numpy().view(np.float16).reshape(num_tokens, vocab).astype(np.float32)

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
        import wgpu as wgpu_lib
        from vllm_webgpu.webgpu.buffer import WebGPUBuffer

        dev = self.wgpu_device.wgpu_device
        rw = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST
        sc = self._sc
        lp = self._lp[layer_idx]
        hidden = self.hidden_size
        inter_shared = self.intermediate_size      # shared expert intermediate size
        inter_moe = self.moe_intermediate_size     # MoE expert intermediate size
        head_dim = lp["head_dim"]
        q_dim = lp["q_dim"]
        kv_dim = lp["kv_dim"]
        num_kv_heads = lp["num_kv_heads"]
        ln_rope = math.log(self.rope_theta)
        p = self._pk(layer_idx)

        h_names = ["h0", "h1", "h2"]
        residual = sc[h_names[(self._hstate + 1) % 3]]
        out      = sc[h_names[(self._hstate + 2) % 3]]
        add_n    = num_tokens * hidden
        _vpt = min((hidden + 255) // 256, 16) if hidden <= 4096 else 0
        _rms = {"HIDDEN_DIM": hidden, "VALS_PER_THREAD": _vpt,
                "GEMMA_NORM": self._gemma_norm_const}

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
                self._dispatch("matmul_quant",
                               [sc["normed"], self.weights[wk],
                                self._scales_buf(wk, uq, sc["normed"]), out_buf],
                               {"K": hidden, "N": dim, "USE_QUANT": uq, "SPLIT_K": 1,
                                **self._quant_extra(wk[:-7], uq)},
                               (dim, 1, 1))
            # v_proj: global attention layers (no separate V; V=K) have no v_proj weight
            vw_key = f"{p}.self_attn.v_proj.weight"
            has_v_proj = vw_key in self.weights
            if has_v_proj:
                uq = self._uq_for_key(vw_key)
                self._dispatch("matmul_quant",
                               [sc["normed"], self.weights[vw_key],
                                self._scales_buf(vw_key, uq, sc["normed"]),
                                sc["v_buf"]],
                               {"K": hidden, "N": kv_dim, "USE_QUANT": uq, "SPLIT_K": 1,
                                **self._quant_extra(vw_key[:-7], uq)},
                               (kv_dim, 1, 1))
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
                                    "HAS_WEIGHT": 1, "GEMMA_NORM": self._gemma_norm_const},
                                   (n_heads, num_tokens, 1))
                else:
                    # Binding 3 (inv_freq_buf): always provided.
                    self._dispatch("rope", [src, pos_buf, dst, _freq_buf],
                                   {**_dg_rope_base, "HEAD_DIM": head_dim, "NUM_HEADS": n_heads},
                                   (num_tokens, n_heads, 1))

            v_to_cache = v_src
            self._dispatch("kv_cache_store", [sc["k_rope"], k_cache, slot_map],
                           {"BLOCK_SIZE": self.block_size, "NUM_KV_HEADS": num_kv_heads,
                            "HEAD_DIM": head_dim}, (num_tokens, num_kv_heads, 1))
            self._dispatch("kv_cache_store", [v_to_cache, v_cache, slot_map],
                           {"BLOCK_SIZE": self.block_size, "NUM_KV_HEADS": num_kv_heads,
                            "HEAD_DIM": head_dim}, (num_tokens, num_kv_heads, 1))
            self._dispatch("attn_score", [sc["q_rope"], k_cache, bt_buf, sc["scores_buf"]],
                           {"BLOCK_SIZE": self.block_size, "NUM_Q_HEADS": self.num_q_heads,
                            "NUM_KV_HEADS": num_kv_heads, "HEAD_DIM": head_dim,
                            "MAX_SEQ_LEN": ctx_len}, (self.num_q_heads, ctx_len, 1))
            self._dispatch("softmax", [sc["scores_buf"], sc["sm_buf"]],
                           {"SEQ_LEN": ctx_len}, (self.num_q_heads, 1, 1))
            self._dispatch("attn_output", [sc["sm_buf"], v_cache, bt_buf, sc["attn_out"]],
                           {"BLOCK_SIZE": self.block_size, "NUM_Q_HEADS": self.num_q_heads,
                            "NUM_KV_HEADS": num_kv_heads, "HEAD_DIM": head_dim,
                            "CTX_LEN": ctx_len}, (self.num_q_heads, 1, 1))

            ow = f"{p}.self_attn.o_proj.weight"
            uq_ow = self._uq_for_key(ow)
            self._dispatch("matmul_quant",
                           [sc["attn_out"], self.weights[ow],
                            self._scales_buf(ow, uq_ow, sc["attn_out"]),
                            sc["o_proj_out"]],
                           {"K": q_dim, "N": hidden, "USE_QUANT": uq_ow, "SPLIT_K": 1,
                            **self._quant_extra(ow[:-7], uq_ow)},
                           (hidden, 1, 1))

            # post_attention norm + residual add
            pan_w = self.weights.get(f"{p}.post_attention_layernorm.weight")
            if pan_w is not None:
                self._dispatch("rms_norm", [sc["o_proj_out"], pan_w, sc["ffn_normed"]], _rms,
                               (num_tokens, 1, 1))
                self._dispatch("add_f32", [x_buf, sc["ffn_normed"], residual],
                               {"N": add_n, "SCALE": 1.0}, ((add_n // 4 + 255) // 256, 1, 1))
            else:
                self._dispatch("add_f32", [x_buf, sc["o_proj_out"], residual],
                               {"N": add_n, "SCALE": 1.0}, ((add_n // 4 + 255) // 256, 1, 1))

            # ── Shared expert FFN ─────────────────────────────────────────────
            gelu_n_shared = num_tokens * inter_shared
            pfn_w = self.weights.get(f"{p}.pre_feedforward_layernorm.weight")
            if pfn_w is not None:
                self._dispatch("rms_norm_f32in", [residual, pfn_w, sc["normed"]],
                               _rms, (num_tokens, 1, 1))
                ffn_in = sc["normed"]
            else:
                ffn_in = residual

            # Shared expert gate + up → SwiGLU (Gemma uses GELU)
            for out_b, proj in [(sc["gate_buf"], "gate_proj"), (sc["up_buf"], "up_proj")]:
                wk = f"{p}.mlp.{proj}.weight"
                uq = self._uq_for_key(wk)
                self._dispatch("matmul_quant",
                               [ffn_in, self.weights[wk],
                                self._scales_buf(wk, uq, ffn_in), out_b],
                               {"K": hidden, "N": inter_shared, "USE_QUANT": uq, "SPLIT_K": 1,
                                **self._quant_extra(wk[:-7], uq)},
                               (inter_shared, 1, 1))
            self._dispatch("gelu_mul", [sc["gate_buf"], sc["up_buf"], sc["ffn_act"]],
                           {"N": gelu_n_shared}, ((gelu_n_shared // 4 + 255) // 256, 1, 1),
                           shader_subdir="gemma")

            dw = f"{p}.mlp.down_proj.weight"
            uq_dw = self._uq_for_key(dw)
            self._dispatch("matmul_quant",
                           [sc["ffn_act"], self.weights[dw],
                            self._scales_buf(dw, uq_dw, sc["ffn_act"]), sc["ffn_out"]],
                           {"K": inter_shared, "N": hidden, "USE_QUANT": uq_dw, "SPLIT_K": 1,
                            **self._quant_extra(dw[:-7], uq_dw)},
                           (hidden, 1, 1))

            pfn1_w = self.weights.get(f"{p}.post_feedforward_layernorm.weight")
            if pfn1_w is not None:
                self._dispatch("rms_norm", [sc["ffn_out"], pfn1_w, sc["o_proj_out"]], _rms,
                               (num_tokens, 1, 1))
                shared_out = sc["o_proj_out"]
            else:
                shared_out = sc["ffn_out"]

            # Accumulate shared expert into dedicated shared_res_buf (avoids h-rotation conflicts).
            self._dispatch("add_f32", [residual, shared_out, self._shared_res_buf],
                           {"N": add_n, "SCALE": 1.0}, ((add_n // 4 + 255) // 256, 1, 1))
            shared_residual = self._shared_res_buf

        # ── MoE expert FFN (all-GPU: router + top-K selection + expert FFNs) ───
        if self.is_moe and f"{p}.router.proj.weight" in self.weights:
            router_logits_buf = self._router_logit_buf
            pfn2_w = self.weights.get(f"{p}.pre_feedforward_layernorm_2.weight")

            with self._batched_dispatch(label=f"L{layer_idx:02d}R"):
                if pfn2_w is not None:
                    self._dispatch("rms_norm_f32in", [shared_residual, pfn2_w, sc["normed"]],
                                   _rms, (num_tokens, 1, 1))
                    moe_in = sc["normed"]
                else:
                    moe_in = shared_residual

                # Gemma4Router preprocessing (vLLM Gemma4Router.forward, line 292-296):
                #   x = norm(x)          — no-weight RMSNorm on the raw residual
                #   x = x * root_size    — 1/sqrt(hidden_size) scalar
                #   x = x * router.scale — learned per-dimension scale
                # Input is shared_residual (pre pre_feedforward_layernorm_2), not moe_in.
                router_scale_w = self.weights.get(f"{p}.router.scale")
                router_in = sc["o_proj_out"]   # reuse free scratch (hidden, f16)
                if router_scale_w is not None:
                    root_size = math.pow(hidden, -0.5)
                    self._dispatch("router_norm_f32in",
                                   [shared_residual, router_scale_w, router_in],
                                   {"HIDDEN_DIM": hidden, "VALS_PER_THREAD": _vpt,
                                    "ROOT_SIZE": root_size},
                                   (num_tokens, 1, 1))
                else:
                    logger.warning("L%d: router.scale missing, routing will be incorrect", layer_idx)
                    router_in = moe_in

                rw_ = f"{p}.router.proj.weight"
                uq_rw = self._uq_for_key(rw_)
                self._dispatch("matmul_quant",
                               [router_in, self.weights[rw_],
                                self._scales_buf(rw_, uq_rw, router_in),
                                router_logits_buf],
                               {"K": hidden, "N": self.num_experts,
                                "USE_QUANT": uq_rw, "SPLIT_K": 0,
                                **self._quant_extra(rw_[:-7], uq_rw)},
                               ((self.num_experts + 255) // 256, 1, 1))
                # GPU top-K: sorts N_EXPERTS logits, picks top-K indices + softmax weights.
                # Eliminates the GPU→CPU readback that previously cost ~1ms per MoE layer.
                self._dispatch("topk_sort",
                               [router_logits_buf, self._topk_idx_buf, self._topk_weight_buf],
                               {"N_EXPERTS": self.num_experts, "K": self.top_k_experts},
                               (1, 1, 1))

            # Read back compact K-element arrays (negligible: K=8 = 32 bytes)
            top_k_idx = self._topk_idx_buf.to_numpy().view(np.uint32)[:self.top_k_experts]
            rw_vals   = self._topk_weight_buf.to_numpy().view(np.float32)[:self.top_k_experts]

            # GPU: run top-K expert FFNs
            gelu_n_moe = num_tokens * inter_moe
            moe_acc = self._moe_acc_buf
            # Zero-initialize the accumulation buffer before the expert loop so
            # moe_accumulate can do in-place += without a ping-pong buffer.
            dev.queue.write_buffer(moe_acc.buf, 0, b"\x00" * (add_n * 2))

            for idx, (eid, ew) in enumerate(zip(top_k_idx, rw_vals)):
                ep = f"{p}.experts.{eid}"
                g_w = self.weights.get(f"{ep}.gate_proj.weight")
                u_w = self.weights.get(f"{ep}.up_proj.weight")
                d_w = self.weights.get(f"{ep}.down_proj.weight")
                if g_w is None or u_w is None or d_w is None:
                    logger.warning("Missing expert %d for L%d", eid, layer_idx)
                    continue

                with self._batched_dispatch(label=f"L{layer_idx:02d}E{eid}"):
                    for ob, ew_key in [(sc["gate_buf"], f"{ep}.gate_proj.weight"),
                                       (sc["up_buf"],   f"{ep}.up_proj.weight")]:
                        uq = self._uq_for_key(ew_key)
                        self._dispatch("matmul_quant",
                                       [moe_in, self.weights[ew_key],
                                        self._scales_buf(ew_key, uq, moe_in), ob],
                                       {"K": hidden, "N": inter_moe,
                                        "USE_QUANT": uq, "SPLIT_K": 1,
                                        **self._quant_extra(ew_key[:-7], uq)},
                                       (inter_moe, 1, 1))
                    self._dispatch("gelu_mul", [sc["gate_buf"], sc["up_buf"], sc["ffn_act"]],
                                   {"N": gelu_n_moe},
                                   ((gelu_n_moe // 4 + 255) // 256, 1, 1),
                                   shader_subdir="gemma")
                    dk = f"{ep}.down_proj.weight"
                    uq_dk = self._uq_for_key(dk)
                    self._dispatch("matmul_quant",
                                   [sc["ffn_act"], self.weights[dk],
                                    self._scales_buf(dk, uq_dk, sc["ffn_act"]),
                                    sc["ffn_out"]],
                                   {"K": inter_moe, "N": hidden,
                                    "USE_QUANT": uq_dk, "SPLIT_K": 1,
                                    **self._quant_extra(dk[:-7], uq_dk)},
                                   (hidden, 1, 1))
                    # Weighted in-place accumulate: moe_acc[i] += w_buf[idx] * ffn_out[i]
                    # K_IDX indexes into _topk_weight_buf (already written by topk_sort).
                    self._dispatch("moe_accumulate",
                                   [moe_acc, sc["ffn_out"], self._topk_weight_buf],
                                   {"N": add_n, "K_IDX": idx},
                                   ((add_n + 255) // 256, 1, 1))

            # Post-MoE norm + residual add
            with self._batched_dispatch(label=f"L{layer_idx:02d}P"):
                pfn2_out_w = self.weights.get(f"{p}.post_feedforward_layernorm_2.weight")
                if pfn2_out_w is not None:
                    self._dispatch("rms_norm", [moe_acc, pfn2_out_w, sc["o_proj_out"]], _rms,
                                   (num_tokens, 1, 1))
                    moe_out = sc["o_proj_out"]
                else:
                    moe_out = moe_acc
                layer_scalar = self._layer_scales[layer_idx]
                self._dispatch("add_f32", [shared_residual, moe_out, out],
                               {"N": add_n},
                               ((add_n // 4 + 255) // 256, 1, 1))
                self._dispatch("f32_scale_inplace", [out],
                               {"N": add_n, "SCALE": layer_scalar},
                               ((add_n + 255) // 256, 1, 1))
        else:
            # No MoE: out = shared_residual
            out = shared_residual

        self._hstate = (self._hstate + 2) % 3
        return out
