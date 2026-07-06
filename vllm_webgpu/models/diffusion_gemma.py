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

    def __init__(self, model_config, wgpu_device: "WebGPUDevice",
                 pipeline_cache: "PipelineCache") -> None:
        super().__init__(model_config, wgpu_device, pipeline_cache)

        # MoE configuration
        self.num_experts: int = getattr(model_config, "num_experts", 0)
        self.top_k_experts: int = getattr(model_config, "top_k_experts", 8)
        self.moe_intermediate_size: int = getattr(model_config, "moe_intermediate_size",
                                                   self.intermediate_size)
        self.is_moe: bool = self.num_experts > 0
        self.canvas_length: int = getattr(model_config, "canvas_length", 256)

        if self.is_moe:
            logger.info("DiffusionGemma MoE: %d experts, top-%d, moe_inter=%d",
                        self.num_experts, self.top_k_experts, self.moe_intermediate_size)

    # ── Weight key helpers ───────────────────────────────────────────────────

    def _pk(self, layer_idx: int) -> str:
        return f"model.decoder.layers.{layer_idx}"

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
        import numpy as np
        import wgpu as wgpu_lib

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
                x_buf = self._transformer_layer(
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

    # ── Override: use decoder prefix for all lookups ─────────────────────────

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

        quant_types = self.weights.get("__quant_types__", {})
        _qt = quant_types if isinstance(quant_types, dict) else {}

        def _uq(key: str) -> int:
            w = self.weights.get(key)
            if w is not None and getattr(w, "dtype", "f16") == "i32":
                return 3
            tt = _qt.get(key, 0)
            if tt == 12:
                return 2
            if self.weights.get(key[:-7] + ".scales") is not None:
                return 1
            return 0

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
                                       (sc["k_buf"], "k_proj", kv_dim),
                                       (sc["v_buf"], "v_proj", kv_dim)]:
                wk = f"{p}.self_attn.{proj}.weight"
                uq = _uq(wk)
                self._dispatch("matmul_quant",
                               [sc["normed"], self.weights[wk],
                                self.weights.get(wk[:-7] + ".scales", sc["normed"]), out_buf],
                               {"K": hidden, "N": dim, "USE_QUANT": uq, "SPLIT_K": 1},
                               (dim, 1, 1))

            for src, dst, n_heads, wk in [
                (sc["q_buf"], sc["q_rope"], self.num_q_heads, f"{p}.self_attn.q_norm.weight"),
                (sc["k_buf"], sc["k_rope"], num_kv_heads, f"{p}.self_attn.k_norm.weight"),
            ]:
                nw = self.weights.get(wk)
                if nw is not None:
                    self._dispatch("fused_per_head_norm_rope",
                                   [src, nw, pos_buf, dst],
                                   {"HEAD_DIM": head_dim, "NUM_HEADS": n_heads,
                                    "ROPE_BASE": float(self.rope_theta),
                                    "LN_ROPE_BASE": ln_rope, "HAS_WEIGHT": 1,
                                    "GEMMA_NORM": self._gemma_norm_const},
                                   (n_heads, num_tokens, 1))
                else:
                    self._dispatch("rope", [src, pos_buf, dst],
                                   {"HEAD_DIM": head_dim, "NUM_HEADS": n_heads,
                                    "LN_ROPE_BASE": ln_rope}, (num_tokens, n_heads, 1))

            v_to_cache = sc["v_buf"]
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
            self._dispatch("matmul_quant",
                           [sc["attn_out"], self.weights[ow],
                            self.weights.get(ow[:-7] + ".scales", sc["attn_out"]),
                            sc["o_proj_out"]],
                           {"K": q_dim, "N": hidden, "USE_QUANT": _uq(ow), "SPLIT_K": 1},
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
                uq = _uq(wk)
                self._dispatch("matmul_quant",
                               [ffn_in, self.weights[wk],
                                self.weights.get(wk[:-7] + ".scales", ffn_in), out_b],
                               {"K": hidden, "N": inter_shared, "USE_QUANT": uq, "SPLIT_K": 1},
                               (inter_shared, 1, 1))
            self._dispatch("gelu_mul", [sc["gate_buf"], sc["up_buf"], sc["ffn_act"]],
                           {"N": gelu_n_shared}, ((gelu_n_shared // 4 + 255) // 256, 1, 1),
                           shader_subdir="gemma")

            dw = f"{p}.mlp.down_proj.weight"
            self._dispatch("matmul_quant",
                           [sc["ffn_act"], self.weights[dw],
                            self.weights.get(dw[:-7] + ".scales", sc["ffn_act"]), sc["ffn_out"]],
                           {"K": inter_shared, "N": hidden, "USE_QUANT": _uq(dw), "SPLIT_K": 1},
                           (hidden, 1, 1))

            pfn1_w = self.weights.get(f"{p}.post_feedforward_layernorm.weight")
            if pfn1_w is not None:
                self._dispatch("rms_norm", [sc["ffn_out"], pfn1_w, sc["o_proj_out"]], _rms,
                               (num_tokens, 1, 1))
                shared_out = sc["o_proj_out"]
            else:
                shared_out = sc["ffn_out"]

            # Accumulate shared expert: residual += shared_out
            self._dispatch("add_f32", [residual, shared_out, sc[h_names[self._hstate]]],
                           {"N": add_n, "SCALE": 1.0}, ((add_n // 4 + 255) // 256, 1, 1))
            shared_residual = sc[h_names[self._hstate]]

        # ── MoE expert FFN (CPU router + GPU expert FFNs) ─────────────────────
        if self.is_moe and f"{p}.router.proj.weight" in self.weights:
            import wgpu as wgpu_lib
            router_logits_buf = WebGPUBuffer.empty(dev, self.num_experts * 2, usage=rw)
            pfn2_w = self.weights.get(f"{p}.pre_feedforward_layernorm_2.weight")
            moe_in_f32 = shared_residual  # f32 residual

            with self._batched_dispatch(label=f"L{layer_idx:02d}R"):
                # Optional pre-norm for MoE path
                if pfn2_w is not None:
                    self._dispatch("rms_norm_f32in", [shared_residual, pfn2_w, sc["normed"]],
                                   _rms, (num_tokens, 1, 1))
                    moe_in = sc["normed"]
                else:
                    moe_in = sc["normed"]  # fallback

                rw_ = f"{p}.router.proj.weight"
                self._dispatch("matmul_quant",
                               [moe_in, self.weights[rw_],
                                self.weights.get(rw_[:-7] + ".scales", moe_in),
                                router_logits_buf],
                               {"K": hidden, "N": self.num_experts,
                                "USE_QUANT": _uq(rw_), "SPLIT_K": 0},
                               ((self.num_experts + 255) // 256, 1, 1))

            # CPU: select top-K experts
            router_logits = router_logits_buf.to_numpy().view(np.float16).astype(np.float32)
            top_k_idx = np.argsort(router_logits)[-self.top_k_experts:]
            rw_vals = router_logits[top_k_idx]
            rw_vals = np.exp(rw_vals - rw_vals.max())
            rw_vals = rw_vals / rw_vals.sum()

            # GPU: run top-K expert FFNs
            gelu_n_moe = num_tokens * inter_moe
            moe_acc = WebGPUBuffer.empty(dev, num_tokens * hidden * 2, usage=rw)

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
                        self._dispatch("matmul_quant",
                                       [moe_in, self.weights[ew_key],
                                        self.weights.get(ew_key[:-7] + ".scales", moe_in), ob],
                                       {"K": hidden, "N": inter_moe,
                                        "USE_QUANT": _uq(ew_key), "SPLIT_K": 1},
                                       (inter_moe, 1, 1))
                    self._dispatch("gelu_mul", [sc["gate_buf"], sc["up_buf"], sc["ffn_act"]],
                                   {"N": gelu_n_moe},
                                   ((gelu_n_moe // 4 + 255) // 256, 1, 1),
                                   shader_subdir="gemma")
                    dk = f"{ep}.down_proj.weight"
                    self._dispatch("matmul_quant",
                                   [sc["ffn_act"], self.weights[dk],
                                    self.weights.get(dk[:-7] + ".scales", sc["ffn_act"]),
                                    sc["ffn_out"]],
                                   {"K": inter_moe, "N": hidden,
                                    "USE_QUANT": _uq(dk), "SPLIT_K": 1},
                                   (hidden, 1, 1))
                    # Accumulate: moe_acc += ew * expert_out
                    scale = float(ew)
                    if idx == 0:
                        self._dispatch("add",
                                       [sc["ffn_out"], sc["ffn_out"], moe_acc],
                                       {"N": add_n, "SCALE": scale},
                                       ((add_n // 4 + 255) // 256, 1, 1))
                    else:
                        self._dispatch("add",
                                       [moe_acc, sc["ffn_out"], moe_acc],
                                       {"N": add_n, "SCALE": scale},
                                       ((add_n // 4 + 255) // 256, 1, 1))

            # Post-MoE norm + residual add
            with self._batched_dispatch(label=f"L{layer_idx:02d}P"):
                pfn2_out_w = self.weights.get(f"{p}.post_feedforward_layernorm_2.weight")
                if pfn2_out_w is not None:
                    self._dispatch("rms_norm", [moe_acc, pfn2_out_w, sc["o_proj_out"]], _rms,
                                   (num_tokens, 1, 1))
                    moe_out = sc["o_proj_out"]
                else:
                    moe_out = moe_acc
                # Scalar from layer
                layer_scalar_w = self.weights.get(f"{p}.layer_scalar")
                layer_scalar = 1.0
                if layer_scalar_w is not None:
                    layer_scalar = float(layer_scalar_w.to_numpy().view(np.float16).ravel()[0])
                self._dispatch("add_f32", [shared_residual, moe_out, out],
                               {"N": add_n, "SCALE": layer_scalar},
                               ((add_n // 4 + 255) // 256, 1, 1))
        else:
            # No MoE: out = shared_residual
            out = shared_residual

        self._hstate = (self._hstate + 2) % 3
        return out
