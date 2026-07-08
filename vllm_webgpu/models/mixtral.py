from __future__ import annotations
import logging
import struct
from typing import TYPE_CHECKING

import numpy as np

from vllm_webgpu.models.llama import LlamaWebGPUModel, _gemv_wg

if TYPE_CHECKING:
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.device import WebGPUDevice
    from vllm_webgpu.webgpu.pipeline import PipelineCache

logger = logging.getLogger(__name__)


class MixtralWebGPUModel(LlamaWebGPUModel):
    """
    Handles Mistral (dense, optional sliding window attention) and
    Mixtral (sparse MoE FFN via block_sparse_moe) architectures.

    Extends LlamaWebGPUModel with:
    - Sliding Window Attention (SWA): caps effective context length passed to
      attention shaders when config.sliding_window is set.
    - MoE FFN: Phase A/B dispatch with CPU readback for expert selection.
      Phase A: router + topk_sort, flush, read indices and weights.
      Phase B: new encoder, per-expert gate/up/down + weighted accumulate.

    Weight naming (Mixtral block_sparse_moe):
      router:  model.layers.{i}.block_sparse_moe.gate.weight
      gate:    model.layers.{i}.block_sparse_moe.experts.{j}.w1.weight
      up:      model.layers.{i}.block_sparse_moe.experts.{j}.w3.weight
      down:    model.layers.{i}.block_sparse_moe.experts.{j}.w2.weight
    """

    def __init__(
        self,
        model_config,
        wgpu_device: "WebGPUDevice",
        pipeline_cache: "PipelineCache",
    ) -> None:
        super().__init__(model_config, wgpu_device, pipeline_cache)
        self._sw: int | None = getattr(model_config, "sliding_window", None)
        if not self._sw:  # treat 0 and None as disabled
            self._sw = None
        self._num_experts: int = getattr(model_config, "num_local_experts", 0)
        self._top_k: int = getattr(model_config, "num_experts_per_tok", 0)
        self._is_moe: bool = self._num_experts > 0 and self._top_k > 0

        if self._is_moe:
            import wgpu as wgpu_lib
            from vllm_webgpu.webgpu.buffer import WebGPUBuffer

            dev = self.wgpu_device.wgpu_device
            rw = (wgpu_lib.BufferUsage.STORAGE
                  | wgpu_lib.BufferUsage.COPY_SRC
                  | wgpu_lib.BufferUsage.COPY_DST)

            def mk(n: int) -> "WebGPUBuffer":
                return WebGPUBuffer.empty(dev, max(n, 8), usage=rw)

            self._moe_sc: dict[str, "WebGPUBuffer"] = {
                "router_out":   mk(self._num_experts * 2),      # [N_E] f16 router logits
                "topk_idx":     mk(self._top_k * 4),             # [K] u32 expert indices
                "topk_w":       mk(self._top_k * 4),             # [K] f32 softmax weights
                "expert_gate":  mk(self.intermediate_size * 2),  # [inter] f16 gate proj
                "expert_up":    mk(self.intermediate_size * 2),  # [inter] f16 up proj
                "expert_act":   mk(self.intermediate_size * 2),  # [inter] f16 activated
                "expert_out":   mk(self.hidden_size * 2),        # [hidden] f16 accumulated
                "expert_tmp":   mk(self.hidden_size * 2),        # [hidden] f16 per-expert
                "moe_w_buf":    mk(self._top_k * 4),             # [K] f32 weights for accumulate
                "dummy_scales": mk(8),                           # fallback scales binding
            }

    def _effective_ctx(self, ctx_len: int) -> int:
        """Cap ctx_len at the sliding window size when SWA is configured."""
        if self._sw and ctx_len > self._sw:
            return self._sw
        return ctx_len

    def forward(
        self,
        input_ids: np.ndarray,
        positions: np.ndarray,
        attn_metadata: object,
    ) -> np.ndarray:
        """Route MoE decode to explicit-encoder path; everything else to parent."""
        if self._is_moe:
            if len(input_ids) > 1:
                raise NotImplementedError(
                    "Mixtral MoE batch prefill is not yet supported on WebGPU"
                )
            return self._moe_decode_forward(input_ids, positions, attn_metadata)
        return super().forward(input_ids, positions, attn_metadata)

    def _moe_decode_forward(
        self,
        input_ids: np.ndarray,
        positions: np.ndarray,
        attn_metadata: object,
    ) -> np.ndarray:
        """MoE decode forward — manages command encoders explicitly.

        Unlike the standard forward(), this does NOT use an outer _batched_dispatch
        context manager. Instead, _active_encoder is set manually so that layer
        methods' inner _batched_dispatch calls are re-entrant no-ops recording into
        this encoder. _moe_ffn_layer flushes and replaces _active_encoder mid-layer
        to handle the CPU readback required for expert index selection.
        """
        dev = self.wgpu_device.wgpu_device
        num_tokens = len(input_ids)
        hidden = self.hidden_size
        vocab = self.vocab_size
        self._hstate = 0
        sc = self._sc
        pre = self._pre

        ctx_len = int(
            attn_metadata.max_decode_seq_len
            if attn_metadata.max_decode_seq_len is not None
            else num_tokens
        )
        if ctx_len <= 0:
            ctx_len = num_tokens

        _vpt = min((hidden + 255) // 256, 16) if hidden <= 4096 else 0
        _rms_base = {"HIDDEN_DIM": hidden, "VALS_PER_THREAD": _vpt}

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

        # Start first encoder manually. Layer methods see _active_encoder is not None
        # and their _batched_dispatch calls become re-entrant no-ops, recording into
        # this encoder. _moe_ffn_layer replaces it after each Phase A flush.
        self._active_encoder = dev.create_command_encoder()

        self._dispatch(
            "embedding_lookup",
            [self.weights["model.embed_tokens.weight"], ids_buf, x_buf],
            {"HIDDEN_DIM": hidden},
            (num_tokens, 1, 1),
        )
        self._dispatch(
            "rms_norm",
            [x_buf, self.weights["model.layers.0.input_layernorm.weight"], sc["normed"]],
            _rms_base, (num_tokens, 1, 1),
        )

        normed_x = sc["normed"]
        for i in range(self.num_layers):
            normed_x, x_buf = self._transformer_layer(
                i, normed_x, x_buf, pos_buf, slot_map, bt_buf, ctx_len, num_tokens)

        self._dispatch(
            "rms_norm",
            [x_buf, self.weights["model.norm.weight"], norm_out],
            _rms_base, (num_tokens, 1, 1),
        )

        lm_head_w = (self.weights.get("lm_head.weight")
                     or self.weights["model.embed_tokens.weight"])
        self._dispatch(
            "matmul_quant",
            [norm_out, lm_head_w, self.weights.get("lm_head.scales", norm_out), logits_buf],
            {"K": hidden, "N": vocab, "USE_QUANT": 0, "SPLIT_K": 0},
            ((vocab + 255) // 256, 1, 1),
        )

        if getattr(self, "_greedy_decode", True):
            self._dispatch("argmax_f16", [logits_buf, self._ensure_sample_buf(vocab)],
                           {"N": vocab}, (1, 1, 1))
            self._copy_sample_to_staging()

        dev.queue.submit([self._active_encoder.finish()])
        self._active_encoder = None

        self._last_logit_buf = logits_buf
        self._last_vocab     = vocab

        if getattr(self, "_greedy_decode", True):
            tok = self._read_sample_tok()
            return np.array([[tok]], dtype=np.int32)
        return self.logit_readback()

    def _transformer_layer(  # noqa: C901
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
        """Overrides LlamaWebGPUModel._transformer_layer with two targeted changes:

        1. SWA: eff = _effective_ctx(ctx_len) caps attention dispatch parameters
           (attn_score MAX_SEQ_LEN, softmax SEQ_LEN, attn_output CTX_LEN,
           flash_attn_decode CTX_LEN). Weight key names are not affected.

        2. MoE FFN: when _is_moe, calls _moe_ffn_layer instead of gate+up+down.
           _moe_ffn_layer flushes and replaces _active_encoder (Phase A/B pattern).
           The subsequent residual-add dispatch lands in the new Phase B encoder.
        """
        eff = self._effective_ctx(ctx_len)

        sc = self._sc
        hidden = self.hidden_size
        p = f"model.layers.{layer_idx}"
        q_dim = self.num_q_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        inter = self.intermediate_size
        ln_rope = self._ln_rope_theta

        _rms_c = self._rms_consts

        h_names = ["h0", "h1", "h2"]
        residual = sc[h_names[(self._hstate + 1) % 3]]
        out = sc[h_names[(self._hstate + 2) % 3]]
        add_n = num_tokens * hidden
        gelu_n = num_tokens * inter

        k_cache, v_cache = self.kv_pool[layer_idx]

        with self._batched_dispatch(label=f"L{layer_idx:02d}"):
            # normed_x is already the pre-normed input (no rms_norm dispatch here).

            # QKV projections
            q_wk = f"{p}.self_attn.q_proj.weight"
            k_wk = f"{p}.self_attn.k_proj.weight"
            v_wk = f"{p}.self_attn.v_proj.weight"
            uq_q, uq_k, uq_v = self._uq_for_key(q_wk), self._uq_for_key(k_wk), self._uq_for_key(v_wk)
            _has_qnorm = self.weights.get(f"{p}.self_attn.q_norm.weight") is not None

            _use_fused_qkv = uq_q == 0 and uq_k == 0 and uq_v == 0 and _has_qnorm

            if _use_fused_qkv:
                self._dispatch("fused_qkv",
                               [normed_x, self.weights[q_wk], self.weights[k_wk],
                                self.weights[v_wk], sc["qkv_buf"]],
                               {"K": hidden, "Q_DIM": q_dim, "KV_DIM": kv_dim},
                               (q_dim + 2 * kv_dim, 1, 1))
                _q_src = sc["qkv_buf"]
                _k_src = sc["qkv_buf"]
                _v_src = sc["qkv_buf"]
                _v_offset = q_dim + kv_dim
            else:
                for out_buf_qkv, proj, dim, uq in [
                    (sc["q_buf"], "q_proj", q_dim, uq_q),
                    (sc["k_buf"], "k_proj", kv_dim, uq_k),
                    (sc["v_buf"], "v_proj", kv_dim, uq_v),
                ]:
                    w_key = f"{p}.self_attn.{proj}.weight"
                    qi = self._quant_extra(f"{p}.self_attn.{proj}", uq)
                    self._dispatch(
                        "matmul_quant",
                        [normed_x, self.weights[w_key], self._scales_buf(w_key, uq, normed_x),
                         out_buf_qkv],
                        {"K": hidden, "N": dim, "USE_QUANT": uq,
                         **({"SPLIT_K": 0} if uq not in (0, 3, 4, 5, 6, 7, 8) else {}),
                         **qi},
                        _gemv_wg(dim, uq),
                    )
                _q_src = sc["q_buf"]
                _k_src = sc["k_buf"]
                _v_src = sc["v_buf"]
                _v_offset = 0

            q_norm_w = self.weights.get(f"{p}.self_attn.q_norm.weight")
            k_norm_w = self.weights.get(f"{p}.self_attn.k_norm.weight")
            _freq_buf = self._rope_freq_buf
            _rope_consts = {
                "HEAD_DIM": self.head_dim,
                "ROPE_BASE": float(self.rope_theta),
                "LN_ROPE_BASE": ln_rope,
                "USE_FREQ_BUF": int(self._use_freq_buf),
            }

            if _use_fused_qkv and q_norm_w is not None:
                # Binding 6 (k_input): dummy (K_SEPARATE=0). Binding 7: inv_freq_buf.
                self._dispatch("fused_qk_norm_rope",
                               [sc["qkv_buf"], q_norm_w, k_norm_w, pos_buf,
                                sc["q_rope"], sc["k_rope"], sc["qkv_buf"], _freq_buf],
                               {**_rope_consts,
                                "NUM_Q_HEADS": self.num_q_heads,
                                "NUM_KV_HEADS": self.num_kv_heads,
                                "HAS_WEIGHT": 1,
                                "INPUT_OFFSET_K": q_dim},
                               (self.num_q_heads + self.num_kv_heads, num_tokens, 1))
            else:
                for src, dst, n_heads, norm_w, in_off in [
                    (_q_src, sc["q_rope"], self.num_q_heads,  q_norm_w, 0),
                    (_k_src, sc["k_rope"], self.num_kv_heads, k_norm_w,
                     q_dim if _use_fused_qkv else 0),
                ]:
                    if norm_w is not None:
                        self._dispatch("fused_per_head_norm_rope",
                                       [src, norm_w, pos_buf, dst, _freq_buf],
                                       {**_rope_consts, "NUM_HEADS": n_heads,
                                        "HAS_WEIGHT": 1, "INPUT_OFFSET": in_off},
                                       (n_heads, num_tokens, 1))
                    else:
                        self._dispatch("rope", [src, pos_buf, dst, _freq_buf],
                                       {**_rope_consts, "NUM_HEADS": n_heads},
                                       (num_tokens, n_heads, 1))

            self._dispatch("kv_cache_store_both",
                           [sc["k_rope"], k_cache, _v_src, v_cache, slot_map],
                           {"BLOCK_SIZE": self.block_size,
                            "NUM_KV_HEADS": self.num_kv_heads,
                            "HEAD_DIM": self.head_dim,
                            "V_IN_OFFSET": _v_offset},
                           (num_tokens, self.num_kv_heads, 1))

            # Always use flash_attn_decode for single-token decode.
            # The 65535 limit applied to attn_score's dispatch dimension; flash_attn_decode
            # loops internally and has no dispatch dimension limit.
            self._dispatch("flash_attn_decode",
                           [sc["q_rope"], k_cache, v_cache, bt_buf, sc["attn_out"]],
                           {"BLOCK_SIZE": self.block_size,
                            "NUM_Q_HEADS": self.num_q_heads,
                            "NUM_KV_HEADS": self.num_kv_heads,
                            "HEAD_DIM": self.head_dim,
                            "CTX_LEN": eff},
                           (self.num_q_heads, 1, 1))

            # Output projection
            w_key = f"{p}.self_attn.o_proj.weight"
            uq = self._uq_for_key(w_key)
            qi = self._quant_extra(f"{p}.self_attn.o_proj", uq)
            self._dispatch("matmul_quant",
                           [sc["attn_out"], self.weights[w_key],
                            self._scales_buf(w_key, uq, sc["attn_out"]), sc["o_proj_out"]],
                           {"K": q_dim, "N": hidden, "USE_QUANT": uq,
                            **({"SPLIT_K": 0} if uq not in (0, 3, 4, 5, 6, 7, 8) else {}),
                            **qi},
                           _gemv_wg(hidden, uq))

            # Fused post-attn residual-add + FFN pre-norm
            self._dispatch("add_rms_norm",
                           [x_buf, sc["o_proj_out"],
                            self.weights[f"{p}.post_attention_layernorm.weight"],
                            residual, sc["ffn_normed"]],
                           _rms_c, (num_tokens, 1, 1))

            # FFN: MoE or standard dense
            if self._is_moe:
                # _moe_ffn_layer flushes and replaces _active_encoder.
                # Subsequent dispatches (residual add below) land in the new encoder.
                self._moe_ffn_layer(sc["ffn_normed"], layer_idx)
                ffn_out = self._moe_sc["expert_out"]
            else:
                gw_k = f"{p}.mlp.gate_proj.weight"
                uw_k = f"{p}.mlp.up_proj.weight"
                uq_g = self._uq_for_key(gw_k)
                uq_u = self._uq_for_key(uw_k)
                if uq_g == 0 and uq_u == 0:
                    self._dispatch("fused_gate_act",
                                   [sc["ffn_normed"], self.weights[gw_k], self.weights[uw_k],
                                    sc["ffn_act"]],
                                   {"K": hidden, "N": inter, "GELU": 0}, (inter, 1, 1))
                else:
                    for out_b, w_k, uq2, mlp_proj in [
                        (sc["gate_buf"], gw_k, uq_g, "gate_proj"),
                        (sc["up_buf"],  uw_k, uq_u, "up_proj"),
                    ]:
                        qi2 = self._quant_extra(f"{p}.mlp.{mlp_proj}", uq2)
                        self._dispatch(
                            "matmul_quant",
                            [sc["ffn_normed"], self.weights[w_k],
                             self._scales_buf(w_k, uq2, sc["ffn_normed"]), out_b],
                            {"K": hidden, "N": inter, "USE_QUANT": uq2,
                             **({"SPLIT_K": 0} if uq2 not in (0, 3, 4, 5, 6, 7, 8) else {}),
                             **qi2},
                            _gemv_wg(inter, uq2),
                        )
                    self._dispatch("gelu_mul",
                                   [sc["gate_buf"], sc["up_buf"], sc["ffn_act"]],
                                   {"N": gelu_n},
                                   ((gelu_n // 4 + 255) // 256, 1, 1))

                w_k = f"{p}.mlp.down_proj.weight"
                uq = self._uq_for_key(w_k)
                qi3 = self._quant_extra(f"{p}.mlp.down_proj", uq)
                self._dispatch(
                    "matmul_quant",
                    [sc["ffn_act"], self.weights[w_k],
                     self._scales_buf(w_k, uq, sc["ffn_act"]), sc["ffn_out"]],
                    {"K": inter, "N": hidden, "USE_QUANT": uq,
                     **({"SPLIT_K": 0} if uq not in (0, 3, 4, 5, 6, 7, 8) else {}),
                     **qi3},
                    _gemv_wg(hidden, uq),
                )
                ffn_out = sc["ffn_out"]

            # Final residual add: fuse with next layer's pre-norm when possible.
            if layer_idx < self.num_layers - 1:
                next_w = self.weights[f"model.layers.{layer_idx + 1}.input_layernorm.weight"]
                self._dispatch("add_rms_norm",
                               [residual, ffn_out, next_w, out, sc["normed"]],
                               _rms_c, (num_tokens, 1, 1))
                normed_out = sc["normed"]
            else:
                self._dispatch("add", [residual, ffn_out, out],
                               {"N": add_n}, ((add_n // 4 + 255) // 256, 1, 1))
                normed_out = sc["normed"]  # stale; unused after last layer

        self._hstate = (self._hstate + 2) % 3
        return normed_out, out

    def _moe_ffn_layer(
        self,
        normed_x: "WebGPUBuffer",
        layer_idx: int,
    ) -> None:
        """MoE FFN using Mixtral block_sparse_moe weight naming (w1/w3/w2).

        Phase A: router + topk_sort dispatched into the current encoder, then
                 flushed and synced so the CPU can read selected expert indices.
        Phase B: new encoder created; selected experts dispatched with
                 weighted accumulation into _moe_sc["expert_out"].

        Caller's self._active_encoder is replaced with the Phase B encoder on
        return. Subsequent dispatches in the calling layer method (the residual
        add_rms_norm or add) land in the Phase B encoder, which is correct.
        """
        dev = self.wgpu_device.wgpu_device
        msc = self._moe_sc
        hidden = self.hidden_size
        inter = self.intermediate_size
        N_E = self._num_experts
        K = self._top_k
        p = f"model.layers.{layer_idx}.block_sparse_moe"

        # ── Phase A: router + top-K (into current encoder) ───────────────────
        rw_k = f"{p}.gate.weight"
        uq_r = self._uq_for_key(rw_k)
        qi_r = self._quant_extra(f"{p}.gate", uq_r)
        extra_r: dict = {"SPLIT_K": 0} if uq_r not in (0, 3, 4, 5, 6, 7, 8) else {}
        self._dispatch(
            "matmul_quant",
            [normed_x, self.weights[rw_k],
             self._scales_buf(rw_k, uq_r, msc["dummy_scales"]),
             msc["router_out"]],
            {"K": hidden, "N": N_E, "USE_QUANT": uq_r, **extra_r, **qi_r},
            _gemv_wg(N_E, uq_r),
        )
        self._dispatch(
            "topk_sort",
            [msc["router_out"], msc["topk_idx"], msc["topk_w"]],
            {"N_EXPERTS": N_E, "K": K},
            (1, 1, 1),
        )

        # Flush current encoder and wait for the router + topk to complete.
        dev.queue.submit([self._active_encoder.finish()])
        dev.queue.on_submitted_work_done_sync()

        # Read back selected expert indices and softmax weights.
        raw_idx = msc["topk_idx"].to_numpy().view(np.uint32)
        raw_w   = msc["topk_w"].to_numpy().view(np.float32)
        expert_indices = [int(raw_idx[k]) for k in range(K)]
        expert_weights = [float(raw_w[k]) for k in range(K)]

        logger.debug(
            "L%02d MoE experts: %s  weights: %s",
            layer_idx, expert_indices,
            [f"{w:.3f}" for w in expert_weights],
        )

        # Write softmax weights into the combined weight buffer so moe_accumulate
        # can read w_buf[K_IDX] without a per-dispatch CPU roundtrip.
        dev.queue.write_buffer(msc["moe_w_buf"].buf, 0,
                               struct.pack(f"<{K}f", *expert_weights))
        # Zero-initialize the accumulation buffer (no shared expert in Mixtral).
        dev.queue.write_buffer(msc["expert_out"].buf, 0, b"\x00" * (hidden * 2))

        # ── Phase B: expert dispatches (new encoder) ──────────────────────────
        # Subsequent _dispatch() calls (including the residual add in the calling
        # _transformer_layer) will land in this new encoder.
        self._active_encoder = dev.create_command_encoder()

        for k_idx, exp_idx in enumerate(expert_indices):
            if expert_weights[k_idx] == 0.0:
                continue
            ep = f"{p}.experts.{exp_idx}"
            w1_key = f"{ep}.w1.weight"  # gate projection
            w3_key = f"{ep}.w3.weight"  # up projection
            w2_key = f"{ep}.w2.weight"  # down projection

            if self.weights.get(w1_key) is None:
                logger.debug("L%02d expert %d weights not loaded, skipping",
                             layer_idx, exp_idx)
                continue

            uq_g = self._uq_for_key(w1_key)
            uq_u = self._uq_for_key(w3_key)

            if uq_g == 0 and uq_u == 0:
                # F16 path: fused gate + up + SiLU in one dispatch.
                self._dispatch(
                    "fused_gate_act",
                    [normed_x, self.weights[w1_key], self.weights[w3_key],
                     msc["expert_act"]],
                    {"K": hidden, "N": inter, "GELU": 0},
                    (inter, 1, 1),
                )
            else:
                # Quantized path: separate gate and up matmuls then SiLU.
                qi_g = self._quant_extra(f"{ep}.w1", uq_g)
                qi_u = self._quant_extra(f"{ep}.w3", uq_u)
                extra_g: dict = {"SPLIT_K": 0} if uq_g not in (0, 3, 4, 5, 6, 7, 8) else {}
                extra_u: dict = {"SPLIT_K": 0} if uq_u not in (0, 3, 4, 5, 6, 7, 8) else {}
                self._dispatch(
                    "matmul_quant",
                    [normed_x, self.weights[w1_key],
                     self._scales_buf(w1_key, uq_g, msc["dummy_scales"]),
                     msc["expert_gate"]],
                    {"K": hidden, "N": inter, "USE_QUANT": uq_g, **extra_g, **qi_g},
                    _gemv_wg(inter, uq_g),
                )
                self._dispatch(
                    "matmul_quant",
                    [normed_x, self.weights[w3_key],
                     self._scales_buf(w3_key, uq_u, msc["dummy_scales"]),
                     msc["expert_up"]],
                    {"K": hidden, "N": inter, "USE_QUANT": uq_u, **extra_u, **qi_u},
                    _gemv_wg(inter, uq_u),
                )
                self._dispatch(
                    "gelu_mul",
                    [msc["expert_gate"], msc["expert_up"], msc["expert_act"]],
                    {"N": inter},
                    ((inter // 4 + 255) // 256, 1, 1),
                )

            # Down projection + weighted accumulate.
            uq_d = self._uq_for_key(w2_key)
            if uq_d == 0:
                # f16: fuse GEMV and accumulate into a single dispatch.
                self._dispatch(
                    "moe_expert_down_accum",
                    [msc["expert_act"], self.weights[w2_key],
                     msc["expert_out"], msc["moe_w_buf"]],
                    {"K": inter, "N": hidden, "K_IDX": k_idx},
                    (hidden, 1, 1),
                )
            else:
                # Quantized path: keep separate dispatches.
                qi_d = self._quant_extra(f"{ep}.w2", uq_d)
                extra_d: dict = {"SPLIT_K": 0} if uq_d not in (0, 3, 4, 5, 6, 7, 8) else {}
                self._dispatch(
                    "matmul_quant",
                    [msc["expert_act"], self.weights[w2_key],
                     self._scales_buf(w2_key, uq_d, msc["dummy_scales"]),
                     msc["expert_tmp"]],
                    {"K": inter, "N": hidden, "USE_QUANT": uq_d, **extra_d, **qi_d},
                    _gemv_wg(hidden, uq_d),
                )
                self._dispatch(
                    "moe_accumulate",
                    [msc["expert_out"], msc["expert_tmp"], msc["moe_w_buf"]],
                    {"N": hidden, "K_IDX": k_idx},
                    ((hidden + 255) // 256, 1, 1),
                )
