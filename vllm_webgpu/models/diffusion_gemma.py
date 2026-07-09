from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from vllm.logger import init_logger
from vllm_webgpu.models.base import _gemv_wg, _vec4_wg, _rows_wg, _H_NAMES
from vllm_webgpu.models.gemma4 import Gemma4WebGPUModel

if TYPE_CHECKING:
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.device import WebGPUDevice
    from vllm_webgpu.webgpu.pipeline import PipelineCache

logger = init_logger(__name__)


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
                 pipeline_cache: "PipelineCache", block_size: int = 16) -> None:
        # Set moe_intermediate_size before super().__init__ because Gemma4.__init__
        # calls _init_scratch_buffers which dispatches to _scratch_inter_size().
        # Use model_config.intermediate_size as fallback (same value as self.intermediate_size
        # after super().__init__; model_config always has this attribute for Gemma models).
        self.moe_intermediate_size: int = getattr(model_config, "moe_intermediate_size",
                                                   getattr(model_config, "expert_intermediate_size",
                                                           model_config.intermediate_size))
        super().__init__(model_config, wgpu_device, pipeline_cache, block_size=block_size)

        # Router scale: constant across all layers and tokens.
        self._router_root_size: float = self.hidden_size ** -0.5

        # MoE configuration
        self.num_experts: int = getattr(model_config, "num_experts", 0)
        self.top_k_experts: int = getattr(model_config, "top_k_experts", 8)
        if self.moe_intermediate_size % 4 != 0:
            raise ValueError(
                f"moe_intermediate_size={self.moe_intermediate_size} must be divisible by 4 "
                f"for vec4<f16> shaders"
            )
        self.is_moe: bool = self.num_experts > 0

        if self.is_moe:
            # _pes_cache is populated by load_weights(); initialize here so that
            # forward() is safe when weights are injected directly (e.g. in tests).
            self._pes_cache: list[np.ndarray | None] = [None] * self.num_layers
            logger.info("DiffusionGemma MoE: %d experts, top-%d, moe_inter=%d",
                        self.num_experts, self.top_k_experts, self.moe_intermediate_size)
            # Extra scratch buffer: shared-expert residual (F16; unlike h0/h1/h2 which are F32).
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
            # Packed routing weights: [num_unique_experts, T] f32, pre-filled before the
            # expert loop so a single write_buffer covers all experts. Sized for worst
            # case: all num_experts active across max_canvas_len tokens.
            self._moe_per_expert_weight_buf = _WB.empty(
                _dev, self.num_experts * max_canvas_len * 4)
            # Tiny f16 dummy for the NO_SCALE=1 router_norm_f32in path: binding 1
            # is declared but never read when NO_SCALE=1; pass this instead of an
            # unrelated weight buffer to make the intent clear.
            self._router_dummy_buf = _WB.empty(_dev, 8)  # 4 x f16

    # ── Scratch buffer sizing ────────────────────────────────────────────────

    def _scratch_token_count(self) -> int:
        return getattr(self.model_config, "canvas_length", 256)

    def _scratch_inter_size(self) -> int:
        return max(self.intermediate_size, self.moe_intermediate_size)

    def _init_scratch_buffers(self, max_ctx: int, max_q_dim: int, max_kv_dim: int) -> None:
        """Allocate scratch buffers without qkv_buf, which _decoder_layer never uses.

        DiffusionGemma overrides forward() and _decoder_layer() entirely; the parent
        _transformer_layer() that reads qkv_buf is never called from this model.
        Calling super() and then deleting qkv_buf wastes a GPU allocation of
        T * (max_q_dim + 2 * max_kv_dim) * 2 bytes (4+ MB at canvas_length=256)
        that is immediately freed. This override replicates only what _decoder_layer
        actually uses.
        """
        from vllm_webgpu.webgpu.buffer import WebGPUBuffer

        dev = self.wgpu_device.wgpu_device
        T = self._scratch_token_count()
        H = self.hidden_size
        I = self._scratch_inter_size()
        NQ = self.num_q_heads
        V = self.vocab_size

        def mk(n: int) -> "WebGPUBuffer":
            return WebGPUBuffer.empty(dev, n)

        self._pre: dict[str, "WebGPUBuffer"] = {
            "ids":      mk(T * 4),
            "pos":      mk(T * 4),
            "slot_map": mk(T * 4),
            "bt":       mk(max(4096, (max_ctx + self.block_size - 1) // self.block_size) * 4),
            "x":        mk(T * H * 4),
            "norm_out": mk(T * H * 2),
            "logits":   mk(T * V * 2),
        }
        if self.softcap is not None and self.softcap > 0:
            self._pre["capped"] = mk(T * V * 2)

        # qkv_buf omitted: _decoder_layer projects Q, K, V separately into q_buf,
        # k_buf, v_buf; the fused [Q|K|V] buffer used by _transformer_layer is
        # never written or read in this model.
        self._sc: dict[str, "WebGPUBuffer"] = {
            "normed":     mk(T * H * 2),
            "q_buf":      mk(T * max_q_dim * 2),
            "k_buf":      mk(T * max_kv_dim * 2),
            "v_buf":      mk(T * max_kv_dim * 2),
            "v_normed":   mk(T * max_kv_dim * 2),
            "q_rope":     mk(T * max_q_dim * 2),
            "k_rope":     mk(T * max_kv_dim * 2),
            "scores_buf": mk(NQ * T * 2),
            "sm_buf":     mk(NQ * T * 2),
            "attn_out":   mk(T * max_q_dim * 2),
            "o_proj_out": mk(T * H * 2),
            "ffn_normed": mk(T * H * 2),
            "gate_buf":   mk(T * I * 2),
            "up_buf":     mk(T * I * 2),
            "ffn_act":    mk(T * I * 2),
            "ffn_out":    mk(T * H * 2),
            "h0":         mk(T * H * 4),
            "h1":         mk(T * H * 4),
            "h2":         mk(T * H * 4),
        }
        self._hstate: int = 0

    # ── Weight key helpers ───────────────────────────────────────────────────

    def _layer_key_prefix(self, layer_idx: int) -> str:
        return f"model.decoder.layers.{layer_idx}"

    def _embed_key(self) -> str:
        """Embedding weight key (DiffusionGemma uses model.decoder.embed_tokens)."""
        return (
            "model.decoder.embed_tokens.weight"
            if "model.decoder.embed_tokens.weight" in self.weights
            else "model.embed_tokens.weight"
        )

    def _norm_key(self) -> str:
        return (
            "model.decoder.norm.weight"
            if "model.decoder.norm.weight" in self.weights
            else "model.norm.weight"
        )

    def _lm_head_key(self) -> str:
        return next(
            (k for k in ("lm_head.weight", "model.decoder.lm_head.weight", "model.lm_head.weight") if k in self.weights),
            self._embed_key(),  # tied weights fallback
        )

    # ── Weight loading ───────────────────────────────────────────────────────

    def load_weights(self, path: str, f32_keys: "frozenset[str] | None" = None) -> None:
        """Load weights and cache per_expert_scale arrays to avoid per-step GPU readbacks."""
        super().load_weights(path, f32_keys=f32_keys)
        # Cache per_expert_scale for each MoE layer. Each to_numpy() is a blocking
        # GPU-CPU sync (~100 µs); caching once at load time avoids N syncs per step.
        self._pes_cache: list[np.ndarray | None] = []
        if self.is_moe:
            for i in range(self.num_layers):
                p = self._layer_key_prefix(i)
                pes_w = self.weights.get(f"{p}.router.per_expert_scale")
                if pes_w is not None:
                    self._pes_cache.append(pes_w.to_numpy().view(np.float16).astype(np.float32))
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

            _lm_key = self._lm_head_key()
            _lm_base = _lm_key.removesuffix(".weight")
            lm_head_w = self.weights[_lm_key]
            uq_lm = self._uq_for_key(_lm_key)
            sc_lm = self._scales_buf(_lm_key, uq_lm, self._dummy_scales_buf)
            if num_tokens > 1:
                # Batched LM head: vocab_size (256128) exceeds the WebGPU 65535
                # per-dimension dispatch limit, so (vocab, num_tokens, 1) is
                # illegal.  matmul_quant_mr4_tiled dispatches
                # ((vocab+255)//256, num_tokens, 1): each workgroup covers 256
                # output columns, one per thread, avoiding the limit.
                self._dispatch("matmul_quant_mr4_tiled",
                               [norm_out, lm_head_w, sc_lm, logits_buf],
                               {"K": hidden, "N": vocab, "M": num_tokens, "USE_QUANT": uq_lm,
                                **self._quant_extra(_lm_base, uq_lm)},
                               ((vocab + 255) // 256, num_tokens, 1))
            else:
                # Single-token decode path.
                self._dispatch("matmul_quant",
                               [norm_out, lm_head_w, sc_lm, logits_buf],
                               {"K": hidden, "N": vocab, "USE_QUANT": uq_lm, "SPLIT_K": 0,
                                **self._quant_extra(_lm_base, uq_lm)},
                               _rows_wg(vocab))

            if self.softcap is not None and self.softcap > 0:
                capped = self._pre["capped"]
                # Dispatch as 2D: x covers vocab elements, y covers tokens.
                # This keeps the x-dimension within the 65535 workgroup-per-dimension
                # limit even when num_tokens * vocab would exceed 65535 * 256.
                self._dispatch("logit_softcap", [logits_buf, capped],
                               {"VOCAB": vocab, "CAP": float(self.softcap)},
                               ((vocab + 255) // 256, num_tokens, 1),
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
        is_kv_shared    = lp.get("is_kv_shared", False)
        kv_shared_target = lp.get("kv_shared_target", -1)
        _kv_layer = kv_shared_target if (is_kv_shared and kv_shared_target >= 0) else layer_idx

        k_cache, v_cache = self.kv_pool[_kv_layer]

        with self._batched_dispatch(label=f"L{layer_idx:02d}"):
            # ── Attention sublayer ────────────────────────────────────────────
            self._dispatch("rms_norm_f32in",
                           [x_buf, self.weights[f"{p}.input_layernorm.weight"], sc["normed"]],
                           _rms, (num_tokens, 1, 1))

            # Q projection: unconditional (KV-shared layers still need Q).
            # K and V projections: skip for KV-shared layers; they reuse the
            # target layer's already-populated cache and never consume these outputs.
            _qk_list = [(sc["q_buf"], "q_proj", q_dim)]
            if not is_kv_shared:
                _qk_list.append((sc["k_buf"], "k_proj", kv_dim))
            for out_buf, proj, dim in _qk_list:
                wk = f"{p}.self_attn.{proj}.weight"
                uq = self._uq_for_key(wk)
                if num_tokens > 1 and uq in (0, 3):
                    _ex: dict = {"K": hidden, "N": dim, "M": num_tokens, "USE_QUANT": uq}
                    _ex.update(self._quant_extra(wk.removesuffix(".weight"), uq))
                    self._dispatch("matmul_quant_mr4",
                                   [sc["normed"], self.weights[wk],
                                    self._scales_buf(wk, uq, self._dummy_scales_buf), out_buf],
                                   _ex, (dim, num_tokens, 1))
                else:
                    self._dispatch("matmul_quant",
                                   [sc["normed"], self.weights[wk],
                                    self._scales_buf(wk, uq, self._dummy_scales_buf), out_buf],
                                   {"K": hidden, "N": dim, "USE_QUANT": uq,
                                    **self._quant_extra(wk.removesuffix(".weight"), uq)},
                                   _gemv_wg(dim))
            if not is_kv_shared:
                # v_proj: global attention layers (no separate V; V=K) have no v_proj weight
                vw_key = f"{p}.self_attn.v_proj.weight"
                has_v_proj = vw_key in self.weights
                if has_v_proj:
                    uq = self._uq_for_key(vw_key)
                    if num_tokens > 1 and uq in (0, 3):
                        _ex_v: dict = {"K": hidden, "N": kv_dim, "M": num_tokens, "USE_QUANT": uq}
                        _ex_v.update(self._quant_extra(vw_key.removesuffix(".weight"), uq))
                        self._dispatch("matmul_quant_mr4",
                                       [sc["normed"], self.weights[vw_key],
                                        self._scales_buf(vw_key, uq, self._dummy_scales_buf),
                                        sc["v_buf"]],
                                       _ex_v, (kv_dim, num_tokens, 1))
                    else:
                        self._dispatch("matmul_quant",
                                       [sc["normed"], self.weights[vw_key],
                                        self._scales_buf(vw_key, uq, self._dummy_scales_buf),
                                        sc["v_buf"]],
                                       {"K": hidden, "N": kv_dim, "USE_QUANT": uq,
                                        **self._quant_extra(vw_key.removesuffix(".weight"), uq)},
                                       _gemv_wg(kv_dim))
                    v_src = sc["v_buf"]
                else:
                    v_src = sc["k_buf"]  # global attention: V = K

            _freq_buf = self._rope_freq_buf
            _dg_rope_base = {"ROPE_BASE": float(self.rope_theta), "LN_ROPE_BASE": ln_rope,
                             "USE_FREQ_BUF": int(self._use_freq_buf)}
            # Q: norm+RoPE unconditionally (KV-shared layers still project and use Q).
            # K: norm+RoPE only for non-shared layers; shared layers read K from cache directly.
            _heads_specs = [(sc["q_buf"], sc["q_rope"], self.num_q_heads, f"{p}.self_attn.q_norm.weight")]
            if not is_kv_shared:
                _heads_specs.append((sc["k_buf"], sc["k_rope"], num_kv_heads, f"{p}.self_attn.k_norm.weight"))
            for src, dst, n_heads, wk in _heads_specs:
                nw = self.weights.get(wk)
                if nw is not None:
                    # Binding 4 (inv_freq_buf): always provided.
                    self._dispatch("fused_per_head_norm_rope",
                                   [src, nw, pos_buf, dst, _freq_buf],
                                   {**_dg_rope_base, "HEAD_DIM": head_dim, "NUM_HEADS": n_heads,
                                    "HAS_WEIGHT": 1, "GEMMA_NORM": self._GEMMA_NORM,
                                    "INPUT_OFFSET": 0},
                                   (n_heads, num_tokens, 1))
                else:
                    # Binding 3 (inv_freq_buf): always provided.
                    self._dispatch("rope", [src, pos_buf, dst, _freq_buf],
                                   {**_dg_rope_base, "HEAD_DIM": head_dim, "NUM_HEADS": n_heads},
                                   (num_tokens, n_heads, 1))

            # Per-head RMSNorm (no weight) on V before caching — required for DiffusionGemma.
            # Matches DiffusionGemmaTextAttention.forward which calls self.v_norm(value_states)
            # unconditionally (DiffusionGemmaRMSNorm, dim=head_dim, with_scale=False).
            # KV-shared layers skip this: V comes from the target layer's cache, not a fresh projection.
            if not is_kv_shared:
                self._dispatch("per_head_rms_norm_no_weight", [v_src, sc["v_normed"]],
                               {"HEAD_DIM": head_dim, "NUM_HEADS": num_kv_heads,
                                "WG_SIZE": min(head_dim, 128),
                                "V_IN_OFFSET": 0},
                               (num_kv_heads, num_tokens, 1), shader_subdir="gemma")
                v_to_cache = sc["v_normed"]
                # Write all T tokens' KV to cache before the attention loop.
                # Each query token then attends to the full ctx_len cache (all T tokens),
                # which is non-causal (bidirectional). For the diffusion denoising use-case
                # this is intentional: the denoising process allows each token to attend
                # to all other tokens in the canvas. If causal attention is ever needed
                # (e.g., for an encoder-only pass), store and attend one token at a time
                # (like _prefill_sequential_fallback) or port flash_attn_prefill here.
                # KV-shared layers reuse the target layer's already-populated cache; skip store.
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
                # SCALE=1.0: Q/K RMSNorm (fused_per_head_norm_rope with GEMMA_NORM=1)
                # implicitly controls magnitudes, so no 1/sqrt(HEAD_DIM) factor is needed.
                self._dispatch("attn_score",
                               [sc["q_rope"], k_cache, bt_buf, sc["scores_buf"]],
                               {"BLOCK_SIZE": self.block_size, "NUM_Q_HEADS": self.num_q_heads,
                                "NUM_KV_HEADS": num_kv_heads, "HEAD_DIM": head_dim,
                                "MAX_SEQ_LEN": ctx_len, "Q_TOKEN_OFFSET": _t_q_off,
                                "SCALE": 1.0},
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
                _ex_ow.update(self._quant_extra(ow.removesuffix(".weight"), uq_ow))
                self._dispatch("matmul_quant_mr4",
                               [sc["attn_out"], self.weights[ow],
                                self._scales_buf(ow, uq_ow, self._dummy_scales_buf),
                                sc["o_proj_out"]],
                               _ex_ow, (hidden, num_tokens, 1))
            else:
                self._dispatch("matmul_quant",
                               [sc["attn_out"], self.weights[ow],
                                self._scales_buf(ow, uq_ow, self._dummy_scales_buf),
                                sc["o_proj_out"]],
                               {"K": q_dim, "N": hidden, "USE_QUANT": uq_ow,
                                **self._quant_extra(ow.removesuffix(".weight"), uq_ow)},
                               _gemv_wg(hidden))

            # post_attention norm + residual add
            pan_w = self.weights.get(f"{p}.post_attention_layernorm.weight")
            if pan_w is not None:
                self._dispatch("rms_norm", [sc["o_proj_out"], pan_w, sc["ffn_normed"]], _rms,
                               (num_tokens, 1, 1))
                self._dispatch("add_f32", [x_buf, sc["ffn_normed"], residual],
                               {"N": add_n}, _vec4_wg(add_n))
            else:
                self._dispatch("add_f32", [x_buf, sc["o_proj_out"], residual],
                               {"N": add_n}, _vec4_wg(add_n))

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
                    _ex_mr4.update(self._quant_extra(wk.removesuffix(".weight"), uq))
                    self._dispatch("matmul_quant_mr4",
                                   [ffn_in, self.weights[wk],
                                    self._scales_buf(wk, uq, self._dummy_scales_buf), out_b],
                                   _ex_mr4, (inter_shared, num_tokens, 1))
                self._dispatch("gelu_mul", [sc["gate_buf"], sc["up_buf"], sc["ffn_act"]],
                               {"N": gelu_n_shared}, _vec4_wg(gelu_n_shared),
                               shader_subdir="gemma")
            elif uq_g == 0 and uq_u == 0:
                self._dispatch("fused_gate_act",
                               [ffn_in, self.weights[gw_k], self.weights[uw_k], sc["ffn_act"]],
                               {"K": hidden, "N": inter_shared, "GELU": 1}, (inter_shared, 1, 1))
            else:
                if num_tokens > 1:
                    raise RuntimeError(
                        "Shared FFN multi-token path requires f16 or GPTQ weights; "
                        "got uq_g=%d uq_u=%d" % (uq_g, uq_u))
                for out_b, wk, uq in [
                        (sc["gate_buf"], gw_k, uq_g),
                        (sc["up_buf"],   uw_k, uq_u)]:
                    self._dispatch("matmul_quant",
                                   [ffn_in, self.weights[wk],
                                    self._scales_buf(wk, uq, self._dummy_scales_buf), out_b],
                                   {"K": hidden, "N": inter_shared, "USE_QUANT": uq,
                                    **self._quant_extra(wk.removesuffix(".weight"), uq)},
                                   _gemv_wg(inter_shared))
                self._dispatch("gelu_mul", [sc["gate_buf"], sc["up_buf"], sc["ffn_act"]],
                               {"N": gelu_n_shared}, _vec4_wg(gelu_n_shared),
                               shader_subdir="gemma")

            dw = f"{p}.mlp.down_proj.weight"
            uq_dw = self._uq_for_key(dw)
            if num_tokens > 1 and uq_dw in (0, 3):
                _ex_dw: dict = {"K": inter_shared, "N": hidden,
                                "M": num_tokens, "USE_QUANT": uq_dw}
                _ex_dw.update(self._quant_extra(dw.removesuffix(".weight"), uq_dw))
                self._dispatch("matmul_quant_mr4",
                               [sc["ffn_act"], self.weights[dw],
                                self._scales_buf(dw, uq_dw, self._dummy_scales_buf), sc["ffn_out"]],
                               _ex_dw, (hidden, num_tokens, 1))
            else:
                self._dispatch("matmul_quant",
                               [sc["ffn_act"], self.weights[dw],
                                self._scales_buf(dw, uq_dw, self._dummy_scales_buf), sc["ffn_out"]],
                               {"K": inter_shared, "N": hidden, "USE_QUANT": uq_dw,
                                **self._quant_extra(dw.removesuffix(".weight"), uq_dw)},
                               _gemv_wg(hidden))

            # MoE layers use post_feedforward_layernorm_1 for the shared MLP stream;
            # non-MoE layers only have the no-suffix key.
            _pfn1_key_1 = f"{p}.post_feedforward_layernorm_1.weight"
            if _pfn1_key_1 in self.weights and self.is_moe:
                pfn1_w = self.weights[_pfn1_key_1]
                self._dispatch("rms_norm", [sc["ffn_out"], pfn1_w, self._shared_res_buf], _rms,
                               (num_tokens, 1, 1))
                hidden_states_1 = self._shared_res_buf
            else:
                hidden_states_1 = sc["ffn_out"]

        layer_scalar = self._layer_scales[layer_idx]

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
                    self._dispatch("router_norm_f32in",
                                   [residual, router_scale_w, router_in],
                                   {"HIDDEN_DIM": hidden, "VALS_PER_THREAD": self._rms_consts["VALS_PER_THREAD"],
                                    "ROOT_SIZE": self._router_root_size},
                                   (num_tokens, 1, 1))
                else:
                    logger.warning("L%d: router.scale missing, routing will be suboptimal (no learned scale)", layer_idx)
                    self._dispatch("router_norm_f32in",
                                   [residual, self._router_dummy_buf, router_in],
                                   {"HIDDEN_DIM": hidden, "VALS_PER_THREAD": self._rms_consts["VALS_PER_THREAD"],
                                    "ROOT_SIZE": self._router_root_size, "NO_SCALE": 1},
                                   (num_tokens, 1, 1))

                rw_ = f"{p}.router.proj.weight"
                uq_rw = self._uq_for_key(rw_)
                _rw_sc = self._scales_buf(rw_, uq_rw, self._dummy_scales_buf)
                if num_tokens > 1 and uq_rw in (0, 3):
                    # Batched router projection: [T, hidden] x [num_experts, hidden]^T -> [T, E]
                    _rw_extra: dict = {"K": hidden, "N": self.num_experts,
                                       "M": num_tokens, "USE_QUANT": uq_rw}
                    _rw_extra.update(self._quant_extra(rw_.removesuffix(".weight"), uq_rw))
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
                                    **self._quant_extra(rw_.removesuffix(".weight"), uq_rw)},
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

            # Vectorized scatter: avoids O(num_tokens * top_k_experts) Python iterations.
            # All (expert, token) index pairs are unique (top-K guarantees distinct
            # expert IDs per token; distinct t values make cross-token duplicates impossible),
            # so buffered fancy-index assignment is equivalent to np.add.at and faster.
            dense_w = np.zeros((self.num_experts, num_tokens), dtype=np.float32)
            t_idx   = np.repeat(np.arange(num_tokens), self.top_k_experts)  # [T*K]
            dense_w[top_k_idx.ravel(), t_idx] += rw_vals.ravel()
            unique_eids = np.unique(top_k_idx).tolist()

            # GPU: run selected expert FFNs
            gelu_n_moe = num_tokens * inter_moe
            moe_acc = self._moe_acc_buf
            # Zero-initialize the accumulation buffer before the expert loop so
            # moe_accumulate_batched can do in-place += without a ping-pong buffer.
            dev.queue.write_buffer(moe_acc.buf, 0, b"\x00" * (add_n * 2))

            # Pre-pack all unique experts' per-token weights into the GPU buffer as a
            # [num_unique_experts, T] f32 array. A single write_buffer here is correct:
            # within one command encoder all write_buffer calls are processed before any
            # recorded compute commands execute, so per-iteration writes inside the loop
            # would leave only the last expert's weights visible to every dispatch. The
            # expert_slot index passed as an override constant lets each shader read its
            # own row without a re-entrant write.
            packed_w = np.stack([dense_w[e] for e in unique_eids], axis=0)  # [num_unique, T]
            dev.queue.write_buffer(self._moe_per_expert_weight_buf.buf, 0, packed_w.tobytes())

            for expert_slot, eid in enumerate(unique_eids):
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

                with self._batched_dispatch(label=f"L{layer_idx:02d}E{eid}"):
                    if use_mr4 and num_tokens > 1:
                        # Batched GEMM path: process all T tokens through this expert.
                        # gate/up: [T, hidden] x [inter_moe, hidden]^T -> [T, inter_moe]
                        for ob, ew_key, uq in [
                            (sc["gate_buf"], f"{ep}.gate_proj.weight", uq_g),
                            (sc["up_buf"],   f"{ep}.up_proj.weight",   uq_u),
                        ]:
                            _sc_e = self._scales_buf(ew_key, uq, self._dummy_scales_buf)
                            _ex_e: dict = {"K": hidden, "N": inter_moe,
                                           "M": num_tokens, "USE_QUANT": uq}
                            _ex_e.update(self._quant_extra(ew_key.removesuffix(".weight"), uq))
                            self._dispatch("matmul_quant_mr4",
                                           [moe_in, self.weights[ew_key], _sc_e, ob],
                                           _ex_e,
                                           (inter_moe, num_tokens, 1))
                        self._dispatch("gelu_mul",
                                       [sc["gate_buf"], sc["up_buf"], sc["ffn_act"]],
                                       {"N": gelu_n_moe},
                                       _vec4_wg(gelu_n_moe),
                                       shader_subdir="gemma")
                        # down: [T, inter_moe] x [hidden, inter_moe]^T -> [T, hidden]
                        _sc_dk = self._scales_buf(dk, uq_dk, self._dummy_scales_buf)
                        _ex_dk: dict = {"K": inter_moe, "N": hidden,
                                        "M": num_tokens, "USE_QUANT": uq_dk}
                        _ex_dk.update(self._quant_extra(dk.removesuffix(".weight"), uq_dk))
                        self._dispatch("matmul_quant_mr4",
                                       [sc["ffn_act"], self.weights[dk], _sc_dk, sc["ffn_out"]],
                                       _ex_dk,
                                       (hidden, num_tokens, 1))
                        # Per-token weighted accumulate: moe_acc[t*H+j] += w[t] * ffn_out[t*H+j]
                        # EXPERT_SLOT selects row expert_slot from packed_w[num_unique, T].
                        self._dispatch("moe_accumulate_batched",
                                       [moe_acc, sc["ffn_out"],
                                        self._moe_per_expert_weight_buf],
                                       {"N": add_n, "H": hidden,
                                        "EXPERT_SLOT": expert_slot},
                                       ((add_n + 255) // 256, 1, 1))
                    else:
                        # Single-token GEMV path (num_tokens==1).
                        for ob, ew_key in [(sc["gate_buf"], f"{ep}.gate_proj.weight"),
                                           (sc["up_buf"],   f"{ep}.up_proj.weight")]:
                            uq = self._uq_for_key(ew_key)
                            self._dispatch("matmul_quant",
                                           [moe_in, self.weights[ew_key],
                                            self._scales_buf(ew_key, uq, self._dummy_scales_buf), ob],
                                           {"K": hidden, "N": inter_moe,
                                            "USE_QUANT": uq,
                                            **self._quant_extra(ew_key.removesuffix(".weight"), uq)},
                                           _gemv_wg(inter_moe))
                        self._dispatch("gelu_mul",
                                       [sc["gate_buf"], sc["up_buf"], sc["ffn_act"]],
                                       {"N": gelu_n_moe},
                                       _vec4_wg(gelu_n_moe),
                                       shader_subdir="gemma")
                        self._dispatch("matmul_quant",
                                       [sc["ffn_act"], self.weights[dk],
                                        self._scales_buf(dk, uq_dk, self._dummy_scales_buf),
                                        sc["ffn_out"]],
                                       {"K": inter_moe, "N": hidden,
                                        "USE_QUANT": uq_dk,
                                        **self._quant_extra(dk.removesuffix(".weight"), uq_dk)},
                                       _gemv_wg(hidden))
                        # K_IDX=expert_slot: reads packed_w[expert_slot] from the pre-filled
                        # [num_unique_experts] f32 array (T=1 so each row is a single scalar).
                        self._dispatch("moe_accumulate",
                                       [moe_acc, sc["ffn_out"],
                                        self._moe_per_expert_weight_buf],
                                       {"N": add_n, "K_IDX": expert_slot},
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
                               {"N": add_n}, _vec4_wg(add_n))

                # Combined post-FFN norm before residual add
                pfn_w = self.weights.get(f"{p}.post_feedforward_layernorm.weight")
                if pfn_w is not None:
                    self._dispatch("rms_norm", [sc["normed"], pfn_w, sc["ffn_out"]], _rms,
                                   (num_tokens, 1, 1))
                    combined_normed = sc["ffn_out"]
                else:
                    combined_normed = sc["normed"]

                self._dispatch("add_f32", [residual, combined_normed, out],
                               {"N": add_n},
                               _vec4_wg(add_n))
        else:
            # Apply post_feedforward_layernorm before residual add, matching vLLM's
            # unconditional application in Gemma4DecoderLayer.forward for all layers.
            pfn_w = self.weights.get(f"{p}.post_feedforward_layernorm.weight")
            if pfn_w is not None:
                self._dispatch("rms_norm", [hidden_states_1, pfn_w, sc["normed"]], _rms,
                               (num_tokens, 1, 1))
                hidden_states_1 = sc["normed"]
            self._dispatch("add_f32", [residual, hidden_states_1, out],
                           {"N": add_n}, _vec4_wg(add_n))

        if abs(layer_scalar - 1.0) > 1e-6:
            self._dispatch("f32_scale_inplace", [out],
                           {"N": add_n, "SCALE": layer_scalar},
                           ((add_n + 255) // 256, 1, 1))

        self._hstate = (self._hstate + 2) % 3
        return out


