from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm_webgpu.models.base import _gemv_wg, _vec4_wg, _rows_wg, _H_NAMES
from vllm_webgpu.models.gemma4 import Gemma4WebGPUModel
from vllm_webgpu.webgpu.buffer import WebGPUBuffer

if TYPE_CHECKING:
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
        # The per-layer "scale" values computed by the parent __init__ are never
        # consumed here because forward() and _decoder_layer() are fully overridden.
        # Clear them to avoid holding stale floats in every lp entry for the
        # lifetime of the model.
        for _lp_e in self._lp:
            _lp_e.pop("scale", None)
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
        self.is_moe: bool = (
            getattr(model_config, "enable_moe_block", False)
            or getattr(model_config, "use_second_mlp_block", False)
        )

        if self.is_moe:
            # _pes_cache is populated by load_weights(); initialize here so that
            # forward() is safe when weights are injected directly (e.g. in tests).
            self._pes_cache: list[np.ndarray | None] = [None] * self.num_layers
            logger.info("DiffusionGemma MoE: %d experts, top-%d, moe_inter=%d",
                        self.num_experts, self.top_k_experts, self.moe_intermediate_size)
            # Extra scratch buffer: shared-expert residual (F16; unlike h0/h1/h2 which are F32).
            # Needed because the 3-buffer h-rotation doesn't accommodate 4 distinct
            # tensor states (x_buf, post-attn, post-shared-expert, post-moe).
            _dev = wgpu_device.wgpu_device
            # canvas_length is the max batch size during diffusion inference (default 256).
            # All per-token scratch buffers must be sized for the full canvas to avoid
            # out-of-bounds writes when num_tokens > 1.
            max_canvas_len = self._scratch_token_count()
            # Pre-allocated row-index array for vectorized MoE scatter; avoids
            # allocating a new array on every _decoder_layer call.
            self._token_arange = np.arange(max_canvas_len, dtype=np.int32)
            self._shared_res_buf = WebGPUBuffer.empty(_dev, max_canvas_len * self.hidden_size * 2)  # F16
            # Pre-allocated GPU top-K buffers — eliminates GPU→CPU router readback.
            self._topk_idx_buf     = WebGPUBuffer.empty(_dev, max_canvas_len * self.top_k_experts * 4)  # [T, K] u32
            self._topk_weight_buf  = WebGPUBuffer.empty(_dev, max_canvas_len * self.top_k_experts * 4)  # [T, K] f32
            self._router_logit_buf     = WebGPUBuffer.empty(_dev, max_canvas_len * self.num_experts * 4)  # [T, E] f32
            self._router_logit_f16_buf = WebGPUBuffer.empty(_dev, max_canvas_len * self.num_experts * 2)  # [T, E] f16 matmul scratch
            self._moe_acc_buf      = WebGPUBuffer.empty(_dev, max_canvas_len * self.hidden_size * 2)    # [T, H] f16
            # Packed routing weights: [num_unique_experts, T] f32, pre-filled before the
            # expert loop so a single write_buffer covers all experts. Sized for worst
            # case: all num_experts active across max_canvas_len tokens.
            self._moe_per_expert_weight_buf = WebGPUBuffer.empty(
                _dev, self.num_experts * max_canvas_len * 4)
            # Pre-allocated dense routing weight matrix: [num_experts, max_canvas_len] f32.
            # Reused across all _decoder_layer calls; only active token columns are zeroed.
            self._dense_w = np.zeros((self.num_experts, max_canvas_len), dtype=np.float32)
            # f16 zero buffer for the NO_SCALE=1 router_norm_f32in path: binding 1
            # is bound but the result is discarded by select when NO_SCALE=1.
            # Sized to hidden_size elements so the binding covers the full scale
            # array the shader declares, avoiding reliance on OOB robustness.
            self._router_dummy_buf = WebGPUBuffer.empty(_dev, self.hidden_size * 2)  # hidden_size x f16

    # ── Guarded parent methods ───────────────────────────────────────────────

    def _transformer_layer(self, *args, **kwargs):
        raise NotImplementedError(
            "DiffusionGemma uses _decoder_layer; parent forward() is fully overridden "
            "and _transformer_layer is never called on this model. "
            "Note: _init_scratch_buffers intentionally omits qkv_buf, so calling "
            "this method would crash with KeyError rather than computing anything useful."
        )

    def _prefill_batch_forward(self, *args, **kwargs):
        raise NotImplementedError(
            "DiffusionGemma does not use the prefill-batch path from the parent; "
            "forward() is fully overridden in this class."
        )

    # ── Scratch buffer sizing ────────────────────────────────────────────────

    def _scratch_token_count(self) -> int:
        return getattr(self.model_config, "canvas_length", 256)

    def _scratch_inter_size(self) -> int:
        return max(super()._scratch_inter_size(), self.moe_intermediate_size)

    def _init_scratch_buffers(self, max_ctx: int, max_q_dim: int, max_kv_dim: int) -> None:
        """Allocate scratch buffers without qkv_buf, which _decoder_layer never uses.

        DiffusionGemma overrides forward() and _decoder_layer() entirely; the parent
        _transformer_layer() that reads qkv_buf is never called from this model.
        Calling super() and then deleting qkv_buf wastes a GPU allocation of
        T * (max_q_dim + 2 * max_kv_dim) * 2 bytes (4+ MB at canvas_length=256)
        that is immediately freed. This override replicates only what _decoder_layer
        actually uses.
        """
        T = self._scratch_token_count()
        H = self.hidden_size
        I = self._max_inter
        NQ = self.num_q_heads
        self._scores_max_ctx = max_ctx

        self._init_pre_buffers(max_ctx)

        # qkv_buf omitted: _decoder_layer projects Q, K, V separately into q_buf,
        # k_buf, v_buf; the fused [Q|K|V] buffer used by _transformer_layer is
        # never written or read in this model.
        self._sc: dict[str, "WebGPUBuffer"] = {
            "normed":     self._make_buf(T * H * 2),
            "q_buf":      self._make_buf(T * max_q_dim * 2),
            "k_buf":      self._make_buf(T * max_kv_dim * 2),
            "v_buf":      self._make_buf(T * max_kv_dim * 2),
            "v_normed":   self._make_buf(T * max_kv_dim * 2),
            "q_rope":     self._make_buf(T * max_q_dim * 2),
            "k_rope":     self._make_buf(T * max_kv_dim * 2),
            # scores_buf and sm_buf are DiffusionGemma-specific: the parent
            # Gemma4WebGPUModel does not allocate them. _decoder_layer uses
            # them for per-token attention score and softmax scratch space.
            "scores_buf": self._make_buf(NQ * max_ctx * 2),  # NQ * max_ctx elements, f16
            "sm_buf":     self._make_buf(NQ * max_ctx * 2),
            "attn_out":   self._make_buf(T * max_q_dim * 2),
            "o_proj_out": self._make_buf(T * H * 2),
            "gate_buf":   self._make_buf(T * I * 2),
            "up_buf":     self._make_buf(T * I * 2),
            "ffn_act":    self._make_buf(T * I * 2),
            "ffn_out":    self._make_buf(T * H * 2),
            # Dedicated buffer for pre_feedforward_layernorm_2 output (MoE input).
            # Using sc["normed"] for this aliased it with three other semantic roles
            # in _decoder_layer, creating an implicit GPU ordering dependency that
            # would silently break if a future read of sc["normed"] were inserted
            # between the L{i}R and L{i}P command encoders.
            "moe_ffn_in": self._make_buf(T * H * 2),
            # Dedicated buffer for router_norm_f32in output (router projection input).
            # Formerly aliased to sc["o_proj_out"], which receives two semantically
            # unrelated writes in the same dispatch sequence (o_proj matmul in L{i},
            # then router norm in L{i}R). Giving it its own buffer removes the implicit
            # ordering dependency that would break if the router norm were ever moved
            # to a separate command encoder.
            "router_in":  self._make_buf(T * H * 2),
            "h0":         self._make_buf(T * H * 4),
            "h1":         self._make_buf(T * H * 4),
            "h2":         self._make_buf(T * H * 4),
        }
        self._hstate: int = 0

    # ── Weight key helpers ───────────────────────────────────────────────────

    def _layer_key_prefix(self, layer_idx: int) -> str:
        return f"model.decoder.layers.{layer_idx}"

    def _embed_key(self) -> str:
        """Embedding weight key (DiffusionGemma uses model.decoder.embed_tokens)."""
        return self._first_weight_key(
            "model.embed_tokens.weight", "model.decoder.embed_tokens.weight",
        )

    def _norm_key(self) -> str:
        return self._first_weight_key(
            "model.norm.weight", "model.decoder.norm.weight",
        )

    def _lm_head_key(self) -> str:
        return self._first_weight_key(
            "lm_head.weight", "model.decoder.lm_head.weight", "model.lm_head.weight",
            self._embed_key(),
        )

    # ── Weight loading ───────────────────────────────────────────────────────

    def load_weights(self, path: str, f32_keys: "frozenset[str] | None" = None,
                     skip_prefixes: "frozenset[str] | None" = None) -> None:
        """Load weights and cache per_expert_scale arrays to avoid per-step GPU readbacks."""
        super().load_weights(path, f32_keys=f32_keys, skip_prefixes=skip_prefixes)
        # Cache per_expert_scale for each MoE layer. Each to_numpy() is a blocking
        # GPU-CPU sync (~100 µs); caching once at load time avoids N syncs per step.
        if self.is_moe:
            for i in range(self.num_layers):
                p = self._layer_key_prefix(i)
                pes_w = self.weights.get(f"{p}.router.per_expert_scale")
                if pes_w is not None:
                    self._pes_cache[i] = pes_w.to_numpy().view(self._buf_np_dtype(pes_w)).astype(np.float32)

    # ── Override forward() for decoder-prefixed keys ─────────────────────────

    def forward(self, input_ids, positions, attn_metadata) -> "np.ndarray":
        """Forward pass using model.decoder.* weight keys."""
        dev = self.wgpu_device.wgpu_device
        num_tokens = len(input_ids)
        max_tokens = self._scratch_token_count()
        if num_tokens > max_tokens:
            raise RuntimeError(
                f"num_tokens={num_tokens} exceeds canvas_length={max_tokens}"
            )
        hidden = self.hidden_size
        vocab = self.vocab_size
        self._hstate = 0

        self._check_single_sequence(attn_metadata)

        ctx_len = int(attn_metadata.max_decode_seq_len)
        if ctx_len > 65535:
            raise RuntimeError(f"ctx_len={ctx_len} exceeds 65535")
        if ctx_len > self._scores_max_ctx:
            raise RuntimeError(
                f"ctx_len={ctx_len} exceeds scores_buf capacity={self._scores_max_ctx}; "
                f"max_position_embeddings in the model config is too small for this sequence"
            )

        pre = self._pre
        dev.queue.write_buffer(pre["ids"].buf, 0, input_ids.astype(np.uint32, copy=False).tobytes())
        dev.queue.write_buffer(pre["pos"].buf, 0, positions.astype(np.uint32, copy=False).tobytes())
        dev.queue.write_buffer(
            pre["slot_map"].buf, 0,
            np.asarray(attn_metadata.slot_mapping, dtype=np.uint32).tobytes())
        bt_arr = self._bt_arr(attn_metadata)
        dev.queue.write_buffer(pre["bt"].buf, 0, bt_arr.tobytes())

        ids_buf = pre["ids"]
        pos_buf = pre["pos"]
        slot_map = pre["slot_map"]
        bt_buf = pre["bt"]
        x_buf = pre["x"]
        norm_out = pre["norm_out"]
        logits_buf = pre["logits"]

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

            _lm_key, lm_head_w, uq_lm, _lm_base = self._lm_head_parts()
            sc_lm = self._scales_buf(_lm_key, uq_lm, self._dummy_buf)
            if num_tokens > 1:
                if uq_lm not in (0, 3):
                    raise RuntimeError(
                        f"batched LM head requires f16 (uq=0) or GPTQ int4 (uq=3); got uq={uq_lm}"
                    )
                # Batched LM head: vocab_size (256128) exceeds the WebGPU 65535
                # per-dimension dispatch limit, so (vocab, num_tokens, 1) is
                # illegal.  matmul_quant_mr4_tiled dispatches
                # ((vocab+255)//256, num_tokens, 1): each workgroup covers 256
                # output columns, one per thread, avoiding the limit.
                self._dispatch("matmul_quant_mr4_tiled",
                               [norm_out, lm_head_w, sc_lm, logits_buf],
                               {"K": hidden, "N": vocab, "M": num_tokens, "USE_QUANT": uq_lm,
                                **self._quant_extra(_lm_base, uq_lm)},
                               (cdiv(vocab, 256), num_tokens, 1))
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
                               (cdiv(vocab, 256), num_tokens, 1),
                               shader_subdir="gemma")
                result = capped
            else:
                result = logits_buf

            dev.queue.submit([self._active_encoder.finish()])
        finally:
            self._active_encoder = None

        return result.to_numpy().view(np.float16)[:num_tokens * vocab].reshape(num_tokens, vocab).astype(np.float32)

    def _gemm_adaptive(
        self,
        src: "WebGPUBuffer",
        wk: str,
        out_b: "WebGPUBuffer",
        K: int,
        N: int,
        num_tokens: int,
    ) -> None:
        """Choose matmul_quant_mr4 for batch or matmul_quant (GEMV) for single token."""
        if num_tokens > 1:
            self._batch_gemm(src, wk, out_b, K, N, num_tokens)
        else:
            uq = self._uq_for_key(wk)
            sc_buf = self._scales_buf(wk, uq, self._dummy_buf)
            self._dispatch("matmul_quant",
                           [src, self.weights[wk], sc_buf, out_b],
                           {"K": K, "N": N, "USE_QUANT": uq, **self._quant_extra(wk.removesuffix(".weight"), uq)},
                           _gemv_wg(N))

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
        inter_shared = lp["intermediate_size"]       # shared expert intermediate size (may be doubled for kv-shared layers with use_double_wide_mlp)
        head_dim = lp["head_dim"]
        q_dim = lp["q_dim"]
        kv_dim = lp["kv_dim"]
        num_kv_heads = lp["num_kv_heads"]
        p = self._layer_key_prefix(layer_idx)

        residual = sc[_H_NAMES[(self._hstate + 1) % 3]]
        out      = sc[_H_NAMES[(self._hstate + 2) % 3]]
        add_n    = num_tokens * hidden
        _rms = self._rms_consts
        is_kv_shared    = lp["is_kv_shared"]
        kv_shared_target = lp["kv_shared_target"]
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
            self._gemm_adaptive(sc["normed"], f"{p}.self_attn.q_proj.weight", sc["q_buf"], hidden, q_dim, num_tokens)
            if not is_kv_shared:
                self._gemm_adaptive(sc["normed"], f"{p}.self_attn.k_proj.weight", sc["k_buf"], hidden, kv_dim, num_tokens)
                # v_proj: global attention layers (no separate V; V=K) have no v_proj weight.
                # Use the precomputed flag from _build_layer_params_from_config as source of truth.
                has_v_proj = lp["has_v_proj"]
                if has_v_proj:
                    self._gemm_adaptive(sc["normed"], f"{p}.self_attn.v_proj.weight", sc["v_buf"], hidden, kv_dim, num_tokens)
                    v_src = sc["v_buf"]
                else:
                    v_src = sc["k_buf"]  # global attention: V = K

            _freq_buf = self._rope_freq_buf
            rc = self._rope_consts[layer_idx]
            # Q: norm+RoPE unconditionally (KV-shared layers still project and use Q).
            # K: norm+RoPE only for non-shared layers; shared layers read K from cache directly.
            _q_nw = self.weights.get(f"{p}.self_attn.q_norm.weight")
            _k_nw = self.weights.get(f"{p}.self_attn.k_norm.weight") if not is_kv_shared else None
            # SCALE=1.0 is correct only when Q/K have been magnitude-normalized by
            # fused_per_head_norm_rope (GEMMA_NORM=1). When the fallback plain rope
            # path runs (q_norm/k_norm weights absent), magnitudes are uncontrolled
            # and the standard 1/sqrt(head_dim) scale applies.
            # Q-norm present but K-norm absent on a non-KV-shared layer is an
            # invariant violation: DiffusionGemma always loads both norms together.
            # Neither 1/sqrt(head_dim) nor 1.0 is clearly correct in this state,
            # so surface it immediately rather than silently producing wrong output.
            if _q_nw is not None and not is_kv_shared and _k_nw is None:
                raise RuntimeError(
                    f"Layer {layer_idx}: Q-norm weight present but K-norm absent "
                    f"on a non-KV-shared layer. DiffusionGemma requires both norms "
                    f"to be loaded together. Check the checkpoint."
                )
            if _q_nw is None and not is_kv_shared and _k_nw is not None:
                raise RuntimeError(
                    f"Layer {layer_idx}: K-norm weight present but Q-norm absent "
                    f"on a non-KV-shared layer. DiffusionGemma requires both norms "
                    f"to be loaded together. Check the checkpoint."
                )
            _has_norms = _q_nw is not None and (is_kv_shared or _k_nw is not None)
            attn_scale = 1.0 if _has_norms else head_dim ** -0.5
            if _q_nw is not None:
                # Binding 4 (inv_freq_buf): always provided.
                self._dispatch("fused_per_head_norm_rope",
                               [sc["q_buf"], _q_nw, pos_buf, sc["q_rope"], _freq_buf],
                               {"ROPE_BASE": rc.rope_base, "LN_ROPE_BASE": rc.ln_rope_base,
                                "USE_FREQ_BUF": rc.use_freq_buf, "ROTARY_DIM": rc.rotary_dim,
                                "FREQ_DIM": rc.freq_dim, "HEAD_DIM": head_dim,
                                "NUM_HEADS": self.num_q_heads,
                                "HAS_WEIGHT": 1, "GEMMA_NORM": self._GEMMA_NORM,
                                "INPUT_OFFSET": 0},
                               (self.num_q_heads, num_tokens, 1))
            else:
                # Binding 3 (inv_freq_buf): always provided.
                self._dispatch("rope", [sc["q_buf"], pos_buf, sc["q_rope"], _freq_buf],
                               {"ROPE_BASE": rc.rope_base, "LN_ROPE_BASE": rc.ln_rope_base,
                                "USE_FREQ_BUF": rc.use_freq_buf,
                                "HEAD_DIM": head_dim, "NUM_HEADS": self.num_q_heads},
                               (num_tokens, self.num_q_heads, 1))
            if not is_kv_shared:
                if _k_nw is not None:
                    # Binding 4 (inv_freq_buf): always provided.
                    self._dispatch("fused_per_head_norm_rope",
                                   [sc["k_buf"], _k_nw, pos_buf, sc["k_rope"], _freq_buf],
                                   {"ROPE_BASE": rc.rope_base, "LN_ROPE_BASE": rc.ln_rope_base,
                                    "USE_FREQ_BUF": rc.use_freq_buf, "ROTARY_DIM": rc.rotary_dim,
                                    "FREQ_DIM": rc.freq_dim, "HEAD_DIM": head_dim,
                                    "NUM_HEADS": num_kv_heads,
                                    "HAS_WEIGHT": 1, "GEMMA_NORM": self._GEMMA_NORM,
                                    "INPUT_OFFSET": 0},
                                   (num_kv_heads, num_tokens, 1))
                else:
                    # Binding 3 (inv_freq_buf): always provided.
                    self._dispatch("rope", [sc["k_buf"], pos_buf, sc["k_rope"], _freq_buf],
                                   {"ROPE_BASE": rc.rope_base, "LN_ROPE_BASE": rc.ln_rope_base,
                                    "USE_FREQ_BUF": rc.use_freq_buf,
                                    "HEAD_DIM": head_dim, "NUM_HEADS": num_kv_heads},
                                   (num_tokens, num_kv_heads, 1))

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
                self._dispatch("attn_score",
                               [sc["q_rope"], k_cache, bt_buf, sc["scores_buf"]],
                               {"BLOCK_SIZE": self.block_size, "NUM_Q_HEADS": self.num_q_heads,
                                "NUM_KV_HEADS": num_kv_heads, "HEAD_DIM": head_dim,
                                "MAX_SEQ_LEN": ctx_len, "Q_TOKEN_OFFSET": _t_q_off,
                                "SCALE": attn_scale},
                               (self.num_q_heads, ctx_len, 1))
                self._dispatch("softmax", [sc["scores_buf"], sc["sm_buf"]],
                               {"SEQ_LEN": ctx_len}, (self.num_q_heads, 1, 1))
                self._dispatch("attn_output",
                               [sc["sm_buf"], v_cache, bt_buf, sc["attn_out"]],
                               {"BLOCK_SIZE": self.block_size, "NUM_Q_HEADS": self.num_q_heads,
                                "NUM_KV_HEADS": num_kv_heads, "HEAD_DIM": head_dim,
                                "CTX_LEN": ctx_len, "ATTN_TOKEN_OFFSET": _t_q_off},
                               (self.num_q_heads, 1, 1))

            self._gemm_adaptive(sc["attn_out"], f"{p}.self_attn.o_proj.weight",
                                sc["o_proj_out"], q_dim, hidden, num_tokens)

            # post_attention norm + residual add + pre_feedforward norm
            pan_w = self.weights.get(f"{p}.post_attention_layernorm.weight")
            pfn_w = self.weights.get(f"{p}.pre_feedforward_layernorm.weight")
            if pfn_w is None:
                raise ValueError(
                    f"Layer {layer_idx} missing pre_feedforward_layernorm.weight "
                    "— f32 residual cannot be fed to f16 FFN projection"
                )

            # ── Shared expert FFN ─────────────────────────────────────────────
            gelu_n_shared = num_tokens * inter_shared
            if pan_w is not None:
                # Fuse rms_norm + add_f32 + rms_norm_f32in into one dispatch,
                # matching the parent Gemma4WebGPUModel._transformer_layer path.
                self._dispatch("rms_norm_add_f32_rms_norm",
                               [sc["o_proj_out"], pan_w, x_buf, pfn_w, residual, sc["normed"]],
                               _rms, (num_tokens, 1, 1))
            else:
                raise ValueError(
                    f"Layer {layer_idx} missing post_attention_layernorm.weight "
                    "— vLLM creates this norm unconditionally; absence indicates a "
                    "corrupt checkpoint"
                )
            ffn_in = sc["normed"]

            # Shared expert gate + up → tanh-GELU activation
            gw_k = f"{p}.mlp.gate_proj.weight"
            uw_k = f"{p}.mlp.up_proj.weight"
            if (num_tokens == 1
                    and self._uq_for_key(gw_k) == 0
                    and self._uq_for_key(uw_k) == 0):
                # Fused single-token f16 path: one shader for gate + up.
                self._dispatch("fused_gate_act",
                               [ffn_in, self.weights[gw_k], self.weights[uw_k], sc["ffn_act"]],
                               {"K": hidden, "N": inter_shared, "GELU": 1}, (inter_shared, 1, 1))
            else:
                self._gemm_adaptive(ffn_in, gw_k, sc["gate_buf"], hidden, inter_shared, num_tokens)
                self._gemm_adaptive(ffn_in, uw_k, sc["up_buf"], hidden, inter_shared, num_tokens)
                self._dispatch("gelu_mul", [sc["gate_buf"], sc["up_buf"], sc["ffn_act"]],
                               {"N": gelu_n_shared}, _vec4_wg(gelu_n_shared),
                               shader_subdir="gemma")

            self._gemm_adaptive(sc["ffn_act"], f"{p}.mlp.down_proj.weight", sc["ffn_out"],
                                inter_shared, hidden, num_tokens)

            # MoE layers use post_feedforward_layernorm_1 for the shared MLP stream;
            # non-MoE layers only have the no-suffix key.
            if self.is_moe:
                pfn1_w = self.weights.get(f"{p}.post_feedforward_layernorm_1.weight")
                if pfn1_w is not None:
                    self._dispatch("rms_norm", [sc["ffn_out"], pfn1_w, self._shared_res_buf], _rms,
                                   (num_tokens, 1, 1))
                    hidden_states_1 = self._shared_res_buf
                else:
                    raise ValueError(
                        f"MoE layer {layer_idx} missing post_feedforward_layernorm_1.weight. "
                        "The shared-MLP output cannot be passed directly to the combine dispatch: "
                        "sc['ffn_out'] is overwritten by the expert loop and the alias would use "
                        "the last expert's down-projection result instead of the shared expert output. "
                        "A correctly loaded DiffusionGemma checkpoint always has this weight."
                    )
            else:
                hidden_states_1 = sc["ffn_out"]

        layer_scalar = self._layer_scales[layer_idx]

        # ── MoE expert FFN (all-GPU: router + top-K selection + expert FFNs) ───
        if self.is_moe:
            inter_moe = self.moe_intermediate_size     # MoE expert intermediate size
            if f"{p}.router.proj.weight" not in self.weights:
                raise RuntimeError(
                    f"L{layer_idx}: is_moe=True but router.proj.weight missing"
                )
            dev = self.wgpu_device.wgpu_device
            router_logits_buf = self._router_logit_buf
            pfn2_w = self.weights.get(f"{p}.pre_feedforward_layernorm_2.weight")

            with self._batched_dispatch(label=f"L{layer_idx:02d}R"):
                if pfn2_w is not None:
                    self._dispatch("rms_norm_f32in", [residual, pfn2_w, sc["moe_ffn_in"]],
                                   _rms, (num_tokens, 1, 1))
                    moe_in = sc["moe_ffn_in"]
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
                router_proj_in = sc["router_in"]
                if router_scale_w is not None:
                    self._dispatch("router_norm_f32in",
                                   [residual, router_scale_w, router_proj_in],
                                   {"HIDDEN_DIM": hidden, "VALS_PER_THREAD": self._rms_consts["VALS_PER_THREAD"],
                                    "ROOT_SIZE": self._router_root_size},
                                   (num_tokens, 1, 1))
                else:
                    logger.warning("L%d: router.scale missing, routing will be suboptimal (no learned scale)", layer_idx)
                    self._dispatch("router_norm_f32in",
                                   [residual, self._router_dummy_buf, router_proj_in],
                                   {"HIDDEN_DIM": hidden, "VALS_PER_THREAD": self._rms_consts["VALS_PER_THREAD"],
                                    "ROOT_SIZE": self._router_root_size, "NO_SCALE": 1},
                                   (num_tokens, 1, 1))

                rw_ = f"{p}.router.proj.weight"
                uq_rw = self._uq_for_key(rw_)
                _rw_sc = self._scales_buf(rw_, uq_rw, self._dummy_buf)
                # Router matmul writes f16 into the scratch buffer; we upcast to f32
                # before top-K so that logits differing by less than one f16 ULP are
                # not collapsed to the same value (matches vLLM's GateLinear f32 path).
                rlogit_f16 = self._router_logit_f16_buf
                if num_tokens > 1 and uq_rw in (0, 3):
                    # Batched router projection: [T, hidden] x [num_experts, hidden]^T -> [T, E]
                    self._dispatch("matmul_quant_mr4",
                                   [router_proj_in, self.weights[rw_], _rw_sc, rlogit_f16],
                                   {"K": hidden, "N": self.num_experts, "M": num_tokens,
                                    "USE_QUANT": uq_rw,
                                    **self._quant_extra(rw_.removesuffix(".weight"), uq_rw)},
                                   (self.num_experts, num_tokens, 1))
                else:
                    if num_tokens > 1:
                        raise RuntimeError(
                            f"L{layer_idx}: router.proj.weight quant uq={uq_rw} is not "
                            f"supported for batched routing (num_tokens={num_tokens}). "
                            f"Only uq=0 (f16) and uq=3 (GPTQ) are handled by "
                            f"matmul_quant_mr4. Routing tokens 1..T-1 via zero logits "
                            f"produces deterministic wrong expert assignments (always "
                            f"experts 0..K-1), not uniform routing."
                        )
                    self._dispatch("matmul_quant",
                                   [router_proj_in, self.weights[rw_], _rw_sc, rlogit_f16],
                                   {"K": hidden, "N": self.num_experts,
                                    "USE_QUANT": uq_rw,
                                    **self._quant_extra(rw_.removesuffix(".weight"), uq_rw)},
                                   _gemv_wg(self.num_experts))
                # Upcast f16 logits to f32 before top-K selection.
                n_logits = num_tokens * self.num_experts
                self._dispatch("f16_to_f32",
                               [rlogit_f16, router_logits_buf],
                               {"N_ELEMS": n_logits},
                               (cdiv(n_logits, 256), 1, 1))
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
            #
            # Invariant: moe_acc zero-init MUST happen after this submit (in the
            # fresh encoder below) so the zero-init is scoped to this layer's
            # expert pass and does not race with accumulations from prior layers.
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
            # np.unique returns sorted deduplicated expert IDs. The sort order only
            # affects expert_slot numbering in packed_w, which is indexed consistently
            # by the same order, so the output is correct regardless of sort order.
            # At K=8, T<=256 (E<=2048 elements) the O(E log E) cost is sub-microsecond.
            unique_eids = np.unique(top_k_idx)
            self._dense_w[unique_eids, :num_tokens] = 0.0
            self._dense_w[top_k_idx, self._token_arange[:num_tokens, None]] = rw_vals

            # GPU: run selected expert FFNs
            gelu_n_moe = num_tokens * inter_moe
            moe_acc = self._moe_acc_buf
            # Zero-initialize the accumulation buffer before the expert loop so
            # moe_accumulate_batched can do in-place += without a ping-pong buffer.
            dev.queue.write_buffer(moe_acc.buf, 0, b"\x00" * (add_n * 2))

            # Pre-pack all unique experts' per-token weights into the GPU buffer as a
            # [num_unique_experts, T] f32 array. A single write_buffer here is correct:
            # all write_buffer calls before a given submit() are visible to that
            # submit's encoded commands (WebGPU submission-ordering guarantee), so
            # the zero-init and weight upload here (after the explicit L{i}R flush
            # above and before the expert dispatches below) are guaranteed to arrive
            # before any expert compute shader reads either buffer. Per-iteration
            # writes inside the loop would leave only the last expert's weights
            # visible to every dispatch. The expert_slot index passed as an override
            # constant lets each shader read its own row without a re-entrant write.
            packed_w = self._dense_w[unique_eids, :num_tokens]  # [num_unique, T]
            dev.queue.write_buffer(self._moe_per_expert_weight_buf.buf, 0, packed_w.tobytes())

            for expert_slot, eid in enumerate(unique_eids):
                ep = f"{p}.experts.{eid}"
                g_w = self.weights.get(f"{ep}.gate_proj.weight")
                u_w = self.weights.get(f"{ep}.up_proj.weight")
                d_w = self.weights.get(f"{ep}.down_proj.weight")
                if g_w is None or u_w is None or d_w is None:
                    raise RuntimeError(f"L{layer_idx}: expert {eid} missing gate/up/down weights")

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
                            _sc_e = self._scales_buf(ew_key, uq, self._dummy_buf)
                            self._dispatch("matmul_quant_mr4",
                                           [moe_in, self.weights[ew_key], _sc_e, ob],
                                           {"K": hidden, "N": inter_moe, "M": num_tokens,
                                            "USE_QUANT": uq,
                                            **self._quant_extra(ew_key.removesuffix(".weight"), uq)},
                                           (inter_moe, num_tokens, 1))
                        self._dispatch("gelu_mul",
                                       [sc["gate_buf"], sc["up_buf"], sc["ffn_act"]],
                                       {"N": gelu_n_moe},
                                       _vec4_wg(gelu_n_moe),
                                       shader_subdir="gemma")
                        # down: [T, inter_moe] x [hidden, inter_moe]^T -> [T, hidden]
                        _sc_dk = self._scales_buf(dk, uq_dk, self._dummy_buf)
                        self._dispatch("matmul_quant_mr4",
                                       [sc["ffn_act"], self.weights[dk], _sc_dk, sc["ffn_out"]],
                                       {"K": inter_moe, "N": hidden, "M": num_tokens,
                                        "USE_QUANT": uq_dk,
                                        **self._quant_extra(dk.removesuffix(".weight"), uq_dk)},
                                       (hidden, num_tokens, 1))
                        # Per-token weighted accumulate: moe_acc[t*H+j] += w[t] * ffn_out[t*H+j]
                        # EXPERT_SLOT selects row expert_slot from packed_w[num_unique, T].
                        self._dispatch("moe_accumulate_batched",
                                       [moe_acc, sc["ffn_out"],
                                        self._moe_per_expert_weight_buf],
                                       {"N": add_n, "H": hidden,
                                        "EXPERT_SLOT": expert_slot},
                                       (cdiv(add_n, 256), 1, 1))
                    else:
                        # Single-token GEMV path (num_tokens==1).
                        # uq_g and uq_u were computed at lines above; reuse to avoid redundant dict lookups.
                        for ob, ew_key, uq in [(sc["gate_buf"], f"{ep}.gate_proj.weight", uq_g),
                                               (sc["up_buf"],   f"{ep}.up_proj.weight",   uq_u)]:
                            self._dispatch("matmul_quant",
                                           [moe_in, self.weights[ew_key],
                                            self._scales_buf(ew_key, uq, self._dummy_buf), ob],
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
                                        self._scales_buf(dk, uq_dk, self._dummy_buf),
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
                                       (cdiv(add_n, 256), 1, 1))
            # Post-MoE norm + single residual add (vLLM Gemma4 pattern)
            with self._batched_dispatch(label=f"L{layer_idx:02d}P"):
                pfn2_out_w = self.weights.get(f"{p}.post_feedforward_layernorm_2.weight")
                if pfn2_out_w is not None:
                    self._dispatch("rms_norm", [moe_acc, pfn2_out_w, sc["o_proj_out"]], _rms,
                                   (num_tokens, 1, 1))
                    hidden_states_2 = sc["o_proj_out"]
                else:
                    raise ValueError(
                        f"MoE layer {layer_idx} missing post_feedforward_layernorm_2.weight. "
                        "The unnormed moe_acc cannot be passed directly to the combine dispatch: "
                        "vLLM's Gemma4DecoderLayer applies post_feedforward_layernorm_2 "
                        "unconditionally when enable_moe_block=True, so skipping it produces "
                        "wrong MoE outputs. A correctly loaded DiffusionGemma checkpoint always "
                        "has this weight."
                    )

                # Combine shared-MLP and MoE streams (f16 + f16 -> f16)
                self._dispatch("add", [hidden_states_1, hidden_states_2, sc["normed"]],
                               {"N": add_n}, _vec4_wg(add_n))

                # Combined post-FFN norm before residual add
                post_ffw_w = self.weights.get(f"{p}.post_feedforward_layernorm.weight")
                if post_ffw_w is None:
                    raise ValueError(
                        f"MoE layer {layer_idx} missing post_feedforward_layernorm.weight "
                        "— vLLM applies this norm unconditionally in Gemma4DecoderLayer.forward; "
                        "skipping it passes the unnormed combined tensor into the residual add "
                        "and produces wrong outputs. A correctly loaded checkpoint always has "
                        "this weight."
                    )
                self._dispatch("rms_norm", [sc["normed"], post_ffw_w, sc["ffn_out"]], _rms,
                               (num_tokens, 1, 1))
                self._dispatch("add_f32", [residual, sc["ffn_out"], out],
                               {"N": add_n}, _vec4_wg(add_n))
                if abs(layer_scalar - 1.0) > 1e-6:
                    self._dispatch("f32_scale_inplace", [out],
                                   {"N": add_n, "SCALE": layer_scalar},
                                   (cdiv(add_n, 256), 1, 1))
        else:
            # Apply post_feedforward_layernorm before residual add, matching vLLM's
            # unconditional application in Gemma4DecoderLayer.forward for all layers.
            with self._batched_dispatch(label=f"L{layer_idx:02d}T"):
                post_ffw_w = self.weights.get(f"{p}.post_feedforward_layernorm.weight")
                if post_ffw_w is None:
                    raise ValueError(
                        f"Layer {layer_idx} missing post_feedforward_layernorm.weight "
                        "— vLLM applies this norm unconditionally; a missing weight "
                        "indicates a corrupt or incomplete checkpoint."
                    )
                self._dispatch("rms_norm", [hidden_states_1, post_ffw_w, sc["normed"]], _rms,
                               (num_tokens, 1, 1))
                hidden_states_1 = sc["normed"]
                self._dispatch("add_f32", [residual, hidden_states_1, out],
                               {"N": add_n}, _vec4_wg(add_n))
                if abs(layer_scalar - 1.0) > 1e-6:
                    self._dispatch("f32_scale_inplace", [out],
                                   {"N": add_n, "SCALE": layer_scalar},
                                   (cdiv(add_n, 256), 1, 1))

        self._hstate = (self._hstate + 2) % 3
        return out


