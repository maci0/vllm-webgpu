from __future__ import annotations
import logging
from typing import TYPE_CHECKING

import numpy as np

from vllm.logger import init_logger
from vllm_webgpu.models.base import _gemv_wg, _rows_wg, _vec4_wg
from vllm_webgpu.models.llama import LlamaWebGPUModel
from vllm_webgpu.webgpu.buffer import WebGPUBuffer

if TYPE_CHECKING:
    from vllm_webgpu.webgpu.device import WebGPUDevice
    from vllm_webgpu.webgpu.pipeline import PipelineCache

logger = init_logger(__name__)


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
        block_size: int = 16,
    ) -> None:
        super().__init__(model_config, wgpu_device, pipeline_cache, block_size=block_size)
        self._sw: int | None = getattr(model_config, "sliding_window", None)
        self._num_experts: int = getattr(model_config, "num_local_experts", 0)
        self._top_k: int = getattr(model_config, "num_experts_per_tok", 0)
        self._is_moe: bool = self._num_experts > 0 and self._top_k > 0

        if self._is_moe:
            import wgpu as _wgpu_lib
            self._wgpu_lib = _wgpu_lib

            dev = self.wgpu_device.wgpu_device

            # Use the larger of intermediate_size and moe_intermediate_size so
            # subclasses that pass expert_inter > intermediate_size to _moe_ffn_layer
            # do not write past the buffer end.
            _moe_act_sz = max(
                self.intermediate_size,
                getattr(model_config, "moe_intermediate_size", 0),
            )
            # expert_gate and expert_up are only needed on the quantized path
            # (uq_g != 0 or uq_u != 0). Allocate lazily on first use in
            # _moe_ffn_layer to avoid wasting GPU memory for f16 MoE models.
            self._moe_act_sz: int = _moe_act_sz
            self._moe_sc = self._alloc_moe_sc(self._num_experts, self._top_k, _moe_act_sz)
            # Pre-allocated MAP_READ staging buffer for topk idx readback.
            # Copies are recorded into the Phase A encoder so no extra GPU submit
            # is needed after on_submitted_work_done_sync().
            _staging_sz = max(self._top_k * 4, 8)
            self._topk_idx_staging = dev.create_buffer(
                size=_staging_sz,
                usage=_wgpu_lib.BufferUsage.COPY_DST | _wgpu_lib.BufferUsage.MAP_READ)
            # Lazy-allocate _topk_w_staging: only needed on the debug-logging path.
            self._topk_w_staging = None
            # Pre-allocated zero buffer for expert_out initialization. Avoids a
            # fresh bytes() allocation per decode token (32 layers × 8 KB each on
            # 8x7B). Reused across both zero-init sites in _moe_ffn_layer.
            self._expert_out_zeros = bytearray(self.hidden_size * 2)

    def _alloc_moe_sc(
        self,
        num_experts: int,
        top_k: int,
        act_sz: int,
    ) -> "dict[str, WebGPUBuffer]":
        """Allocate the six shared MoE scratch buffers.

        Called from __init__ (Mixtral) and _init_scratch_buffers (Qwen35) to
        avoid duplicating the same dict literal in both subclasses.
        """
        dev = self.wgpu_device.wgpu_device

        def mk(n: int) -> "WebGPUBuffer":
            return WebGPUBuffer.empty(dev, max(n, 8))
        return {
            "router_out":   mk(num_experts * 2),       # [N_E] f16 router logits
            "topk_idx":     mk(top_k * 4),             # [K] u32 expert indices
            "topk_w":       mk(top_k * 4),             # [K] f32 softmax weights
            "expert_act":   mk(act_sz * 2),            # [max_inter] f16 activated
            "expert_out":   mk(self.hidden_size * 2),  # [hidden] f16 accumulated
            "expert_tmp":   mk(self.hidden_size * 2),  # [hidden] f16 per-expert
        }

    def _effective_ctx_len(self, ctx_len: int) -> int:
        """Cap ctx_len at the sliding window size when SWA is configured."""
        return min(ctx_len, self._sw) if self._sw is not None else ctx_len

    def _ffn_dispatch(
        self,
        normed_x: "WebGPUBuffer",
        layer_idx: int,
        num_tokens: int,
    ) -> "WebGPUBuffer":
        """MoE FFN for Mixtral; falls back to dense FFN for non-MoE (Mistral) models."""
        if self._is_moe:
            self._moe_ffn_layer(normed_x, layer_idx)
            return self._moe_sc["expert_out"]
        return super()._ffn_dispatch(normed_x, layer_idx, num_tokens)

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
        self._check_single_sequence(attn_metadata)
        if self.profiling:
            raise RuntimeError(
                "profiling=True is not supported for MoE decode; "
                "disable profiling before calling forward() on a Mixtral MoE model"
            )
        dev = self.wgpu_device.wgpu_device
        num_tokens = len(input_ids)
        vocab = self.vocab_size
        self._hstate = 0

        ids_buf, pos_buf, slot_map, bt_buf, x_buf, norm_out, logits_buf, ctx_len = \
            self._decode_setup(input_ids, positions, attn_metadata)

        greedy = self._greedy_decode

        # Start first encoder manually so that _moe_ffn_layer can flush and
        # recreate it mid-loop for the Phase A CPU expert-index readback.
        # Layer methods see _active_encoder is not None and their _batched_dispatch
        # calls become re-entrant no-ops recording into this encoder.
        self._active_encoder = dev.create_command_encoder()
        try:
            self._run_decode_dispatches(
                ids_buf, pos_buf, slot_map, bt_buf, x_buf,
                norm_out, logits_buf, ctx_len, num_tokens, vocab, greedy,
            )
            dev.queue.submit([self._active_encoder.finish()])
        finally:
            self._active_encoder = None

        if greedy:
            tok = self._read_sample_tok()
            return np.array([[tok]], dtype=np.int32)
        return self.logit_readback()

    def _dispatch_expert_gate_up(
        self,
        normed_x: "WebGPUBuffer",
        gw_key: str,
        uw_key: str,
        inter: int,
        extra_gate_consts: dict,
    ) -> None:
        """Dispatch gate + up projections into self._moe_sc["expert_act"].

        Handles both the fused f16 path (fused_gate_act) and the separate
        quantized matmul + gelu_mul path. gw_key and uw_key are full weight
        keys (ending in '.weight'); the quant-extra prefix is derived by
        stripping the suffix.
        """
        msc = self._moe_sc
        hidden = self.hidden_size
        uq_g = self._uq_for_key(gw_key)
        uq_u = self._uq_for_key(uw_key)
        if uq_g == 0 and uq_u == 0:
            self._dispatch(
                "fused_gate_act",
                [normed_x, self.weights[gw_key], self.weights[uw_key], msc["expert_act"]],
                {"K": hidden, "N": inter, **extra_gate_consts},
                (inter, 1, 1),
            )
        else:
            self._ensure_moe_expert_bufs()
            qi_g = self._quant_extra(gw_key.removesuffix(".weight"), uq_g)
            qi_u = self._quant_extra(uw_key.removesuffix(".weight"), uq_u)
            self._dispatch(
                "matmul_quant",
                [normed_x, self.weights[gw_key],
                 self._scales_buf(gw_key, uq_g, self._dummy_scales_buf),
                 msc["expert_gate"]],
                {"K": hidden, "N": inter, "USE_QUANT": uq_g, **qi_g},
                _gemv_wg(inter),
            )
            self._dispatch(
                "matmul_quant",
                [normed_x, self.weights[uw_key],
                 self._scales_buf(uw_key, uq_u, self._dummy_scales_buf),
                 msc["expert_up"]],
                {"K": hidden, "N": inter, "USE_QUANT": uq_u, **qi_u},
                _gemv_wg(inter),
            )
            self._dispatch(
                "gelu_mul",
                [msc["expert_gate"], msc["expert_up"], msc["expert_act"]],
                {"N": inter, **extra_gate_consts},
                _vec4_wg(inter),
            )

    def _ensure_moe_expert_bufs(self) -> None:
        """Lazily allocate expert_gate and expert_up scratch buffers on first quantized call."""
        msc = self._moe_sc
        if "expert_gate" not in msc:
            dev = self.wgpu_device.wgpu_device
            _act_sz = self._moe_act_sz
            msc["expert_gate"] = WebGPUBuffer.empty(dev, max(_act_sz * 2, 8))
            msc["expert_up"]   = WebGPUBuffer.empty(dev, max(_act_sz * 2, 8))

    def _moe_ffn_layer(
        self,
        normed_x: "WebGPUBuffer",
        layer_idx: int,
        bsm_prefix: str = "block_sparse_moe",
        router_subkey: str = "gate",
        extra_gate_consts: dict | None = None,
        gate_key: str = "w1",
        up_key: str = "w3",
        down_key: str = "w2",
        expert_inter: int | None = None,
        shared_expert_prefix: str | None = None,
        shared_expert_inter: int | None = None,
    ) -> None:
        """MoE FFN, parameterised over weight-key prefix, router key, and expert key names.

        Phase A: router + topk_sort dispatched into the current encoder, then
                 flushed and synced so the CPU can read selected expert indices.
        Phase B: new encoder created; optional shared expert seeded into
                 expert_out, then selected experts accumulated with weights.

        Caller's self._active_encoder is replaced with the Phase B encoder on
        return. Subsequent dispatches in the calling layer method (the residual
        add_rms_norm or add) land in the Phase B encoder, which is correct.

        Args:
            bsm_prefix: Weight-key namespace under model.layers.{i}, e.g.
                        'block_sparse_moe' (Mixtral) or 'mlp' (GPT-OSS/Qwen35).
            router_subkey: Sub-key for the router weight, e.g. 'gate'
                           (Mixtral/Qwen35) or 'router' (GPT-OSS).
            extra_gate_consts: Extra shader constants merged into fused_gate_act
                               and gelu_mul dispatches (quantized path); all keys
                               must be declared as overrides AND identically
                               implemented in both shaders.
            gate_key: Expert weight sub-key for the gate projection, e.g.
                      'w1' (Mixtral/GPT-OSS) or 'gate_proj' (Qwen35).
            up_key: Expert weight sub-key for the up projection, e.g.
                    'w3' (Mixtral/GPT-OSS) or 'up_proj' (Qwen35).
            down_key: Expert weight sub-key for the down projection, e.g.
                      'w2' (Mixtral/GPT-OSS) or 'down_proj' (Qwen35).
            expert_inter: Intermediate size for each expert. Defaults to
                          self.intermediate_size when None.
            shared_expert_prefix: Sub-key under bsm_prefix for the always-active
                                  shared expert (e.g. 'shared_expert' for Qwen35).
                                  When None, expert_out is zero-initialized instead.
            shared_expert_inter: Intermediate size for the shared expert. Defaults
                                 to expert_inter when None.
        """
        if extra_gate_consts is None:
            extra_gate_consts = {}
        dev = self.wgpu_device.wgpu_device
        msc = self._moe_sc
        hidden = self.hidden_size
        inter = expert_inter if expert_inter is not None else self.intermediate_size
        N_E = self._num_experts
        K = self._top_k
        p = f"model.layers.{layer_idx}.{bsm_prefix}"

        # ── Phase A: router + top-K (into current encoder) ───────────────────
        rw_k = f"{p}.{router_subkey}.weight"
        uq_r = self._uq_for_key(rw_k)
        qi_r = self._quant_extra(f"{p}.{router_subkey}", uq_r)
        rb_k = f"{p}.{router_subkey}.bias"
        router_bias = self.weights.get(rb_k)
        router_bindings = [normed_x, self.weights[rw_k],
                           self._scales_buf(rw_k, uq_r, self._dummy_scales_buf),
                           msc["router_out"]]
        router_consts: dict = {"K": hidden, "N": N_E, "USE_QUANT": uq_r, **qi_r}
        if router_bias is not None:
            router_bindings.append(router_bias)
            router_consts["HAS_BIAS"] = 1
        self._dispatch(
            "matmul_quant",
            router_bindings,
            router_consts,
            _gemv_wg(N_E),
        )
        self._dispatch(
            "topk_sort",
            [msc["router_out"], msc["topk_idx"], msc["topk_w"]],
            {"N_EXPERTS": N_E, "K": K},
            (1, 1, 1),
        )

        # Copy topk results into pre-allocated staging buffers inside the Phase A
        # encoder so no extra GPU submit is needed for the readback.
        self._active_encoder.copy_buffer_to_buffer(
            msc["topk_idx"].buf, 0, self._topk_idx_staging, 0, K * 4)
        _debug_weights = logger.isEnabledFor(logging.DEBUG)
        if _debug_weights:
            if self._topk_w_staging is None:
                self._topk_w_staging = dev.create_buffer(
                    size=max(self._top_k * 4, 8),
                    usage=self._wgpu_lib.BufferUsage.COPY_DST | self._wgpu_lib.BufferUsage.MAP_READ)
            self._active_encoder.copy_buffer_to_buffer(
                msc["topk_w"].buf, 0, self._topk_w_staging, 0, K * 4)

        # Flush current encoder and wait for the router + topk to complete.
        dev.queue.submit([self._active_encoder.finish()])
        dev.queue.on_submitted_work_done_sync()

        # Map the pre-allocated staging buffers — no extra GPU submit needed.
        self._topk_idx_staging.map_sync(mode=self._wgpu_lib.MapMode.READ)
        raw_idx = np.frombuffer(self._topk_idx_staging.read_mapped(), dtype=np.uint32).copy()
        self._topk_idx_staging.unmap()
        expert_indices = raw_idx[:K].tolist()
        if _debug_weights:
            self._topk_w_staging.map_sync(mode=self._wgpu_lib.MapMode.READ)
            raw_w = np.frombuffer(self._topk_w_staging.read_mapped(), dtype=np.float32).copy()
            self._topk_w_staging.unmap()
            logger.debug(
                "L%02d MoE experts: %s  weights: %s",
                layer_idx, expert_indices,
                [f"{w:.3f}" for w in raw_w[:K]],
            )

        # Guard: verify scratch buffers are large enough for both inter sizes.
        # _init_scratch_buffers (or __init__) must allocate with the maximum
        # possible intermediate size; assert here so buffer overruns fail fast.
        _max_inter = inter
        if shared_expert_prefix is not None and shared_expert_inter is not None:
            _max_inter = max(inter, shared_expert_inter)
        _buf_capacity = msc["expert_act"].nbytes // 2  # bytes -> f16 elements
        if _max_inter > _buf_capacity:
            raise RuntimeError(
                f"MoE scratch buffer too small: need {_max_inter} f16 elements "
                f"but expert_act holds {_buf_capacity}. "
                f"Override _init_scratch_buffers to allocate max(intermediate_size, "
                f"moe_intermediate_size) elements."
            )

        # Without a shared expert, zero-initialize the accumulation buffer so
        # the first expert's weighted output accumulates from zero.
        if shared_expert_prefix is None:
            dev.queue.write_buffer(msc["expert_out"].buf, 0, self._expert_out_zeros)

        # ── Phase B: expert dispatches (new encoder) ──────────────────────────
        # Subsequent _dispatch() calls (including the residual add in the calling
        # _transformer_layer) will land in this new encoder.
        self._active_encoder = dev.create_command_encoder()

        if shared_expert_prefix is not None:
            # Shared expert is always active with coefficient 1.0. Dispatch it
            # first so its output seeds expert_out before the weighted expert loop.
            _sinter = shared_expert_inter if shared_expert_inter is not None else inter
            sp = f"{p}.{shared_expert_prefix}"
            sgw_k = f"{sp}.{gate_key}.weight"
            suw_k = f"{sp}.{up_key}.weight"
            sdw_k = f"{sp}.{down_key}.weight"
            if all(k in self.weights for k in (sgw_k, suw_k, sdw_k)):
                self._dispatch_expert_gate_up(normed_x, sgw_k, suw_k, _sinter, extra_gate_consts)
                uq_sd = self._uq_for_key(sdw_k)
                qi_sd = self._quant_extra(f"{sp}.{down_key}", uq_sd)
                self._dispatch(
                    "matmul_quant",
                    [msc["expert_act"], self.weights[sdw_k],
                     self._scales_buf(sdw_k, uq_sd, self._dummy_scales_buf), msc["expert_out"]],
                    {"K": _sinter, "N": hidden, "USE_QUANT": uq_sd, **qi_sd},
                    _gemv_wg(hidden),
                )
            else:
                # Shared expert weights not loaded; fall back to zero-init.
                dev.queue.write_buffer(msc["expert_out"].buf, 0, self._expert_out_zeros)

        for k_idx, exp_idx in enumerate(expert_indices):
            ep = f"{p}.experts.{exp_idx}"
            w1_key = f"{ep}.{gate_key}.weight"
            w3_key = f"{ep}.{up_key}.weight"
            w2_key = f"{ep}.{down_key}.weight"

            if any(k not in self.weights for k in (w1_key, w3_key, w2_key)):
                logger.debug("L%02d expert %d weights not loaded, skipping",
                             layer_idx, exp_idx)
                continue

            self._dispatch_expert_gate_up(normed_x, w1_key, w3_key, inter, extra_gate_consts)

            # Down projection + weighted accumulate.
            uq_d = self._uq_for_key(w2_key)
            if uq_d == 0:
                # f16: fuse GEMV and accumulate into a single dispatch.
                self._dispatch(
                    "moe_expert_down_accum",
                    [msc["expert_act"], self.weights[w2_key],
                     msc["expert_out"], msc["topk_w"]],
                    {"K": inter, "N": hidden, "K_IDX": k_idx},
                    (hidden, 1, 1),
                )
            else:
                # Quantized path: keep separate dispatches.
                qi_d = self._quant_extra(f"{ep}.{down_key}", uq_d)
                self._dispatch(
                    "matmul_quant",
                    [msc["expert_act"], self.weights[w2_key],
                     self._scales_buf(w2_key, uq_d, self._dummy_scales_buf),
                     msc["expert_tmp"]],
                    {"K": inter, "N": hidden, "USE_QUANT": uq_d, **qi_d},
                    _gemv_wg(hidden),
                )
                self._dispatch(
                    "moe_accumulate",
                    [msc["expert_out"], msc["expert_tmp"], msc["topk_w"]],
                    {"N": hidden, "K_IDX": k_idx},
                    _rows_wg(hidden),
                )
