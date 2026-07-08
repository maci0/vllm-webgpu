from __future__ import annotations
import logging
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
        return min(ctx_len, self._sw) if self._sw else ctx_len

    def _effective_ctx_len(self, ctx_len: int) -> int:
        return self._effective_ctx(ctx_len)

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
        dev = self.wgpu_device.wgpu_device
        num_tokens = len(input_ids)
        hidden = self.hidden_size
        vocab = self.vocab_size
        self._hstate = 0
        sc = self._sc

        _rms_base = self._rms_consts

        ids_buf, pos_buf, slot_map, bt_buf, x_buf, norm_out, logits_buf, ctx_len = \
            self._decode_setup(input_ids, positions, attn_metadata)

        greedy = getattr(self, "_greedy_decode", True)

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

        self._decode_teardown(norm_out, logits_buf, vocab, greedy)

        dev.queue.submit([self._active_encoder.finish()])
        self._active_encoder = None

        if greedy:
            tok = self._read_sample_tok()
            return np.array([[tok]], dtype=np.int32)
        return self.logit_readback()

    def _moe_ffn_layer(
        self,
        normed_x: "WebGPUBuffer",
        layer_idx: int,
        bsm_prefix: str = "block_sparse_moe",
        router_subkey: str = "gate",
        extra_gate_consts: dict | None = None,
    ) -> None:
        """MoE FFN, parameterised over weight-key prefix and router key name.

        Phase A: router + topk_sort dispatched into the current encoder, then
                 flushed and synced so the CPU can read selected expert indices.
        Phase B: new encoder created; selected experts dispatched with
                 weighted accumulation into _moe_sc["expert_out"].

        Caller's self._active_encoder is replaced with the Phase B encoder on
        return. Subsequent dispatches in the calling layer method (the residual
        add_rms_norm or add) land in the Phase B encoder, which is correct.

        Args:
            bsm_prefix: Weight-key namespace under model.layers.{i}, e.g.
                        'block_sparse_moe' (Mixtral) or 'mlp' (GPT-OSS).
            router_subkey: Sub-key for the router weight, e.g. 'gate'
                           (Mixtral) or 'router' (GPT-OSS).
            extra_gate_consts: Extra shader constants merged into fused_gate_act
                               dispatches, e.g. {'CLAMP_MAX': limit}.
        """
        if extra_gate_consts is None:
            extra_gate_consts = {}
        dev = self.wgpu_device.wgpu_device
        msc = self._moe_sc
        hidden = self.hidden_size
        inter = self.intermediate_size
        N_E = self._num_experts
        K = self._top_k
        p = f"model.layers.{layer_idx}.{bsm_prefix}"

        # ── Phase A: router + top-K (into current encoder) ───────────────────
        rw_k = f"{p}.{router_subkey}.weight"
        uq_r = self._uq_for_key(rw_k)
        qi_r = self._quant_extra(f"{p}.{router_subkey}", uq_r)
        self._dispatch(
            "matmul_quant",
            [normed_x, self.weights[rw_k],
             self._scales_buf(rw_k, uq_r, msc["dummy_scales"]),
             msc["router_out"]],
            {"K": hidden, "N": N_E, "USE_QUANT": uq_r, **self._split_k_extra(uq_r), **qi_r},
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
        expert_indices = raw_idx[:K].tolist()
        expert_weights = raw_w[:K].tolist()

        logger.debug(
            "L%02d MoE experts: %s  weights: %s",
            layer_idx, expert_indices,
            [f"{w:.3f}" for w in expert_weights],
        )

        # Write softmax weights into the combined weight buffer so moe_accumulate
        # can read w_buf[K_IDX] without a per-dispatch CPU roundtrip.
        dev.queue.write_buffer(msc["moe_w_buf"].buf, 0,
                               np.array(expert_weights, dtype=np.float32).tobytes())
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
                    {"K": hidden, "N": inter, "GELU": 0, **extra_gate_consts},
                    (inter, 1, 1),
                )
            else:
                # Quantized path: separate gate and up matmuls then SiLU.
                qi_g = self._quant_extra(f"{ep}.w1", uq_g)
                qi_u = self._quant_extra(f"{ep}.w3", uq_u)
                self._dispatch(
                    "matmul_quant",
                    [normed_x, self.weights[w1_key],
                     self._scales_buf(w1_key, uq_g, msc["dummy_scales"]),
                     msc["expert_gate"]],
                    {"K": hidden, "N": inter, "USE_QUANT": uq_g, **self._split_k_extra(uq_g), **qi_g},
                    _gemv_wg(inter, uq_g),
                )
                self._dispatch(
                    "matmul_quant",
                    [normed_x, self.weights[w3_key],
                     self._scales_buf(w3_key, uq_u, msc["dummy_scales"]),
                     msc["expert_up"]],
                    {"K": hidden, "N": inter, "USE_QUANT": uq_u, **self._split_k_extra(uq_u), **qi_u},
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
                self._dispatch(
                    "matmul_quant",
                    [msc["expert_act"], self.weights[w2_key],
                     self._scales_buf(w2_key, uq_d, msc["dummy_scales"]),
                     msc["expert_tmp"]],
                    {"K": inter, "N": hidden, "USE_QUANT": uq_d, **self._split_k_extra(uq_d), **qi_d},
                    _gemv_wg(hidden, uq_d),
                )
                self._dispatch(
                    "moe_accumulate",
                    [msc["expert_out"], msc["expert_tmp"], msc["moe_w_buf"]],
                    {"N": hidden, "K_IDX": k_idx},
                    ((hidden + 255) // 256, 1, 1),
                )
