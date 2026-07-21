from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from vllm.logger import init_logger
from vllm_webgpu.models.base import _vec4_wg, _H_NAMES
from vllm_webgpu.models.llama import LlamaWebGPUModel

if TYPE_CHECKING:
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer

logger = init_logger(__name__)


class Olmo2WebGPUModel(LlamaWebGPUModel):
    """OLMo-2 WebGPU backend.

    OLMo-2 uses a post-norm architecture (no input_layernorm):

      y_attn   = post_attention_layernorm(attn(x))
      x_mid    = x + y_attn
      y_ffn    = post_feedforward_layernorm(mlp(x_mid))
      x_next   = x_mid + y_ffn

    A final model.norm is applied after all layers.

    Weight key differences from Llama:
    - NO model.layers.{i}.input_layernorm.weight
    - model.layers.{i}.post_attention_layernorm.weight   (applied to attn output)
    - model.layers.{i}.post_feedforward_layernorm.weight (applied to FFN output)
    - model.layers.{i}.self_attn.q_norm.weight           (per-tensor, shape [hidden_size])
    - model.layers.{i}.self_attn.k_norm.weight           (per-tensor, shape [kv_dim])

    The q_norm and k_norm have shape [hidden_size] and [kv_dim] respectively.
    LlamaWebGPUModel's fused_per_head_norm_rope applies these per-head, which is
    a per-head approximation of OLMo-2's per-tensor norm. The approximation
    differs from the reference when the weight is non-uniform across heads.

    The final norm weight key is model.norm.weight (same as Llama).
    """

    # OLMo-2 has no input_layernorm: disable the fused-final-norm path so that
    # _run_decode_dispatches uses the explicit rms_norm dispatch for model.norm.
    _norm_fusion: bool = False

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
        _final_norm_w: "WebGPUBuffer | None" = None,
        _final_norm_out: "WebGPUBuffer | None" = None,
    ) -> "tuple[WebGPUBuffer, WebGPUBuffer]":
        """OLMo-2 post-norm transformer layer.

        For OLMo-2, normed_x == x_buf on entry (both are the raw residual that
        attention operates on directly). The post-attention and post-feedforward
        norms are applied to the branch outputs (not to the residual stream).

        Returns (out, out) where out = x_mid + norm_ffn(ffn(x_mid)).
        Both return values point to the same buffer so the caller's normed_x
        and x_buf for the next layer are identical, which is correct since
        OLMo-2's next-layer attention also operates on the raw residual.
        """
        sc     = self._sc
        p      = f"model.layers.{layer_idx}"
        rms_c  = self._rms_consts

        residual = sc[_H_NAMES[(self._hstate + 1) % 3]]
        out      = sc[_H_NAMES[(self._hstate + 2) % 3]]

        with self._batched_dispatch(label=f"L{layer_idx:02d}"):
            # Attention on raw input (normed_x == x_buf for OLMo-2).
            o_proj_out = self._attn_block(
                layer_idx, normed_x, pos_buf, slot_map, bt_buf, ctx_len, num_tokens)

            # Fused: rms_norm(attn_out) then add to residual.
            # residual = x + rms_norm(o_proj_out, post_attention_layernorm)
            self._dispatch(
                "rms_norm_add",
                [x_buf,
                 o_proj_out,
                 self.weights[f"{p}.post_attention_layernorm.weight"],
                 residual],
                rms_c,
                (num_tokens, 1, 1),
            )

            # FFN on the post-attention-normed residual.
            ffn_out = self._ffn_dispatch(residual, layer_idx)

            # Fused: rms_norm(ffn_out) then add to residual.
            # out = residual + rms_norm(ffn_out, post_feedforward_layernorm)
            self._dispatch(
                "rms_norm_add",
                [residual,
                 ffn_out,
                 self.weights[f"{p}.post_feedforward_layernorm.weight"],
                 out],
                rms_c,
                (num_tokens, 1, 1),
            )

        self._hstate = (self._hstate + 2) % 3
        # Both outputs point to the same buffer because the next layer's attention
        # also reads the raw residual (no separate pre-norm in OLMo-2).
        return out, out

    # ── Decode path override ──────────────────────────────────────────────────

    def _run_decode_dispatches(
        self,
        ids_buf, pos_buf, slot_map, bt_buf, x_buf,
        norm_out, logits_buf, ctx_len, num_tokens, vocab, greedy,
    ) -> None:
        """OLMo-2 decode dispatches: skip per-layer pre-norm, use x_buf directly.

        OLMo-2 has no input_layernorm. Attention in every layer operates on the
        raw residual stream. The first layer's attention thus receives the embedding
        output without any normalization applied.
        """
        rms_base = self._rms_consts

        self._dispatch(
            "embedding_lookup",
            [self.weights["model.embed_tokens.weight"], ids_buf, x_buf],
            {"HIDDEN_DIM": self.hidden_size},
            (num_tokens, 1, 1),
        )

        # OLMo-2: no initial pre-norm. Every layer's attention sees the raw residual.
        # normed_x == x_buf; we skip the rms_norm dispatch that Llama issues here.
        normed_x = x_buf

        for i in range(self.num_layers):
            normed_x, x_buf = self._transformer_layer(
                i, normed_x, x_buf, pos_buf, slot_map, bt_buf, ctx_len, num_tokens)

        # Final model norm (model.norm.weight) applied after all layers.
        # x_buf here is the last layer's output (already includes per-layer post-norms).
        self._dispatch(
            "rms_norm",
            [x_buf, self.weights["model.norm.weight"], norm_out],
            rms_base,
            (num_tokens, 1, 1),
        )

        self._decode_teardown(norm_out, logits_buf, vocab, greedy)

    # ── Prefill overrides ─────────────────────────────────────────────────────

    def _prefill_sequential_fallback(
        self,
        input_ids: "np.ndarray",
        positions: "np.ndarray",
        attn_metadata: object,
        T: int,
    ) -> "np.ndarray":
        """Per-token prefill without initial pre-norm (OLMo-2 has no input_layernorm)."""
        hidden   = self.hidden_size
        vocab    = self.vocab_size
        rms_base = self._rms_consts
        pre      = self._pre

        bt_bytes = self._bt_arr(attn_metadata).tobytes()
        slot_arr = np.asarray(attn_metadata.slot_mapping, dtype=np.uint32)

        self.wgpu_device.wgpu_device.queue.write_buffer(pre["bt"].buf, 0, bt_bytes)

        for t in range(T):
            self._hstate = 0
            tok_pos = int(positions[t])
            tok_ctx = tok_pos + 1

            ids_t = input_ids[t : t + 1]
            pos_t = positions[t : t + 1]
            self._write_token_bufs(ids_t, pos_t, slot_arr[t : t + 1].tobytes(), write_bt=False)

            with self._batched_dispatch():
                self._dispatch(
                    "embedding_lookup",
                    [self.weights["model.embed_tokens.weight"], pre["ids"], pre["x"]],
                    {"HIDDEN_DIM": hidden},
                    (1, 1, 1),
                )

                # OLMo-2: no pre-norm before the first layer.
                normed_x: "WebGPUBuffer" = pre["x"]
                x_buf:    "WebGPUBuffer" = pre["x"]

                for layer_idx in range(self.num_layers):
                    normed_x, x_buf = self._transformer_layer(
                        layer_idx, normed_x, x_buf,
                        pre["pos"], pre["slot_map"], pre["bt"],
                        tok_ctx, 1,
                    )

        # Final model norm on the last token's output.
        with self._batched_dispatch():
            self._dispatch(
                "rms_norm",
                [x_buf, self.weights["model.norm.weight"], pre["norm_out"]],
                rms_base,
                (1, 1, 1),
            )
            greedy = self._greedy_decode
            self._decode_teardown(pre["norm_out"], pre["logits"], vocab, greedy)

        return self._finish_forward(greedy)

    def _prefill_batch_forward(
        self,
        input_ids: "np.ndarray",
        positions: "np.ndarray",
        attn_metadata: object,
        T: int,
    ) -> "np.ndarray":
        """Batch prefill for OLMo-2: skip initial pre-norm, use copy instead.

        The batch prefill path initialises b["normed"] with a GPU copy of
        b["x"] (the embedding output) rather than an rms_norm dispatch because
        OLMo-2 has no input_layernorm. The layer loop and the rest of the batch
        path are then identical to the Llama version.
        """
        from itertools import batched
        from vllm.utils.math_utils import cdiv
        from vllm_webgpu.webgpu.buffer import WebGPUBuffer

        if T > 65535 or cdiv(T * max(self.intermediate_size, self.hidden_size), 1024) > 65535:
            return self._prefill_sequential_fallback(input_ids, positions, attn_metadata, T)
        if int(positions[0]) > 0:
            return self._prefill_sequential_fallback(input_ids, positions, attn_metadata, T)
        if not self._batch_matmul_supported:
            return self._prefill_sequential_fallback(input_ids, positions, attn_metadata, T)
        if self._sw is not None:
            return self._prefill_sequential_fallback(input_ids, positions, attn_metadata, T)

        hidden = self.hidden_size
        vocab  = self.vocab_size
        rms_base = self._rms_consts
        dev    = self.wgpu_device.wgpu_device

        q_dim  = self.q_dim
        kv_dim = self.kv_dim
        inter  = self.intermediate_size

        b: dict = {
            "x":        self._make_buf(T * hidden * 2),
            "normed":   self._make_buf(T * hidden * 2),
            "q_buf":    self._make_buf(T * q_dim * 2),
            "k_buf":    self._make_buf(T * kv_dim * 2),
            "v_buf":    self._make_buf(T * kv_dim * 2),
            "q_rope":   self._make_buf(T * q_dim * 2),
            "k_rope":   self._make_buf(T * kv_dim * 2),
            "attn_out": self._make_buf(T * q_dim * 2),
            "o_proj":   self._make_buf(T * hidden * 2),
            "gate_buf": self._make_buf(T * inter * 2),
            "up_buf":   self._make_buf(T * inter * 2),
            "ffn_act":  self._make_buf(T * inter * 2),
            "ffn_out":  self._make_buf(T * hidden * 2),
            "h0":       self._make_buf(T * hidden * 2),
            "h1":       self._make_buf(T * hidden * 2),
            "h2":       self._make_buf(T * hidden * 2),
            "last_tok":  self._pre["x"],
            "last_norm": self._pre["norm_out"],
            "logits":    self._pre["logits"],
        }

        slot_map_arr = np.asarray(attn_metadata.slot_mapping, dtype=np.uint32)
        slot_map_buf = WebGPUBuffer.from_numpy(dev, slot_map_arr)
        pos_buf      = WebGPUBuffer.from_numpy(dev, positions.astype(np.uint32, copy=False))
        ids_buf      = WebGPUBuffer.from_numpy(dev, input_ids.astype(np.uint32, copy=False))
        _hstate      = 0
        x_res        = b["x"]
        _pfill_rope_base = self._rope_consts
        _freq_buf    = self._rope_freq_buf

        with self._batched_dispatch():
            self._dispatch("embedding_lookup",
                           [self.weights["model.embed_tokens.weight"], ids_buf, b["x"]],
                           {"HIDDEN_DIM": hidden}, (T, 1, 1))
            # OLMo-2: no initial pre-norm. Use GPU copy to initialise b["normed"]
            # with the embedding output (= identity "norm").
            self._active_encoder.copy_buffer_to_buffer(
                b["x"].buf, 0, b["normed"].buf, 0, T * hidden * 2)

        for chunk_layers in batched(range(self.num_layers), self._PREFILL_CHUNK):
            with self._batched_dispatch():
                for i in chunk_layers:
                    p    = f"model.layers.{i}"
                    q_wk = f"{p}.self_attn.q_proj.weight"
                    k_wk = f"{p}.self_attn.k_proj.weight"
                    v_wk = f"{p}.self_attn.v_proj.weight"
                    ow   = f"{p}.self_attn.o_proj.weight"
                    gw_k = f"{p}.mlp.gate_proj.weight"
                    uw_k = f"{p}.mlp.up_proj.weight"
                    dw_k = f"{p}.mlp.down_proj.weight"

                    # b["normed"] is the input to attention (= raw residual for OLMo-2)
                    self._gemm_batch(b["normed"], q_wk, b["q_buf"],    hidden, q_dim,  T)
                    self._gemm_batch(b["normed"], k_wk, b["k_buf"],    hidden, kv_dim, T)
                    self._gemm_batch(b["normed"], v_wk, b["v_buf"],    hidden, kv_dim, T)

                    for src, dst, n_h, wk in [
                        (b["q_buf"],  b["q_rope"], self.num_q_heads,  f"{p}.self_attn.q_norm.weight"),
                        (b["k_buf"],  b["k_rope"], self.num_kv_heads, f"{p}.self_attn.k_norm.weight"),
                    ]:
                        nw = self.weights.get(wk)
                        if nw is not None:
                            self._dispatch("fused_per_head_norm_rope",
                                           [src, nw, pos_buf, dst, _freq_buf],
                                           {**_pfill_rope_base, "NUM_HEADS": n_h, "HAS_WEIGHT": 1, "INPUT_OFFSET": 0},
                                           (n_h, T, 1))
                        else:
                            self._dispatch("rope", [src, pos_buf, dst, _freq_buf],
                                           {**_pfill_rope_base, "NUM_HEADS": n_h, "INPUT_OFFSET": 0},
                                           (T, n_h, 1))

                    k_cache, v_cache = self.kv_pool[i]
                    self._dispatch("kv_cache_store_both",
                                   [b["k_rope"], k_cache, b["v_buf"], v_cache, slot_map_buf],
                                   {"BLOCK_SIZE": self.block_size,
                                    "NUM_KV_HEADS": self.num_kv_heads,
                                    "HEAD_DIM": self.head_dim,
                                    "V_IN_OFFSET": 0},
                                   (T, self.num_kv_heads, 1))
                    self._dispatch("flash_attn_prefill",
                                   [b["q_rope"], b["k_rope"], b["v_buf"], b["attn_out"]],
                                   {"NUM_Q_HEADS": self.num_q_heads,
                                    "NUM_KV_HEADS": self.num_kv_heads,
                                    "HEAD_DIM": self.head_dim,
                                    "NUM_T": T,
                                    "SCALE": self._attn_scale},
                                   (self.num_q_heads, T, 1))

                    self._gemm_batch(b["attn_out"], ow, b["o_proj"], q_dim, hidden, T)

                    # Fused: rms_norm(attn_out) then add to residual.
                    residual = b[_H_NAMES[(_hstate + 1) % 3]]
                    self._dispatch("rms_norm_add",
                                   [x_res, b["o_proj"],
                                    self.weights[f"{p}.post_attention_layernorm.weight"],
                                    residual],
                                   rms_base, (T, 1, 1))

                    # FFN
                    self._gemm_batch(residual, gw_k, b["gate_buf"], hidden, inter, T)
                    self._gemm_batch(residual, uw_k, b["up_buf"],   hidden, inter, T)
                    self._dispatch("gelu_mul",
                                   [b["gate_buf"], b["up_buf"], b["ffn_act"]],
                                   {"N": T * inter}, _vec4_wg(T * inter))
                    self._gemm_batch(b["ffn_act"], dw_k, b["ffn_out"], inter, hidden, T)

                    # Fused: rms_norm(ffn_out) then add to residual.
                    out_h = b[_H_NAMES[(_hstate + 2) % 3]]
                    self._dispatch("rms_norm_add",
                                   [residual, b["ffn_out"],
                                    self.weights[f"{p}.post_feedforward_layernorm.weight"],
                                    out_h],
                                   rms_base, (T, 1, 1))

                    # For the next layer: copy out_h into b["normed"] as the attention input.
                    # (OLMo-2 next-layer attention receives the raw residual = out_h.)
                    if i < self.num_layers - 1:
                        self._active_encoder.copy_buffer_to_buffer(
                            out_h.buf, 0, b["normed"].buf, 0, T * hidden * 2)

                    x_res   = out_h
                    _hstate = (_hstate + 2) % 3

        with self._batched_dispatch():
            last_token_byte_offset = (T - 1) * hidden * 2
            self._active_encoder.copy_buffer_to_buffer(
                x_res.buf, last_token_byte_offset,
                b["last_tok"].buf, 0,
                hidden * 2,
            )
            self._dispatch("rms_norm",
                           [b["last_tok"], self.weights["model.norm.weight"], b["last_norm"]],
                           rms_base, (1, 1, 1))
            greedy = self._greedy_decode
            self._decode_teardown(b["last_norm"], b["logits"], vocab, greedy)

        return self._finish_forward(greedy)
