"""GptOssForCausalLM — OpenAI GPT-OSS hybrid SWA+MoE with attention biases."""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING

from vllm_webgpu.models.mixtral import MixtralWebGPUModel, _gemv_wg

if TYPE_CHECKING:
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.device import WebGPUDevice
    from vllm_webgpu.webgpu.pipeline import PipelineCache

logger = logging.getLogger(__name__)


class GptOssWebGPUModel(MixtralWebGPUModel):
    """GPT-OSS: per-layer SWA/full attention, sparse MoE, attention biases.

    Weight key differences from Mixtral:
    - MoE router:  model.layers.{i}.mlp.router.weight
    - MoE experts: model.layers.{i}.mlp.experts.{j}.w1/w3/w2.weight
    - Attention:   model.layers.{i}.self_attn.{q,k,v,o}_proj.{weight,bias}

    GPT-OSS-specific features:
    - attention_bias: Q/K/V/O projections have additive bias vectors.
    - swiglu_limit: gate activation clamped to [-limit, limit] before up-multiply.
    - layer_types: per-layer "sliding_attention" or "full_attention" (not uniform SWA).
    """

    def __init__(
        self,
        model_config,
        wgpu_device: "WebGPUDevice",
        pipeline_cache: "PipelineCache",
    ) -> None:
        super().__init__(model_config, wgpu_device, pipeline_cache)
        self._swiglu_limit: float = getattr(model_config, "swiglu_limit", 0.0)
        self._attn_bias: bool = bool(getattr(model_config, "attention_bias", False))
        self._layer_types: list[str] = list(
            getattr(model_config, "layer_types", None) or []
        )

    def _layer_eff_ctx(self, layer_idx: int, ctx_len: int) -> int:
        """Return effective context for this layer respecting per-layer attention type."""
        if self._layer_types and layer_idx < len(self._layer_types):
            if self._layer_types[layer_idx] == "full_attention":
                return ctx_len
        return self._effective_ctx(ctx_len)

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
        """Mixtral _transformer_layer extended with:

        1. Per-layer effective context: full_attention layers ignore the sliding
           window cap; sliding_attention layers use _effective_ctx() as normal.
        2. Attention bias: after each Q/K/V matmul dispatch and after the O
           projection, the bias vector is added in-place via the add.wgsl shader.
           Bias is added before RoPE so that the rotary encoding is applied to
           the correct biased values.
        3. MoE FFN: delegates to _moe_ffn_layer which uses mlp.experts prefix
           and respects swiglu_limit via the CLAMP_MAX shader override.
        """
        # Per-layer effective context (sliding vs full attention).
        eff = self._layer_eff_ctx(layer_idx, ctx_len)

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

            # QKV projections.
            # GPT-OSS has no q_norm/k_norm weights, so fused_qkv is never used.
            q_wk = f"{p}.self_attn.q_proj.weight"
            k_wk = f"{p}.self_attn.k_proj.weight"
            v_wk = f"{p}.self_attn.v_proj.weight"
            uq_q, uq_k, uq_v = self._uq_for_key(q_wk), self._uq_for_key(k_wk), self._uq_for_key(v_wk)

            for out_buf_qkv, proj, dim, uq in [
                (sc["q_buf"], "q_proj", q_dim, uq_q),
                (sc["k_buf"], "k_proj", kv_dim, uq_k),
                (sc["v_buf"], "v_proj", kv_dim, uq_v),
            ]:
                w_key = f"{p}.self_attn.{proj}.weight"
                qi = self._quant_extra(f"{p}.self_attn.{proj}", uq)
                self._dispatch(
                    "matmul_quant",
                    [normed_x, self.weights[w_key],
                     self._scales_buf(w_key, uq, normed_x), out_buf_qkv],
                    {"K": hidden, "N": dim, "USE_QUANT": uq,
                     **self._split_k_extra(uq),
                     **qi},
                    _gemv_wg(dim, uq),
                )

            # Bias addition before RoPE: WebGPU forbids a buffer appearing as both
            # STORAGE_READ (binding 0) and STORAGE_READ_WRITE (binding 2) in the
            # same dispatch. Use free scratch buffers as add destinations instead:
            #   Q bias: q_buf → gate_buf    K bias: k_buf → up_buf
            #   V bias: v_buf → ffn_act     (these are unused until the FFN phase)
            _q_src = sc["q_buf"]
            _k_src = sc["k_buf"]
            _v_src = sc["v_buf"]

            if self._attn_bias:
                q_bias = self.weights.get(f"{p}.self_attn.q_proj.bias")
                k_bias = self.weights.get(f"{p}.self_attn.k_proj.bias")
                v_bias = self.weights.get(f"{p}.self_attn.v_proj.bias")
                if q_bias is not None:
                    self._dispatch("add", [sc["q_buf"], q_bias, sc["gate_buf"]],
                                   {"N": q_dim}, ((q_dim // 4 + 255) // 256, 1, 1))
                    _q_src = sc["gate_buf"]
                if k_bias is not None:
                    self._dispatch("add", [sc["k_buf"], k_bias, sc["up_buf"]],
                                   {"N": kv_dim}, ((kv_dim // 4 + 255) // 256, 1, 1))
                    _k_src = sc["up_buf"]
                if v_bias is not None:
                    self._dispatch("add", [sc["v_buf"], v_bias, sc["ffn_act"]],
                                   {"N": kv_dim}, ((kv_dim // 4 + 255) // 256, 1, 1))
                    _v_src = sc["ffn_act"]

            q_norm_w = self.weights.get(f"{p}.self_attn.q_norm.weight")
            k_norm_w = self.weights.get(f"{p}.self_attn.k_norm.weight")
            _rope_consts = {
                "HEAD_DIM": self.head_dim,
                "ROPE_BASE": float(self.rope_theta),
                "LN_ROPE_BASE": ln_rope,
                "USE_FREQ_BUF": int(self._use_freq_buf),
                "ATTN_SCALE": self._yarn_mscale,
            }

            for src, dst, n_heads, norm_w, in_off in [
                (_q_src, sc["q_rope"], self.num_q_heads,  q_norm_w, 0),
                (_k_src, sc["k_rope"], self.num_kv_heads, k_norm_w, 0),
            ]:
                if norm_w is not None:
                    self._dispatch(
                        "fused_per_head_norm_rope",
                        [src, norm_w, pos_buf, dst],
                        {**_rope_consts, "NUM_HEADS": n_heads,
                         "HAS_WEIGHT": 1, "INPUT_OFFSET": in_off},
                        (n_heads, num_tokens, 1),
                    )
                else:
                    self._dispatch(
                        "rope",
                        [src, pos_buf, dst],
                        {**_rope_consts, "NUM_HEADS": n_heads},
                        (num_tokens, n_heads, 1),
                    )

            self._dispatch(
                "kv_cache_store_both",
                [sc["k_rope"], k_cache, _v_src, v_cache, slot_map],
                {"BLOCK_SIZE": self.block_size,
                 "NUM_KV_HEADS": self.num_kv_heads,
                 "HEAD_DIM": self.head_dim,
                 "V_IN_OFFSET": 0},
                (num_tokens, self.num_kv_heads, 1),
            )

            # Always use flash_attn_decode for single-token decode.
            # The 65535 limit applied to attn_score's dispatch dimension; flash_attn_decode
            # loops internally and has no dispatch dimension limit.
            self._dispatch(
                "flash_attn_decode",
                [sc["q_rope"], k_cache, v_cache, bt_buf, sc["attn_out"]],
                {"BLOCK_SIZE": self.block_size,
                 "NUM_Q_HEADS": self.num_q_heads,
                 "NUM_KV_HEADS": self.num_kv_heads,
                 "HEAD_DIM": self.head_dim,
                 "CTX_LEN": eff},
                (self.num_q_heads, 1, 1),
            )

            # Output projection + optional O-projection bias.
            w_key = f"{p}.self_attn.o_proj.weight"
            uq = self._uq_for_key(w_key)
            qi = self._quant_extra(f"{p}.self_attn.o_proj", uq)
            self._dispatch(
                "matmul_quant",
                [sc["attn_out"], self.weights[w_key],
                 self._scales_buf(w_key, uq, sc["attn_out"]), sc["o_proj_out"]],
                {"K": q_dim, "N": hidden, "USE_QUANT": uq,
                 **self._split_k_extra(uq),
                 **qi},
                _gemv_wg(hidden, uq),
            )
            # O-projection bias: write to gate_buf (free now — Q bias phase is done).
            # Using gate_buf avoids the STORAGE_READ vs STORAGE_READ_WRITE conflict
            # that would occur if we tried to add bias in-place to o_proj_out.
            _o_proj_src = sc["o_proj_out"]
            if self._attn_bias:
                o_bias = self.weights.get(f"{p}.self_attn.o_proj.bias")
                if o_bias is not None:
                    self._dispatch(
                        "add",
                        [sc["o_proj_out"], o_bias, sc["gate_buf"]],
                        {"N": hidden},
                        ((hidden // 4 + 255) // 256, 1, 1),
                    )
                    _o_proj_src = sc["gate_buf"]

            # Fused post-attention residual add + FFN pre-norm.
            self._dispatch(
                "add_rms_norm",
                [x_buf, _o_proj_src,
                 self.weights[f"{p}.post_attention_layernorm.weight"],
                 residual, sc["ffn_normed"]],
                _rms_c,
                (num_tokens, 1, 1),
            )

            # FFN: MoE (GPT-OSS always uses MoE) or dense fallback.
            if self._is_moe:
                # _moe_ffn_layer (overridden) uses mlp.experts prefix and CLAMP_MAX.
                self._moe_ffn_layer(sc["ffn_normed"], layer_idx)
                ffn_out = self._moe_sc["expert_out"]
            else:
                gw_k = f"{p}.mlp.gate_proj.weight"
                uw_k = f"{p}.mlp.up_proj.weight"
                uq_g = self._uq_for_key(gw_k)
                uq_u = self._uq_for_key(uw_k)
                clamp_extra = ({"CLAMP_MAX": self._swiglu_limit}
                               if self._swiglu_limit > 0 else {})
                if uq_g == 0 and uq_u == 0:
                    self._dispatch(
                        "fused_gate_act",
                        [sc["ffn_normed"], self.weights[gw_k], self.weights[uw_k],
                         sc["ffn_act"]],
                        {"K": hidden, "N": inter, "GELU": 0, **clamp_extra},
                        (inter, 1, 1),
                    )
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
                             **self._split_k_extra(uq2),
                             **qi2},
                            _gemv_wg(inter, uq2),
                        )
                    self._dispatch(
                        "gelu_mul",
                        [sc["gate_buf"], sc["up_buf"], sc["ffn_act"]],
                        {"N": gelu_n},
                        ((gelu_n // 4 + 255) // 256, 1, 1),
                    )
                w_k = f"{p}.mlp.down_proj.weight"
                uq = self._uq_for_key(w_k)
                qi3 = self._quant_extra(f"{p}.mlp.down_proj", uq)
                self._dispatch(
                    "matmul_quant",
                    [sc["ffn_act"], self.weights[w_k],
                     self._scales_buf(w_k, uq, sc["ffn_act"]), sc["ffn_out"]],
                    {"K": inter, "N": hidden, "USE_QUANT": uq,
                     **self._split_k_extra(uq),
                     **qi3},
                    _gemv_wg(hidden, uq),
                )
                ffn_out = sc["ffn_out"]

            # Final residual add fused with next-layer pre-norm when possible.
            if layer_idx < self.num_layers - 1:
                next_w = self.weights[
                    f"model.layers.{layer_idx + 1}.input_layernorm.weight"
                ]
                self._dispatch(
                    "add_rms_norm",
                    [residual, ffn_out, next_w, out, sc["normed"]],
                    _rms_c,
                    (num_tokens, 1, 1),
                )
                normed_out = sc["normed"]
            else:
                self._dispatch(
                    "add",
                    [residual, ffn_out, out],
                    {"N": add_n},
                    ((add_n // 4 + 255) // 256, 1, 1),
                )
                normed_out = out

        self._hstate = (self._hstate + 2) % 3
        return normed_out, out

    def _moe_ffn_layer(
        self,
        normed_x: "WebGPUBuffer",
        layer_idx: int,
    ) -> None:
        """GPT-OSS MoE FFN: delegates to parent with mlp prefix and swiglu clamp.

        Weight keys under model.layers.{i}.mlp:
          router:  mlp.router.weight
          experts: mlp.experts.{j}.w1/w3/w2.weight
        """
        clamp_extra: dict = (
            {"CLAMP_MAX": self._swiglu_limit} if self._swiglu_limit > 0 else {}
        )
        super()._moe_ffn_layer(
            normed_x,
            layer_idx,
            bsm_prefix="mlp",
            router_subkey="router",
            extra_gate_consts=clamp_extra,
        )
