"""GptOssForCausalLM — OpenAI GPT-OSS hybrid SWA+MoE with attention biases."""
from __future__ import annotations
from typing import TYPE_CHECKING

from vllm_webgpu.models.base import _rows_wg, _vec4_wg
from vllm_webgpu.models.mixtral import MixtralWebGPUModel

if TYPE_CHECKING:
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.device import WebGPUDevice
    from vllm_webgpu.webgpu.pipeline import PipelineCache


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
        block_size: int = 16,
    ) -> None:
        self._swiglu_limit: float = getattr(model_config, "swiglu_limit", 0.0)
        self._attn_bias: bool = bool(getattr(model_config, "attention_bias", False))
        self._layer_types: list[str] = getattr(model_config, "layer_types", None) or []
        # SwigluOAI: x*sigmoid(1.702*x) with (up+1) bias; optional symmetric up clamp when swiglu_limit > 0
        self._clamp_extra: dict = {"ACTIVATION": 1, "UP_BIAS": 1.0}
        if self._swiglu_limit > 0:
            self._clamp_extra["CLAMP_MAX"] = self._swiglu_limit
            self._clamp_extra["CLAMP_MIN"] = -self._swiglu_limit

        super().__init__(model_config, wgpu_device, pipeline_cache, block_size=block_size)

        # _moe_inter does not need to precede super().__init__() because no code
        # invoked during that call (including _init_scratch_buffers) references it.
        # Moving it here consolidates config reads and lets us use self.intermediate_size
        # (set by the parent) as the natural fallback instead of reaching back to model_config.
        # Use walrus-operator None-check so an explicit moe_intermediate_size=0 is not
        # silently treated as absent (the `or` form would fall back to intermediate_size
        # for zero, which is wrong).
        self._moe_inter: int = v if (v := getattr(model_config, "moe_intermediate_size", None)) is not None else self.intermediate_size
        # Batch prefill bypasses _attn_block and cannot honour per-layer context
        # overrides or inject attention biases. Force sequential prefill whenever
        # either condition is present. The dangerous case for layer_types is
        # sliding_attention (batch prefill applies no per-layer SWA cap); full_attention
        # layers require no special handling.
        # Note: _force_sequential_prefill is intentionally not set here. GptOss has no
        # forward() override for batch prefill: MoE+T>1 raises NotImplementedError before
        # _prefill_batch_forward is reached, and MoE+T=1 uses _moe_decode_forward. The flag
        # would have no observable effect. Restore it alongside a forward() override if
        # batch-prefill support is added for GptOss.

    def _init_scratch_buffers(self, max_ctx: int) -> None:
        """Extend parent scratch buffers with dedicated Q/K/V bias temporaries.

        Allocates dedicated Q/K/V/O bias temporaries sized at q_dim and kv_dim.
        Using the parent FFN scratch buffers (gate_buf/up_buf, sized at
        intermediate_size) would overflow when q_dim or kv_dim > intermediate_size.
        """
        super()._init_scratch_buffers(max_ctx)
        if self._attn_bias:
            Q      = self.num_q_heads  * self.head_dim
            KV     = self.num_kv_heads * self.head_dim
            hidden = self.hidden_size
            self._sc["q_bias_tmp"]  = self._make_buf(Q      * 2)   # [Q]      f16
            self._sc["k_bias_tmp"]  = self._make_buf(KV     * 2)   # [KV]     f16
            self._sc["v_bias_tmp"]  = self._make_buf(KV     * 2)   # [KV]     f16
            self._sc["o_bias_tmp"]  = self._make_buf(hidden * 2)   # [hidden] f16

    def _attn_block(
        self,
        layer_idx: int,
        normed_x: "WebGPUBuffer",
        pos_buf: "WebGPUBuffer",
        slot_map: "WebGPUBuffer",
        bt_buf: "WebGPUBuffer",
        ctx_len: int,
        num_tokens: int,
    ) -> "WebGPUBuffer":
        """Attention block with Q/K/V/O bias injection and per-layer ctx override.

        GPT-OSS has no q_norm/k_norm weights so the fused_qkv path is never
        taken. Bias vectors are added after each projection and before RoPE,
        using scratch buffers to avoid the WebGPU STORAGE_READ / STORAGE_READ_WRITE
        aliasing restriction. The O-projection bias uses sc['ffn_out'] as its
        destination (free at this call site). Per-layer context length respects
        the layer_types list: full_attention layers ignore the sliding window cap.
        """
        is_full = (layer_idx < len(self._layer_types)
                   and self._layer_types[layer_idx] == "full_attention")
        eff = ctx_len if is_full else self._effective_ctx_len(ctx_len)

        sc = self._sc
        hidden = self.hidden_size
        p = f"model.layers.{layer_idx}"
        q_dim = self.num_q_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        k_cache, v_cache = self.kv_pool[layer_idx]

        # QKV projections (always separate; GPT-OSS has no q_norm/k_norm).
        _q_src, _k_src, _v_src = self._qkv_proj(
            normed_x, layer_idx,
            self._uq_for_key(f"{p}.self_attn.q_proj.weight"),
            self._uq_for_key(f"{p}.self_attn.k_proj.weight"),
            self._uq_for_key(f"{p}.self_attn.v_proj.weight"),
        )

        # Bias addition before RoPE: WebGPU forbids a buffer appearing as both
        # STORAGE_READ (binding 0) and STORAGE_READ_WRITE (binding 2) in the
        # same dispatch. Use dedicated scratch buffers sized at q_dim / kv_dim:
        #   Q bias: q_buf -> q_bias_tmp
        #   K bias: k_buf -> k_bias_tmp
        #   V bias: v_buf -> v_bias_tmp
        # These are allocated in _init_scratch_buffers with correct sizes, avoiding
        # the overflow that would occur if gate_buf/up_buf/ffn_act (sized at
        # intermediate_size) were used when q_dim or kv_dim > intermediate_size.

        if self._attn_bias:
            q_bias = self.weights.get(f"{p}.self_attn.q_proj.bias")
            k_bias = self.weights.get(f"{p}.self_attn.k_proj.bias")
            v_bias = self.weights.get(f"{p}.self_attn.v_proj.bias")
            if q_bias is not None:
                self._dispatch("add", [_q_src, q_bias, sc["q_bias_tmp"]],
                               {"N": q_dim}, _vec4_wg(q_dim))
                _q_src = sc["q_bias_tmp"]
            if k_bias is not None:
                self._dispatch("add", [_k_src, k_bias, sc["k_bias_tmp"]],
                               {"N": kv_dim}, _vec4_wg(kv_dim))
                _k_src = sc["k_bias_tmp"]
            if v_bias is not None:
                self._dispatch("add", [_v_src, v_bias, sc["v_bias_tmp"]],
                               {"N": kv_dim}, _vec4_wg(kv_dim))
                _v_src = sc["v_bias_tmp"]

        _rope_consts = self._rope_consts
        _freq_buf = self._rope_freq_buf

        # GPT-OSS has no q_norm/k_norm weights; dispatch rope directly.
        # INPUT_OFFSET=0 is explicit to share the compiled pipeline with other
        # rope call sites that always pass it (omitting it yields a distinct
        # PipelineKey that never gets reused, wasting JIT compilation).
        self._dispatch(
            "rope",
            [_q_src, pos_buf, sc["q_rope"], _freq_buf],
            {**_rope_consts, "NUM_HEADS": self.num_q_heads, "INPUT_OFFSET": 0},
            (num_tokens, self.num_q_heads, 1),
        )
        self._dispatch(
            "rope",
            [_k_src, pos_buf, sc["k_rope"], _freq_buf],
            {**_rope_consts, "NUM_HEADS": self.num_kv_heads, "INPUT_OFFSET": 0},
            (num_tokens, self.num_kv_heads, 1),
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

        self._dispatch(
            "flash_attn_decode",
            [sc["q_rope"], k_cache, v_cache, bt_buf, sc["attn_out"]],
            {"BLOCK_SIZE": self.block_size,
             "NUM_Q_HEADS": self.num_q_heads,
             "NUM_KV_HEADS": self.num_kv_heads,
             "HEAD_DIM": self.head_dim,
             "CTX_LEN": eff,
             "START_BLOCK": 0 if is_full else self._start_block(ctx_len)},
            (self.num_q_heads, 1, 1),
        )

        # Output projection.
        w_key = f"{p}.self_attn.o_proj.weight"
        uq = self._uq_for_key(w_key)
        qi = self._quant_extra(f"{p}.self_attn.o_proj", uq)
        self._dispatch(
            "matmul_quant",
            [sc["attn_out"], self.weights[w_key],
             self._scales_buf(w_key, uq, self._dummy_buf), sc["o_proj_out"]],
            {"K": q_dim, "N": hidden, "USE_QUANT": uq,
             **qi},
            (hidden, 1, 1),
        )

        # O-projection bias: write to sc['o_bias_tmp'], a dedicated buffer sized
        # at hidden that does not alias sc['ffn_out']. Using sc['ffn_out'] as the
        # destination created an implicit ordering contract: _ffn_dispatch must not
        # write ffn_out before add_rms_norm consumes the O-bias result. The dedicated
        # buffer makes both uses independent and safe to reorder.
        _o_proj_src = sc["o_proj_out"]
        if self._attn_bias:
            o_bias = self.weights.get(f"{p}.self_attn.o_proj.bias")
            if o_bias is not None:
                self._dispatch(
                    "add",
                    [sc["o_proj_out"], o_bias, sc["o_bias_tmp"]],
                    {"N": hidden},
                    _vec4_wg(hidden),
                )
                _o_proj_src = sc["o_bias_tmp"]

        return _o_proj_src

    def _ffn_dispatch(
        self,
        normed_x: "WebGPUBuffer",
        layer_idx: int,
        num_tokens: int,
    ) -> "WebGPUBuffer":
        """GPT-OSS FFN dispatch. GPT-OSS is always MoE; raise if that invariant breaks.

        Without this guard, a non-MoE fallback would silently use standard SiLU
        activation instead of SwigluOAI (ACTIVATION=1, UP_BIAS=1.0), producing
        wrong outputs with no error.
        """
        if not self._is_moe:
            raise RuntimeError(
                "GptOssWebGPUModel._ffn_dispatch: _is_moe is False. "
                "GPT-OSS is always MoE; a non-MoE path would produce wrong "
                "outputs (standard SiLU instead of SwigluOAI). Check the "
                "model config (num_local_experts, num_experts_per_tok)."
            )
        return super()._ffn_dispatch(normed_x, layer_idx, num_tokens)

    def _dispatch_expert_gate_up(
        self,
        normed_x: "WebGPUBuffer",
        gw_key: str,
        uw_key: str,
        inter: int,
        extra_gate_consts: dict,
    ) -> None:
        """Extend parent with per-expert gate/up bias injection.

        Derives bias keys from weight keys by replacing the '.weight' suffix with
        '.bias'. If neither key is present in self.weights, delegates directly to
        the parent (fused f16 or quantized path). When at least one bias is found,
        falls back to separate gate/up matmul dispatches (even for f16 weights) so
        the bias vectors can be injected between the matmuls and the activation.

        Buffer routing to avoid intra-dispatch read/write aliasing (WebGPU forbids
        a buffer at two bindings with conflicting access modes in one dispatch):
          expert_gate       <- raw gate GEMV output
          expert_up         <- raw up GEMV output
          expert_gate_biased <- bias-injected gate (or bias-injected up when only
                               u_bias is present and g_bias is absent)
          expert_gate        <- bias-injected up when both biases are present
                               (expert_gate is safe to overwrite after expert_gate_biased
                                has been written as the gate destination)
        The final gelu_mul always writes to msc["expert_act"] so callers are
        unaffected.
        """
        gb_key = gw_key.removesuffix(".weight") + ".bias"
        ub_key = uw_key.removesuffix(".weight") + ".bias"
        g_bias = self.weights.get(gb_key)
        u_bias = self.weights.get(ub_key)

        if g_bias is None and u_bias is None:
            super()._dispatch_expert_gate_up(normed_x, gw_key, uw_key, inter, extra_gate_consts)
            return

        if "K" in extra_gate_consts or "N" in extra_gate_consts:
            raise ValueError(
                f"extra_gate_consts must not contain 'K' or 'N'; "
                f"got {list(extra_gate_consts.keys())}"
            )

        # Separate gate and up dispatches (needed to inject bias between matmul and activation).
        # Allocate expert_gate, expert_up, and expert_tmp together to maintain the three-buffer
        # invariant expected by _ensure_moe_expert_bufs and the quantized _dispatch_expert_down path.
        self._ensure_moe_expert_bufs()
        msc = self._moe_sc
        if "expert_gate_biased" not in msc:
            msc["expert_gate_biased"] = self._make_buf(self._moe_act_sz * 2)
        hidden = self.hidden_size
        uq_g = self._uq_for_key(gw_key)
        uq_u = self._uq_for_key(uw_key)
        qi_g = self._quant_extra(gw_key.removesuffix(".weight"), uq_g)
        qi_u = self._quant_extra(uw_key.removesuffix(".weight"), uq_u)

        self._dispatch(
            "matmul_quant",
            [normed_x, self.weights[gw_key],
             self._scales_buf(gw_key, uq_g, self._dummy_buf), msc["expert_gate"]],
            {"K": hidden, "N": inter, "USE_QUANT": uq_g, **qi_g},
            (inter, 1, 1),
        )
        self._dispatch(
            "matmul_quant",
            [normed_x, self.weights[uw_key],
             self._scales_buf(uw_key, uq_u, self._dummy_buf), msc["expert_up"]],
            {"K": hidden, "N": inter, "USE_QUANT": uq_u, **qi_u},
            (inter, 1, 1),
        )

        # Inject gate bias: expert_gate → expert_gate_biased (different src/dst: no alias).
        # When g_bias is absent use expert_gate directly as the gate source for gelu_mul.
        if g_bias is not None:
            self._dispatch("add", [msc["expert_gate"], g_bias, msc["expert_gate_biased"]],
                           {"N": inter}, _vec4_wg(inter))
            gate_src = msc["expert_gate_biased"]
        else:
            gate_src = msc["expert_gate"]

        # Inject up bias.
        # When both biases are present: write biased up to expert_gate (now free because
        #   raw gate was already consumed into expert_gate_biased above).
        # When only up bias is present: write to expert_gate_biased (gate is read from
        #   expert_gate, which must not be overwritten before gelu_mul).
        if u_bias is not None:
            up_dst = msc["expert_gate"] if g_bias is not None else msc["expert_gate_biased"]
            self._dispatch("add", [msc["expert_up"], u_bias, up_dst],
                           {"N": inter}, _vec4_wg(inter))
            up_src = up_dst
        else:
            up_src = msc["expert_up"]

        self._dispatch(
            "gelu_mul",
            [gate_src, up_src, msc["expert_act"]],
            {**extra_gate_consts, "N": inter},
            _vec4_wg(inter),
        )

    def _dispatch_expert_down(
        self,
        ep: str,
        down_key_name: str,
        w2_key: str,
        k_idx: int,
        inter: int,
    ) -> None:
        """Extend parent with per-expert down bias injection.

        Derives the bias key as '{ep}.{down_key_name}.bias'. If absent, delegates
        to the parent. When present, switches to the separate matmul + add + accumulate
        sequence (bypassing the fused moe_expert_down_accum shader which has no bias
        binding) and uses expert_down_tmp as the staging buffer for the biased output.
        """
        w2_bias = self.weights.get(f"{ep}.{down_key_name}.bias")
        if w2_bias is None:
            super()._dispatch_expert_down(ep, down_key_name, w2_key, k_idx, inter)
            return

        msc = self._moe_sc
        # expert_tmp is allocated by _ensure_moe_expert_bufs() when the biased
        # gate/up path runs. This guard only fires on the f16 non-bias gate/up +
        # biased-down combination, where _ensure_moe_expert_bufs() was not called.
        if "expert_tmp" not in msc:
            msc["expert_tmp"] = self._make_buf(self.hidden_size * 2)
        hidden = self.hidden_size
        uq_d = self._uq_for_key(w2_key)
        qi_d = self._quant_extra(f"{ep}.{down_key_name}", uq_d)

        # Down GEMV with fused bias (HAS_BIAS=1) → expert_tmp.
        # Mirrors the router bias pattern used in _moe_ffn_layer; avoids a
        # separate add dispatch and removes the expert_down_tmp staging buffer.
        self._dispatch(
            "matmul_quant",
            [msc["expert_act"], self.weights[w2_key],
             self._scales_buf(w2_key, uq_d, self._dummy_buf),
             msc["expert_tmp"], w2_bias],
            {"K": inter, "N": hidden, "USE_QUANT": uq_d, "HAS_BIAS": 1, **qi_d},
            (hidden, 1, 1),
        )
        # Weighted accumulate into expert_out from the biased down output.
        self._dispatch(
            "moe_accumulate",
            [msc["expert_out"], msc["expert_tmp"], msc["topk_w"]],
            {"N": hidden, "K_IDX": k_idx},
            _rows_wg(hidden),
        )

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
        super()._moe_ffn_layer(
            normed_x,
            layer_idx,
            bsm_prefix="mlp",
            router_subkey="router",
            extra_gate_consts=self._clamp_extra,
            expert_inter=self._moe_inter,
        )
