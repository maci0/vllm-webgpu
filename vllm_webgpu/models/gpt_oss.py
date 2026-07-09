"""GptOssForCausalLM — OpenAI GPT-OSS hybrid SWA+MoE with attention biases."""
from __future__ import annotations
from typing import TYPE_CHECKING

from vllm_webgpu.models.base import _gemv_wg, _vec4_wg
from vllm_webgpu.models.mixtral import MixtralWebGPUModel
from vllm_webgpu.webgpu.buffer import WebGPUBuffer

if TYPE_CHECKING:
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
        self._clamp_extra: dict = {"CLAMP_MAX": self._swiglu_limit} if self._swiglu_limit > 0 else {}
        self._moe_inter: int = (
            getattr(model_config, "moe_intermediate_size", None)
            or model_config.intermediate_size
        )

        # _prefill_batch_forward bypasses _attn_block entirely, so it cannot
        # apply attention biases or per-layer layer_types context overrides.
        # Force the sequential fallback path whenever either feature is active.
        # Use a dedicated flag rather than mutating _sw: setting _sw = -1 poisons
        # _effective_ctx_len (min(ctx_len, -1) == -1), which then passes -1 as
        # CTX_LEN to flash_attn_decode and wraps to max-u32 on the GPU side.
        self._force_sequential_prefill: bool = bool(self._attn_bias or self._layer_types)
        super().__init__(model_config, wgpu_device, pipeline_cache, block_size=block_size)

    def _init_scratch_buffers(self, max_ctx: int) -> None:
        """Extend parent scratch buffers with dedicated Q/K/V bias temporaries.

        The parent class reuses gate_buf/up_buf/ffn_act (each sized intermediate_size)
        as bias-addition destinations for Q, K, and V projections. For architectures
        where num_q_heads * head_dim > intermediate_size or
        num_kv_heads * head_dim > intermediate_size, those writes overflow.
        Dedicated buffers sized at the correct Q and KV dimensions avoid the overflow.
        """
        super()._init_scratch_buffers(max_ctx)
        if self._attn_bias:
            dev = self.wgpu_device.wgpu_device
            Q  = self.num_q_heads  * self.head_dim
            KV = self.num_kv_heads * self.head_dim
            self._sc["q_bias_tmp"] = WebGPUBuffer.empty(dev, Q  * 2)   # [Q]  f16
            self._sc["k_bias_tmp"] = WebGPUBuffer.empty(dev, KV * 2)   # [KV] f16
            self._sc["v_bias_tmp"] = WebGPUBuffer.empty(dev, KV * 2)   # [KV] f16

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
        eff = (ctx_len if (self._layer_types and layer_idx < len(self._layer_types)
               and self._layer_types[layer_idx] == "full_attention")
               else self._effective_ctx_len(ctx_len))

        sc = self._sc
        hidden = self.hidden_size
        p = f"model.layers.{layer_idx}"
        q_dim = self.num_q_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        k_cache, v_cache = self.kv_pool[layer_idx]

        # QKV projections (always separate; GPT-OSS has no q_norm/k_norm).
        _q_src, _k_src, _v_src = self._qkv_proj(normed_x, layer_idx)

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
                self._dispatch("add", [sc["q_buf"], q_bias, sc["q_bias_tmp"]],
                               {"N": q_dim}, _vec4_wg(q_dim))
                _q_src = sc["q_bias_tmp"]
            if k_bias is not None:
                self._dispatch("add", [sc["k_buf"], k_bias, sc["k_bias_tmp"]],
                               {"N": kv_dim}, _vec4_wg(kv_dim))
                _k_src = sc["k_bias_tmp"]
            if v_bias is not None:
                self._dispatch("add", [sc["v_buf"], v_bias, sc["v_bias_tmp"]],
                               {"N": kv_dim}, _vec4_wg(kv_dim))
                _v_src = sc["v_bias_tmp"]

        _rope_consts = self._rope_consts
        _freq_buf = self._rope_freq_buf

        # GPT-OSS has no q_norm/k_norm weights; dispatch rope directly.
        self._dispatch(
            "rope",
            [_q_src, pos_buf, sc["q_rope"], _freq_buf],
            {**_rope_consts, "NUM_HEADS": self.num_q_heads},
            (num_tokens, self.num_q_heads, 1),
        )
        self._dispatch(
            "rope",
            [_k_src, pos_buf, sc["k_rope"], _freq_buf],
            {**_rope_consts, "NUM_HEADS": self.num_kv_heads},
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
             "CTX_LEN": eff},
            (self.num_q_heads, 1, 1),
        )

        # Output projection.
        w_key = f"{p}.self_attn.o_proj.weight"
        uq = self._uq_for_key(w_key)
        qi = self._quant_extra(f"{p}.self_attn.o_proj", uq)
        self._dispatch(
            "matmul_quant",
            [sc["attn_out"], self.weights[w_key],
             self._scales_buf(w_key, uq, self._dummy_scales_buf), sc["o_proj_out"]],
            {"K": q_dim, "N": hidden, "USE_QUANT": uq,
             **qi},
            _gemv_wg(hidden),
        )

        # O-projection bias: write to sc['ffn_out'], which is free at this point.
        # _attn_block runs before _ffn_dispatch writes ffn_out and before
        # add_rms_norm writes ffn_normed, so there is no aliasing conflict.
        # The add reads sc['o_proj_out'] (binding 0) and writes sc['ffn_out']
        # (binding 2); the next add_rms_norm reads ffn_out (binding 1) and
        # writes ffn_normed (binding 4).
        _o_proj_src = sc["o_proj_out"]
        if self._attn_bias:
            o_bias = self.weights.get(f"{p}.self_attn.o_proj.bias")
            if o_bias is not None:
                self._dispatch(
                    "add",
                    [sc["o_proj_out"], o_bias, sc["ffn_out"]],
                    {"N": hidden},
                    _vec4_wg(hidden),
                )
                _o_proj_src = sc["ffn_out"]

        return _o_proj_src

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
