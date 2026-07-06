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
    """DiffusionGemma: Gemma4 backbone with MoE FFN + discrete diffusion.

    Architecture differences from Gemma4:
      - MoE FFN: 128 experts, top-8 active, moe_intermediate_size per expert
      - Mixed sliding/global attention (sliding_window=1024 for most layers)
      - Partial RoPE (partial_rotary_factor from rope_parameters)
      - Canvas-based block diffusion generation (canvas_length=256)
      - Self-conditioning: previous predictions fed back as probability-weighted embeddings

    Current implementation: single-token autoregressive decode using the Gemma4
    backbone. MoE routing uses CPU-side top-K selection + GPU expert FFNs.
    Bidirectional attention and diffusion sampling loop are future work.
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

        # Canvas / diffusion
        self.canvas_length: int = getattr(model_config, "canvas_length", 256)

        if self.is_moe:
            logger.info("DiffusionGemma: %d experts, top-%d active, moe_inter=%d",
                        self.num_experts, self.top_k_experts, self.moe_intermediate_size)

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
        """Override to add MoE FFN for layers that have expert routing."""
        p = f"model.layers.{layer_idx}"

        # Check if this layer uses MoE FFN (has a router weight)
        if self.is_moe and f"{p}.mlp.router.weight" in self.weights:
            return self._transformer_layer_moe(
                layer_idx, x_buf, pos_buf, slot_map, bt_buf, ctx_len, num_tokens)

        # Standard Gemma4 layer (global attention layers or non-MoE)
        return super()._transformer_layer(
            layer_idx, x_buf, pos_buf, slot_map, bt_buf, ctx_len, num_tokens)

    def _transformer_layer_moe(
        self,
        layer_idx: int,
        x_buf: "WebGPUBuffer",
        pos_buf: "WebGPUBuffer",
        slot_map: "WebGPUBuffer",
        bt_buf: "WebGPUBuffer",
        ctx_len: int,
        num_tokens: int,
    ) -> "WebGPUBuffer":
        """MoE transformer layer: attention + router + top-K expert FFNs.

        Expert selection uses CPU readback (router logits are tiny: num_experts floats).
        Expert FFNs run on GPU using existing matmul_quant infrastructure.
        """
        import wgpu as wgpu_lib
        from vllm_webgpu.webgpu.buffer import WebGPUBuffer

        dev = self.wgpu_device.wgpu_device
        sc = self._sc
        lp = self._lp[layer_idx]
        hidden = self.hidden_size
        head_dim = lp["head_dim"]
        q_dim = lp["q_dim"]
        kv_dim = lp["kv_dim"]
        num_kv_heads = lp["num_kv_heads"]
        has_v = lp["has_v_proj"]
        inter = self.moe_intermediate_size
        ln_rope = math.log(self.rope_theta)
        p = f"model.layers.{layer_idx}"

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
        out = sc[h_names[(self._hstate + 2) % 3]]
        add_n = num_tokens * hidden
        _vpt = min((hidden + 255) // 256, 16) if hidden <= 4096 else 0
        _rms_consts = {"HIDDEN_DIM": hidden, "VALS_PER_THREAD": _vpt,
                       "GEMMA_NORM": self._gemma_norm_const}

        k_cache, v_cache = self.kv_pool[layer_idx]

        with self._batched_dispatch(label=f"L{layer_idx:02d}"):
            # Attention sublayer (same as Gemma4)
            self._dispatch("rms_norm_f32in",
                           [x_buf, self.weights[f"{p}.input_layernorm.weight"], sc["normed"]],
                           _rms_consts, (num_tokens, 1, 1))

            qw = f"{p}.self_attn.q_proj.weight"
            self._dispatch("matmul_quant",
                           [sc["normed"], self.weights[qw],
                            self.weights.get(f"{p}.self_attn.q_proj.scales", sc["normed"]),
                            sc["q_buf"]],
                           {"K": hidden, "N": q_dim, "USE_QUANT": _uq(qw), "SPLIT_K": 1},
                           (q_dim, 1, 1))

            kw = f"{p}.self_attn.k_proj.weight"
            self._dispatch("matmul_quant",
                           [sc["normed"], self.weights[kw],
                            self.weights.get(f"{p}.self_attn.k_proj.scales", sc["normed"]),
                            sc["k_buf"]],
                           {"K": hidden, "N": kv_dim, "USE_QUANT": _uq(kw), "SPLIT_K": 1},
                           (kv_dim, 1, 1))

            if has_v:
                vw = f"{p}.self_attn.v_proj.weight"
                self._dispatch("matmul_quant",
                               [sc["normed"], self.weights[vw],
                                self.weights.get(f"{p}.self_attn.v_proj.scales", sc["normed"]),
                                sc["v_buf"]],
                               {"K": hidden, "N": kv_dim, "USE_QUANT": _uq(vw), "SPLIT_K": 1},
                               (kv_dim, 1, 1))
                v_src = sc["v_buf"]
            else:
                v_src = sc["k_buf"]

            for src, dst, n_heads, w_key in [
                (sc["q_buf"], sc["q_rope"], self.num_q_heads,
                 f"{p}.self_attn.q_norm.weight"),
                (sc["k_buf"], sc["k_rope"], num_kv_heads,
                 f"{p}.self_attn.k_norm.weight"),
            ]:
                norm_w = self.weights.get(w_key)
                if norm_w is not None:
                    self._dispatch("fused_per_head_norm_rope",
                                   [src, norm_w, pos_buf, dst],
                                   {"HEAD_DIM": head_dim, "NUM_HEADS": n_heads,
                                    "ROPE_BASE": float(self.rope_theta),
                                    "LN_ROPE_BASE": ln_rope, "HAS_WEIGHT": 1,
                                    "GEMMA_NORM": self._gemma_norm_const},
                                   (n_heads, num_tokens, 1))
                else:
                    self._dispatch("rope", [src, pos_buf, dst],
                                   {"HEAD_DIM": head_dim, "NUM_HEADS": n_heads,
                                    "LN_ROPE_BASE": ln_rope},
                                   (num_tokens, n_heads, 1))

            if self._apply_v_norm:
                self._dispatch("per_head_rms_norm_no_weight", [v_src, sc["v_normed"]],
                               {"HEAD_DIM": head_dim, "NUM_HEADS": num_kv_heads,
                                "WG_SIZE": min(head_dim, 128)},
                               (num_kv_heads, num_tokens, 1), shader_subdir="gemma")
                v_to_cache = sc["v_normed"]
            else:
                v_to_cache = v_src

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
                            self.weights.get(f"{p}.self_attn.o_proj.scales", sc["attn_out"]),
                            sc["o_proj_out"]],
                           {"K": q_dim, "N": hidden, "USE_QUANT": _uq(ow), "SPLIT_K": 1},
                           (hidden, 1, 1))

            post_attn_w = self.weights.get(f"{p}.post_attention_layernorm.weight")
            if post_attn_w is not None:
                self._dispatch("rms_norm", [sc["o_proj_out"], post_attn_w, sc["ffn_normed"]],
                               _rms_consts, (num_tokens, 1, 1))
                self._dispatch("add_f32", [x_buf, sc["ffn_normed"], residual],
                               {"N": add_n, "SCALE": 1.0},
                               ((add_n // 4 + 255) // 256, 1, 1))
            else:
                self._dispatch("add_f32", [x_buf, sc["o_proj_out"], residual],
                               {"N": add_n, "SCALE": 1.0},
                               ((add_n // 4 + 255) // 256, 1, 1))

            # MoE FFN sublayer
            pre_ffn_w = self.weights.get(f"{p}.pre_feedforward_layernorm.weight")
            if pre_ffn_w is not None:
                self._dispatch("rms_norm_f32in", [residual, pre_ffn_w, sc["normed"]],
                               _rms_consts, (num_tokens, 1, 1))
                ffn_normed = sc["normed"]
            else:
                ffn_normed = residual

            # Router: normed → logits[num_experts]
            router_w = self.weights.get(f"{p}.mlp.router.weight")

        # CPU readback for router logits (tiny: num_experts × 4 bytes)
        if router_w is not None:
            import wgpu as wgpu_lib
            rw_usage = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST
            router_logits_buf = WebGPUBuffer.empty(dev, self.num_experts * 2, usage=rw_usage)
            with self._batched_dispatch(label=f"L{layer_idx:02d}R"):
                self._dispatch("matmul_quant",
                               [ffn_normed, router_w,
                                self.weights.get(f"{p}.mlp.router.scales", ffn_normed),
                                router_logits_buf],
                               {"K": hidden, "N": self.num_experts, "USE_QUANT": 0,
                                "SPLIT_K": 0},
                               ((self.num_experts + 255) // 256, 1, 1))

            router_logits = router_logits_buf.to_numpy().view(np.float16).astype(np.float32)
            top_k_idx = np.argsort(router_logits)[-self.top_k_experts:]
            router_weights = router_logits[top_k_idx]
            router_weights = np.exp(router_weights - router_weights.max())
            router_weights = router_weights / router_weights.sum()

            # Run top-K expert FFNs and accumulate weighted outputs
            # sc["ffn_out"] accumulates the combined expert output
            expert_out_acc = WebGPUBuffer.empty(dev, num_tokens * hidden * 2, usage=rw_usage)
            gelu_n = num_tokens * inter

            first_expert = True
            for eid, ew in zip(top_k_idx, router_weights):
                ep = f"{p}.mlp.experts.{eid}"
                gw = self.weights.get(f"{ep}.gate_proj.weight")
                uw = self.weights.get(f"{ep}.up_proj.weight")
                dw = self.weights.get(f"{ep}.down_proj.weight")
                if gw is None or uw is None or dw is None:
                    logger.warning("Missing expert %d weights for layer %d", eid, layer_idx)
                    continue

                with self._batched_dispatch(label=f"L{layer_idx:02d}E{eid}"):
                    self._dispatch("matmul_quant",
                                   [ffn_normed, gw, self.weights.get(f"{ep}.gate_proj.scales", gw),
                                    sc["gate_buf"]],
                                   {"K": hidden, "N": inter, "USE_QUANT": _uq(f"{ep}.gate_proj.weight"),
                                    "SPLIT_K": 1},
                                   (inter, 1, 1))
                    self._dispatch("matmul_quant",
                                   [ffn_normed, uw, self.weights.get(f"{ep}.up_proj.scales", uw),
                                    sc["up_buf"]],
                                   {"K": hidden, "N": inter, "USE_QUANT": _uq(f"{ep}.up_proj.weight"),
                                    "SPLIT_K": 1},
                                   (inter, 1, 1))
                    self._dispatch("gelu_mul", [sc["gate_buf"], sc["up_buf"], sc["ffn_act"]],
                                   {"N": gelu_n}, ((gelu_n // 4 + 255) // 256, 1, 1),
                                   shader_subdir="gemma")
                    self._dispatch("matmul_quant",
                                   [sc["ffn_act"], dw, self.weights.get(f"{ep}.down_proj.scales", dw),
                                    sc["ffn_out"]],
                                   {"K": inter, "N": hidden, "USE_QUANT": _uq(f"{ep}.down_proj.weight"),
                                    "SPLIT_K": 1},
                                   (hidden, 1, 1))
                    # Accumulate: expert_out_acc += ew * ffn_out
                    if first_expert:
                        self._dispatch("add", [sc["ffn_out"], sc["ffn_out"], expert_out_acc],
                                       {"N": add_n, "SCALE": float(ew)},
                                       ((add_n // 4 + 255) // 256, 1, 1))
                        first_expert = False
                    else:
                        self._dispatch("add", [expert_out_acc, sc["ffn_out"], expert_out_acc],
                                       {"N": add_n, "SCALE": float(ew)},
                                       ((add_n // 4 + 255) // 256, 1, 1))
        else:
            expert_out_acc = sc["ffn_out"]

        with self._batched_dispatch(label=f"L{layer_idx:02d}F"):
            post_ffw_w = self.weights.get(f"{p}.post_feedforward_layernorm.weight")
            if post_ffw_w is not None:
                self._dispatch("rms_norm", [expert_out_acc, post_ffw_w, sc["o_proj_out"]],
                               _rms_consts, (num_tokens, 1, 1))
                self._dispatch("add_f32", [residual, sc["o_proj_out"], out],
                               {"N": add_n, "SCALE": 1.0},
                               ((add_n // 4 + 255) // 256, 1, 1))
            else:
                self._dispatch("add_f32", [residual, expert_out_acc, out],
                               {"N": add_n, "SCALE": 1.0},
                               ((add_n // 4 + 255) // 256, 1, 1))

        self._hstate = (self._hstate + 2) % 3
        return out
