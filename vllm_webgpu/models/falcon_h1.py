from __future__ import annotations
import math
from itertools import chain
from typing import TYPE_CHECKING

import numpy as np

from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator, is_conv_state_dim_first
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm_webgpu.models.base import _vals_per_thread, _vec4_wg, _rows_wg, _H_NAMES
from vllm_webgpu.models.nemotron_h import NemotronHWebGPUModel, _a_log_transform
from vllm_webgpu.webgpu.buffer import WebGPUBuffer, _ELEM_BYTES

if TYPE_CHECKING:
    from vllm_webgpu.webgpu.device import WebGPUDevice
    from vllm_webgpu.webgpu.pipeline import PipelineCache

logger = init_logger(__name__)


def _falconh1_mamba_int(cfg) -> int:
    """Compute FalconH1's Mamba intermediate size.

    Mirrors FalconH1SSMDecoderLayer.__init__:
        d_ssm = mamba_d_ssm if mamba_d_ssm is not None else int(mamba_expand * hidden_size)
    """
    d_ssm = getattr(cfg, "mamba_d_ssm", None)
    return d_ssm if d_ssm is not None else int(cfg.mamba_expand * cfg.hidden_size)


class FalconH1WebGPUModel(NemotronHWebGPUModel):
    """FalconH1 parallel-hybrid WebGPU backend.

    Architecture: every layer runs attention AND Mamba-2 SSM in parallel on the
    same pre-normed input. Their outputs are summed (with multiplier scaling) and
    added to the residual, then a feed-forward MLP follows.

    Per-layer structure:
      normed   = rms_norm(x, input_layernorm)
      attn_out = attention(normed * attn_in_mult) * attn_out_mult
      ssm_out  = mamba(normed * ssm_in_mult)      * ssm_out_mult
      combined = x + attn_out + ssm_out
      ffn_nrm  = rms_norm(combined, pre_ff_layernorm)
      ffn_out  = mlp(ffn_nrm)
      x_next   = combined + ffn_out

    Multipliers != 1.0 require a scalar-multiplication shader not present in
    this backend. Construction raises NotImplementedError for any non-unit value.

    Weight keys (FalconH1 HF checkpoint uses model. prefix, no backbone. mapper):
    - Attention:  model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
    - Mamba:      model.layers.{i}.mamba.{in_proj,out_proj,conv1d,A_log,D,dt_bias,norm}
    - FFN:        model.layers.{i}.feed_forward.{gate,up,down}_proj.weight
    - Pre-norms:  model.layers.{i}.{input_layernorm,pre_ff_layernorm}.weight
    - Final norm: model.final_layernorm.weight

    Config mapping from FalconH1Config to NemotronH-compatible names:
    - mamba_n_heads  -> mamba_num_heads
    - mamba_d_head   -> mamba_head_dim
    - mamba_n_groups -> n_groups
    - mamba_d_state  -> ssm_state_size
    - mamba_d_conv   -> conv_kernel
    """

    def __init__(
        self,
        model_config,
        wgpu_device: "WebGPUDevice",
        pipeline_cache: "PipelineCache",
        block_size: int = 16,
    ) -> None:
        # Guard: scalar multipliers must all be 1.0.
        for attr in ("attention_in_multiplier", "attention_out_multiplier",
                     "ssm_in_multiplier", "ssm_out_multiplier",
                     "embedding_multiplier", "key_multiplier"):
            val = float(getattr(model_config, attr, 1.0))
            if val != 1.0:
                raise NotImplementedError(
                    f"FalconH1WebGPUModel: {attr}={val!r} != 1.0. "
                    "Non-unit scalar multipliers require a vec_scale shader that is "
                    "not yet implemented. Only multiplier=1.0 is supported."
                )
        # Guard: mlp_multipliers = [gate_multiplier, down_multiplier] must be [1.0, 1.0].
        # Non-unit values are silently dropped by the FFN dispatch (no vec_scale shader).
        mlp_mults = list(getattr(model_config, "mlp_multipliers", (1.0, 1.0)))
        if any(float(m) != 1.0 for m in mlp_mults):
            raise NotImplementedError(
                f"FalconH1WebGPUModel: mlp_multipliers={mlp_mults!r} != [1.0, 1.0]. "
                "Non-unit FFN scale factors (gate_multiplier, down_multiplier) are not "
                "yet implemented. Only [1.0, 1.0] is supported."
            )

        # Map FalconH1Config field names to NemotronH-compatible attribute names.
        model_config.mamba_num_heads  = model_config.mamba_n_heads
        model_config.mamba_head_dim   = model_config.mamba_d_head
        model_config.n_groups         = model_config.mamba_n_groups
        model_config.ssm_state_size   = model_config.mamba_d_state
        model_config.conv_kernel      = model_config.mamba_d_conv
        if not hasattr(model_config, "intermediate_size"):
            model_config.intermediate_size = _falconh1_mamba_int(model_config)

        # NemotronH guards: provide required but FalconH1-named attrs.
        model_config.mlp_bias         = getattr(model_config, "mlp_bias",       False)
        model_config.use_bias         = getattr(model_config, "mamba_proj_bias", False)
        model_config.mlp_hidden_act   = "relu2"   # placeholder; no "mlp" type layers
        model_config.mamba_hidden_act = getattr(model_config, "hidden_act", "silu")

        # All layers are "attention" for KV cache spec (every layer has attention).
        # This avoids the mlp_hidden_act guard (no "mlp" type layers) and ensures
        # every layer is included in the KV pool.
        num_layers = model_config.num_hidden_layers
        model_config.layers_block_type = ["attention"] * num_layers

        super().__init__(model_config, wgpu_device, pipeline_cache, block_size)

        # RoPE constants for the attention branch.
        _rp = getattr(model_config, "rope_parameters", {}) or {}
        self.rope_theta: float = float(_rp.get("rope_theta", 10000.0))
        self._rope_consts: dict = {
            "HEAD_DIM":     self.head_dim,
            "LN_ROPE_BASE": math.log(self.rope_theta),
            "USE_FREQ_BUF": 0,
            "YARN_MSCALE":  1.0,
        }

        # FalconH1 intermediate size for the feed-forward MLP.
        self._ffn_inter: int = model_config.intermediate_size

    # ── Scratch buffer allocation ─────────────────────────────────────────────

    def _init_scratch_buffers(self) -> None:
        """Extend NemotronH scratch buffers with FalconH1-specific additions.

        Added buffers:
        - pos:           [1] u32 position for the current decode token
        - attn_proj_out: [hidden_size] f16 attention output after o_proj
        - ffn_normed:    [hidden_size] f16 pre-FFN normed input
        - gate_buf:      [intermediate_size] f16 gate projection output (GPTQ path)
        - up_buf:        [intermediate_size] f16 up projection output (GPTQ path);
                         overrides the 4-byte placeholder from NemotronH (all layers
                         are "attention"-typed so _layer_int_size is all zeros there)
        - ffn_proj_out:  [hidden_size] f16 down projection output; dedicated buffer
                         so the size is always H regardless of mamba_int
        """
        super()._init_scratch_buffers()
        H     = self.hidden_size
        I_max = max(self._layer_int_size) if any(self._layer_int_size) else H
        inter = max(I_max, getattr(self.model_config, "intermediate_size", H))

        # Position buffer for RoPE in the attention branch.
        self._pre["pos"] = self._make_buf(4)

        # Attention output scratch (avoids aliasing with sc["mixer_out"] from mamba).
        self._sc["attn_proj_out"] = self._make_buf(H * 2)
        # Pre-FFN norm output.
        self._sc["ffn_normed"]    = self._make_buf(H * 2)
        # Gate, up, and FFN activation buffers for the FFN path.
        # NemotronH allocates these as max(_layer_int_size)*2 bytes, which is
        # 4 bytes for FalconH1 (all "attention" layers → int_size=0 for all).
        # Override all three to the correct FFN intermediate size.
        self._sc["gate_buf"]      = self._make_buf(inter * 2)
        self._sc["up_buf"]        = self._make_buf(inter * 2)
        self._sc["ffn_act"]       = self._make_buf(inter * 2)
        # Dedicated FFN down-proj output buffer. Using sc["mamba_norm_out"] as a
        # temporary would overflow when mamba_int < hidden_size (e.g. test configs
        # or checkpoints with small SSM expand ratios).
        self._sc["ffn_proj_out"]  = self._make_buf(H * 2)

    # ── Mamba state allocation (all layers) ──────────────────────────────────

    def _init_mamba_states(self, num_spec: int = 0) -> None:
        """Allocate Mamba conv and SSM state buffers for ALL FalconH1 layers.

        NemotronH's _init_mamba_states allocates only for "mamba"-typed layers.
        FalconH1 has a parallel Mamba branch in every layer, so override to
        allocate for all num_hidden_layers instead.
        """
        if num_spec != 0:
            raise NotImplementedError(
                "speculative decoding not yet supported (mamba2_causal_conv shader)"
            )
        if is_conv_state_dim_first():
            raise NotImplementedError(
                "VLLM_SSM_CONV_STATE_LAYOUT=DS is not supported on the WebGPU path"
            )
        conv_shape, ssm_shape = MambaStateShapeCalculator.mamba2_state_shape(
            tp_world_size=1,
            intermediate_size=self.mamba_int,
            n_groups=self.n_groups,
            num_heads=self.mamba_num_heads,
            head_dim=self.mamba_head_dim,
            state_size=self.ssm_state_size,
            conv_kernel=self.conv_kernel,
            num_spec=num_spec,
        )
        conv_bytes = math.prod(conv_shape) * _ELEM_BYTES["f16"]
        ssm_bytes  = math.prod(ssm_shape)  * _ELEM_BYTES["f32"]
        for i in range(self.num_layers):
            self._conv_states[i] = self._make_buf(conv_bytes)
            self._ssm_states[i]  = self._make_buf(ssm_bytes)

    # ── Recurrent state management ────────────────────────────────────────────

    def reset_recurrent_states(self) -> None:
        for buf in chain(self._conv_states.values(), self._ssm_states.values()):
            self._zero_write(buf)

    def save_recurrent_states(self) -> dict:
        bufs = list(chain(
            (("conv", i, b) for i, b in self._conv_states.items()),
            (("ssm",  i, b) for i, b in self._ssm_states.items()),
        ))
        return self._readback_recurrent_states(bufs)

    def restore_recurrent_states(self, states: dict) -> None:
        dev = self.wgpu_device.wgpu_device
        for i, data in states.get("conv", {}).items():
            dev.queue.write_buffer(self._conv_states[i].buf, 0, data)
        for i, data in states.get("ssm", {}).items():
            dev.queue.write_buffer(self._ssm_states[i].buf, 0, data)

    # ── Weight loading ────────────────────────────────────────────────────────

    def load_weights(self, path: str, *, num_spec: int = 0) -> None:
        """Load FalconH1 weights with A_log -> -exp(A) transform and QKV packing.

        FalconH1 uses the model. prefix (no backbone. remapping). The HF mapper
        stacking (.q_proj -> .qkv_proj etc.) is not applied; raw HF key names are
        used and packed manually in _pack_falconh1_attn_weights.
        """
        # Upload Mamba SSM parameters as f32 (same rationale as NemotronH).
        f32_keys = frozenset(
            f"model.layers.{i}.mamba.{wk}"
            for i in range(self.num_layers)
            for wk in ("D", "dt_bias", "A_log")
        )
        # Register A_log -> -exp(A) transforms before loading.
        for i in range(self.num_layers):
            self._weight_transforms[f"model.layers.{i}.mamba.A_log"] = _a_log_transform

        from vllm_webgpu.models.base import BaseWebGPUModel
        BaseWebGPUModel.load_weights(self, path, f32_keys=f32_keys)

        # Rename A_log -> A (transform already applied, just the key needs renaming).
        for i in range(self.num_layers):
            old_k = f"model.layers.{i}.mamba.A_log"
            new_k = f"model.layers.{i}.mamba.A"
            if old_k in self.weights:
                self.weights[new_k] = self.weights.pop(old_k)

        self._pack_falconh1_attn_weights()
        self._init_mamba_states(num_spec)

        # Cache per-layer pre-norm weight buffers (input_layernorm in FalconH1).
        self._layer_norm_weights = [
            self.weights[f"model.layers.{i}.input_layernorm.weight"]
            for i in range(self.num_layers)
        ]
        self._layer0_norm_w = self._layer_norm_weights[0]

        logger.info(
            "FalconH1: loaded %d weight tensors (%d parallel-hybrid layers)",
            len(self.weights), self.num_layers,
        )

    def _pack_falconh1_attn_weights(self) -> None:
        """Pack separate q/k/v proj tensors into a single qkv_proj buffer per layer.

        FalconH1 HF checkpoint stores q_proj, k_proj, v_proj as separate tensors
        under self_attn.*. Concatenate them row-wise into qkv_proj.weight.
        """
        dev = self.wgpu_device.wgpu_device
        for i in range(self.num_layers):
            p = f"model.layers.{i}.self_attn"
            q_key = f"{p}.q_proj.weight"
            k_key = f"{p}.k_proj.weight"
            v_key = f"{p}.v_proj.weight"
            if q_key not in self.weights:
                continue
            q_nb, k_nb, v_nb = (self.weights[k].nbytes for k in (q_key, k_key, v_key))
            total_nb = q_nb + k_nb + v_nb
            src_dtype = self.weights[q_key].dtype

            qkv_buf = self._make_buf(total_nb)
            qkv_buf.dtype = src_dtype
            enc = dev.create_command_encoder()
            enc.copy_buffer_to_buffer(self.weights[q_key].buf, 0, qkv_buf.buf, 0,         q_nb)
            enc.copy_buffer_to_buffer(self.weights[k_key].buf, 0, qkv_buf.buf, q_nb,      k_nb)
            enc.copy_buffer_to_buffer(self.weights[v_key].buf, 0, qkv_buf.buf, q_nb+k_nb, v_nb)
            dev.queue.submit([enc.finish()])

            self.weights[f"{p}.qkv_proj.weight"] = qkv_buf
            del self.weights[q_key], self.weights[k_key], self.weights[v_key]

    # ── Per-layer branch dispatches ───────────────────────────────────────────

    def _mamba_branch(self, layer_idx: int, normed_x: "WebGPUBuffer") -> "WebGPUBuffer":
        """Mamba-2 SSM branch for FalconH1 (weight prefix: model.layers.{i}.mamba).

        Returns sc["mixer_out"] (the SSM output).
        """
        sc  = self._sc
        p   = f"model.layers.{layer_idx}.mamba"
        H   = self.hidden_size
        MI  = self.mamba_int
        CD  = self.conv_dim
        MNH = self.mamba_num_heads
        MHD = self.mamba_head_dim
        NS  = self.ssm_state_size
        NG  = self.n_groups

        in_w = f"{p}.in_proj.weight"
        uq   = self._uq_for_key(in_w)
        self._dispatch("matmul_quant",
                       [normed_x, self.weights[in_w],
                        self._scales_buf(in_w, uq, self._dummy_buf), sc["mamba_inproj"]],
                       {"K": H, "N": self.in_proj_dim, "USE_QUANT": uq,
                        **self._quant_extra(f"{p}.in_proj", uq)},
                       (self.in_proj_dim, 1, 1))

        enc = self._active_encoder
        enc.copy_buffer_to_buffer(sc["mamba_inproj"].buf, 0,           sc["mamba_gate"].buf,    0, MI * 2)
        enc.copy_buffer_to_buffer(sc["mamba_inproj"].buf, MI * 2,      sc["mamba_conv_in"].buf, 0, CD * 2)
        enc.copy_buffer_to_buffer(sc["mamba_inproj"].buf, (MI+CD) * 2, sc["mamba_dt"].buf,      0, MNH * 2)

        conv_w   = f"{p}.conv1d.weight"
        conv_b   = f"{p}.conv1d.bias"
        bias_buf = self.weights.get(conv_b, self._dummy_buf)
        has_bias = int(bias_buf is not self._dummy_buf)
        self._dispatch("mamba2_causal_conv",
                       [sc["mamba_conv_in"], self.weights[conv_w], bias_buf,
                        self._conv_states[layer_idx], sc["mamba_conv_out"]],
                       {"CONV_DIM": CD, "KERNEL": self.conv_kernel, "WG_SIZE": 256, "HAS_BIAS": has_bias},
                       _rows_wg(CD))

        self._dispatch("mamba2_ssm_step",
                       [sc["mamba_conv_out"], sc["mamba_dt"],
                        self.weights[f"{p}.A"], self.weights[f"{p}.dt_bias"],
                        self.weights[f"{p}.D"],
                        self._ssm_states[layer_idx], sc["mamba_ssm_y"]],
                       {"NUM_HEADS": MNH, "HEAD_DIM": MHD, "STATE_SIZE": NS, "N_GROUPS": NG, "WG_SIZE": 256},
                       (MNH, 1, 1))

        self._dispatch("mamba2_norm_gate",
                       [sc["mamba_ssm_y"], sc["mamba_gate"],
                        self.weights[f"{p}.norm.weight"], sc["mamba_norm_out"]],
                       {"MAMBA_INT": MI, "N_GROUPS": NG, "WG_SIZE": 256},
                       (NG, 1, 1))

        out_w = f"{p}.out_proj.weight"
        uq2   = self._uq_for_key(out_w)
        self._dispatch("matmul_quant",
                       [sc["mamba_norm_out"], self.weights[out_w],
                        self._scales_buf(out_w, uq2, self._dummy_buf), sc["mixer_out"]],
                       {"K": MI, "N": H, "USE_QUANT": uq2,
                        **self._quant_extra(f"{p}.out_proj", uq2)},
                       (H, 1, 1))
        return sc["mixer_out"]

    def _attn_branch(
        self,
        layer_idx: int,
        normed_x: "WebGPUBuffer",
        pos_buf: "WebGPUBuffer",
        slot_map: "WebGPUBuffer",
        bt_buf: "WebGPUBuffer",
        ctx_len: int,
        num_tokens: int,
    ) -> "WebGPUBuffer":
        """Attention branch for FalconH1 (weight prefix: model.layers.{i}.self_attn).

        Uses the packed qkv_proj.weight. Applies standard RoPE to Q and K.
        Returns sc["attn_proj_out"] (the attention output after o_proj).
        """
        sc    = self._sc
        p     = f"model.layers.{layer_idx}.self_attn"
        H     = self.hidden_size
        q_dim = self._q_dim
        k_dim = self._k_dim
        scale = self.head_dim ** -0.5
        _freq = self._rope_freq_buf
        _rc   = self._rope_consts

        qkv_w     = f"{p}.qkv_proj.weight"
        total_qkv = q_dim + 2 * k_dim
        uq = self._uq_for_key(qkv_w)
        self._dispatch("matmul_quant",
                       [normed_x, self.weights[qkv_w],
                        self._scales_buf(qkv_w, uq, self._dummy_buf), sc["qkv_buf"]],
                       {"K": H, "N": total_qkv, "USE_QUANT": uq,
                        **self._quant_extra(f"{p}.qkv_proj", uq)},
                       (total_qkv, 1, 1))

        enc = self._active_encoder
        enc.copy_buffer_to_buffer(sc["qkv_buf"].buf, 0,            sc["q_buf"].buf, 0, q_dim * 2)
        enc.copy_buffer_to_buffer(sc["qkv_buf"].buf, q_dim * 2,    sc["k_buf"].buf, 0, k_dim * 2)
        enc.copy_buffer_to_buffer(sc["qkv_buf"].buf, (q_dim+k_dim)*2, sc["v_buf"].buf, 0, k_dim * 2)

        # Apply RoPE: write to q_rope / k_rope so q_buf / k_buf are preserved.
        self._dispatch("rope", [sc["q_buf"], pos_buf, sc["q_rope"], _freq],
                       {**_rc, "NUM_HEADS": self.num_q_heads, "INPUT_OFFSET": 0},
                       (num_tokens, self.num_q_heads, 1))
        self._dispatch("rope", [sc["k_buf"], pos_buf, sc["k_rope"], _freq],
                       {**_rc, "NUM_HEADS": self.num_kv_heads, "INPUT_OFFSET": 0},
                       (num_tokens, self.num_kv_heads, 1))

        k_cache, v_cache = self.kv_pool[layer_idx]
        if not self._replay_mode:
            self._dispatch("kv_cache_store_both",
                           [sc["k_rope"], k_cache, sc["v_buf"], v_cache, slot_map],
                           {"BLOCK_SIZE": self.block_size, "NUM_KV_HEADS": self.num_kv_heads,
                            "HEAD_DIM": self.head_dim, "V_IN_OFFSET": 0},
                           (num_tokens, self.num_kv_heads, 1))

        self._dispatch("flash_attn_decode",
                       [sc["q_rope"], k_cache, v_cache, bt_buf, sc["attn_out"]],
                       {"BLOCK_SIZE": self.block_size, "NUM_Q_HEADS": self.num_q_heads,
                        "NUM_KV_HEADS": self.num_kv_heads, "HEAD_DIM": self.head_dim,
                        "CTX_LEN": ctx_len, "SCALE": scale, "START_BLOCK": 0},
                       (self.num_q_heads, 1, 1))

        ow  = f"{p}.o_proj.weight"
        uq2 = self._uq_for_key(ow)
        # Write to sc["attn_proj_out"] to avoid aliasing with sc["mixer_out"] (mamba output).
        self._dispatch("matmul_quant",
                       [sc["attn_out"], self.weights[ow],
                        self._scales_buf(ow, uq2, self._dummy_buf), sc["attn_proj_out"]],
                       {"K": q_dim, "N": H, "USE_QUANT": uq2,
                        **self._quant_extra(f"{p}.o_proj", uq2)},
                       (H, 1, 1))
        return sc["attn_proj_out"]

    def _layer_dispatch(
        self,
        layer_idx: int,
        normed_x: "WebGPUBuffer | None",
        x_buf: "WebGPUBuffer",
        slot_map: "WebGPUBuffer",
        bt_buf: "WebGPUBuffer",
        ctx_len: int,
        num_tokens: int,
    ) -> "tuple[WebGPUBuffer | None, WebGPUBuffer]":
        """FalconH1 parallel-hybrid layer dispatch.

        normed_x is unused (FalconH1 computes the pre-norm internally).
        pos_buf is read from self._pre["pos"] which is set before the dispatch loop.
        """
        sc    = self._sc
        H     = self.hidden_size
        p     = f"model.layers.{layer_idx}"
        rms_c = self._rms_base
        add_n = num_tokens * H

        merged = sc[_H_NAMES[(self._hstate + 1) % 3]]
        out    = sc[_H_NAMES[(self._hstate + 2) % 3]]
        pos_buf = self._pre["pos"]

        with self._batched_dispatch(label=f"L{layer_idx:02d}"):
            # Pre-norm shared between both branches.
            if self._layer_norm_weights:
                pre_norm_w = self._layer_norm_weights[layer_idx]
            else:
                pre_norm_w = self.weights[f"{p}.input_layernorm.weight"]
            self._dispatch("rms_norm",
                           [x_buf, pre_norm_w, sc["normed"]],
                           rms_c, (num_tokens, 1, 1))
            normed = sc["normed"]

            # Parallel branches.
            ssm_out  = self._mamba_branch(layer_idx, normed)
            attn_out = self._attn_branch(layer_idx, normed, pos_buf, slot_map, bt_buf, ctx_len, num_tokens)

            # Parallel residual merge (all multipliers are 1.0, checked at init).
            # merged = x + ssm_out + attn_out
            # Use 'out' as a non-aliased intermediate for the first add. Binding
            # the same buffer to both the read (binding 0) and read_write (binding 2)
            # slots of the add shader is invalid per the WebGPU spec; 'out' is a
            # distinct h-buffer and is overwritten later by add_rms_norm / add.
            self._dispatch("add", [x_buf, ssm_out, out],    {"N": add_n}, _vec4_wg(add_n))
            self._dispatch("add", [out,   attn_out, merged], {"N": add_n}, _vec4_wg(add_n))

            # Feed-forward with pre-FFN norm.
            pre_ff_w = self.weights[f"{p}.pre_ff_layernorm.weight"]
            self._dispatch("rms_norm",
                           [merged, pre_ff_w, sc["ffn_normed"]],
                           rms_c, (num_tokens, 1, 1))

            inter = self._ffn_inter
            gw_k = f"{p}.feed_forward.gate_proj.weight"
            uw_k = f"{p}.feed_forward.up_proj.weight"
            uq_g = self._uq_for_key(gw_k)
            uq_u = self._uq_for_key(uw_k)
            if uq_g == 0 and uq_u == 0:
                self._dispatch("fused_gate_act",
                               [sc["ffn_normed"], self.weights[gw_k], self.weights[uw_k], sc["ffn_act"]],
                               {"K": H, "N": inter}, (inter, 1, 1))
            else:
                for out_b, w_k, uq2 in [(sc["gate_buf"], gw_k, uq_g), (sc["up_buf"], uw_k, uq_u)]:
                    qi2 = self._quant_extra(w_k.removesuffix(".weight"), uq2)
                    self._dispatch("matmul_quant",
                                   [sc["ffn_normed"], self.weights[w_k],
                                    self._scales_buf(w_k, uq2, self._dummy_buf), out_b],
                                   {"K": H, "N": inter, "USE_QUANT": uq2, **qi2},
                                   (inter, 1, 1))
                self._dispatch("gelu_mul", [sc["gate_buf"], sc["up_buf"], sc["ffn_act"]],
                               {"N": inter}, _vec4_wg(inter))

            dw_k = f"{p}.feed_forward.down_proj.weight"
            uq_d = self._uq_for_key(dw_k)
            ffn_proj_out = sc["ffn_proj_out"]  # dedicated H-element buffer; mamba_norm_out
            self._dispatch("matmul_quant",
                           [sc["ffn_act"], self.weights[dw_k],
                            self._scales_buf(dw_k, uq_d, self._dummy_buf), ffn_proj_out],
                           {"K": inter, "N": H, "USE_QUANT": uq_d,
                            **self._quant_extra(f"{p}.feed_forward.down_proj", uq_d)},
                           (H, 1, 1))

            # Final residual: out = merged + ffn_proj_out.
            # Fuse with next layer's pre-norm for non-last layers.
            if layer_idx < self.num_layers - 1:
                if self._layer_norm_weights:
                    next_norm_w = self._layer_norm_weights[layer_idx + 1]
                else:
                    next_norm_w = self.weights[f"model.layers.{layer_idx+1}.input_layernorm.weight"]
                self._dispatch("add_rms_norm",
                               [merged, ffn_proj_out, next_norm_w, out, sc["normed"]],
                               rms_c, (num_tokens, 1, 1))
                normed_out = sc["normed"]
            else:
                self._dispatch("add", [merged, ffn_proj_out, out],
                               {"N": add_n}, _vec4_wg(add_n))
                normed_out = None

        self._hstate = (self._hstate + 2) % 3
        return normed_out, out

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: np.ndarray,
        positions: np.ndarray,
        attn_metadata: object,
    ) -> np.ndarray:
        """FalconH1 decode/prefill forward pass."""
        num_tokens = len(input_ids)
        self._hstate = 0
        self._check_single_sequence(attn_metadata)

        if num_tokens > 1:
            return self._prefill_forward(input_ids, positions, attn_metadata, num_tokens)

        dev = self.wgpu_device.wgpu_device
        pre = self._pre
        ctx_len = int(attn_metadata.max_decode_seq_len)

        dev.queue.write_buffer(pre["ids"].buf,      0, input_ids.astype(np.uint32, copy=False).tobytes())
        dev.queue.write_buffer(pre["pos"].buf,      0, positions.astype(np.uint32, copy=False).tobytes())
        dev.queue.write_buffer(pre["slot_map"].buf, 0,
                               np.asarray(attn_metadata.slot_mapping, dtype=np.uint32).tobytes())
        dev.queue.write_buffer(pre["bt"].buf, 0, self._bt_arr(attn_metadata).tobytes())

        with self._batched_dispatch():
            self._dispatch("embedding_lookup",
                           [self.weights["model.embed_tokens.weight"], pre["ids"], pre["x"]],
                           {"HIDDEN_DIM": self.hidden_size}, (1, 1, 1))

            x_buf = pre["x"]
            for i in range(self.num_layers):
                _, x_buf = self._layer_dispatch(
                    i, None, x_buf, pre["slot_map"], pre["bt"], ctx_len, 1)

            self._run_final_norm_and_lm_head(x_buf, 1)

        return self._finalize_output()

    def _prefill_forward(
        self,
        input_ids: np.ndarray,
        positions: np.ndarray,
        attn_metadata: object,
        T: int,
    ) -> np.ndarray:
        """FalconH1 prefill: process T tokens one at a time."""
        dev = self.wgpu_device.wgpu_device
        pre = self._pre
        dev.queue.write_buffer(pre["bt"].buf, 0, self._bt_arr(attn_metadata).tobytes())
        slot_arr = np.asarray(attn_metadata.slot_mapping, dtype=np.uint32)

        for t in range(T):
            self._hstate = 0
            tok_ctx = int(positions[t]) + 1

            dev.queue.write_buffer(pre["ids"].buf,      0, input_ids[t:t+1].astype(np.uint32, copy=False).tobytes())
            dev.queue.write_buffer(pre["pos"].buf,      0, positions[t:t+1].astype(np.uint32, copy=False).tobytes())
            dev.queue.write_buffer(pre["slot_map"].buf, 0, slot_arr[t:t+1].tobytes())

            with self._batched_dispatch():
                self._dispatch("embedding_lookup",
                               [self.weights["model.embed_tokens.weight"], pre["ids"], pre["x"]],
                               {"HIDDEN_DIM": self.hidden_size}, (1, 1, 1))
                x_buf = pre["x"]
                for i in range(self.num_layers):
                    _, x_buf = self._layer_dispatch(
                        i, None, x_buf, pre["slot_map"], pre["bt"], tok_ctx, 1)
                if t == T - 1:
                    self._run_final_norm_and_lm_head(x_buf, 1)

        return self._finalize_output()

    def _run_final_norm_and_lm_head(self, x_buf: "WebGPUBuffer", num_tokens: int) -> None:
        """FalconH1 final norm: model.final_layernorm.weight (not model.norm_f.weight)."""
        pre   = self._pre
        vocab = self.vocab_size
        H     = self.hidden_size
        self._dispatch("rms_norm",
                       [x_buf, self.weights["model.final_layernorm.weight"], pre["norm_out"]],
                       self._rms_base, (num_tokens, 1, 1))
        lm_key = self._lm_head_key()
        uq = self._uq_for_key(lm_key)
        self._dispatch("matmul_quant",
                       [pre["norm_out"], self.weights[lm_key],
                        self._scales_buf(lm_key, uq, self._dummy_buf), pre["logits"]],
                       {"K": H, "N": vocab, "USE_QUANT": uq, "SPLIT_K": 0,
                        **self._quant_extra(lm_key.removesuffix(".weight"), uq)},
                       _rows_wg(vocab))
        if self._greedy_decode:
            self._dispatch("argmax_f16",
                           [pre["logits"], self._ensure_sample_buf()],
                           {"N": vocab}, (1, 1, 1))
            self._copy_sample_to_staging()

    def _finalize_output(self) -> np.ndarray:
        self._last_logit_buf = self._pre["logits"]
        self._last_vocab = self.vocab_size
        return self._finish_forward(self._greedy_decode)
