from __future__ import annotations
import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM
from vllm_webgpu.models.base import BaseWebGPUModel
from vllm_webgpu.models.llama import _gemv_wg

if TYPE_CHECKING:
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.device import WebGPUDevice
    from vllm_webgpu.webgpu.pipeline import PipelineCache

logger = logging.getLogger(__name__)


class NemotronHWebGPUModel(BaseWebGPUModel):
    """
    Nemotron-H hybrid Mamba-2 SSM / Attention model (WebGPU decode backend).

    Architecture: NemotronHForCausalLM from vLLM / HuggingFace.
    Each layer is one of:
      M = Mamba-2 SSM  (in_proj -> conv -> SSM -> gated_norm -> out_proj)
      * = Attention     (qkv_proj -> rope -> flash_attn -> o_proj)
      - = MLP-only      (up_proj -> relu^2 -> down_proj)
      E = MoE (not implemented; raises at runtime)

    The layer sequence is encoded in config.hybrid_override_pattern as a
    string of the characters above, one per layer.

    Weight key remapping: HuggingFace checkpoints use 'backbone.' prefix and
    store all mixer weights under '.mixer.' regardless of block type. Keys are
    remapped to the vLLM-canonical 'model.' prefix; no per-type renaming is needed.
    """

    logit_returns_token_id: bool = True

    def __init__(
        self,
        model_config,
        wgpu_device: "WebGPUDevice",
        pipeline_cache: "PipelineCache",
    ) -> None:
        super().__init__(model_config, wgpu_device, pipeline_cache)

        self.hidden_size: int = model_config.hidden_size
        self.num_layers: int = model_config.num_hidden_layers
        self.vocab_size: int = model_config.vocab_size

        # Attention layer parameters
        self.num_q_heads: int = model_config.num_attention_heads
        self.num_kv_heads: int = model_config.num_key_value_heads
        self.head_dim: int = getattr(
            model_config, "head_dim", self.hidden_size // self.num_q_heads
        )
        self.rope_theta: float = getattr(model_config, "rope_theta", 10000.0)
        self._ln_rope_theta: float = math.log(self.rope_theta)

        # MLP parameters (used in '-' layers).
        # intermediate_size may be a list for heterogeneous (puzzle) configs;
        # store the per-layer list and use the max for scratch buffer sizing.
        _raw_int = model_config.intermediate_size
        if isinstance(_raw_int, list):
            self._intermediate_sizes: list[int] | None = _raw_int
            self.intermediate_size: int = max(_raw_int)
        else:
            self._intermediate_sizes = None
            self.intermediate_size: int = _raw_int

        # Mamba-2 parameters
        self.mamba_num_heads: int = model_config.mamba_num_heads
        self.mamba_head_dim: int = model_config.mamba_head_dim
        # mamba_int: the Mamba "intermediate size" = num_heads * head_dim
        self.mamba_int: int = self.mamba_num_heads * self.mamba_head_dim
        self.n_groups: int = model_config.n_groups
        self.ssm_state_size: int = model_config.ssm_state_size
        self.conv_kernel: int = model_config.conv_kernel
        self.groups_state_size: int = self.n_groups * self.ssm_state_size
        # conv_dim: size of the vector passed through the causal conv
        # = x (mamba_int) + B (n_groups*state_size) + C (n_groups*state_size)
        self.conv_dim: int = self.mamba_int + 2 * self.groups_state_size
        # in_proj output: [gate (mamba_int) | x_B_C (conv_dim) | dt (mamba_num_heads)]
        self.in_proj_dim: int = (
            self.mamba_int + self.conv_dim + self.mamba_num_heads
        )

        from vllm_webgpu.config import get_config

        self.block_size: int = get_config().block_size

        self._layer_types: list[str] = model_config.layers_block_type
        # Length invariant is enforced by NemotronHConfig.__init__ asserting
        # len(hybrid_override_pattern) == num_hidden_layers.

        # Pre-build O(1) lookup from layer_idx to MLP rank (its index among MLP layers).
        # Only needed for heterogeneous configs where _intermediate_sizes varies per MLP layer.
        self._mlp_rank: dict[int, int] = {}
        if self._intermediate_sizes is not None:
            _mlp_count = 0
            for _i, _lt in enumerate(self._layer_types):
                if _lt == "mlp":
                    self._mlp_rank[_i] = _mlp_count
                    _mlp_count += 1

        # Persistent Mamba state buffers — allocated in _init_mamba_states()
        # after weights are loaded (device is available from __init__).
        self._conv_states: dict[int, "WebGPUBuffer"] = {}
        self._ssm_states: dict[int, "WebGPUBuffer"] = {}

        self._rms_base: dict = {
            "HIDDEN_DIM": self.hidden_size,
            "VALS_PER_THREAD": self._vals_per_thread(self.hidden_size),
        }
        self._init_scratch_buffers()

    # ── Scratch buffer allocation ─────────────────────────────────────────────

    def _init_scratch_buffers(self) -> None:
        from vllm_webgpu.webgpu.buffer import WebGPUBuffer

        dev = self.wgpu_device.wgpu_device
        rw = self._rw_flags()

        H   = self.hidden_size
        MI  = self.mamba_int
        CD  = self.conv_dim
        IPD = self.in_proj_dim
        MNH = self.mamba_num_heads
        I   = self.intermediate_size
        V   = self.vocab_size
        qd  = self.num_q_heads * self.head_dim
        kd  = self.num_kv_heads * self.head_dim

        def mk(n: int) -> "WebGPUBuffer":
            return WebGPUBuffer.empty(dev, max(n, 8), usage=rw)

        # Fixed pre-allocated decode buffers (zero-alloc hot path for T=1).
        self._pre: dict[str, "WebGPUBuffer"] = {
            "ids":      mk(4),           # [1] u32
            "pos":      mk(4),           # [1] u32
            "slot_map": mk(4),           # [1] u32
            "bt":       mk(4096 * 4),    # block table, up to 4096 blocks
            "x":        mk(H * 2),       # [H] f16 embedding output
            "norm_out": mk(H * 2),       # [H] f16 final norm output
            "logits":   mk(V * 2),       # [V] f16 LM head output
        }

        # Scratch buffers shared across layers.
        self._sc: dict[str, "WebGPUBuffer"] = {
            # 3-buffer residual rotation to prevent aliasing between layers.
            "h0": mk(H * 2),
            "h1": mk(H * 2),
            "h2": mk(H * 2),
            # Pre-normed input for current layer's mixer.
            "normed": mk(H * 2),
            # Mixer output (all layer types write here before residual add).
            "mixer_out": mk(H * 2),

            # Mamba-2 intermediates
            "mamba_inproj":  mk(IPD * 2),  # [in_proj_dim] f16
            "mamba_conv_in": mk(CD * 2),   # [conv_dim] f16 — x_B_C extracted
            "mamba_conv_out": mk(CD * 2),  # [conv_dim] f16 — after conv+SiLU
            "mamba_dt":      mk(MNH * 2),  # [mamba_num_heads] f16 — dt
            "mamba_gate":    mk(MI * 2),   # [mamba_int] f16 — gate portion
            "mamba_ssm_y":   mk(MI * 2),   # [mamba_int] f16 — SSM step output
            "mamba_norm_out": mk(MI * 2),  # [mamba_int] f16 — after gated norm

            # Attention intermediates
            "qkv_buf":     mk((qd + 2 * kd) * 2),
            "q_buf":       mk(qd * 2),
            "k_buf":       mk(kd * 2),
            "v_buf":       mk(kd * 2),
            "q_rope":      mk(qd * 2),
            "k_rope":      mk(kd * 2),
            "attn_out":    mk(qd * 2),

            # MLP intermediates
            "up_buf":  mk(I * 2),
            "ffn_act": mk(I * 2),
        }
        self._hstate = 0

    # ── Mamba state management ────────────────────────────────────────────────

    def _init_mamba_states(self) -> None:
        """Allocate zero-initialized GPU buffers for each Mamba layer's state."""
        from vllm_webgpu.webgpu.buffer import WebGPUBuffer

        dev = self.wgpu_device.wgpu_device
        rw = self._rw_flags()

        for i, lt in enumerate(self._layer_types):
            if lt != "mamba":
                continue
            # conv_state: [(kernel-1), conv_dim] f16
            conv_bytes = (self.conv_kernel - 1) * self.conv_dim * 2
            self._conv_states[i] = WebGPUBuffer.empty(
                dev, max(conv_bytes, 8), usage=rw
            )
            # ssm_state: [num_heads, head_dim, state_size] f32
            ssm_bytes = self.mamba_num_heads * self.mamba_head_dim * self.ssm_state_size * 4
            self._ssm_states[i] = WebGPUBuffer.empty(
                dev, max(ssm_bytes, 8), usage=rw
            )

        self.reset_recurrent_states()

    def reset_recurrent_states(self) -> None:
        """Zero all Mamba conv and SSM states. Call before each new request."""
        dev = self.wgpu_device.wgpu_device
        for buf in self._conv_states.values():
            dev.queue.write_buffer(buf.buf, 0, bytes(buf.nbytes))
        for buf in self._ssm_states.values():
            dev.queue.write_buffer(buf.buf, 0, bytes(buf.nbytes))

    # ── Weight loading ────────────────────────────────────────────────────────

    _hf_to_vllm_mapper = NemotronHForCausalLM.hf_to_vllm_mapper

    def load_weights(self, path: str) -> None:
        """Load weights with key remapping and Mamba-specific postprocessing."""
        # D, dt_bias, and A_log are F32 in the checkpoint and are read as array<f32>
        # by the SSM shader. The base loader would downcast them to F16, losing 13
        # mantissa bits. A_log near 0 (slow-decay states) would suffer 5-10% relative
        # error, corrupting the SSM state transition coefficient A after -exp().
        # Pass their HF-side key names so they are uploaded as F32 directly.
        # HF prefix is 'backbone.' (mapper swaps it to 'model.').
        f32_keys = frozenset(
            f"backbone.layers.{i}.mixer.{wk}"
            for i, lt in enumerate(self._layer_types)
            if lt == "mamba"
            for wk in ("D", "dt_bias", "A_log")
        )
        super().load_weights(path, f32_keys=f32_keys)
        self.weights = self._hf_to_vllm_mapper.apply_dict(self.weights)
        qmeta = self.weights.get("__quant_meta__")
        if qmeta:
            self.weights["__quant_meta__"] = self._hf_to_vllm_mapper.apply_dict(qmeta)
        self._pack_attn_weights()
        self._postprocess_mamba_weights()
        self._init_mamba_states()
        logger.info(
            "NemotronH: loaded %d weight tensors (%d Mamba layers, %d attn layers)",
            len(self.weights),
            self._layer_types.count("mamba"),
            self._layer_types.count("attention"),
        )

    def _pack_attn_weights(self) -> None:
        """Fuse separate q/k/v projection weights into a single qkv_proj buffer.

        HF NemotronH checkpoints store three tensors per attention layer:
          {p}.q_proj.weight, {p}.k_proj.weight, {p}.v_proj.weight

        The forward pass dispatches a single fused matmul against
        {p}.qkv_proj.weight, so the three row-major matrices are concatenated
        along axis 0 (i.e., their flat byte arrays are concatenated in order
        Q, K, V).  The originals are deleted after packing.
        """
        from vllm_webgpu.webgpu.buffer import WebGPUBuffer

        dev = self.wgpu_device.wgpu_device
        rw = self._rw_flags()

        for i, lt in enumerate(self._layer_types):
            if lt != "attention":
                continue
            p = f"model.layers.{i}.mixer"
            q_key = f"{p}.q_proj.weight"
            k_key = f"{p}.k_proj.weight"
            v_key = f"{p}.v_proj.weight"

            if q_key not in self.weights:
                # Already packed or checkpoint uses a different layout.
                continue

            # Preserve the source weight dtype so _uq_for_key resolves the
            # correct USE_QUANT for GPU-quantized formats (GPTQ i32, FP8/INT8
            # u8 with fmt tag).  to_numpy() always returns raw u8 bytes, so
            # the dtype must be carried forward explicitly after concatenation.
            src_dtype = self.weights[q_key].dtype

            q_bytes = self.weights[q_key].to_numpy()
            k_bytes = self.weights[k_key].to_numpy()
            v_bytes = self.weights[v_key].to_numpy()
            qkv_bytes = np.concatenate([q_bytes, k_bytes, v_bytes])

            qkv_key = f"{p}.qkv_proj.weight"
            packed_buf = WebGPUBuffer.from_numpy(
                dev, np.ascontiguousarray(qkv_bytes), usage=rw
            )
            # Override the dtype that from_numpy() inferred from the uint8
            # concatenation; the underlying GPU bytes are correct already.
            packed_buf.dtype = src_dtype
            self.weights[qkv_key] = packed_buf

            # Propagate quant_meta from q_proj to qkv_proj so _uq_for_key
            # and _quant_extra find the correct fmt / group_size / global_scale.
            qmeta = self.weights.get("__quant_meta__")
            if qmeta is not None:
                q_base = q_key.removesuffix(".weight")
                qkv_base = f"{p}.qkv_proj"
                if q_base in qmeta:
                    qmeta[qkv_base] = dict(qmeta[q_base])

            # Also pack per-weight scales (GPU quant formats store them
            # alongside the weight at w_key + ".scales").
            q_s = f"{q_key}.scales"
            k_s = f"{k_key}.scales"
            v_s = f"{v_key}.scales"
            if q_s in self.weights and k_s in self.weights and v_s in self.weights:
                scales_dtype = self.weights[q_s].dtype
                packed_scales = np.concatenate([
                    self.weights[q_s].to_numpy(),
                    self.weights[k_s].to_numpy(),
                    self.weights[v_s].to_numpy(),
                ])
                scales_buf = WebGPUBuffer.from_numpy(
                    dev, np.ascontiguousarray(packed_scales), usage=rw
                )
                scales_buf.dtype = scales_dtype
                self.weights[f"{qkv_key}.scales"] = scales_buf
                del self.weights[q_s], self.weights[k_s], self.weights[v_s]

            del self.weights[q_key], self.weights[k_key], self.weights[v_key]

    def _postprocess_mamba_weights(self) -> None:
        """Convert A_log -> A and normalize conv1d.weight shape.

        HF checkpoints store A as A_log (raw log values). Apply -exp() here to
        match what vLLM's composed_weight_loader does in the CUDA path
        (mamba_mixer2.py line 461). The WebGPU loader does not run PyTorch
        weight-loaders, so this transformation must be applied manually.

        D and dt_bias are uploaded as F32 directly by load_weights() — no
        conversion is needed here.

        The conv1d.weight may have an extra dim [conv_dim, 1, kernel] which
        we flatten to [conv_dim, kernel].
        """
        from vllm_webgpu.webgpu.buffer import WebGPUBuffer

        dev = self.wgpu_device.wgpu_device
        rw = self._rw_flags()

        for i, lt in enumerate(self._layer_types):
            if lt != "mamba":
                continue
            p = f"model.layers.{i}.mixer"

            # A: checkpoint stores as A_log; apply -exp() to get actual A.
            a_key = f"{p}.A"
            if a_key in self.weights:
                raw = self.weights[a_key].to_numpy().view(np.float32)
                a_f32 = -np.exp(raw)
                self.weights[a_key] = WebGPUBuffer.from_numpy(dev, a_f32, usage=rw)

            # conv1d.weight: flatten any extra dimension.
            # Shape may be [conv_dim, 1, kernel] (from PyTorch unsqueeze) or
            # [conv_dim, kernel]. Flatten to [conv_dim * kernel] f16 elements.
            cw_key = f"{p}.conv1d.weight"
            if cw_key in self.weights:
                raw_bytes = self.weights[cw_key].to_numpy()  # u8
                expected = self.conv_dim * self.conv_kernel
                arr = raw_bytes.view(np.float16).ravel()
                if len(arr) != expected:
                    raise ValueError(
                        f"conv1d.weight layer {i}: got {len(arr)} elements, expected {expected}"
                    )
                self.weights[cw_key] = WebGPUBuffer.from_numpy(
                    dev, np.ascontiguousarray(arr), usage=rw
                )

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: np.ndarray,
        positions: np.ndarray,
        attn_metadata: object,
    ) -> np.ndarray:
        dev = self.wgpu_device.wgpu_device
        num_tokens = len(input_ids)
        hidden = self.hidden_size
        self._hstate = 0

        if (
            hasattr(attn_metadata, "block_tables")
            and len(attn_metadata.block_tables) > 1
        ):
            raise RuntimeError(
                f"multi-sequence batching not supported: "
                f"{len(attn_metadata.block_tables)} block tables"
            )

        ctx_len = int(
            attn_metadata.max_decode_seq_len
            if getattr(attn_metadata, "max_decode_seq_len", None) is not None
            else int(positions[-1]) + 1
        )
        if ctx_len <= 0:
            ctx_len = int(positions[-1]) + 1

        vocab = self.vocab_size

        if num_tokens > 1:
            return self._prefill_forward(
                input_ids, positions, attn_metadata,
                num_tokens, ctx_len, vocab, self._rms_base,
            )

        # Decode path (T=1): zero-alloc hot path via pre-allocated buffers.
        pre = self._pre
        dev.queue.write_buffer(pre["ids"].buf, 0, input_ids.astype(np.uint32).tobytes())
        dev.queue.write_buffer(pre["pos"].buf, 0, positions.astype(np.uint32).tobytes())
        dev.queue.write_buffer(
            pre["slot_map"].buf, 0,
            np.array(attn_metadata.slot_mapping, dtype=np.uint32).tobytes(),
        )
        bt_arr = self._bt_arr(attn_metadata)
        dev.queue.write_buffer(pre["bt"].buf, 0, bt_arr.tobytes())

        with self._batched_dispatch():
            # Embedding lookup.
            self._dispatch(
                "embedding_lookup",
                [self.weights["model.embed_tokens.weight"], pre["ids"], pre["x"]],
                {"HIDDEN_DIM": hidden},
                (num_tokens, 1, 1),
            )

            # Layer 0 pre-norm (initial case: no residual add yet).
            self._dispatch(
                "rms_norm",
                [pre["x"],
                 self.weights["model.layers.0.norm.weight"],
                 self._sc["normed"]],
                self._rms_base,
                (num_tokens, 1, 1),
            )

            normed_x = self._sc["normed"]
            x_buf    = pre["x"]  # initial residual = embedding

            for i in range(self.num_layers):
                normed_x, x_buf = self._layer_dispatch(
                    i, normed_x, x_buf,
                    pre["pos"], pre["slot_map"], pre["bt"],
                    ctx_len, num_tokens, self._rms_base,
                )

            # Final norm: x_buf holds the fully accumulated residual after all layers.
            self._dispatch(
                "rms_norm",
                [x_buf, self.weights["model.norm_f.weight"], pre["norm_out"]],
                self._rms_base,
                (num_tokens, 1, 1),
            )

            lm_head_w = self.weights.get(
                "lm_head.weight", self.weights["model.embed_tokens.weight"]
            )
            self._dispatch(
                "matmul_quant",
                [pre["norm_out"], lm_head_w,
                 self.weights.get("lm_head.scales", pre["norm_out"]),
                 pre["logits"]],
                {"K": hidden, "N": vocab, "USE_QUANT": 0, "SPLIT_K": 0},
                ((vocab + 255) // 256, 1, 1),
            )
            if getattr(self, "_greedy_decode", True):
                self._dispatch(
                    "argmax_f16",
                    [pre["logits"], self._ensure_sample_buf(vocab)],
                    {"N": vocab},
                    (1, 1, 1),
                )
                self._copy_sample_to_staging()

        self._last_logit_buf = pre["logits"]
        self._last_vocab = vocab
        if getattr(self, "_greedy_decode", True):
            tok = self._read_sample_tok()
            return np.array([[tok]], dtype=np.int32)
        return self.logit_readback()

    def _layer_dispatch(
        self,
        layer_idx: int,
        normed_x: "WebGPUBuffer",
        x_buf: "WebGPUBuffer",
        pos_buf: "WebGPUBuffer",
        slot_map: "WebGPUBuffer",
        bt_buf: "WebGPUBuffer",
        ctx_len: int,
        num_tokens: int,
        rms_base: dict,
    ) -> "tuple[WebGPUBuffer, WebGPUBuffer]":
        """Dispatch one Nemotron-H layer (Mamba, Attention, or MLP).

        The layer:
          1. Runs the mixer on the pre-normed input (normed_x).
          2. Fuses residual-add with the next layer's pre-norm (or plain add
             for the final layer so the caller can apply norm_f).

        Returns (normed_for_next_layer, raw_accumulated_residual).
        """
        sc = self._sc
        lt = self._layer_types[layer_idx]
        h_names = ["h0", "h1", "h2"]
        out = sc[h_names[(self._hstate + 2) % 3]]
        add_n = num_tokens * self.hidden_size

        with self._batched_dispatch(label=f"L{layer_idx:02d}"):
            if lt == "mamba":
                self._mamba_layer(layer_idx, normed_x, num_tokens)
            elif lt == "attention":
                self._attn_layer(
                    layer_idx, normed_x, pos_buf, slot_map, bt_buf, ctx_len, num_tokens
                )
            elif lt == "mlp":
                self._mlp_layer(layer_idx, normed_x, num_tokens)
            else:
                raise NotImplementedError(
                    f"Layer type {lt!r} at index {layer_idx} not implemented"
                )

            # sc["mixer_out"] now holds the mixer result.
            mixer_out = sc["mixer_out"]

            # Fuse add + pre-norm for the next layer (saves one dispatch per layer).
            # For the last layer: plain add; norm_f applied in forward() after the loop.
            if layer_idx < self.num_layers - 1:
                next_norm_w = self.weights[
                    f"model.layers.{layer_idx + 1}.norm.weight"
                ]
                self._dispatch(
                    "add_rms_norm",
                    [x_buf, mixer_out, next_norm_w, out, sc["normed"]],
                    rms_base,
                    (num_tokens, 1, 1),
                )
                normed_out = sc["normed"]
            else:
                self._dispatch(
                    "add",
                    [x_buf, mixer_out, out],
                    {"N": add_n},
                    ((add_n // 4 + 255) // 256, 1, 1),
                )
                normed_out = sc["normed"]  # stale; unused after last layer

        self._hstate = (self._hstate + 2) % 3
        return normed_out, out

    # ── Mixer implementations ─────────────────────────────────────────────────

    def _mamba_layer(
        self,
        layer_idx: int,
        normed_x: "WebGPUBuffer",
        num_tokens: int,
    ) -> None:
        """Mamba-2 SSM layer.

        Pipeline: in_proj -> [GPU-side extract] -> causal_conv -> SSM step
                  -> grouped gated RMSNorm -> out_proj.
        Result goes to sc["mixer_out"].
        """
        sc  = self._sc
        p   = f"model.layers.{layer_idx}.mixer"
        H   = self.hidden_size
        MI  = self.mamba_int
        CD  = self.conv_dim
        MNH = self.mamba_num_heads
        MHD = self.mamba_head_dim
        NS  = self.ssm_state_size
        NG  = self.n_groups

        # Step 1: in_proj — hidden -> [gate | x_B_C | dt]
        in_w = f"{p}.in_proj.weight"
        uq   = self._uq_for_key(in_w)
        self._dispatch(
            "matmul_quant",
            [normed_x, self.weights[in_w],
             self._scales_buf(in_w, uq, normed_x), sc["mamba_inproj"]],
            {"K": H, "N": self.in_proj_dim, "USE_QUANT": uq,
             **self._split_k_extra(uq),
             **self._quant_extra(f"{p}.in_proj", uq)},
            _gemv_wg(self.in_proj_dim, uq),
        )

        # GPU-side byte copies to extract the three portions of in_proj output.
        # gate:   bytes [0          .. MI*2)        -> sc["mamba_gate"]
        # x_B_C:  bytes [MI*2      .. (MI+CD)*2)   -> sc["mamba_conv_in"]
        # dt:     bytes [(MI+CD)*2 .. (MI+CD+MNH)*2) -> sc["mamba_dt"]
        enc = self._active_encoder
        if enc is not None:
            enc.copy_buffer_to_buffer(
                sc["mamba_inproj"].buf, 0,
                sc["mamba_gate"].buf,   0,
                MI * 2,
            )
            enc.copy_buffer_to_buffer(
                sc["mamba_inproj"].buf, MI * 2,
                sc["mamba_conv_in"].buf, 0,
                CD * 2,
            )
            enc.copy_buffer_to_buffer(
                sc["mamba_inproj"].buf, (MI + CD) * 2,
                sc["mamba_dt"].buf,      0,
                MNH * 2,
            )

        # Step 2: Causal conv1d on x_B_C with SiLU activation.
        conv_w = f"{p}.conv1d.weight"
        conv_b = f"{p}.conv1d.bias"
        has_bias = int(conv_b in self.weights)
        bias_buf = self.weights.get(conv_b, sc["mamba_conv_in"])  # dummy when absent
        self._dispatch(
            "mamba2_causal_conv",
            [sc["mamba_conv_in"], self.weights[conv_w], bias_buf,
             self._conv_states[layer_idx], sc["mamba_conv_out"]],
            {"CONV_DIM": CD, "KERNEL": self.conv_kernel,
             "WG_SIZE": 256, "HAS_BIAS": has_bias},
            ((CD + 255) // 256, 1, 1),
        )

        # Step 3: Mamba-2 SSM state update.
        # x_B_C layout after conv: [x(MI) | B(N_GROUPS*NS) | C(N_GROUPS*NS)]
        self._dispatch(
            "mamba2_ssm_step",
            [sc["mamba_conv_out"], sc["mamba_dt"],
             self.weights[f"{p}.A"], self.weights[f"{p}.dt_bias"],
             self.weights[f"{p}.D"],
             self._ssm_states[layer_idx], sc["mamba_ssm_y"]],
            {"NUM_HEADS": MNH, "HEAD_DIM": MHD,
             "STATE_SIZE": NS, "N_GROUPS": NG, "WG_SIZE": 256},
            (MNH, 1, 1),
        )

        # Step 4: Grouped gated RMSNorm.
        self._dispatch(
            "mamba2_norm_gate",
            [sc["mamba_ssm_y"], sc["mamba_gate"],
             self.weights[f"{p}.norm.weight"], sc["mamba_norm_out"]],
            {"MAMBA_INT": MI, "N_GROUPS": NG, "WG_SIZE": 256},
            (NG, 1, 1),
        )

        # Step 5: out_proj — mamba_int -> hidden.
        out_w = f"{p}.out_proj.weight"
        uq2   = self._uq_for_key(out_w)
        self._dispatch(
            "matmul_quant",
            [sc["mamba_norm_out"], self.weights[out_w],
             self._scales_buf(out_w, uq2, sc["mamba_norm_out"]), sc["mixer_out"]],
            {"K": MI, "N": H, "USE_QUANT": uq2,
             **self._split_k_extra(uq2),
             **self._quant_extra(f"{p}.out_proj", uq2)},
            _gemv_wg(H, uq2),
        )

    def _attn_layer(
        self,
        layer_idx: int,
        normed_x: "WebGPUBuffer",
        pos_buf: "WebGPUBuffer",
        slot_map: "WebGPUBuffer",
        bt_buf: "WebGPUBuffer",
        ctx_len: int,
        num_tokens: int,
    ) -> None:
        """Full-attention layer (no per-head norm in NemotronH).

        Pipeline: qkv_proj -> [extract Q/K/V] -> RoPE -> KV cache -> flash_attn -> o_proj.
        Result goes to sc["mixer_out"].
        """
        sc    = self._sc
        p     = f"model.layers.{layer_idx}.mixer"
        H     = self.hidden_size
        q_dim = self.num_q_heads * self.head_dim
        k_dim = self.num_kv_heads * self.head_dim
        ln_rope = self._ln_rope_theta

        # Fused QKV projection.
        qkv_w    = f"{p}.qkv_proj.weight"
        total_qkv = q_dim + 2 * k_dim
        uq = self._uq_for_key(qkv_w)
        self._dispatch(
            "matmul_quant",
            [normed_x, self.weights[qkv_w],
             self._scales_buf(qkv_w, uq, normed_x), sc["qkv_buf"]],
            {"K": H, "N": total_qkv, "USE_QUANT": uq,
             **self._split_k_extra(uq),
             **self._quant_extra(f"{p}.qkv_proj", uq)},
            _gemv_wg(total_qkv, uq),
        )

        # GPU-side extraction: split QKV buffer into Q, K, V.
        enc = self._active_encoder
        if enc is not None:
            enc.copy_buffer_to_buffer(
                sc["qkv_buf"].buf, 0,           sc["q_buf"].buf, 0, q_dim * 2)
            enc.copy_buffer_to_buffer(
                sc["qkv_buf"].buf, q_dim * 2,   sc["k_buf"].buf, 0, k_dim * 2)
            enc.copy_buffer_to_buffer(
                sc["qkv_buf"].buf, (q_dim + k_dim) * 2,
                sc["v_buf"].buf, 0, k_dim * 2)

        # RoPE for Q and K. NemotronH has no per-head RMSNorm on Q/K.
        _rope_c = {
            "HEAD_DIM":    self.head_dim,
            "ROPE_BASE":   float(self.rope_theta),
            "LN_ROPE_BASE": ln_rope,
            "USE_FREQ_BUF": 0,
        }
        self._dispatch(
            "rope",
            [sc["q_buf"], pos_buf, sc["q_rope"], self._rope_freq_buf],
            {**_rope_c, "NUM_HEADS": self.num_q_heads},
            (num_tokens, self.num_q_heads, 1),
        )
        self._dispatch(
            "rope",
            [sc["k_buf"], pos_buf, sc["k_rope"], self._rope_freq_buf],
            {**_rope_c, "NUM_HEADS": self.num_kv_heads},
            (num_tokens, self.num_kv_heads, 1),
        )

        # Fused KV cache store.
        k_cache, v_cache = self.kv_pool[layer_idx]
        self._dispatch(
            "kv_cache_store_both",
            [sc["k_rope"], k_cache, sc["v_buf"], v_cache, slot_map],
            {"BLOCK_SIZE": self.block_size, "NUM_KV_HEADS": self.num_kv_heads,
             "HEAD_DIM": self.head_dim, "V_IN_OFFSET": 0},
            (num_tokens, self.num_kv_heads, 1),
        )

        # Flash attention decode.
        self._dispatch(
            "flash_attn_decode",
            [sc["q_rope"], k_cache, v_cache, bt_buf, sc["attn_out"]],
            {"BLOCK_SIZE": self.block_size, "NUM_Q_HEADS": self.num_q_heads,
             "NUM_KV_HEADS": self.num_kv_heads, "HEAD_DIM": self.head_dim,
             "CTX_LEN": ctx_len},
            (self.num_q_heads, 1, 1),
        )

        # Output projection.
        ow  = f"{p}.o_proj.weight"
        uq2 = self._uq_for_key(ow)
        self._dispatch(
            "matmul_quant",
            [sc["attn_out"], self.weights[ow],
             self._scales_buf(ow, uq2, sc["attn_out"]), sc["mixer_out"]],
            {"K": q_dim, "N": H, "USE_QUANT": uq2,
             **self._split_k_extra(uq2),
             **self._quant_extra(f"{p}.o_proj", uq2)},
            _gemv_wg(H, uq2),
        )

    def _mlp_layer(
        self,
        layer_idx: int,
        normed_x: "WebGPUBuffer",
        num_tokens: int,
    ) -> None:
        """MLP-only layer with squared-ReLU activation (no gate projection).

        Pipeline: up_proj -> relu^2 -> down_proj.
        Result goes to sc["mixer_out"].
        """
        sc  = self._sc
        p   = f"model.layers.{layer_idx}.mixer"
        H   = self.hidden_size
        if self._intermediate_sizes is not None:
            mlp_idx = self._mlp_rank[layer_idx]
            I = self._intermediate_sizes[mlp_idx % len(self._intermediate_sizes)]
        else:
            I = self.intermediate_size

        # up_proj: hidden -> intermediate
        uw  = f"{p}.up_proj.weight"
        uq  = self._uq_for_key(uw)
        self._dispatch(
            "matmul_quant",
            [normed_x, self.weights[uw],
             self._scales_buf(uw, uq, normed_x), sc["up_buf"]],
            {"K": H, "N": I, "USE_QUANT": uq,
             **self._split_k_extra(uq),
             **self._quant_extra(f"{p}.up_proj", uq)},
            _gemv_wg(I, uq),
        )

        # relu^2 element-wise activation
        relu_n = num_tokens * I
        self._dispatch(
            "relu_sq",
            [sc["up_buf"], sc["ffn_act"]],
            {"N": relu_n, "WG_SIZE": 256},
            ((relu_n + 255) // 256, 1, 1),
        )

        # down_proj: intermediate -> hidden
        dw  = f"{p}.down_proj.weight"
        uq2 = self._uq_for_key(dw)
        self._dispatch(
            "matmul_quant",
            [sc["ffn_act"], self.weights[dw],
             self._scales_buf(dw, uq2, sc["ffn_act"]), sc["mixer_out"]],
            {"K": I, "N": H, "USE_QUANT": uq2,
             **self._split_k_extra(uq2),
             **self._quant_extra(f"{p}.down_proj", uq2)},
            _gemv_wg(H, uq2),
        )

    # ── Prefill fallback ──────────────────────────────────────────────────────

    def _prefill_forward(
        self,
        input_ids: np.ndarray,
        positions: np.ndarray,
        attn_metadata: object,
        T: int,
        ctx_len: int,
        vocab: int,
        rms_base: dict,
    ) -> np.ndarray:
        """Process T prompt tokens one at a time through the decode path.

        Each token is processed sequentially so the Mamba conv/SSM states
        accumulate correctly. KV cache is filled token-by-token for causal
        attention. Only the last token's logits are returned.
        """
        dev = self.wgpu_device.wgpu_device
        pre = self._pre
        sc  = self._sc
        bt_arr = self._bt_arr(attn_metadata)

        for t in range(T):
            self._hstate = 0
            tok_ctx = int(positions[t]) + 1

            dev.queue.write_buffer(
                pre["ids"].buf, 0, input_ids[t:t+1].astype(np.uint32).tobytes())
            dev.queue.write_buffer(
                pre["pos"].buf, 0, positions[t:t+1].astype(np.uint32).tobytes())
            dev.queue.write_buffer(
                pre["slot_map"].buf, 0,
                np.array(attn_metadata.slot_mapping[t:t+1], dtype=np.uint32).tobytes())
            dev.queue.write_buffer(pre["bt"].buf, 0, bt_arr.tobytes())

            with self._batched_dispatch():
                self._dispatch(
                    "embedding_lookup",
                    [self.weights["model.embed_tokens.weight"], pre["ids"], pre["x"]],
                    {"HIDDEN_DIM": self.hidden_size},
                    (1, 1, 1),
                )
                self._dispatch(
                    "rms_norm",
                    [pre["x"],
                     self.weights["model.layers.0.norm.weight"],
                     sc["normed"]],
                    rms_base,
                    (1, 1, 1),
                )

                normed_x = sc["normed"]
                x_buf    = pre["x"]

                for i in range(self.num_layers):
                    normed_x, x_buf = self._layer_dispatch(
                        i, normed_x, x_buf,
                        pre["pos"], pre["slot_map"], pre["bt"],
                        tok_ctx, 1, rms_base,
                    )

                if t == T - 1:
                    self._dispatch(
                        "rms_norm",
                        [x_buf, self.weights["model.norm_f.weight"], pre["norm_out"]],
                        rms_base,
                        (1, 1, 1),
                    )
                    lm_head_w = self.weights.get(
                        "lm_head.weight", self.weights["model.embed_tokens.weight"]
                    )
                    self._dispatch(
                        "matmul_quant",
                        [pre["norm_out"], lm_head_w,
                         self.weights.get("lm_head.scales", pre["norm_out"]),
                         pre["logits"]],
                        {"K": self.hidden_size, "N": vocab, "USE_QUANT": 0, "SPLIT_K": 0},
                        ((vocab + 255) // 256, 1, 1),
                    )
                    if getattr(self, "_greedy_decode", True):
                        self._dispatch(
                            "argmax_f16",
                            [pre["logits"], self._ensure_sample_buf(vocab)],
                            {"N": vocab},
                            (1, 1, 1),
                        )
                        self._copy_sample_to_staging()

        self._last_logit_buf = pre["logits"]
        self._last_vocab     = vocab
        if getattr(self, "_greedy_decode", True):
            tok = self._read_sample_tok()
            return np.array([[tok]], dtype=np.int32)
        return self.logit_readback()

