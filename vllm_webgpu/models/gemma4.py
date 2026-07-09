from __future__ import annotations
import math
from itertools import batched
from typing import TYPE_CHECKING

import numpy as np

from vllm.logger import init_logger
from vllm_webgpu.models.base import BaseWebGPUModel, _gemv_wg, _vec4_wg, _rows_wg, _H_NAMES

if TYPE_CHECKING:
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.device import WebGPUDevice
    from vllm_webgpu.webgpu.pipeline import PipelineCache

logger = init_logger(__name__)


class Gemma4WebGPUModel(BaseWebGPUModel):
    """
    Gemma 4 transformer with heterogeneous per-layer attention.

    Gemma4-12B mixes two attention types:
    - Local layers (head_dim=256, 8 KV heads): standard GQA with per-head RMSNorm on V
    - Global layers (head_dim=512, 1 KV head): MQA, no separate V projection (V=K)

    Every 6th layer (indices 5, 11, 17, ...) is a global attention layer.
    Scratch buffers are allocated at maximum dimensions to handle both types.
    """
    _GEMMA_NORM: int = 1  # all Gemma models use (1+w) RMSNorm

    # GPU argmax path returns (1,1) int32; logit_readback() provides full logits.
    logit_returns_token_id: bool = True

    def __init__(self, model_config, wgpu_device: "WebGPUDevice", pipeline_cache: "PipelineCache", block_size: int = 16) -> None:
        super().__init__(model_config, wgpu_device, pipeline_cache)
        self.num_layers: int = model_config.num_hidden_layers
        self.num_q_heads: int = model_config.num_attention_heads
        self.hidden_size: int = model_config.hidden_size
        self.intermediate_size: int = model_config.intermediate_size
        self.vocab_size: int = model_config.vocab_size
        # Softcap is optional — Gemma4 uses 30.0, Gemma3 uses None (no cap)
        self.softcap: float | None = getattr(model_config, "final_logit_softcapping", None)
        self.rope_theta: float = getattr(model_config, "rope_theta", 10000.0)
        self.block_size: int = block_size

        # Gemma3 vs Gemma4 capability flags:
        # - GEMMA_NORM=1: all Gemma models use (1+w) RMSNorm (weights trained as deviations from 0)
        # - _apply_v_norm: only Gemma4 applies per-head RMS norm to V before caching
        # Per-layer attention parameters (head_dim, num_kv_heads, q_dim, kv_dim, has_v_proj).
        # Set from _layer_attention_params if available (parsed from GGUF), otherwise derive
        # using the heuristic that every 6th layer (idx%6==5) is global attention.
        raw_lp = getattr(model_config, "_layer_attention_params", None)
        default_hd = getattr(model_config, "head_dim",
                             self.hidden_size // self.num_q_heads)
        default_kv = getattr(model_config, "num_key_value_heads", 1)

        # Gemma4 safetensors: derive per-layer params from layer_types + global_head_dim.
        layer_types = getattr(model_config, "layer_types", None)

        # Capability flag: only Gemma4 applies per-head RMS norm to V before caching.
        # Gemma3 does NOT normalize V; DiffusionGemma handles V-norm unconditionally in _decoder_layer.
        # Use layer_types as the discriminator: Gemma4 configs carry a per-layer attention type list
        # (one entry per layer), Gemma3 configs do not. This avoids fragile architecture name matching.
        self._apply_v_norm = (layer_types is not None and len(layer_types) == self.num_layers)
        global_hd   = getattr(model_config, "global_head_dim", default_hd)
        global_kv   = getattr(model_config, "num_global_key_value_heads", default_kv)
        _k_eq_v     = getattr(model_config, "attention_k_eq_v", False)

        if raw_lp and len(raw_lp) == self.num_layers:
            for lp_entry in raw_lp:
                lp_entry.setdefault("intermediate_size", self.intermediate_size)
                lp_entry.setdefault("is_kv_shared", False)
                lp_entry.setdefault("kv_shared_target", -1)
            self._lp: list[dict] = raw_lp
        elif layer_types and len(layer_types) == self.num_layers:
            # Build per-layer params from layer_types list (Gemma4 safetensors config).
            # sliding_attention: local GQA, head_dim=default_hd, has_v_proj=True
            # full_attention:    global, head_dim=global_hd; V=K only when attention_k_eq_v=True
            #
            # use_double_wide_mlp: the last num_kv_shared_layers layers have FFN width * 2.
            # is_kv_shared: the last num_kv_shared_layers layers share KV with an earlier
            # layer of the same type (mirrors vLLM Gemma4Attention.is_kv_shared_layer).
            first_kv_shared = self.num_layers - getattr(model_config, "num_kv_shared_layers", 0)
            use_dwm = getattr(model_config, "use_double_wide_mlp", False)
            self._lp = []
            for i, lt in enumerate(layer_types):
                inter_l = self.intermediate_size * (2 if use_dwm and 0 < first_kv_shared < self.num_layers and i >= first_kv_shared else 1)
                is_kv_shared = (first_kv_shared < self.num_layers) and (i >= first_kv_shared)
                kv_shared_target = -1
                if is_kv_shared:
                    # Find the last non-shared layer with the same layer_type (matches vLLM).
                    _prev = layer_types[:first_kv_shared]
                    kv_shared_target = next((j for j in range(len(_prev)-1, -1, -1) if _prev[j] == lt), -1)
                if lt == "full_attention":
                    hd_l  = global_hd
                    nkv_l = global_kv if _k_eq_v else default_kv
                    hv    = not _k_eq_v
                else:
                    hd_l = default_hd
                    nkv_l = default_kv
                    hv = True
                self._lp.append({
                    "head_dim":         hd_l,
                    "num_q_heads":      self.num_q_heads,
                    "num_kv_heads":     nkv_l,
                    "q_dim":            self.num_q_heads * hd_l,
                    "kv_dim":           nkv_l * hd_l,
                    "has_v_proj":       hv,
                    "intermediate_size": inter_l,
                    "is_kv_shared":     is_kv_shared,
                    "kv_shared_target": kv_shared_target,
                })
        else:
            # Uniform fallback: all layers use the config defaults.
            # For Gemma3 safetensors (uniform attention) this is correct.
            hd = default_hd
            nkv = default_kv
            uniform_lp = {
                "head_dim":         hd,
                "num_q_heads":      self.num_q_heads,
                "num_kv_heads":     nkv,
                "q_dim":            self.num_q_heads * hd,
                "kv_dim":           nkv * hd,
                "has_v_proj":       True,
                "intermediate_size": self.intermediate_size,
                "is_kv_shared":     False,
                "kv_shared_target": -1,
            }
            self._lp = [uniform_lp.copy() for _ in range(self.num_layers)]

        # Validate even dimensions required by WGSL shaders
        for name, val in [("hidden_size", self.hidden_size),
                          ("intermediate_size", self.intermediate_size)]:
            if val % 4 != 0:
                raise ValueError(f"{name}={val} must be divisible by 4 for vec4<f16> shaders")

        # Compute max dimensions across all layers for scratch buffer sizing
        self._max_q_dim = max(lp["q_dim"] for lp in self._lp)
        self._max_kv_dim = max(lp["kv_dim"] for lp in self._lp)
        max_ctx = getattr(model_config, "max_position_embeddings", 8192)
        self._init_scratch_buffers(max_ctx, self._max_q_dim, self._max_kv_dim)

        _vpt = self._vals_per_thread(self.hidden_size)
        self._rms_consts = {"HIDDEN_DIM": self.hidden_size, "VALS_PER_THREAD": _vpt, "GEMMA_NORM": self._GEMMA_NORM}
        self._ln_rope_theta: float = math.log(self.rope_theta)
        self._g4_rope_base = {
            "ROPE_BASE": float(self.rope_theta),
            "LN_ROPE_BASE": self._ln_rope_theta,
            "USE_FREQ_BUF": int(self._use_freq_buf),
        }

    def _scratch_token_count(self) -> int:
        """Number of tokens to size T-dependent scratch buffers for. Override in subclasses."""
        return 1

    def _scratch_inter_size(self) -> int:
        """Intermediate size for FFN scratch buffers. Override in subclasses."""
        return max(lp["intermediate_size"] for lp in self._lp)

    def _init_scratch_buffers(self, max_ctx: int, max_q_dim: int, max_kv_dim: int) -> None:
        """Pre-allocate scratch buffers at maximum layer dimensions."""
        from vllm_webgpu.webgpu.buffer import WebGPUBuffer

        dev = self.wgpu_device.wgpu_device
        T = self._scratch_token_count()
        H = self.hidden_size
        I = self._scratch_inter_size()

        def mk(n: int) -> "WebGPUBuffer":
            return WebGPUBuffer.empty(dev, n)

        # Pre-allocated per-step buffers (reused every decode via write_buffer).
        V = self.vocab_size
        self._pre: dict[str, "WebGPUBuffer"] = {
            "ids":      mk(T * 4),         # [1] uint32 token id
            "pos":      mk(T * 4),         # [1] uint32 position
            "slot_map": mk(T * 4),         # [1] uint32 physical slot
            "bt":       mk(max(4096, (max_ctx + self.block_size - 1) // self.block_size) * 4),  # block table
            "x":        mk(T * H * 4),     # [1, H] f32 residual
            "norm_out": mk(T * H * 2),     # [1, H] f16 final norm
            "logits":   mk(T * V * 2),     # [1, V] f16 logits
        }
        if self.softcap is not None and self.softcap > 0:
            self._pre["capped"] = mk(T * V * 2)  # [1, V] f16 softcapped logits (Gemma4)

        self._sc: dict[str, "WebGPUBuffer"] = {
            "normed":     mk(T * H * 2),                              # f16
            "qkv_buf":    mk(T * (max_q_dim + 2 * max_kv_dim) * 2),  # f16 [Q|K|V]
            "q_buf":      mk(T * max_q_dim * 2),                      # f16
            "k_buf":      mk(T * max_kv_dim * 2),                     # f16
            "v_buf":      mk(T * max_kv_dim * 2),                     # f16
            "v_normed":   mk(T * max_kv_dim * 2),                     # f16
            "q_rope":     mk(T * max_q_dim * 2),  # f16
            "k_rope":     mk(T * max_kv_dim * 2), # f16
            "attn_out":   mk(T * max_q_dim * 2),  # f16
            "o_proj_out": mk(T * H * 2),           # f16
            "ffn_normed": mk(T * H * 2),           # f16
            "gate_buf":   mk(T * I * 2),           # f16
            "up_buf":     mk(T * I * 2),           # f16
            "ffn_act":    mk(T * I * 2),
            "ffn_out":    mk(T * H * 2),
            # Residual buffers stored in f32 for precision.
            # Gemma4 has output_norm weights up to 600 which cause f16 saturation
            # when accumulated across 48 layers — f32 residuals prevent this.
            "h0":         mk(T * H * 4),           # f32 (4 bytes)
            "h1":         mk(T * H * 4),           # f32
            "h2":         mk(T * H * 4),           # f32
        }
        self._hstate: int = 0

    def _postprocess_weights(self) -> None:
        """Tile per-head norm weights that have shape (head_dim,) to (num_heads * head_dim,).

        Gemma4 q_norm/k_norm weights are per-head (shape = head_dim) but the
        fused_per_head_norm_rope shader indexes weight[head_idx * HEAD_DIM + i],
        requiring shape (num_heads * head_dim). Tile to match, using per-layer params.
        """
        from vllm_webgpu.webgpu.buffer import WebGPUBuffer
        dev = self.wgpu_device.wgpu_device

        for i, lp in enumerate(self._lp):
            p = self._layer_key_prefix(i)
            hd = lp["head_dim"]
            for norm_key, num_heads in [
                (f"{p}.self_attn.q_norm.weight", lp["num_q_heads"]),
                (f"{p}.self_attn.k_norm.weight", lp["num_kv_heads"]),
            ]:
                buf = self.weights.get(norm_key)
                if buf is None:
                    continue
                expected_len = num_heads * hd
                raw = buf.to_numpy().view(np.float16)
                if len(raw) == expected_len:
                    continue
                if len(raw) == hd:
                    tiled = np.tile(raw, num_heads)
                    self.weights[norm_key] = WebGPUBuffer.from_numpy(dev, tiled)
                else:
                    logger.warning("Unexpected q/k_norm shape for %s: got %d, expected %d or %d",
                                   norm_key, len(raw), expected_len, hd)

    def _layer_key_prefix(self, layer_idx: int) -> str:
        """Return the weight key prefix for layer i. Subclasses may override."""
        return f"model.layers.{layer_idx}"

    def _embed_key(self) -> str:
        """Return the weight key for the embedding table. Subclasses may override."""
        return "model.embed_tokens.weight"

    def _norm_key(self) -> str:
        """Return the weight key for the final layer norm. Subclasses may override."""
        return "model.norm.weight"


    def _load_layer_scales(self) -> None:
        """Cache layer_scalar values on CPU at load time.

        Avoids 48 GPU→CPU readbacks per token (each to_numpy() is a blocking ~100µs sync).
        Subclasses with a different key scheme should override _layer_key_prefix instead.
        """
        self._layer_scales: list[float] = []
        for i in range(self.num_layers):
            p = self._layer_key_prefix(i)
            ls_buf = self.weights.get(f"{p}.layer_scalar")
            if ls_buf is not None:
                self._layer_scales.append(float(ls_buf.to_numpy().view(np.float16)[0]))
            else:
                self._layer_scales.append(1.0)

    def load_weights(self, path: str, f32_keys: "frozenset[str] | None" = None) -> None:
        super().load_weights(path, f32_keys=f32_keys)
        self._postprocess_weights()
        self._load_layer_scales()

    def forward(
        self,
        input_ids: np.ndarray,
        positions: np.ndarray,
        attn_metadata: object,
    ) -> np.ndarray:
        dev = self.wgpu_device.wgpu_device
        num_tokens = len(input_ids)
        hidden = self.hidden_size
        vocab = self.vocab_size

        self._hstate = 0

        if hasattr(attn_metadata, "block_tables") and len(attn_metadata.block_tables) > 1:
            # Each forward() call handles exactly one sequence. The model runner
            # calls forward() once per decode request. Batching N sequences requires
            # N separate pre-alloc buffer sets and per-sequence attention dispatch.
            raise RuntimeError(
                f"multi-sequence batching not supported: got {len(attn_metadata.block_tables)} "
                "block tables; call forward() once per decode request"
            )

        # Prefill path: T>1 tokens use matmul_quant_mr4 (batch GEMM) and flash_attn_prefill.
        if num_tokens > 1:
            return self._prefill_batch_forward(input_ids, positions, attn_metadata, num_tokens)

        ctx_len = int(attn_metadata.max_decode_seq_len
                      if attn_metadata.max_decode_seq_len is not None
                      else int(positions[-1]) + 1)
        if ctx_len <= 0:
            ctx_len = int(positions[-1]) + 1

        # Update pre-allocated buffers via write_buffer — no GPU allocation per step.
        pre = self._pre
        dev.queue.write_buffer(pre["ids"].buf, 0, input_ids.astype(np.uint32).tobytes())
        dev.queue.write_buffer(pre["pos"].buf, 0, positions.astype(np.uint32).tobytes())
        dev.queue.write_buffer(
            pre["slot_map"].buf, 0,
            np.array(attn_metadata.slot_mapping, dtype=np.uint32).tobytes())
        bt_arr = self._bt_arr(attn_metadata)
        dev.queue.write_buffer(pre["bt"].buf, 0, bt_arr.tobytes())

        ids_buf    = pre["ids"]
        pos_buf    = pre["pos"]
        slot_map   = pre["slot_map"]
        bt_buf     = pre["bt"]
        x_buf      = pre["x"]
        norm_out   = pre["norm_out"]
        logits_buf = pre["logits"]

        _rms_base = self._rms_consts
        sc = self._sc

        with self._batched_dispatch():
            # Embedding lookup → f32 output for f32 residual pipeline
            self._dispatch("embedding_lookup_f32",
                           [self.weights[self._embed_key()], ids_buf, x_buf],
                           {"HIDDEN_DIM": hidden}, (num_tokens, 1, 1))

            # Initial pre-norm for layer 0 (subsequent pre-norms are fused into each
            # layer's final add_f32_rms_norm dispatch).
            self._dispatch("rms_norm_f32in",
                           [x_buf, self.weights[f"{self._layer_key_prefix(0)}.input_layernorm.weight"], sc["normed"]],
                           _rms_base, (num_tokens, 1, 1))

            normed_x = sc["normed"]
            for i in range(self.num_layers):
                normed_x, x_buf = self._transformer_layer(
                    i, normed_x, x_buf, pos_buf, slot_map, bt_buf, ctx_len, num_tokens)

            # Final norm: reads f32 residual, writes f16 norm_out
            self._dispatch("rms_norm_f32in",
                           [x_buf, self.weights[self._norm_key()], norm_out],
                           _rms_base, (num_tokens, 1, 1))

            _lm_key = self._lm_head_key()
            lm_head_w = self.weights[_lm_key]
            uq_lm = self._uq_for_key(_lm_key)
            _lm_base = _lm_key.removesuffix('.weight')
            # vocab_size exceeds the 65535 workgroup-per-dimension limit, so the split-K
            # path is unusable. Force SPLIT_K=0 (row-per-thread) with ceil(vocab/256) WGs.
            self._dispatch("matmul_quant",
                           [norm_out, lm_head_w,
                            self._scales_buf(_lm_key, uq_lm, self._dummy_scales_buf), logits_buf],
                           {"K": hidden, "N": vocab, "USE_QUANT": uq_lm, "SPLIT_K": 0,
                            **self._quant_extra(_lm_base, uq_lm)},
                           _rows_wg(vocab))

            if self.softcap is not None and self.softcap > 0:
                capped_buf = pre["capped"]
                self._dispatch("logit_softcap", [logits_buf, capped_buf],
                               {"VOCAB": vocab, "CAP": float(self.softcap)},
                               ((vocab + 255) // 256, num_tokens, 1),
                               shader_subdir="gemma")
                result_buf = capped_buf
            else:
                result_buf = logits_buf

            self._dispatch("argmax_f16", [result_buf, self._ensure_sample_buf()],
                           {"N": vocab}, (1, 1, 1))
            self._copy_sample_to_staging()

        self._last_logit_buf = result_buf
        self._last_vocab     = vocab
        tok = self._read_sample_tok()
        return np.array([[tok]], dtype=np.int32)

    def _mr4_quant_supported(self) -> bool:
        """Return True when every layer weight uses a quant format compatible with matmul_quant_mr4.

        Only USE_QUANT=0 (f16) and USE_QUANT=3 (GPTQ int4) are supported in the batch
        prefill path. Scanning all layers up front catches mixed-quant checkpoints before
        any KV-cache population occurs rather than crashing mid-forward.
        """
        for i in range(self.num_layers):
            p  = self._layer_key_prefix(i)
            lp = self._lp[i]
            keys = [
                f"{p}.self_attn.q_proj.weight",
                f"{p}.self_attn.k_proj.weight",
                f"{p}.self_attn.o_proj.weight",
                f"{p}.mlp.gate_proj.weight",
                f"{p}.mlp.up_proj.weight",
                f"{p}.mlp.down_proj.weight",
            ]
            if lp.get("has_v_proj", True):
                keys.append(f"{p}.self_attn.v_proj.weight")
            for k in keys:
                if k in self.weights and self._uq_for_key(k) not in (0, 3):
                    return False
        return True

    def _prefill_batch_forward(  # noqa: C901
        self,
        input_ids: "np.ndarray",
        positions: "np.ndarray",
        attn_metadata: object,
        T: int,
    ) -> "np.ndarray":
        """Batch prefill: process T prompt tokens in a series of GPU command encoders.

        GEMM ops use matmul_quant_mr4 (T rows at once).
        Attention uses flash_attn_prefill (dense causal Q/K/V, not paged cache).
        Layers are chunked across separate command encoders (4 per submit) to stay
        under Metal's per-command-buffer GPU timeout.
        Returns shape (1, 1) int32 (GPU argmax of last-token logits).
        """
        from vllm_webgpu.webgpu.buffer import WebGPUBuffer

        if not self._mr4_quant_supported():
            return self._prefill_sequential_fallback(input_ids, positions, attn_metadata, T)

        dev = self.wgpu_device.wgpu_device
        hidden = self.hidden_size
        vocab  = self.vocab_size
        max_inter  = max(lp["intermediate_size"] for lp in self._lp)
        max_q_dim  = self._max_q_dim
        max_kv_dim = self._max_kv_dim

        def alloc(n_bytes: int) -> WebGPUBuffer:
            return WebGPUBuffer.empty(dev, max(n_bytes, 8))

        # T-token batch buffers. Allocated once per prefill call;
        # allocation cost is negligible vs the GEMM savings.
        b: dict = {
            "x":        alloc(T * hidden * 4),       # f32 embedding residual
            "normed":   alloc(T * hidden * 2),        # f16 normed (pre-attn and pre-FFN)
            "q_buf":    alloc(T * max_q_dim * 2),     # f16 Q projection output
            "k_buf":    alloc(T * max_kv_dim * 2),    # f16 K projection output
            "v_buf":    alloc(T * max_kv_dim * 2),    # f16 V projection output
            "q_rope":   alloc(T * max_q_dim * 2),     # f16 Q after norm+rope
            "k_rope":   alloc(T * max_kv_dim * 2),    # f16 K after norm+rope
            "v_normed": alloc(T * max_kv_dim * 2),    # f16 V after per-head RMS norm
            "attn_out": alloc(T * max_q_dim * 2),     # f16 attention output
            "o_proj":   alloc(T * hidden * 2),         # f16 output projection
            "ffn_n":    alloc(T * hidden * 2),         # f16 FFN normed (intermediate)
            "gate_buf": alloc(T * max_inter * 2),          # f16 FFN gate
            "up_buf":   alloc(T * max_inter * 2),          # f16 FFN up
            "ffn_act":  alloc(T * max_inter * 2),          # f16 activated gate*up
            "ffn_out":  alloc(T * hidden * 2),         # f16 FFN output
            "h0":       alloc(T * hidden * 4),         # f32 residual (rotation slot 0)
            "h1":       alloc(T * hidden * 4),         # f32 residual (rotation slot 1)
            "h2":       alloc(T * hidden * 4),         # f32 residual (rotation slot 2)
            # Single-token scratch for final norm + LM head
            "last_f32":  alloc(hidden * 4),            # f32 last-token residual copy
            "last_norm": alloc(hidden * 2),            # f16 last-token after final norm
            "logits":    alloc(vocab * 2),             # f16 LM head output
        }
        if self.softcap is not None and self.softcap > 0:
            b["capped"] = alloc(vocab * 2)             # f16 softcapped logits (Gemma4)
        slot_map_buf = WebGPUBuffer.from_numpy(
            dev, np.array(attn_metadata.slot_mapping, dtype=np.uint32))
        pos_buf = WebGPUBuffer.from_numpy(dev, positions.astype(np.uint32))
        ids_buf = WebGPUBuffer.from_numpy(dev, input_ids.astype(np.uint32))
        bt_buf  = WebGPUBuffer.from_numpy(dev, self._bt_arr(attn_metadata))

        _rms = self._rms_consts

        def gemm_batch(x_b: WebGPUBuffer, w_key: str,
                       out_b: WebGPUBuffer, K_in: int, N_out: int) -> None:
            """Dispatch matmul_quant_mr4: out[T, N_out] = x[T, K_in] @ w[N_out, K_in].T."""
            uq = self._uq_for_key(w_key)
            if uq == 3:
                sc_b = self._scales_buf(w_key, uq, self._dummy_scales_buf)
                self._dispatch("matmul_quant_mr4",
                               [x_b, self.weights[w_key], sc_b, out_b],
                               {"K": K_in, "N": N_out, "M": T, "USE_QUANT": 3,
                                **self._quant_extra(w_key.removesuffix(".weight"), uq)},
                               (N_out, T, 1))
            elif uq == 0:
                self._dispatch("matmul_quant_mr4",
                               [x_b, self.weights[w_key], self._dummy_scales_buf, out_b],
                               {"K": K_in, "N": N_out, "M": T, "USE_QUANT": 0},
                               (N_out, T, 1))
            else:
                raise RuntimeError(
                    f"Batch prefill does not support USE_QUANT={uq} for weight {w_key}. "
                    f"Only f16 (USE_QUANT=0) and GPTQ int4 (USE_QUANT=3) are handled by "
                    f"matmul_quant_mr4. Other quant types must use the sequential path."
                )

        _CHUNK   = 4
        _hstate  = 0
        _freq_buf = self._rope_freq_buf
        _g4_rope_base = self._g4_rope_base

        for chunk_idx, chunk_layers in enumerate(batched(range(self.num_layers), _CHUNK)):
            with self._batched_dispatch():
                if chunk_idx == 0:
                    self._dispatch("embedding_lookup_f32",
                                   [self.weights[self._embed_key()], ids_buf, b["x"]],
                                   {"HIDDEN_DIM": hidden}, (T, 1, 1))
                    self._dispatch(
                        "rms_norm_f32in",
                        [b["x"],
                         self.weights[f"{self._layer_key_prefix(0)}.input_layernorm.weight"],
                         b["normed"]],
                        _rms, (T, 1, 1))
                    normed_x = b["normed"]
                    x_res    = b["x"]

                for i in chunk_layers:
                    lp              = self._lp[i]
                    p               = self._layer_key_prefix(i)
                    q_dim           = lp["q_dim"]
                    kv_dim          = lp["kv_dim"]
                    head_dim        = lp["head_dim"]
                    num_kv_heads    = lp["num_kv_heads"]
                    has_v           = lp["has_v_proj"]
                    inter           = lp["intermediate_size"]
                    is_kv_shared    = lp.get("is_kv_shared", False)
                    kv_shared_target = lp.get("kv_shared_target", -1)
                    add_n           = T * hidden
                    gelu_n          = T * inter
                    _ls             = self._layer_scales[i]

                    residual = b[_H_NAMES[(_hstate + 1) % 3]]
                    out_h    = b[_H_NAMES[(_hstate + 2) % 3]]

                    qw = f"{p}.self_attn.q_proj.weight"
                    kw = f"{p}.self_attn.k_proj.weight"

                    # QKV projections (always separate in batch path — no fused_qkv).
                    # For KV-shared layers only Q is used; K and V come from the target cache.
                    gemm_batch(normed_x, qw, b["q_buf"], hidden, q_dim)
                    if not is_kv_shared:
                        gemm_batch(normed_x, kw, b["k_buf"], hidden, kv_dim)
                        if has_v:
                            gemm_batch(normed_x, f"{p}.self_attn.v_proj.weight",
                                       b["v_buf"], hidden, kv_dim)
                            v_src = b["v_buf"]
                        else:
                            v_src = b["k_buf"]   # global attention: V = K (pre-RoPE)
                    # Resolve which KV pool slot to read/write.
                    # KV-shared layers use the target layer's cache; non-shared use their own.
                    _kv_layer = kv_shared_target if (is_kv_shared and kv_shared_target >= 0) else i
                    k_cache, v_cache = self.kv_pool[_kv_layer]

                    # Per-head RMSNorm + RoPE for Q and K
                    q_norm_w  = self.weights.get(f"{p}.self_attn.q_norm.weight")
                    k_norm_wl = self.weights.get(f"{p}.self_attn.k_norm.weight")

                    if is_kv_shared:
                        # KV-shared layer: Q norm + RoPE only.
                        # K norm, K RoPE, V norm, and KV cache store are skipped.
                        # K and V come from the target layer's paged cache (k_cache, v_cache).
                        if q_norm_w is not None:
                            self._dispatch(
                                "fused_per_head_norm_rope",
                                [b["q_buf"], q_norm_w, pos_buf, b["q_rope"], _freq_buf],
                                {**_g4_rope_base, "HEAD_DIM": head_dim,
                                 "NUM_HEADS": self.num_q_heads, "HAS_WEIGHT": 1,
                                 "GEMMA_NORM": self._GEMMA_NORM, "INPUT_OFFSET": 0},
                                (self.num_q_heads, T, 1))
                        else:
                            self._dispatch(
                                "rope",
                                [b["q_buf"], pos_buf, b["q_rope"], _freq_buf],
                                {**_g4_rope_base, "HEAD_DIM": head_dim,
                                 "NUM_HEADS": self.num_q_heads},
                                (T, self.num_q_heads, 1))
                        # Load the target layer's cached K and V into dense buffers so that
                        # flash_attn_prefill can consume them (flash_attn_prefill needs dense K/V).
                        self._dispatch(
                            "kv_cache_load_dense",
                            [k_cache, v_cache, slot_map_buf, b["k_rope"], b["v_normed"]],
                            {"BLOCK_SIZE":   self.block_size,
                             "NUM_KV_HEADS": num_kv_heads,
                             "HEAD_DIM":     head_dim},
                            (T, num_kv_heads, 1))
                        v_for_attn = b["v_normed"]
                    else:
                        # Non-shared: Q + K norm + RoPE, V norm, KV cache store.
                        if q_norm_w is not None and k_norm_wl is not None:
                            # K_SEPARATE=1: Q and K are in separate buffers.
                            # Binding 0=Q buf, binding 6=K buf, INPUT_OFFSET_K=0.
                            self._dispatch(
                                "fused_qk_norm_rope",
                                [b["q_buf"], q_norm_w, k_norm_wl, pos_buf,
                                 b["q_rope"], b["k_rope"], b["k_buf"], _freq_buf],
                                {**_g4_rope_base,
                                 "HEAD_DIM":      head_dim,
                                 "NUM_Q_HEADS":   self.num_q_heads,
                                 "NUM_KV_HEADS":  num_kv_heads,
                                 "HAS_WEIGHT":    1,
                                 "GEMMA_NORM":    self._GEMMA_NORM,
                                 "INPUT_OFFSET_K": 0,
                                 "K_SEPARATE":    1},
                                (self.num_q_heads + num_kv_heads, T, 1))
                        else:
                            for src, dst, n_heads, wk in [
                                (b["q_buf"], b["q_rope"], self.num_q_heads,
                                 f"{p}.self_attn.q_norm.weight"),
                                (b["k_buf"], b["k_rope"], num_kv_heads,
                                 f"{p}.self_attn.k_norm.weight"),
                            ]:
                                nw = self.weights.get(wk)
                                if nw is not None:
                                    self._dispatch(
                                        "fused_per_head_norm_rope",
                                        [src, nw, pos_buf, dst, _freq_buf],
                                        {**_g4_rope_base, "HEAD_DIM": head_dim,
                                         "NUM_HEADS": n_heads, "HAS_WEIGHT": 1,
                                         "GEMMA_NORM": self._GEMMA_NORM,
                                         "INPUT_OFFSET": 0},
                                        (n_heads, T, 1))
                                else:
                                    self._dispatch(
                                        "rope",
                                        [src, pos_buf, dst, _freq_buf],
                                        {**_g4_rope_base, "HEAD_DIM": head_dim,
                                         "NUM_HEADS": n_heads},
                                        (T, n_heads, 1))

                        # Per-head RMS norm on V before caching (Gemma4 only; not Gemma3).
                        # v_src is b["v_buf"] for local layers and b["k_buf"] for global;
                        # both are standalone buffers so V_IN_OFFSET=0.
                        if self._apply_v_norm:
                            self._dispatch(
                                "per_head_rms_norm_no_weight",
                                [v_src, b["v_normed"]],
                                {"HEAD_DIM": head_dim, "NUM_HEADS": num_kv_heads,
                                 "WG_SIZE": min(head_dim, 128), "V_IN_OFFSET": 0},
                                (num_kv_heads, T, 1), shader_subdir="gemma")
                            v_for_attn = b["v_normed"]
                        else:
                            v_for_attn = v_src

                        # KV cache store (paged, for subsequent decode steps).
                        # V_IN_OFFSET=0: v_for_attn is always a standalone buffer here.
                        self._dispatch(
                            "kv_cache_store_both",
                            [b["k_rope"], k_cache, v_for_attn, v_cache, slot_map_buf],
                            {"BLOCK_SIZE":   self.block_size,
                             "NUM_KV_HEADS": num_kv_heads,
                             "HEAD_DIM":     head_dim,
                             "V_IN_OFFSET":  0},
                            (T, num_kv_heads, 1))

                    # Causal attention over all T query tokens (dense Q/K/V, not paged).
                    # flash_attn_prefill applies causal masking: token t_q attends to [0, t_q].
                    # For KV-shared layers, b["k_rope"] and b["v_normed"] were loaded from
                    # the target layer's paged cache via kv_cache_load_dense above.
                    self._dispatch(
                        "flash_attn_prefill",
                        [b["q_rope"], b["k_rope"], v_for_attn, b["attn_out"]],
                        {"NUM_Q_HEADS":  self.num_q_heads,
                         "NUM_KV_HEADS": num_kv_heads,
                         "HEAD_DIM":     head_dim,
                         "NUM_T":        T,
                         "SCALE":        1.0 if self._apply_v_norm else (1.0 / head_dim ** 0.5)},
                        (self.num_q_heads, T, 1))

                    # Output projection (batch GEMM)
                    gemm_batch(b["attn_out"], f"{p}.self_attn.o_proj.weight",
                               b["o_proj"], q_dim, hidden)

                    # Post-attention norm + residual add + pre-FFN norm.
                    # Gemma4 correct sublayer order:
                    #   residual = x
                    #   hidden   = input_layernorm(x) → attn → o_proj
                    #   hidden   = post_attention_layernorm(hidden)
                    #   residual = residual + hidden
                    #   ffn_in   = pre_feedforward_layernorm(residual)
                    post_attn_w = self.weights.get(
                        f"{p}.post_attention_layernorm.weight")
                    pre_ffn_w   = self.weights.get(
                        f"{p}.pre_feedforward_layernorm.weight")

                    if post_attn_w is not None and pre_ffn_w is not None:
                        self._dispatch(
                            "rms_norm_add_f32_rms_norm",
                            [b["o_proj"], post_attn_w,
                             x_res, pre_ffn_w,
                             residual, b["normed"]],
                            _rms, (T, 1, 1))
                        ffn_normed = b["normed"]
                    else:
                        if post_attn_w is not None:
                            self._dispatch(
                                "rms_norm",
                                [b["o_proj"], post_attn_w, b["ffn_n"]],
                                _rms, (T, 1, 1))
                            attn_delta = b["ffn_n"]
                        else:
                            attn_delta = b["o_proj"]
                        if pre_ffn_w is not None:
                            self._dispatch(
                                "add_f32_rms_norm",
                                [x_res, attn_delta, pre_ffn_w, residual, b["normed"]],
                                _rms, (T, 1, 1))
                            ffn_normed = b["normed"]
                        else:
                            raise ValueError(
                                f"Layer {i} missing pre_feedforward_layernorm.weight "
                                "— f32 residual cannot be fed to f16 FFN projection"
                            )

                    # FFN gate + up projections (batch GEMM) + tanh-GELU activation
                    gw_k = f"{p}.mlp.gate_proj.weight"
                    uw_k = f"{p}.mlp.up_proj.weight"
                    gemm_batch(ffn_normed, gw_k, b["gate_buf"], hidden, inter)
                    gemm_batch(ffn_normed, uw_k, b["up_buf"],   hidden, inter)
                    self._dispatch(
                        "gelu_mul",
                        [b["gate_buf"], b["up_buf"], b["ffn_act"]],
                        {"N": gelu_n},
                        _vec4_wg(gelu_n),
                        shader_subdir="gemma")

                    # FFN down projection (batch GEMM)
                    gemm_batch(b["ffn_act"], f"{p}.mlp.down_proj.weight",
                               b["ffn_out"], inter, hidden)

                    # Post-FFN norm + residual add (+ pre-norm for next layer if not last)
                    post_ffw_w = self.weights.get(
                        f"{p}.post_feedforward_layernorm.weight")
                    if i < self.num_layers - 1:
                        next_w = self.weights[
                            f"{self._layer_key_prefix(i + 1)}.input_layernorm.weight"]
                        if post_ffw_w is not None:
                            self._dispatch(
                                "rms_norm_add_f32_rms_norm",
                                [b["ffn_out"], post_ffw_w,
                                 residual, next_w,
                                 out_h, b["normed"]],
                                _rms, (T, 1, 1))
                        else:
                            self._dispatch(
                                "add_f32_rms_norm",
                                [residual, b["ffn_out"], next_w, out_h, b["normed"]],
                                _rms, (T, 1, 1))
                    else:
                        # Last layer: no next pre-norm, just update residual.
                        if post_ffw_w is not None:
                            self._dispatch(
                                "rms_norm", [b["ffn_out"], post_ffw_w, b["o_proj"]],
                                _rms, (T, 1, 1))
                            ffn_delta = b["o_proj"]
                        else:
                            ffn_delta = b["ffn_out"]
                        self._dispatch(
                            "add_f32", [residual, ffn_delta, out_h],
                            {"N": add_n}, _vec4_wg(add_n))

                    # Apply layer_scalar to the full f32 residual (matches vLLM).
                    if abs(_ls - 1.0) > 1e-6:
                        self._dispatch(
                            "f32_scale_inplace", [out_h],
                            {"N": add_n, "SCALE": _ls},
                            ((add_n + 255) // 256, 1, 1))

                    normed_x = b["normed"]
                    x_res    = out_h
                    _hstate  = (_hstate + 2) % 3

        # Extract last token, apply final norm, run LM head.
        # copy_buffer_to_buffer is a GPU-side operation (no CPU round-trip).
        _lm_key = self._lm_head_key()
        lm_head_w = self.weights[_lm_key]
        uq_lm = self._uq_for_key(_lm_key)
        _lm_base = _lm_key.removesuffix('.weight')
        with self._batched_dispatch():
            last_byte_offset = (T - 1) * hidden * 4   # f32: 4 bytes per element
            enc = self._active_encoder
            assert enc is not None
            enc.copy_buffer_to_buffer(
                x_res.buf, last_byte_offset,
                b["last_f32"].buf, 0,
                hidden * 4,
            )
            self._dispatch(
                "rms_norm_f32in",
                [b["last_f32"], self.weights[self._norm_key()], b["last_norm"]],
                _rms, (1, 1, 1))
            self._dispatch(
                "matmul_quant",
                [b["last_norm"], lm_head_w,
                 self._scales_buf(_lm_key, uq_lm, self._dummy_scales_buf), b["logits"]],
                {"K": hidden, "N": vocab, "USE_QUANT": uq_lm, "SPLIT_K": 0,
                 **self._quant_extra(_lm_base, uq_lm)},
                _rows_wg(vocab))

            if self.softcap is not None and self.softcap > 0:
                self._dispatch(
                    "logit_softcap", [b["logits"], b["capped"]],
                    {"VOCAB": vocab, "CAP": float(self.softcap)},
                    _rows_wg(vocab),
                    shader_subdir="gemma")
                result_buf = b["capped"]
            else:
                result_buf = b["logits"]

            self._dispatch("argmax_f16", [result_buf, self._ensure_sample_buf()],
                           {"N": vocab}, (1, 1, 1))
            self._copy_sample_to_staging()

        self._last_logit_buf = result_buf
        self._last_vocab     = vocab
        tok = self._read_sample_tok()
        return np.array([[tok]], dtype=np.int32)

    def _prefill_sequential_fallback(
        self,
        input_ids: "np.ndarray",
        positions: "np.ndarray",
        attn_metadata: object,
        T: int,
    ) -> "np.ndarray":
        """Process T prefill tokens one at a time via the decode-path infrastructure.

        Used when projection weights use a quant format not supported by
        matmul_quant_mr4 (anything other than f16 or GPTQ int4). Each token runs
        through _transformer_layer with num_tokens=1, building the KV cache
        incrementally to maintain causal attention. Only the last token's logits
        are returned.
        """
        dev = self.wgpu_device.wgpu_device
        pre = self._pre
        sc  = self._sc
        hidden = self.hidden_size
        vocab  = self.vocab_size
        _rms   = self._rms_consts
        bt_arr = self._bt_arr(attn_metadata)

        for t in range(T):
            self._hstate = 0
            tok_pos = int(positions[t])
            tok_ctx = tok_pos + 1

            ids_t  = input_ids[t : t + 1]
            pos_t  = positions[t : t + 1]
            slot_t = np.array(attn_metadata.slot_mapping[t : t + 1], dtype=np.uint32)

            dev.queue.write_buffer(pre["ids"].buf,      0, ids_t.astype(np.uint32).tobytes())
            dev.queue.write_buffer(pre["pos"].buf,      0, pos_t.astype(np.uint32).tobytes())
            dev.queue.write_buffer(pre["slot_map"].buf, 0, slot_t.tobytes())
            dev.queue.write_buffer(pre["bt"].buf,       0, bt_arr.tobytes())

            with self._batched_dispatch():
                self._dispatch(
                    "embedding_lookup_f32",
                    [self.weights[self._embed_key()], pre["ids"], pre["x"]],
                    {"HIDDEN_DIM": hidden}, (1, 1, 1))
                self._dispatch(
                    "rms_norm_f32in",
                    [pre["x"],
                     self.weights[f"{self._layer_key_prefix(0)}.input_layernorm.weight"],
                     sc["normed"]],
                    _rms, (1, 1, 1))

            normed_x = sc["normed"]
            x_buf    = pre["x"]
            for layer_idx in range(self.num_layers):
                normed_x, x_buf = self._transformer_layer(
                    layer_idx, normed_x, x_buf,
                    pre["pos"], pre["slot_map"], pre["bt"], tok_ctx, 1,
                )

        # Final norm and LM head on the last token's hidden state.
        _lm_key = self._lm_head_key()
        lm_head_w = self.weights[_lm_key]
        uq_lm = self._uq_for_key(_lm_key)
        _lm_base = _lm_key.removesuffix('.weight')
        with self._batched_dispatch():
            self._dispatch(
                "rms_norm_f32in",
                [x_buf, self.weights[self._norm_key()], pre["norm_out"]],
                _rms, (1, 1, 1))
            self._dispatch(
                "matmul_quant",
                [pre["norm_out"], lm_head_w,
                 self._scales_buf(_lm_key, uq_lm, self._dummy_scales_buf), pre["logits"]],
                {"K": hidden, "N": vocab, "USE_QUANT": uq_lm, "SPLIT_K": 0,
                 **self._quant_extra(_lm_base, uq_lm)},
                _rows_wg(vocab))

            if self.softcap is not None and self.softcap > 0:
                self._dispatch(
                    "logit_softcap", [pre["logits"], pre["capped"]],
                    {"VOCAB": vocab, "CAP": float(self.softcap)},
                    _rows_wg(vocab),
                    shader_subdir="gemma")
                result_buf = pre["capped"]
            else:
                result_buf = pre["logits"]

            self._dispatch("argmax_f16", [result_buf, self._ensure_sample_buf()],
                           {"N": vocab}, (1, 1, 1))
            self._copy_sample_to_staging()

        self._last_logit_buf = result_buf
        self._last_vocab     = vocab
        tok = self._read_sample_tok()
        return np.array([[tok]], dtype=np.int32)

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
    ) -> "tuple[WebGPUBuffer, WebGPUBuffer]":
        """Returns (normed_out, raw_out) — normed_out is sc['normed'] for next layer."""
        sc = self._sc
        lp = self._lp[layer_idx]
        hidden = self.hidden_size
        p = self._layer_key_prefix(layer_idx)
        q_dim = lp["q_dim"]
        kv_dim = lp["kv_dim"]
        head_dim = lp["head_dim"]
        num_kv_heads = lp["num_kv_heads"]
        has_v = lp["has_v_proj"]
        inter = lp["intermediate_size"]
        # Per-layer scalar from GGUF (layer_scalar weight, e.g. ~0.97 or ~0.053 depending on model).
        # Applied to the full residual once after both attn and FFN sublayers, matching vLLM:
        #   hidden_states = hidden_states * self.layer_scalar
        # Cached at load_weights() — no GPU-to-CPU readback per token.
        _ls = self._layer_scales[layer_idx]
        # KV-shared layers reuse the target layer's KV cache (mirrors vLLM is_kv_shared_layer).
        is_kv_shared    = lp.get("is_kv_shared", False)
        kv_shared_target = lp.get("kv_shared_target", -1)

        residual = sc[_H_NAMES[(self._hstate + 1) % 3]]
        out = sc[_H_NAMES[(self._hstate + 2) % 3]]
        add_n = num_tokens * hidden
        gelu_n = num_tokens * inter
        _rms_consts = self._rms_consts

        _kv_layer = kv_shared_target if (is_kv_shared and kv_shared_target >= 0) else layer_idx
        k_cache, v_cache = self.kv_pool[_kv_layer]

        with self._batched_dispatch(label=f"L{layer_idx:02d}"):
            # normed_x already pre-normalized by caller (or previous layer's fused add_f32_rms_norm).

            # QKV projections: fused for f16 local layers; separate for quantized weights or
            # global layers (has_v=False, no v_proj weight).
            qw = f"{p}.self_attn.q_proj.weight"
            kw = f"{p}.self_attn.k_proj.weight"
            uq_q = self._uq_for_key(qw)
            uq_k = self._uq_for_key(kw)
            if has_v:
                vw = f"{p}.self_attn.v_proj.weight"
                uq_v = self._uq_for_key(vw)
                _use_fused_qkv = (uq_q == 0 and uq_k == 0 and uq_v == 0)
            else:
                vw = None
                _use_fused_qkv = False

            if _use_fused_qkv and not is_kv_shared:
                # All f16, non-shared: single fused_qkv dispatch → sc["qkv_buf"] laid out as [Q | K | V].
                self._dispatch("fused_qkv",
                               [normed_x, self.weights[qw], self.weights[kw], self.weights[vw],
                                sc["qkv_buf"]],
                               {"K": hidden, "Q_DIM": q_dim, "KV_DIM": kv_dim},
                               (q_dim + 2 * kv_dim, 1, 1))
                _q_src = sc["qkv_buf"]       # Q at element offset 0
                _k_src = sc["qkv_buf"]       # K at element offset q_dim
                _v_src = sc["qkv_buf"]       # V at element offset q_dim + kv_dim
                _v_src_offset = q_dim + kv_dim
            else:
                # Separate projections (quantized weights, global attention, or KV-shared layer).
                # KV-shared layers only need Q; K and V come from the target layer's KV cache.
                uq = uq_q
                self._dispatch("matmul_quant",
                               [normed_x, self.weights[qw],
                                self._scales_buf(qw, uq, self._dummy_scales_buf), sc["q_buf"]],
                               {"K": hidden, "N": q_dim, "USE_QUANT": uq,
                                **self._quant_extra(f"{p}.self_attn.q_proj", uq)},
                               _gemv_wg(q_dim))
                if not is_kv_shared:
                    uq = uq_k
                    self._dispatch("matmul_quant",
                                   [normed_x, self.weights[kw],
                                    self._scales_buf(kw, uq, self._dummy_scales_buf), sc["k_buf"]],
                                   {"K": hidden, "N": kv_dim, "USE_QUANT": uq,
                                    **self._quant_extra(f"{p}.self_attn.k_proj", uq)},
                                   _gemv_wg(kv_dim))
                    if has_v:
                        uq = uq_v
                        self._dispatch("matmul_quant",
                                       [normed_x, self.weights[vw],
                                        self._scales_buf(vw, uq, self._dummy_scales_buf), sc["v_buf"]],
                                       {"K": hidden, "N": kv_dim, "USE_QUANT": uq,
                                        **self._quant_extra(f"{p}.self_attn.v_proj", uq)},
                                       _gemv_wg(kv_dim))
                        _v_src = sc["v_buf"]
                    else:
                        _v_src = sc["k_buf"]  # global attention: V = K
                    _k_src = sc["k_buf"]
                else:
                    pass  # KV-shared path: _k_src/_v_src are not used
                _q_src = sc["q_buf"]
                _v_src_offset = 0

            # Per-head RMSNorm + RoPE for Q and K.
            # When both norm weights exist: fused_qk_norm_rope handles both in one dispatch.
            #   Fused QKV path: K_SEPARATE=0, both Q and K in qkv_buf (INPUT_OFFSET_K=q_dim).
            #   Separate buffer path: K_SEPARATE=1, Q in q_buf (binding 0), K in k_buf (binding 6).
            q_norm_w = self.weights.get(f"{p}.self_attn.q_norm.weight")
            k_norm_w_l = self.weights.get(f"{p}.self_attn.k_norm.weight")
            _freq_buf = self._rope_freq_buf
            _g4_rope_base = self._g4_rope_base

            if is_kv_shared:
                # KV-shared layer (last N sliding-attention layers in laptop Gemma4 variant):
                # Q gets norm + RoPE; K norm, K RoPE, V norm, and KV cache store are all
                # skipped. K and V for attention come from the target layer's KV cache.
                # Matches vLLM Gemma4Attention.forward when is_kv_shared_layer=True.
                if q_norm_w is not None:
                    self._dispatch("fused_per_head_norm_rope",
                                   [_q_src, q_norm_w, pos_buf, sc["q_rope"], _freq_buf],
                                   {**_g4_rope_base, "HEAD_DIM": head_dim,
                                    "NUM_HEADS": self.num_q_heads, "HAS_WEIGHT": 1,
                                    "GEMMA_NORM": self._GEMMA_NORM, "INPUT_OFFSET": 0},
                                   (self.num_q_heads, num_tokens, 1))
                elif _use_fused_qkv:
                    self._dispatch("fused_per_head_norm_rope",
                                   [_q_src, _q_src, pos_buf, sc["q_rope"], _freq_buf],
                                   {**_g4_rope_base, "HEAD_DIM": head_dim,
                                    "NUM_HEADS": self.num_q_heads, "HAS_WEIGHT": 0,
                                    "GEMMA_NORM": 0, "INPUT_OFFSET": 0},
                                   (self.num_q_heads, num_tokens, 1))
                else:
                    self._dispatch("rope", [_q_src, pos_buf, sc["q_rope"], _freq_buf],
                                   {**_g4_rope_base, "HEAD_DIM": head_dim,
                                    "NUM_HEADS": self.num_q_heads},
                                   (num_tokens, self.num_q_heads, 1))
            else:
                # Non-shared: Q + K norm + RoPE, then V norm and KV cache store.
                if q_norm_w is not None and k_norm_w_l is not None:
                    _k_separate = 0 if _use_fused_qkv else 1
                    _k_in_offset = q_dim if _use_fused_qkv else 0
                    _k_bind = sc["qkv_buf"] if _use_fused_qkv else sc["k_buf"]
                    self._dispatch("fused_qk_norm_rope",
                                   [_q_src, q_norm_w, k_norm_w_l, pos_buf,
                                    sc["q_rope"], sc["k_rope"], _k_bind, _freq_buf],
                                   {**_g4_rope_base,
                                    "HEAD_DIM": head_dim,
                                    "NUM_Q_HEADS": self.num_q_heads,
                                    "NUM_KV_HEADS": num_kv_heads,
                                    "HAS_WEIGHT": 1,
                                    "GEMMA_NORM": self._GEMMA_NORM,
                                    "INPUT_OFFSET_K": _k_in_offset,
                                    "K_SEPARATE": _k_separate},
                                   (self.num_q_heads + num_kv_heads, num_tokens, 1))
                else:
                    # Fallback: separate per-head norm+rope or plain rope for each of Q and K.
                    _k_in_off = q_dim if _use_fused_qkv else 0
                    for src, dst, n_heads, w_key, in_off in [
                        (_q_src, sc["q_rope"], self.num_q_heads, f"{p}.self_attn.q_norm.weight", 0),
                        (_k_src, sc["k_rope"], num_kv_heads, f"{p}.self_attn.k_norm.weight", _k_in_off),
                    ]:
                        norm_w = self.weights.get(w_key)
                        if norm_w is not None:
                            self._dispatch("fused_per_head_norm_rope",
                                           [src, norm_w, pos_buf, dst, _freq_buf],
                                           {**_g4_rope_base, "HEAD_DIM": head_dim,
                                            "NUM_HEADS": n_heads, "HAS_WEIGHT": 1,
                                            "GEMMA_NORM": self._GEMMA_NORM,
                                            "INPUT_OFFSET": in_off},
                                           (n_heads, num_tokens, 1))
                        elif not _use_fused_qkv:
                            # Plain rope from standalone buffer (no offset needed).
                            self._dispatch("rope", [src, pos_buf, dst, _freq_buf],
                                           {**_g4_rope_base, "HEAD_DIM": head_dim,
                                            "NUM_HEADS": n_heads},
                                           (num_tokens, n_heads, 1))
                        else:
                            # Rope from fused QKV buffer at in_off — use HAS_WEIGHT=0 variant.
                            self._dispatch("fused_per_head_norm_rope",
                                           [src, src, pos_buf, dst, _freq_buf],
                                           {**_g4_rope_base, "HEAD_DIM": head_dim,
                                            "NUM_HEADS": n_heads, "HAS_WEIGHT": 0,
                                            "GEMMA_NORM": 0, "INPUT_OFFSET": in_off},
                                           (n_heads, num_tokens, 1))

                # Per-head RMSNorm (no weight) on V before caching — Gemma4 only.
                # Gemma3 does NOT apply V normalization (no v_norm weight in the model).
                # V_IN_OFFSET is non-zero when V lives inside the fused qkv_buf.
                if self._apply_v_norm:
                    self._dispatch("per_head_rms_norm_no_weight", [_v_src, sc["v_normed"]],
                                   {"HEAD_DIM": head_dim, "NUM_HEADS": num_kv_heads,
                                    "WG_SIZE": min(head_dim, 128),
                                    "V_IN_OFFSET": _v_src_offset},
                                   (num_kv_heads, num_tokens, 1), shader_subdir="gemma")
                    v_to_cache = sc["v_normed"]
                    _v_cache_offset = 0
                else:
                    v_to_cache = _v_src  # Gemma3: use V directly without normalization
                    _v_cache_offset = _v_src_offset  # 0 when not fused; q_dim+kv_dim when fused

                # Fused K+V cache store — single dispatch saves 1 overhead per layer.
                self._dispatch("kv_cache_store_both",
                               [sc["k_rope"], k_cache, v_to_cache, v_cache, slot_map],
                               {"BLOCK_SIZE": self.block_size, "NUM_KV_HEADS": num_kv_heads,
                                "HEAD_DIM": head_dim, "V_IN_OFFSET": _v_cache_offset},
                               (num_tokens, num_kv_heads, 1))

            # Always use flash_attn_decode for single-token decode.
            # The 65535 limit applied to attn_score's dispatch dimension; flash_attn_decode
            # loops internally and has no dispatch dimension limit.
            self._dispatch("flash_attn_decode",
                           [sc["q_rope"], k_cache, v_cache, bt_buf, sc["attn_out"]],
                           {"BLOCK_SIZE": self.block_size, "NUM_Q_HEADS": self.num_q_heads,
                            "NUM_KV_HEADS": num_kv_heads, "HEAD_DIM": head_dim,
                            "CTX_LEN": ctx_len,
                            "SCALE": 1.0 if self._apply_v_norm else (1.0 / head_dim ** 0.5)},
                           (self.num_q_heads, 1, 1))

            # Output projection → sc["o_proj_out"]
            ow = f"{p}.self_attn.o_proj.weight"
            uq = self._uq_for_key(ow)
            self._dispatch("matmul_quant",
                           [sc["attn_out"], self.weights[ow],
                            self._scales_buf(ow, uq, self._dummy_scales_buf), sc["o_proj_out"]],
                           {"K": q_dim, "N": hidden, "USE_QUANT": uq,
                            **self._quant_extra(f"{p}.self_attn.o_proj", uq)},
                           _gemv_wg(hidden))

            # Correct Gemma4 attention sublayer (matches HF Gemma3DecoderLayer.forward):
            #   residual = x
            #   hidden = input_layernorm(x)     → attn → o_proj
            #   hidden = post_attention_layernorm(hidden)   ← norm on ATTN OUTPUT (before residual)
            #   residual = residual + hidden                ← residual add AFTER norm
            #
            # When both post_attn_norm and pre_ffn_norm are present, fuse them with
            # rms_norm_add_f32_rms_norm to keep the intermediate in registers only.
            post_attn_norm_w = self.weights.get(f"{p}.post_attention_layernorm.weight")
            pre_ffn_norm_w = self.weights.get(f"{p}.pre_feedforward_layernorm.weight")
            if post_attn_norm_w is not None and pre_ffn_norm_w is not None:
                # Fused: rms_norm(o_proj_out, post_attn_w) + residual_add + rms_norm(residual, pre_ffn_w)
                # Eliminates 1 dispatch vs the rms_norm → add_f32_rms_norm pair.
                # No SCALE here: vLLM does not apply layer_scalar at the attention sublayer.
                self._dispatch("rms_norm_add_f32_rms_norm",
                               [sc["o_proj_out"], post_attn_norm_w,
                                x_buf, pre_ffn_norm_w,
                                residual, sc["normed"]],
                               _rms_consts, (num_tokens, 1, 1))
                ffn_normed = sc["normed"]
            else:
                if post_attn_norm_w is not None:
                    self._dispatch("rms_norm", [sc["o_proj_out"], post_attn_norm_w, sc["ffn_normed"]],
                                   _rms_consts, (num_tokens, 1, 1))
                    attn_delta = sc["ffn_normed"]
                else:
                    attn_delta = sc["o_proj_out"]

                if pre_ffn_norm_w is not None:
                    self._dispatch("add_f32_rms_norm",
                                   [x_buf, attn_delta, pre_ffn_norm_w, residual, sc["normed"]],
                                   _rms_consts, (num_tokens, 1, 1))
                    ffn_normed = sc["normed"]
                else:
                    raise ValueError(
                        f"Layer {layer_idx} missing pre_feedforward_layernorm.weight "
                        "— f32 residual cannot be fed to f16 FFN projection"
                    )

            # Gate + up projection
            # Fused gate+up (f16 only); Gemma uses tanh-GELU.
            gw_k = f"{p}.mlp.gate_proj.weight"
            uw_k = f"{p}.mlp.up_proj.weight"
            uq_g = self._uq_for_key(gw_k)
            uq_u = self._uq_for_key(uw_k)
            if uq_g == 0 and uq_u == 0:
                self._dispatch("fused_gate_act",
                               [ffn_normed, self.weights[gw_k], self.weights[uw_k],
                                sc["ffn_act"]],
                               {"K": hidden, "N": inter, "GELU": 1}, (inter, 1, 1))
            else:
                for out_b, proj, w_k, uq2 in [
                        (sc["gate_buf"], "gate_proj", gw_k, uq_g),
                        (sc["up_buf"],   "up_proj",   uw_k, uq_u)]:
                    self._dispatch("matmul_quant",
                                   [ffn_normed, self.weights[w_k],
                                    self._scales_buf(w_k, uq2, self._dummy_scales_buf), out_b],
                                   {"K": hidden, "N": inter, "USE_QUANT": uq2,
                                    **self._quant_extra(f"{p}.mlp.{proj}", uq2)},
                                   _gemv_wg(inter))
                self._dispatch("gelu_mul", [sc["gate_buf"], sc["up_buf"], sc["ffn_act"]],
                               {"N": gelu_n}, _vec4_wg(gelu_n),
                               shader_subdir="gemma")

            # Down projection → sc["ffn_out"]
            w_k = f"{p}.mlp.down_proj.weight"
            uq = self._uq_for_key(w_k)
            self._dispatch("matmul_quant",
                           [sc["ffn_act"], self.weights[w_k],
                            self._scales_buf(w_k, uq, self._dummy_scales_buf), sc["ffn_out"]],
                           {"K": inter, "N": hidden, "USE_QUANT": uq,
                            **self._quant_extra(f"{p}.mlp.down_proj", uq)},
                           _gemv_wg(hidden))

            # Post-FFN norm on FFN output (before residual add), then fused residual + next pre-norm.
            # When post_ffw_w and next input_layernorm both exist (all non-last layers),
            # fuse into rms_norm_add_f32_rms_norm to keep the intermediate in registers.
            post_ffw_w = self.weights.get(f"{p}.post_feedforward_layernorm.weight")
            if layer_idx < self.num_layers - 1:
                next_w = self.weights[f"{self._layer_key_prefix(layer_idx + 1)}.input_layernorm.weight"]
                if post_ffw_w is not None:
                    # Fused: rms_norm(ffn_out, post_ffw_w) + residual_add + rms_norm(residual, next_w)
                    # SCALE=1.0 (default): layer_scalar applied separately below via f32_scale_inplace.
                    # RMSNorm is scale-invariant so normed_out is correct even after scaling out.
                    self._dispatch("rms_norm_add_f32_rms_norm",
                                   [sc["ffn_out"], post_ffw_w,
                                    residual, next_w,
                                    out, sc["normed"]],
                                   _rms_consts, (num_tokens, 1, 1))
                else:
                    self._dispatch("add_f32_rms_norm",
                                   [residual, sc["ffn_out"], next_w, out, sc["normed"]],
                                   _rms_consts, (num_tokens, 1, 1))
            else:
                # Last layer: no next pre-norm, just update residual.
                if post_ffw_w is not None:
                    self._dispatch("rms_norm", [sc["ffn_out"], post_ffw_w, sc["o_proj_out"]],
                                   _rms_consts, (num_tokens, 1, 1))
                    ffn_delta = sc["o_proj_out"]
                else:
                    ffn_delta = sc["ffn_out"]
                self._dispatch("add_f32", [residual, ffn_delta, out],
                               {"N": add_n}, _vec4_wg(add_n))

            # Apply layer_scalar to the full residual once per decoder layer.
            # Matches vLLM: hidden_states = hidden_states * self.layer_scalar,
            # which scales (x + delta_attn + delta_ffn), not just the deltas.
            if abs(_ls - 1.0) > 1e-6:
                self._dispatch("f32_scale_inplace", [out],
                               {"N": add_n, "SCALE": _ls},
                               ((add_n + 255) // 256, 1, 1))

        self._hstate = (self._hstate + 2) % 3
        return sc["normed"], out
