from __future__ import annotations
from functools import partial
from itertools import batched
from typing import TYPE_CHECKING

import numpy as np

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm_webgpu.models.base import BaseWebGPUModel, compute_yarn_freqs, _gemv_wg, _rows_wg, _vals_per_thread, _vec4_wg, _H_NAMES
from vllm_webgpu.webgpu.buffer import WebGPUBuffer

if TYPE_CHECKING:
    from vllm_webgpu.webgpu.device import WebGPUDevice
    from vllm_webgpu.webgpu.pipeline import PipelineCache

logger = init_logger(__name__)


class LlamaWebGPUModel(BaseWebGPUModel):
    """
    Handles Llama 3.x and Qwen 2.5/3.x (architecturally identical).
    Fused dispatch sequence per token:
      embedding_lookup
      -> N x (add_rms_norm -> fused_qkv -> fused_qk_norm_rope ->
              kv_cache_store_both -> flash_attn_decode ->
              matmul_quant(o_proj) -> add_rms_norm ->
              fused_gate_act -> matmul_quant(down_proj) -> add_rms_norm)
      -> rms_norm -> matmul_quant(lm_head) -> argmax_f16
    """

    # GPU argmax path returns (1,1) int32; logit_readback() provides full logits.
    logit_returns_token_id: bool = True

    # Sliding-window size (set by MixtralWebGPUModel); None means full attention.
    _sw: int | None = None
    # MoE flag (set by subclasses such as MixtralWebGPUModel); False in base class.
    _is_moe: bool = False

    def __init__(self, model_config, wgpu_device: "WebGPUDevice", pipeline_cache: "PipelineCache", block_size: int = 16) -> None:
        super().__init__(model_config, wgpu_device, pipeline_cache)
        self.num_layers: int = model_config.num_hidden_layers
        self.num_q_heads: int = model_config.num_attention_heads
        self.num_kv_heads: int = model_config.num_key_value_heads
        self.hidden_size: int = model_config.hidden_size
        self.intermediate_size: int = model_config.intermediate_size
        self.vocab_size: int = model_config.vocab_size
        # Use explicit head_dim when present (e.g. Qwen3: head_dim=128, hidden=2560, heads=32,
        # so hidden//heads=80 but actual Q dim per head is 128).
        self.head_dim: int = getattr(model_config, "head_dim", self.hidden_size // self.num_q_heads)
        self.rope_theta: float = getattr(model_config, "rope_theta", 10000.0)
        self.block_size: int = block_size
        # add.wgsl and gelu_mul.wgsl use vec4<f16>: dimensions must be divisible by 4.
        # The % 4 check subsumes the % 2 check for hidden_size and intermediate_size.
        for name, val in [("hidden_size", self.hidden_size),
                          ("intermediate_size", self.intermediate_size)]:
            if val % 4 != 0:
                raise ValueError(f"{name}={val} must be divisible by 4 for vec4<f16> shaders")
        # matmul_quant f16 path packs two f16 values per u32; head_dim must be even.
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim={self.head_dim} must be even for f16 GEMV")
        max_ctx = getattr(model_config, "max_position_embeddings", 8192)
        # Precompute constants that are used every forward pass.
        self._rms_consts: dict = {"HIDDEN_DIM": self.hidden_size, "VALS_PER_THREAD": _vals_per_thread(self.hidden_size)}
        self._init_scratch_buffers(max_ctx)
        self._init_rope_freq_buf()
        self._rope_consts: dict = {
            "HEAD_DIM": self.head_dim,
            "ROPE_BASE": float(self.rope_theta),
            "LN_ROPE_BASE": float(np.log(self.rope_theta)),
            "USE_FREQ_BUF": int(self._use_freq_buf),
            "ATTN_SCALE": self._yarn_mscale,
        }
        # Pre-register norm tiling transforms so load_weights can tile q_norm/k_norm
        # weights at upload time, avoiding a GPU roundtrip (to_numpy → tile → re-upload).
        # Qwen3 checkpoints store shared norm as (head_dim,); the shader expects
        # (num_heads * head_dim,) with each head using the same values.
        head_dim = self.head_dim
        num_q = self.num_q_heads
        num_kv = self.num_kv_heads

        def _tile_norm(a, n_heads):
            return np.tile(a, n_heads) if a.shape == (head_dim,) else a

        _q_xform = partial(_tile_norm, n_heads=num_q)
        _k_xform = partial(_tile_norm, n_heads=num_kv)
        self._weight_transforms.update({
            f"model.layers.{i}.self_attn.{k}.weight": xf
            for i in range(self.num_layers)
            for k, xf in (("q_norm", _q_xform), ("k_norm", _k_xform))
        })
        # Cached after load_weights: True iff all *_proj weights are USE_QUANT=0 or 3.
        # None means not yet computed (weights not yet loaded).
        self._batch_matmul_supported: bool | None = None

    def _init_scratch_buffers(self, max_ctx: int, qkv_size: "int | None" = None) -> None:
        """Pre-allocate all intermediate scratch buffers used in _transformer_layer.

        Eliminates 17 GPU buffer allocations per layer per decode token.
        Decode path only (num_tokens=1). Sizes are fixed by model dimensions.

        qkv_size: optional override for qkv_buf byte size. Subclasses that need
            a non-standard qkv buffer (e.g. Qwen35 GDN layers) pass this to
            avoid allocating the standard-sized buffer only to immediately replace it.
        """
        T = 1  # decode: num_tokens == 1
        H = self.hidden_size
        I = self.intermediate_size
        Q = self.num_q_heads * self.head_dim
        KV = self.num_kv_heads * self.head_dim

        # Pre-allocated per-step buffers: reused every decode call via write_buffer.
        # Eliminates GPU allocation overhead (~5-10ms per token on Metal).
        max_bt_blocks = max(4096, cdiv(max_ctx, self.block_size))
        self._pre: dict[str, "WebGPUBuffer"] = {
            "ids":      self._make_buf(T * 4),              # [1] uint32 token id
            "pos":      self._make_buf(T * 4),              # [1] uint32 position
            "slot_map": self._make_buf(T * 4),              # [1] uint32 physical slot
            "bt":       self._make_buf(max_bt_blocks * 4),  # [max_blocks] uint32 block table
            "x":        self._make_buf(T * H * 2),          # [1, H] f16 residual / embedding
            "norm_out": self._make_buf(T * H * 2),          # [1, H] f16 final norm output
            "logits":   self._make_buf(T * self.vocab_size * 2),  # [1, vocab] f16 logits
        }

        self._sc: dict[str, "WebGPUBuffer"] = {
            "normed":  self._make_buf(T * H * 2),
            "qkv_buf": self._make_buf(qkv_size or T * (Q + 2 * KV) * 2),  # [Q|K|V] f16
            "q_buf":       self._make_buf(T * Q * 2),
            "k_buf":       self._make_buf(T * KV * 2),
            "v_buf":       self._make_buf(T * KV * 2),
            "q_rope":      self._make_buf(T * Q * 2),
            "k_rope":      self._make_buf(T * KV * 2),
            "attn_out":   self._make_buf(T * Q * 2),
            "o_proj_out": self._make_buf(T * H * 2),
            "ffn_normed": self._make_buf(T * H * 2),
            "gate_buf":   self._make_buf(T * I * 2),
            "up_buf":     self._make_buf(T * I * 2),
            "ffn_act":    self._make_buf(T * I * 2),
            "ffn_out":    self._make_buf(T * H * 2),
            # Three hidden-state buffers: ping-pong between h0/h1/h2 so that
            # x_buf, residual, and out are always distinct within a single layer.
            "h0":         self._make_buf(T * H * 2),
            "h1":         self._make_buf(T * H * 2),
            "h2":         self._make_buf(T * H * 2),
        }
        # Index into hidden-state rotation: the layer output cycles h0 -> h1 -> h2 -> h0 ...
        self._hstate: int = 0

    def _init_rope_freq_buf(self) -> None:
        """Detect YaRN rope_scaling and upload precomputed inverse frequencies to GPU.

        When rope_type == 'yarn', replaces the base-class dummy buffer with actual
        YaRN-scaled frequencies. All other rope types keep the dummy (_use_freq_buf=False).
        """
        _rope_parameters = getattr(self.model_config, "rope_parameters", None)
        rope_scaling = dict(
            _rope_parameters
            if _rope_parameters is not None
            else getattr(self.model_config, "rope_scaling", None)
            or {}
        )
        rope_type = rope_scaling.get("rope_type", "")

        if rope_type != "yarn":
            if rope_type not in ("", "default"):
                logger.warning(
                    "rope_type=%r not implemented; using standard RoPE "
                    "(long-context accuracy reduced beyond 8192 tokens)",
                    rope_type,
                )
            return

        dev = self.wgpu_device.wgpu_device
        # rotary_dim derivation is delegated to compute_yarn_freqs, which mirrors
        # vllm/model_executor/layers/rotary_embedding/__init__.py:66-72.
        freqs, mscale = compute_yarn_freqs(self.head_dim, self.rope_theta, rope_scaling)
        self._rope_freq_buf = WebGPUBuffer.from_numpy(dev, freqs)
        self._yarn_mscale = mscale
        self._use_freq_buf = True
        logger.info(
            "YaRN RoPE: factor=%.1f beta_fast=%.1f beta_slow=%.1f orig_ctx=%d",
            rope_scaling.get("factor", 1.0),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("original_max_position_embeddings", 4096),
        )

    def load_weights(self, path: str, f32_keys: "frozenset[str] | None" = None,
                     skip_prefixes: "frozenset[str] | None" = None,
                     scale_transforms: "dict | None" = None) -> None:
        super().load_weights(path, f32_keys=f32_keys, skip_prefixes=skip_prefixes,
                             scale_transforms=scale_transforms)

        # _uq_for_key returns 0 (f16) or 3 (GPTQ int4) for formats supported by
        # matmul_quant_mr4 batch-prefill path. Any other value (AWQ=4, FP8=5,
        # NVFP4=6, INT8=7, NF4=8) falls back to the sequential decode path.
        # MoE models never reach _prefill_batch_forward, so skip the scan entirely.
        if self._is_moe:
            self._batch_matmul_supported = False
        else:
            proj_keys = [k for k in self.weights if k.endswith('.weight') and 'model.layers.' in k and '_proj' in k]
            self._batch_matmul_supported = proj_keys and all(self._uq_for_key(k) in (0, 3) for k in proj_keys)

    def _decode_setup(
        self,
        input_ids: "np.ndarray",
        positions: "np.ndarray",
        attn_metadata: object,
    ) -> tuple:
        """Write per-step input buffers and return pre-allocated buffer aliases + ctx_len.

        Shared between LlamaWebGPUModel.forward() and MixtralWebGPUModel._moe_decode_forward().
        Returns (ids_buf, pos_buf, slot_map, bt_buf, x_buf, norm_out, logits_buf, ctx_len).
        """
        pre = self._pre
        bt_arr = self._bt_arr(attn_metadata)
        self._write_token_bufs(
            input_ids, positions,
            np.asarray(attn_metadata.slot_mapping, dtype=np.uint32).tobytes(),
            bt_arr.tobytes(),
        )
        ctx_len = int(attn_metadata.max_decode_seq_len)
        return (
            pre["ids"], pre["pos"], pre["slot_map"], pre["bt"],
            pre["x"], pre["norm_out"], pre["logits"], ctx_len,
        )

    def _write_token_bufs(
        self,
        ids_1d: "np.ndarray",
        pos_1d: "np.ndarray",
        slot_1d: bytes,
        bt_bytes: bytes,
    ) -> None:
        """Write the four per-token input buffers (ids, pos, slot_map, bt).

        Extracted so _decode_setup and _prefill_sequential_fallback share the
        same write pattern without duplicating queue.write_buffer calls.
        """
        pre = self._pre
        dev = self.wgpu_device.wgpu_device
        dev.queue.write_buffer(pre["ids"].buf,      0, ids_1d.astype(np.uint32, copy=False).tobytes())
        dev.queue.write_buffer(pre["pos"].buf,      0, pos_1d.astype(np.uint32, copy=False).tobytes())
        dev.queue.write_buffer(pre["slot_map"].buf, 0, slot_1d)
        dev.queue.write_buffer(pre["bt"].buf,       0, bt_bytes)

    def _decode_teardown(
        self,
        norm_out: "WebGPUBuffer",
        logits_buf: "WebGPUBuffer",
        vocab: int,
        greedy: bool,
    ) -> None:
        """Dispatch the LM-head and optionally the GPU argmax + copy-to-staging.

        Must be called inside an active encoder context (either _batched_dispatch
        or a manually managed encoder in MixtralWebGPUModel._moe_decode_forward).
        Sets _last_logit_buf and _last_vocab. The caller is responsible for
        submitting the encoder and reading back the result.
        """
        hidden = self.hidden_size
        lm_key = self._lm_head_key()
        uq = self._uq_for_key(lm_key)
        # Always use SPLIT_K=0 (row-per-thread, ceil(vocab/256) WGs) for the LM
        # head: SPLIT_K=1 dispatches (vocab, 1, 1) WGs which exceeds the 65535
        # per-dimension WebGPU limit for large vocabularies (Llama3: 128256,
        # Qwen2.5/3: 152064). SPLIT_K=0 supports USE_QUANT values 0 (f16),
        # 3 (gptq_sym), 4 (awq_sym), 5 (fp8_gpu), 6 (nvfp4_gpu), 7 (int8_gpu),
        # 8 (nf4_gpu) — the full set that _uq_for_key() can return. Values 1 and
        # 2 are phantom types that _uq_for_key never produces.
        self._dispatch(
            "matmul_quant",
            [norm_out,
             self.weights[lm_key],
             self._scales_buf(lm_key, uq, self._dummy_buf),
             logits_buf],
            {"K": hidden, "N": vocab, "USE_QUANT": uq, "SPLIT_K": 0, **self._quant_extra(lm_key.removesuffix(".weight"), uq)},
            _rows_wg(vocab),
        )
        if greedy:
            self._dispatch("argmax_f16", [logits_buf, self._ensure_sample_buf()],
                           {"N": vocab}, (1, 1, 1))
            self._copy_sample_to_staging()
        self._last_logit_buf = logits_buf
        self._last_vocab = vocab

    def _run_decode_dispatches(
        self,
        ids_buf, pos_buf, slot_map, bt_buf, x_buf,
        norm_out, logits_buf, ctx_len, num_tokens, vocab, greedy,
    ) -> None:
        """Emit all GPU dispatches for a single-token decode step.

        Must be called inside an active command encoder (either via
        _batched_dispatch() or with _active_encoder set manually, as
        MixtralWebGPUModel does for MoE expert routing).
        """
        sc = self._sc
        _rms_base = self._rms_consts

        # Embed (single dispatch; removed the duplicate standalone dispatch)
        self._dispatch(
            "embedding_lookup",
            [self.weights["model.embed_tokens.weight"], ids_buf, x_buf],
            {"HIDDEN_DIM": self.hidden_size},
            (num_tokens, 1, 1),
        )

        # Pre-norm for layer 0 - subsequent layers' pre-norms are fused into
        # the previous layer's final add_rms_norm dispatch.
        self._dispatch(
            "rms_norm",
            [x_buf, self.weights["model.layers.0.input_layernorm.weight"], sc["normed"]],
            _rms_base, (num_tokens, 1, 1),
        )

        normed_x = sc["normed"]
        for i in range(self.num_layers):
            normed_x, x_buf = self._transformer_layer(
                i, normed_x, x_buf, pos_buf, slot_map, bt_buf, ctx_len, num_tokens)

        # Final norm
        self._dispatch(
            "rms_norm",
            [x_buf, self.weights["model.norm.weight"], norm_out],
            _rms_base,
            (num_tokens, 1, 1),
        )

        self._decode_teardown(norm_out, logits_buf, vocab, greedy)

    def forward(
        self,
        input_ids: np.ndarray,
        positions: np.ndarray,
        attn_metadata: object,
    ) -> np.ndarray:
        """
        Args:
            input_ids:    [num_tokens]  uint32
            positions:    [num_tokens]  uint32
            attn_metadata: carries slot_mapping and block_table

        Returns:
            int32 [1, 1] token ID when greedy (default); float32 [1, vocab_size] logits for the
            last token when non-greedy. Even during prefill only the last token is read back.
        """
        num_tokens = len(input_ids)
        self._hstate = 0

        self._check_single_sequence(attn_metadata)

        vocab = self.vocab_size

        # Batch prefill: T>1 tokens use matmul_quant_mr4 (all T rows at once) plus
        # sequential causal attention. Layers are chunked across separate command
        # encoders (4 layers each) to stay under Metal's per-command-buffer GPU timeout.
        if num_tokens > 1:
            return self._prefill_batch_forward(
                input_ids, positions, attn_metadata,
                num_tokens,
            )

        # Decode path (num_tokens=1): use pre-allocated buffers for zero-alloc hot path.
        ids_buf, pos_buf, slot_map, bt_buf, x_buf, norm_out, logits_buf, ctx_len = \
            self._decode_setup(input_ids, positions, attn_metadata)

        greedy = self._greedy_decode
        with self._batched_dispatch():
            self._run_decode_dispatches(
                ids_buf, pos_buf, slot_map, bt_buf, x_buf,
                norm_out, logits_buf, ctx_len, num_tokens, vocab, greedy,
            )

        return self._finish_forward(greedy)

    def _prefill_batch_forward(  # noqa: C901
        self,
        input_ids: "np.ndarray",
        positions: "np.ndarray",
        attn_metadata: object,
        T: int,
    ) -> "np.ndarray":
        """Batch prefill: process T prompt tokens in one GPU command encoder.

        GEMM ops use matmul_quant_mr4 (T rows at once).
        Attention is a fused T-token causal dispatch via flash_attn_prefill; no per-token looping occurs.
        Last-token prediction extracted via GPU copy_buffer_to_buffer.
        Returns: (1, 1) int32 (GPU argmax token id) when greedy, or (1, vocab) float32 logits when greedy is False.
        """
        # Guard: load_weights() must run before forward() so that _batch_matmul_supported
        # and self.weights are populated. Check before any early-return path so that a
        # missing load_weights() call always surfaces as a clear RuntimeError, even on
        # APC prefix-cache hits where positions[0] > 0.
        if self._batch_matmul_supported is None:
            raise RuntimeError("load_weights() must be called before forward()")

        # APC prefix-cache hit: the first token's absolute position is > 0, meaning
        # num_computed cached K/V blocks already exist in the KV cache. The batch
        # prefill shader has no KV-cache binding and applies a batch-local causal
        # mask starting at index 0, so it cannot attend to the prefix. Fall back to
        # the sequential path, which drives flash_attn_decode with the full block
        # table and ctx_len = tok_pos + 1.
        if int(positions[0]) > 0:
            return self._prefill_sequential_fallback(input_ids, positions, attn_metadata, T)

        # Fall back to per-token sequential only for formats matmul_quant_mr4 cannot handle.
        # USE_QUANT=0 (f16) and USE_QUANT=3 (GPTQ INT4) are both supported in the batch path.
        # All other quant types (AWQ, FP8, NF4, Q4_K, ...) fall through to _transformer_layer
        # which dispatches matmul_quant with the correct USE_QUANT per key.
        # _batch_matmul_supported is computed once in load_weights; no re-scan per call.
        if not self._batch_matmul_supported:
            return self._prefill_sequential_fallback(
                input_ids, positions, attn_metadata, T,
            )

        # Sliding Window Attention models must use the sequential path so that
        # each token's ctx_len is capped by _effective_ctx_len (overridden in
        # MixtralWebGPUModel). flash_attn_prefill applies standard causal masking
        # and has no WINDOW_SIZE constant, so batch prefill would attend across the
        # full context and produce wrong attention beyond the window.
        if self._sw is not None:
            return self._prefill_sequential_fallback(
                input_ids, positions, attn_metadata, T,
            )

        hidden = self.hidden_size
        vocab = self.vocab_size
        rms_base = self._rms_consts
        dev  = self.wgpu_device.wgpu_device

        q_dim  = self.num_q_heads  * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        inter  = self.intermediate_size

        # Temporary batch buffers (T × size). Allocated once per prefill call;
        # only reached when the batch path is confirmed, so overhead is negligible
        # vs the GEMM savings.
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
            "ffn_n":    self._make_buf(T * hidden * 2),
            "gate_buf": self._make_buf(T * inter * 2),
            "up_buf":   self._make_buf(T * inter * 2),
            "ffn_act":  self._make_buf(T * inter * 2),
            "ffn_out":  self._make_buf(T * hidden * 2),
            "h0":       self._make_buf(T * hidden * 2),
            "h1":       self._make_buf(T * hidden * 2),
            "h2":       self._make_buf(T * hidden * 2),
            # Single-token scratch for final norm + LM head.
            # Reuse the decode-path pre-allocated buffers: _pre["x"] and _pre["norm_out"]
            # are both hidden*2 bytes and _pre["logits"] is vocab*2 bytes, matching exactly.
            # These buffers are idle during batch prefill (the batch path does not call
            # _prefill_sequential_fallback or _decode_setup), so there is no aliasing risk.
            "last_tok":  self._pre["x"],
            "last_norm": self._pre["norm_out"],
            "logits":    self._pre["logits"],
        }

        slot_map_arr = np.asarray(attn_metadata.slot_mapping, dtype=np.uint32)
        slot_map_buf = WebGPUBuffer.from_numpy(dev, slot_map_arr)
        pos_buf      = WebGPUBuffer.from_numpy(dev, positions.astype(np.uint32, copy=False))
        ids_buf      = WebGPUBuffer.from_numpy(dev, input_ids.astype(np.uint32, copy=False))
        def gemm_batch(x_buf: "WebGPUBuffer", w_key: str, out_buf: "WebGPUBuffer",
                       K_in: int, N_out: int) -> None:
            """Batch GEMM: out[T, N_out] = x[T, K_in] @ w[N_out, K_in].T.

            Supports USE_QUANT=0 (f16) and USE_QUANT=3 (GPTQ INT4).
            """
            uq = self._uq_for_key(w_key)
            sc_buf = self._scales_buf(w_key, uq, self._dummy_buf)
            self._dispatch("matmul_quant_mr4",
                           [x_buf, self.weights[w_key], sc_buf, out_buf],
                           {"K": K_in, "N": N_out, "M": T,
                            "USE_QUANT": uq, **self._quant_extra(w_key.removesuffix('.weight'), uq)},
                           (N_out, T, 1))

        # _CHUNK layers per command encoder keeps each submit under Metal's GPU timeout.
        # At T=19 and inter=9728, a single 36-layer encoder generates ~37M threads and
        # exceeds the ~4-8 s per-command-buffer limit. 4 layers at a time stays safe.
        _CHUNK = 4
        _hstate = 0
        x_res    = b["x"]
        _pfill_rope_base = self._rope_consts
        _freq_buf = self._rope_freq_buf

        for chunk_idx, chunk_layers in enumerate(batched(range(self.num_layers), _CHUNK)):
            with self._batched_dispatch():
                if chunk_idx == 0:
                    # ── Embedding (T tokens) ──────────────────────────────────────────
                    self._dispatch("embedding_lookup",
                                   [self.weights["model.embed_tokens.weight"], ids_buf, b["x"]],
                                   {"HIDDEN_DIM": hidden}, (T, 1, 1))

                    self._dispatch("rms_norm",
                                   [b["x"], self.weights["model.layers.0.input_layernorm.weight"],
                                    b["normed"]],
                                   rms_base, (T, 1, 1))

                for i in chunk_layers:
                    p    = f"model.layers.{i}"
                    q_wk = f"{p}.self_attn.q_proj.weight"
                    k_wk = f"{p}.self_attn.k_proj.weight"
                    v_wk = f"{p}.self_attn.v_proj.weight"
                    ow   = f"{p}.self_attn.o_proj.weight"
                    gw_k = f"{p}.mlp.gate_proj.weight"
                    uw_k = f"{p}.mlp.up_proj.weight"
                    dw_k = f"{p}.mlp.down_proj.weight"

                    gemm_batch(b["normed"], q_wk, b["q_buf"],    hidden, q_dim)
                    gemm_batch(b["normed"], k_wk, b["k_buf"],    hidden, kv_dim)
                    gemm_batch(b["normed"], v_wk, b["v_buf"],    hidden, kv_dim)

                    # ── Per-head RMSNorm + RoPE for all T tokens ──────────────────
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

                    # ── KV store: all T tokens at once ────────────────────────────
                    self._dispatch("kv_cache_store_both",
                                   [b["k_rope"], k_cache, b["v_buf"], v_cache, slot_map_buf],
                                   {"BLOCK_SIZE": self.block_size,
                                    "NUM_KV_HEADS": self.num_kv_heads,
                                    "HEAD_DIM": self.head_dim,
                                    "V_IN_OFFSET": 0},
                                   (T, self.num_kv_heads, 1))

                    # ── Causal attention: all T tokens in one fused dispatch ──────
                    # flash_attn_prefill reads dense Q/K/V (already in b["q_rope"],
                    # b["k_rope"], b["v_buf"]) and applies causal masking internally.
                    # Replaces T×3 dispatches (attn_score + softmax + attn_output per token).
                    self._dispatch("flash_attn_prefill",
                                   [b["q_rope"], b["k_rope"], b["v_buf"], b["attn_out"]],
                                   {"NUM_Q_HEADS": self.num_q_heads,
                                    "NUM_KV_HEADS": self.num_kv_heads,
                                    "HEAD_DIM": self.head_dim,
                                    "NUM_T": T},
                                   (self.num_q_heads, T, 1))

                    # ── O projection (batch GEMM) ─────────────────────────────────
                    gemm_batch(b["attn_out"], ow, b["o_proj"], q_dim, hidden)

                    # ── Fused post-attn add + FFN pre-norm ───────────────────────
                    residual = b[_H_NAMES[(_hstate + 1) % 3]]
                    out_h    = b[_H_NAMES[(_hstate + 2) % 3]]
                    self._dispatch("add_rms_norm",
                                   [x_res, b["o_proj"],
                                    self.weights[f"{p}.post_attention_layernorm.weight"],
                                    residual, b["ffn_n"]],
                                   rms_base, (T, 1, 1))

                    # ── FFN (batch GEMMs + SiLU) ──────────────────────────────────
                    gemm_batch(b["ffn_n"], gw_k, b["gate_buf"], hidden, inter)
                    gemm_batch(b["ffn_n"], uw_k, b["up_buf"],   hidden, inter)
                    self._dispatch("gelu_mul",
                                   [b["gate_buf"], b["up_buf"], b["ffn_act"]],
                                   {"N": T * inter},
                                   _vec4_wg(T * inter))
                    gemm_batch(b["ffn_act"], dw_k, b["ffn_out"], inter, hidden)

                    # ── Residual add (cross-layer fused if not last) ──────────────
                    if i < self.num_layers - 1:
                        next_w = self.weights[f"model.layers.{i+1}.input_layernorm.weight"]
                        self._dispatch("add_rms_norm",
                                       [residual, b["ffn_out"], next_w, out_h, b["normed"]],
                                       rms_base, (T, 1, 1))
                        # normed_x stays as b["normed"] — set once before the loop
                    else:
                        add_n = T * hidden
                        self._dispatch("add",
                                       [residual, b["ffn_out"], out_h],
                                       {"N": add_n},
                                       _vec4_wg(add_n))

                    x_res   = out_h
                    _hstate = (_hstate + 2) % 3

        # ── Extract last token, apply final norm, run LM head ─────────────
        # Separate final encoder: copy_buffer_to_buffer is a GPU-side copy with
        # no CPU roundtrip. Recorded into this encoder alongside norm and LM head.
        with self._batched_dispatch():
            last_token_byte_offset = (T - 1) * hidden * 2  # f16 bytes
            self._active_encoder.copy_buffer_to_buffer(
                x_res.buf, last_token_byte_offset,
                b["last_tok"].buf, 0,
                hidden * 2,
            )

            # Final norm on the single last-token vector — separate input/output buffers.
            self._dispatch("rms_norm",
                           [b["last_tok"], self.weights["model.norm.weight"], b["last_norm"]],
                           rms_base, (1, 1, 1))

            # LM head + optional argmax (mirrors _decode_teardown; sets _last_logit_buf/_last_vocab)
            greedy = self._greedy_decode
            self._decode_teardown(b["last_norm"], b["logits"], vocab, greedy)

        return self._finish_forward(greedy)

    def _prefill_sequential_fallback(
        self,
        input_ids: "np.ndarray",
        positions: "np.ndarray",
        attn_metadata: object,
        T: int,
    ) -> "np.ndarray":
        """Process T prefill tokens one at a time through the decode-path infrastructure.

        Used in two cases:
        1. Unsupported quant type for matmul_quant_mr4: USE_QUANT=0 (f16) and USE_QUANT=3
           (GPTQ INT4) are supported in the batch path; all other types (AWQ, FP8, NF4,
           Q4_K, ...) fall through here so _transformer_layer can dispatch the correct
           USE_QUANT per key.
        2. Sliding-window attention (_sw is not None): flash_attn_prefill uses standard
           causal masking with no WINDOW_SIZE support, so batch prefill would attend across
           the full context past the window boundary.

        KV entries are stored token-by-token so causal attention is satisfied at each step.
        Only the last token's logits are returned (prefill next-token prediction).
        """
        hidden = self.hidden_size
        vocab = self.vocab_size
        rms_base = self._rms_consts
        pre = self._pre
        sc  = self._sc

        bt_arr = self._bt_arr(attn_metadata)
        bt_bytes = bt_arr.tobytes()
        slot_arr = np.asarray(attn_metadata.slot_mapping, dtype=np.uint32)

        # x_buf is updated inside the loop; initialize here so the final-norm
        # dispatch is always bound even if T were ever 0.
        # In practice T >= 1 (enforced by _prefill_batch_forward callers), but
        # Python would raise UnboundLocalError without this pre-assignment.
        x_buf = self._pre["x"]

        for t in range(T):
            self._hstate = 0
            tok_pos   = int(positions[t])
            tok_ctx   = tok_pos + 1

            ids_t  = input_ids[t : t + 1]
            pos_t  = positions[t : t + 1]
            self._write_token_bufs(ids_t, pos_t, slot_arr[t : t + 1].tobytes(), bt_bytes)

            with self._batched_dispatch():
                self._dispatch(
                    "embedding_lookup",
                    [self.weights["model.embed_tokens.weight"], pre["ids"], pre["x"]],
                    {"HIDDEN_DIM": hidden}, (1, 1, 1),
                )
                self._dispatch(
                    "rms_norm",
                    [pre["x"], self.weights["model.layers.0.input_layernorm.weight"],
                     sc["normed"]],
                    rms_base, (1, 1, 1),
                )

                normed_x: "WebGPUBuffer" = sc["normed"]
                x_buf                    = pre["x"]

                for layer_idx in range(self.num_layers):
                    normed_x, x_buf = self._transformer_layer(
                        layer_idx, normed_x, x_buf,
                        pre["pos"], pre["slot_map"], pre["bt"],
                        tok_ctx, 1,
                    )

        # Final norm + LM head on the last token's hidden state.
        with self._batched_dispatch():
            self._dispatch(
                "rms_norm",
                [x_buf, self.weights["model.norm.weight"], pre["norm_out"]],
                rms_base, (1, 1, 1),
            )
            # LM head + optional argmax (mirrors _decode_teardown; sets _last_logit_buf/_last_vocab)
            greedy = self._greedy_decode
            self._decode_teardown(pre["norm_out"], pre["logits"], vocab, greedy)

        return self._finish_forward(greedy)

    def _qkv_proj(
        self,
        normed_x: "WebGPUBuffer",
        layer_idx: int,
        uq_q: int,
        uq_k: int,
        uq_v: int,
    ) -> "tuple[WebGPUBuffer, WebGPUBuffer, WebGPUBuffer]":
        """Dispatch separate Q/K/V matmul projections into pre-allocated scratch buffers.

        Returns (q_buf, k_buf, v_buf) from self._sc. Used by the non-fused attention
        path in LlamaWebGPUModel and GptOssWebGPUModel.

        uq_q/uq_k/uq_v are the quantization modes for each projection, already
        computed by the caller (avoids redundant _uq_for_key lookups).
        """
        sc = self._sc
        hidden = self.hidden_size
        p = f"model.layers.{layer_idx}"
        q_dim = self.num_q_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        for (out_buf, proj, dim), uq in zip([
            (sc["q_buf"], "q_proj", q_dim),
            (sc["k_buf"], "k_proj", kv_dim),
            (sc["v_buf"], "v_proj", kv_dim),
        ], [uq_q, uq_k, uq_v]):
            w_key = f"{p}.self_attn.{proj}.weight"
            qi = self._quant_extra(f"{p}.self_attn.{proj}", uq)
            self._dispatch(
                "matmul_quant",
                [normed_x, self.weights[w_key], self._scales_buf(w_key, uq, self._dummy_buf), out_buf],
                {"K": hidden, "N": dim, "USE_QUANT": uq, **qi},
                _gemv_wg(dim),
            )
        return sc["q_buf"], sc["k_buf"], sc["v_buf"]

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
        """QKV projections, RoPE, KV-cache store, attention decode, and O projection.

        Returns sc["o_proj_out"]. Override in subclasses to modify the attention
        computation (e.g. inject bias vectors or change context length per layer).
        Must be called inside an active _batched_dispatch context.
        """
        sc = self._sc
        hidden = self.hidden_size
        p = f"model.layers.{layer_idx}"
        q_dim = self.num_q_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        k_cache, v_cache = self.kv_pool[layer_idx]

        # QKV projections: fused for f16 with per-head norm weights; separate otherwise.
        q_wk = f"{p}.self_attn.q_proj.weight"
        k_wk = f"{p}.self_attn.k_proj.weight"
        v_wk = f"{p}.self_attn.v_proj.weight"
        uq_q, uq_k, uq_v = self._uq_for_key(q_wk), self._uq_for_key(k_wk), self._uq_for_key(v_wk)
        q_norm_w = self.weights.get(f"{p}.self_attn.q_norm.weight")
        k_norm_w = self.weights.get(f"{p}.self_attn.k_norm.weight")
        _use_fused_qkv = uq_q == 0 and uq_k == 0 and uq_v == 0

        if _use_fused_qkv:
            # All f16 + per-head norms: single fused_qkv -> qkv_buf[Q|K|V].
            self._dispatch("fused_qkv",
                           [normed_x, self.weights[q_wk], self.weights[k_wk], self.weights[v_wk],
                            sc["qkv_buf"]],
                           {"K": hidden, "Q_DIM": q_dim, "KV_DIM": kv_dim},
                           (q_dim + 2 * kv_dim, 1, 1))
            _q_src = sc["qkv_buf"]
            _k_src = sc["qkv_buf"]
            _v_src = sc["qkv_buf"]
            _v_offset = q_dim + kv_dim  # f16 elements before V section
        else:
            _q_src, _k_src, _v_src = self._qkv_proj(normed_x, layer_idx, uq_q, uq_k, uq_v)
            _v_offset = 0

        # Per-head norm + RoPE for Q and K.
        # When using fused QKV (f16 + per-head norms): single fused_qk_norm_rope dispatch.
        # Otherwise: two separate fused_per_head_norm_rope (or plain rope) calls.
        _freq_buf = self._rope_freq_buf
        _rope_consts = self._rope_consts

        if _use_fused_qkv and q_norm_w is not None and k_norm_w is not None:
            # fused_qk_norm_rope: Q+K norm+rope in one dispatch.
            # Binding 6 (k_input): unused here (K_SEPARATE=0), bind qkv_buf as dummy.
            # Binding 7 (inv_freq_buf): always provided (wgpu requires all declared bindings).
            self._dispatch("fused_qk_norm_rope",
                           [_q_src, q_norm_w, k_norm_w, pos_buf,
                            sc["q_rope"], sc["k_rope"], _k_src, _freq_buf],
                           {**_rope_consts,
                            "NUM_Q_HEADS": self.num_q_heads,
                            "NUM_KV_HEADS": self.num_kv_heads,
                            "HAS_WEIGHT": 1,
                            "INPUT_OFFSET_K": q_dim},
                           (self.num_q_heads + self.num_kv_heads, num_tokens, 1))
        else:
            for src, dst, n_heads, norm_w, in_off in [
                (_q_src, sc["q_rope"], self.num_q_heads,  q_norm_w, 0),
                (_k_src, sc["k_rope"], self.num_kv_heads, k_norm_w, q_dim if _use_fused_qkv else 0),
            ]:
                if norm_w is not None:
                    # Binding 4 (inv_freq_buf): always provided.
                    self._dispatch("fused_per_head_norm_rope",
                                   [src, norm_w, pos_buf, dst, _freq_buf],
                                   {**_rope_consts, "NUM_HEADS": n_heads,
                                    "HAS_WEIGHT": 1, "INPUT_OFFSET": in_off},
                                   (n_heads, num_tokens, 1))
                else:
                    # Binding 3 (inv_freq_buf): always provided.
                    self._dispatch("rope", [src, pos_buf, dst, _freq_buf],
                                   {**_rope_consts, "NUM_HEADS": n_heads,
                                    "INPUT_OFFSET": in_off},
                                   (num_tokens, n_heads, 1))

        # Fused K+V cache store.
        # When using fused QKV, V lives in qkv_buf starting at element (q_dim+kv_dim).
        self._dispatch("kv_cache_store_both",
                       [sc["k_rope"], k_cache, _v_src, v_cache, slot_map],
                       {"BLOCK_SIZE": self.block_size, "NUM_KV_HEADS": self.num_kv_heads,
                        "HEAD_DIM": self.head_dim, "V_IN_OFFSET": _v_offset},
                       (num_tokens, self.num_kv_heads, 1))

        # Always use flash_attn_decode for single-token decode.
        # The 65535 limit applied to attn_score's dispatch dimension; flash_attn_decode
        # loops internally and has no dispatch dimension limit.
        self._dispatch("flash_attn_decode",
                       [sc["q_rope"], k_cache, v_cache, bt_buf, sc["attn_out"]],
                       {"BLOCK_SIZE": self.block_size,
                        "NUM_Q_HEADS": self.num_q_heads,
                        "NUM_KV_HEADS": self.num_kv_heads,
                        "HEAD_DIM": self.head_dim,
                        "CTX_LEN": self._effective_ctx_len(ctx_len),
                        "START_BLOCK": self._start_block(ctx_len)},
                       (self.num_q_heads, 1, 1))

        # Output projection.
        w_key = f"{p}.self_attn.o_proj.weight"
        uq = self._uq_for_key(w_key)
        qi = self._quant_extra(f"{p}.self_attn.o_proj", uq)
        self._dispatch("matmul_quant", [sc["attn_out"], self.weights[w_key],
                                        self._scales_buf(w_key, uq, self._dummy_buf), sc["o_proj_out"]],
                       {"K": q_dim, "N": hidden, "USE_QUANT": uq, **qi},
                       _gemv_wg(hidden))

        return sc["o_proj_out"]

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
        """Returns (normed_out, raw_out).

        normed_out: pre-normalized hidden state for next layer QKV input.
        raw_out: the updated hidden state (raw residual) for the next layer.

        The initial rms_norm is handled by the CALLER before the loop. This
        allows fusing the final residual-add with the next layer's pre-norm into
        a single add_rms_norm dispatch, saving 2 dispatches per non-last layer.
        """
        sc = self._sc
        hidden = self.hidden_size
        p = f"model.layers.{layer_idx}"
        _rms_c = self._rms_consts

        residual = sc[_H_NAMES[(self._hstate + 1) % 3]]
        out = sc[_H_NAMES[(self._hstate + 2) % 3]]
        add_n = num_tokens * hidden

        with self._batched_dispatch(label=f"L{layer_idx:02d}"):
            # normed_x is already the pre-normed input (no rms_norm dispatch here).
            o_proj_out = self._attn_block(
                layer_idx, normed_x, pos_buf, slot_map, bt_buf, ctx_len, num_tokens)

            # Fused post-attn residual-add + FFN pre-norm: saves 1 dispatch/layer.
            # residual = x_buf + o_proj_out; ffn_normed = rms_norm(residual, weight)
            self._dispatch("add_rms_norm",
                           [x_buf, o_proj_out,
                            self.weights[f"{p}.post_attention_layernorm.weight"],
                            residual, sc["ffn_normed"]],
                           _rms_c, (num_tokens, 1, 1))

            ffn_out = self._ffn_dispatch(sc["ffn_normed"], layer_idx, num_tokens)

            # Final residual add: fuse with next layer's pre-norm when possible.
            # Last layer: plain add; intermediate layers: add_rms_norm saves 1 dispatch.
            if layer_idx < self.num_layers - 1:
                next_w = self.weights[f"model.layers.{layer_idx+1}.input_layernorm.weight"]
                self._dispatch("add_rms_norm",
                               [residual, ffn_out, next_w, out, sc["normed"]],
                               _rms_c, (num_tokens, 1, 1))
                normed_out = sc["normed"]
            else:
                self._dispatch("add", [residual, ffn_out, out],
                               {"N": add_n}, _vec4_wg(add_n))
                # Callers ignore the first return element after the last layer;
                # yield out as a harmless placeholder to satisfy the return tuple.
                normed_out = out

        self._hstate = (self._hstate + 2) % 3
        return normed_out, out

    def _effective_ctx_len(self, ctx_len: int) -> int:
        """Effective context length for flash_attn_decode. Subclasses may cap (e.g. SWA)."""
        return ctx_len

    def _start_block(self, ctx_len: int) -> int:
        """Block table offset for flash_attn_decode. Default is 0 (full attention).

        Subclasses with sliding-window attention override this to skip old blocks
        so the shader reads the most-recent window blocks instead of block 0.
        """
        return 0

    def _ffn_dispatch(
        self,
        normed_x: "WebGPUBuffer",
        layer_idx: int,
        num_tokens: int,
    ) -> "WebGPUBuffer":
        """Dense SiLU FFN (gate_proj + up_proj + SiLU + down_proj).

        Returns sc["ffn_out"]. Subclasses may override to replace with MoE or
        another FFN variant without duplicating the surrounding transformer scaffolding.
        """
        sc = self._sc
        p = f"model.layers.{layer_idx}"
        hidden = self.hidden_size
        inter = self.intermediate_size

        gw_k = f"{p}.mlp.gate_proj.weight"
        uw_k = f"{p}.mlp.up_proj.weight"
        uq_g = self._uq_for_key(gw_k)
        uq_u = self._uq_for_key(uw_k)
        if uq_g == 0 and uq_u == 0:
            # Single dispatch: GEMV for gate+up with inline SiLU -> ffn_act.
            self._dispatch("fused_gate_act",
                           [normed_x, self.weights[gw_k], self.weights[uw_k], sc["ffn_act"]],
                           {"K": hidden, "N": inter}, (inter, 1, 1))
        else:
            gelu_n = num_tokens * inter
            for out_b, w_k, uq2, mlp_proj in [
                    (sc["gate_buf"], gw_k, uq_g, "gate_proj"),
                    (sc["up_buf"],  uw_k, uq_u, "up_proj")]:
                qi2 = self._quant_extra(f"{p}.mlp.{mlp_proj}", uq2)
                self._dispatch("matmul_quant",
                               [normed_x, self.weights[w_k],
                                self._scales_buf(w_k, uq2, self._dummy_buf), out_b],
                               {"K": hidden, "N": inter, "USE_QUANT": uq2, **qi2},
                               _gemv_wg(inter))
            self._dispatch("gelu_mul", [sc["gate_buf"], sc["up_buf"], sc["ffn_act"]],
                           {"N": gelu_n}, _vec4_wg(gelu_n))

        # Down projection
        w_k = f"{p}.mlp.down_proj.weight"
        uq = self._uq_for_key(w_k)
        qi3 = self._quant_extra(f"{p}.mlp.down_proj", uq)
        self._dispatch("matmul_quant",
                       [sc["ffn_act"], self.weights[w_k],
                        self._scales_buf(w_k, uq, self._dummy_buf), sc["ffn_out"]],
                       {"K": inter, "N": hidden, "USE_QUANT": uq, **qi3},
                       _gemv_wg(hidden))
        return sc["ffn_out"]
