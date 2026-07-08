from __future__ import annotations
import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from vllm_webgpu.models.base import BaseWebGPUModel, compute_yarn_freqs

if TYPE_CHECKING:
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.device import WebGPUDevice
    from vllm_webgpu.webgpu.pipeline import PipelineCache

logger = logging.getLogger(__name__)


def _gemv_wg(N: int, uq: int) -> tuple:
    """Workgroup count for matmul_quant dispatch.

    SPLIT_K=1 (one workgroup per output row): USE_QUANT in (0,3,4,5,6,7,8).
    Row-per-thread: USE_QUANT in (1,2).
    """
    if uq in (0, 3, 4, 5, 6, 7, 8):
        return (N, 1, 1)
    return ((N + 255) // 256, 1, 1)


class LlamaWebGPUModel(BaseWebGPUModel):
    """
    Handles Llama 3.x and Qwen 2.5/3.x (architecturally identical).
    Layer order per token:
      embedding_lookup
      -> N x (rms_norm -> qkv_proj -> fused_per_head_norm_rope ->
              kv_cache_store -> attn_score -> softmax -> attn_output ->
              o_proj -> add -> rms_norm -> gate_proj + up_proj ->
              gelu_mul -> down_proj -> add)
      -> rms_norm -> lm_head -> logits
    """

    # GPU argmax path returns (1,1) int32; logit_readback() provides full logits.
    logit_returns_token_id: bool = True

    def __init__(self, model_config, wgpu_device: "WebGPUDevice", pipeline_cache: "PipelineCache") -> None:
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
        from vllm_webgpu.config import get_config
        self.block_size: int = get_config().block_size
        # matmul_quant f16 path packs two f16 values per u32. Row boundaries only
        # align to u32 boundaries when K is even; odd K silently produces wrong results.
        for name, val in [("hidden_size", self.hidden_size),
                          ("intermediate_size", self.intermediate_size),
                          ("head_dim", self.head_dim)]:
            if val % 2 != 0:
                raise ValueError(f"{name}={val} must be even for f16 GEMV")
        # add.wgsl and gelu_mul.wgsl use vec4<f16>: dimensions must be divisible by 4.
        for name, val in [("hidden_size", self.hidden_size),
                          ("intermediate_size", self.intermediate_size)]:
            if val % 4 != 0:
                raise ValueError(f"{name}={val} must be divisible by 4 for vec4<f16> shaders")
        max_ctx = getattr(model_config, "max_position_embeddings", 8192)
        # Precompute constants that are used every forward pass.
        self._ln_rope_theta: float = math.log(self.rope_theta)
        _wg_size = 256
        _vpt = min((self.hidden_size + _wg_size - 1) // _wg_size, 16) if self.hidden_size <= _wg_size * 16 else 0
        self._rms_consts: dict = {"HIDDEN_DIM": self.hidden_size, "VALS_PER_THREAD": _vpt}
        self._init_scratch_buffers(max_ctx)
        self._init_rope_freq_buf()

    def _init_scratch_buffers(self, max_ctx: int) -> None:
        """Pre-allocate all intermediate scratch buffers used in _transformer_layer.

        Eliminates 17 GPU buffer allocations per layer per decode token.
        Decode path only (num_tokens=1). Sizes are fixed by model dimensions.
        """
        import wgpu as wgpu_lib
        from vllm_webgpu.webgpu.buffer import WebGPUBuffer

        dev = self.wgpu_device.wgpu_device
        rw = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST
        T = 1  # decode: num_tokens == 1
        H = self.hidden_size
        I = self.intermediate_size
        Q = self.num_q_heads * self.head_dim
        KV = self.num_kv_heads * self.head_dim

        def mk(n: int) -> "WebGPUBuffer":
            return WebGPUBuffer.empty(dev, n, usage=rw)

        # Pre-allocated per-step buffers: reused every decode call via write_buffer.
        # Eliminates GPU allocation overhead (~5-10ms per token on Metal).
        max_bt_blocks = max(4096, (max_ctx + self.block_size - 1) // self.block_size)
        self._pre: dict[str, "WebGPUBuffer"] = {
            "ids":      mk(T * 4),              # [1] uint32 token id
            "pos":      mk(T * 4),              # [1] uint32 position
            "slot_map": mk(T * 4),              # [1] uint32 physical slot
            "bt":       mk(max_bt_blocks * 4),  # [max_blocks] uint32 block table
            "x":        mk(T * H * 2),          # [1, H] f16 residual / embedding
            "norm_out": mk(T * H * 2),          # [1, H] f16 final norm output
            "logits":   mk(T * self.vocab_size * 2),  # [1, vocab] f16 logits
        }

        self._sc: dict[str, "WebGPUBuffer"] = {
            "normed":  mk(T * H * 2),
            "qkv_buf": mk(T * (Q + 2 * KV) * 2),  # [Q|K|V] f16 - fused QKV output
            "q_buf":       mk(T * Q * 2),
            "k_buf":       mk(T * KV * 2),
            "v_buf":       mk(T * KV * 2),
            "q_rope":      mk(T * Q * 2),
            "k_rope":      mk(T * KV * 2),
            "attn_out":   mk(T * Q * 2),
            "o_proj_out": mk(T * H * 2),
            "ffn_normed": mk(T * H * 2),
            "gate_buf":   mk(T * I * 2),
            "up_buf":     mk(T * I * 2),
            "ffn_act":    mk(T * I * 2),
            "ffn_out":    mk(T * H * 2),
            # Three hidden-state buffers: ping-pong between h0/h1/h2 so that
            # x_buf, residual, and out are always distinct within a single layer.
            "h0":         mk(T * H * 2),
            "h1":         mk(T * H * 2),
            "h2":         mk(T * H * 2),
        }
        # Index into hidden-state rotation: the layer output cycles h0 -> h1 -> h2 -> h0 ...
        self._hstate: int = 0

    def _init_rope_freq_buf(self) -> None:
        """Detect YaRN rope_scaling and upload precomputed inverse frequencies to GPU.

        When rope_type == 'yarn', replaces the base-class dummy buffer with actual
        YaRN-scaled frequencies. All other rope types keep the dummy (_use_freq_buf=False).
        """
        from vllm_webgpu.webgpu.buffer import WebGPUBuffer
        import wgpu as wgpu_lib

        rope_scaling = getattr(self.model_config, "rope_scaling", None) or {}
        rope_type = (rope_scaling.get("rope_type", "") or
                     rope_scaling.get("type", ""))

        if rope_type != "yarn":
            return  # base-class dummy buffer is sufficient; _use_freq_buf stays False

        dev = self.wgpu_device.wgpu_device
        rw = (wgpu_lib.BufferUsage.STORAGE
              | wgpu_lib.BufferUsage.COPY_SRC
              | wgpu_lib.BufferUsage.COPY_DST)
        freqs, mscale = compute_yarn_freqs(self.head_dim, self.rope_theta, rope_scaling)
        self._rope_freq_buf = WebGPUBuffer.from_numpy(dev, freqs, usage=rw)
        self._yarn_mscale = mscale
        self._use_freq_buf = True
        logger.info(
            "YaRN RoPE: factor=%.1f beta_fast=%.1f beta_slow=%.1f orig_ctx=%d",
            rope_scaling.get("factor", 1.0),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("original_max_position_embeddings", 4096),
        )

    def _postprocess_weights(self) -> None:
        """Fix weight shapes that differ between model variants.

        Qwen3's q_norm/k_norm weights are shared across all heads: shape (head_dim,).
        Our fused_per_head_norm_rope shader indexes weight[head_idx * HEAD_DIM + i],
        expecting shape (num_heads * head_dim,). Tile if the loaded shape is just (head_dim,).
        """
        import wgpu as wgpu_lib
        from vllm_webgpu.webgpu.buffer import WebGPUBuffer
        dev = self.wgpu_device.wgpu_device
        rw = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST

        for i in range(self.num_layers):
            p = f"model.layers.{i}"
            for norm_key, num_heads in [
                (f"{p}.self_attn.q_norm.weight", self.num_q_heads),
                (f"{p}.self_attn.k_norm.weight", self.num_kv_heads),
            ]:
                buf = self.weights.get(norm_key)
                if buf is None:
                    continue
                expected = (num_heads * self.head_dim,)
                if buf.shape == expected:
                    continue
                if buf.shape == (self.head_dim,):
                    # Shared norm: tile to (num_heads * head_dim,) so each head uses same weights.
                    tiled = np.tile(buf.to_numpy().view(np.float16), num_heads)
                    self.weights[norm_key] = WebGPUBuffer.from_numpy(dev, tiled, usage=rw)
                else:
                    raise ValueError(
                        f"{norm_key}: unexpected shape {buf.shape}, "
                        f"expected {expected} or ({self.head_dim},)"
                    )

    def load_weights(self, path: str) -> None:
        super().load_weights(path)
        self._postprocess_weights()

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
            logits: [num_tokens, vocab_size]  float32
        """
        dev = self.wgpu_device.wgpu_device
        num_tokens = len(input_ids)
        hidden = self.hidden_size
        self._hstate = 0

        if hasattr(attn_metadata, "block_tables") and len(attn_metadata.block_tables) > 1:
            # Each forward() call handles exactly one sequence. The model runner
            # calls forward() once per decode request. Batching N sequences requires
            # N separate pre-alloc buffer sets and per-sequence attention dispatch.
            raise RuntimeError(
                f"multi-sequence batching not supported: got {len(attn_metadata.block_tables)} "
                "block tables; call forward() once per decode request"
            )

        ctx_len = int(attn_metadata.max_decode_seq_len
                      if attn_metadata.max_decode_seq_len is not None
                      else num_tokens)
        if ctx_len <= 0:
            ctx_len = num_tokens

        vocab = self.vocab_size
        _rms_base = self._rms_consts
        sc = self._sc

        # Batch prefill: T>1 tokens use matmul_quant_mr4 (all T rows at once) plus
        # sequential causal attention. Layers are chunked across separate command
        # encoders (4 layers each) to stay under Metal's per-command-buffer GPU timeout.
        if num_tokens > 1:
            return self._prefill_batch_forward(
                input_ids, positions, attn_metadata,
                num_tokens, hidden, ctx_len, vocab, _rms_base,
            )

        # Decode path (num_tokens=1): use pre-allocated buffers for zero-alloc hot path.
        pre = self._pre
        dev.queue.write_buffer(pre["ids"].buf, 0, input_ids.astype(np.uint32).tobytes())
        dev.queue.write_buffer(pre["pos"].buf, 0, positions.astype(np.uint32).tobytes())
        dev.queue.write_buffer(
            pre["slot_map"].buf, 0,
            np.array(attn_metadata.slot_mapping, dtype=np.uint32).tobytes())
        bt_arr = self._bt_arr(attn_metadata)
        dev.queue.write_buffer(pre["bt"].buf, 0, bt_arr.tobytes())

        ids_buf   = pre["ids"]
        pos_buf   = pre["pos"]
        slot_map  = pre["slot_map"]
        bt_buf    = pre["bt"]
        x_buf     = pre["x"]
        norm_out  = pre["norm_out"]
        logits_buf = pre["logits"]

        with self._batched_dispatch():
            # Embed (single dispatch; removed the duplicate standalone dispatch)
            self._dispatch(
                "embedding_lookup",
                [self.weights["model.embed_tokens.weight"], ids_buf, x_buf],
                {"HIDDEN_DIM": hidden},
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

            # LM head: vocab_size (e.g. 151936) exceeds the WebGPU
            # maxComputeWorkgroupsPerDimension limit of 65535, so the split-K path
            # (one workgroup per output row) is unusable. Force SPLIT_K=0 to use the
            # row-per-thread path, which dispatches ceil(vocab/256) workgroups instead.
            self._dispatch(
                "matmul_quant",
                [norm_out, self.weights.get("lm_head.weight", self.weights["model.embed_tokens.weight"]),
                 self.weights.get("lm_head.scales", norm_out),  # unused for f16
                 logits_buf],
                {"K": hidden, "N": vocab, "USE_QUANT": 0, "SPLIT_K": 0},
                ((vocab + 255) // 256, 1, 1),
            )
            # GPU argmax only when the caller has confirmed greedy decoding.
            # For non-greedy sampling the model runner reads full logits on CPU.
            if getattr(self, "_greedy_decode", True):
                self._dispatch("argmax_f16", [logits_buf, self._ensure_sample_buf(vocab)],
                               {"N": vocab}, (1, 1, 1))
                self._copy_sample_to_staging()

        self._last_logit_buf = logits_buf
        self._last_vocab     = vocab
        if getattr(self, "_greedy_decode", True):
            # Greedy path: 4-byte readback from staging buffer (mapped during main sync).
            tok = self._read_sample_tok()
            return np.array([[tok]], dtype=np.int32)  # shape (1, 1)
        # Non-greedy path: return full float32 logits for CPU sampling.
        return self.logit_readback()  # shape (1, vocab)

    def _prefill_batch_forward(  # noqa: C901
        self,
        input_ids: "np.ndarray",
        positions: "np.ndarray",
        attn_metadata: object,
        T: int,
        hidden: int,
        ctx_len: int,
        vocab: int,
        rms_base: dict,
    ) -> "np.ndarray":
        """Batch prefill: process T prompt tokens in one GPU command encoder.

        GEMM ops use matmul_quant_mr4 (T rows at once).
        Attention is sequential per token (causal masking via Q_TOKEN_OFFSET override).
        Last-token prediction extracted via GPU copy_buffer_to_buffer.
        Returns shape (1, 1) int32 (GPU argmax of last-token logits).
        """
        import wgpu as wgpu_lib
        from vllm_webgpu.webgpu.buffer import WebGPUBuffer

        dev  = self.wgpu_device.wgpu_device
        rw   = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST

        def alloc(n_f16: int) -> "WebGPUBuffer":
            return WebGPUBuffer.empty(dev, max(n_f16 * 2, 8), usage=rw)

        q_dim  = self.num_q_heads  * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        inter  = self.intermediate_size
        ln_rope = self._ln_rope_theta

        # Temporary batch buffers (T × size). Allocated once per prefill call;
        # overhead is negligible vs the GEMM savings.
        b: dict = {
            "x":        alloc(T * hidden),
            "normed":   alloc(T * hidden),
            "q_buf":    alloc(T * q_dim),
            "k_buf":    alloc(T * kv_dim),
            "v_buf":    alloc(T * kv_dim),
            "q_rope":   alloc(T * q_dim),
            "k_rope":   alloc(T * kv_dim),
            "attn_out": alloc(T * q_dim),
            "o_proj":   alloc(T * hidden),
            "ffn_n":    alloc(T * hidden),
            "gate_buf": alloc(T * inter),
            "up_buf":   alloc(T * inter),
            "ffn_act":  alloc(T * inter),
            "ffn_out":  alloc(T * hidden),
            "h0":       alloc(T * hidden),
            "h1":       alloc(T * hidden),
            "h2":       alloc(T * hidden),
            # Single-token scratch for final norm + LM head
            "last_tok":  alloc(hidden),   # raw last-token hidden state (from copy)
            "last_norm": alloc(hidden),   # normed last-token hidden state
            "logits":    alloc(vocab),
        }

        slot_map_arr = np.array(attn_metadata.slot_mapping, dtype=np.uint32)
        slot_map_buf = WebGPUBuffer.from_numpy(dev, slot_map_arr, usage=rw)
        pos_buf      = WebGPUBuffer.from_numpy(dev, positions.astype(np.uint32), usage=rw)
        ids_buf      = WebGPUBuffer.from_numpy(dev, input_ids.astype(np.uint32), usage=rw)
        bt_arr = self._bt_arr(attn_metadata)
        bt_buf = WebGPUBuffer.from_numpy(dev, bt_arr, usage=rw)

        # Small dummy scales buffer for USE_QUANT=0 f16 path (binding 2 not read).
        _dummy = alloc(4)

        def gemm_batch(x_buf: "WebGPUBuffer", w_key: str, out_buf: "WebGPUBuffer",
                       K_in: int, N_out: int) -> None:
            """Batch GEMM: out[T, N_out] = x[T, K_in] @ w[N_out, K_in].T.

            Supports USE_QUANT=0 (f16) and USE_QUANT=3 (GPTQ INT4).
            """
            uq = self._uq_for_key(w_key)
            if uq == 3:
                base_key = w_key[:-7]  # strip ".weight" suffix
                group_k = self._quant_extra(base_key, uq).get("GROUP_K", 128)
                sc_buf = self.weights.get(w_key + ".scales", _dummy)
                self._dispatch("matmul_quant_mr4",
                               [x_buf, self.weights[w_key], sc_buf, out_buf],
                               {"K": K_in, "N": N_out, "M": T,
                                "USE_QUANT": 3, "GROUP_K": group_k},
                               (N_out, T, 1))
            else:
                # f16 path (uq == 0)
                self._dispatch("matmul_quant_mr4",
                               [x_buf, self.weights[w_key], _dummy, out_buf],
                               {"K": K_in, "N": N_out, "M": T, "USE_QUANT": 0},
                               (N_out, T, 1))

        # Fall back to per-token sequential only for formats matmul_quant_mr4 cannot handle.
        # USE_QUANT=0 (f16) and USE_QUANT=3 (GPTQ INT4) are both supported in the batch path.
        # All other quant types (AWQ, FP8, NF4, Q4_K, ...) fall through to _transformer_layer
        # which dispatches matmul_quant with the correct USE_QUANT per key.
        _rep_quant_key = "model.layers.0.self_attn.q_proj.weight"
        _rep_uq = self._uq_for_key(_rep_quant_key)
        if _rep_uq not in (0, 3):
            return self._prefill_sequential_fallback(
                input_ids, positions, attn_metadata, T, hidden, ctx_len, vocab, rms_base,
            )

        # _CHUNK layers per command encoder keeps each submit under Metal's GPU timeout.
        # At T=19 and inter=9728, a single 36-layer encoder generates ~37M threads and
        # exceeds the ~4-8 s per-command-buffer limit. 4 layers at a time stays safe.
        _CHUNK = 4
        _hstate = 0
        h_names = ["h0", "h1", "h2"]
        normed_x = b["normed"]
        x_res    = b["x"]

        chunks = [list(range(i, min(i + _CHUNK, self.num_layers))) for i in range(0, self.num_layers, _CHUNK)]

        for chunk_idx, chunk_layers in enumerate(chunks):
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
                    normed_x = b["normed"]
                    x_res    = b["x"]

                for i in chunk_layers:
                    p    = f"model.layers.{i}"
                    q_wk = f"{p}.self_attn.q_proj.weight"
                    k_wk = f"{p}.self_attn.k_proj.weight"
                    v_wk = f"{p}.self_attn.v_proj.weight"
                    ow   = f"{p}.self_attn.o_proj.weight"
                    gw_k = f"{p}.mlp.gate_proj.weight"
                    uw_k = f"{p}.mlp.up_proj.weight"
                    dw_k = f"{p}.mlp.down_proj.weight"

                    gemm_batch(normed_x, q_wk, b["q_buf"],    hidden, q_dim)
                    gemm_batch(normed_x, k_wk, b["k_buf"],    hidden, kv_dim)
                    gemm_batch(normed_x, v_wk, b["v_buf"],    hidden, kv_dim)

                    # ── Per-head RMSNorm + RoPE for all T tokens ──────────────────
                    _pfill_rope_base = {"HEAD_DIM": self.head_dim,
                                        "ROPE_BASE": float(self.rope_theta),
                                        "LN_ROPE_BASE": ln_rope,
                                        "USE_FREQ_BUF": int(self._use_freq_buf),
                                        "ATTN_SCALE": self._yarn_mscale}
                    _freq_buf = self._rope_freq_buf
                    for src, dst, n_h, wk in [
                        (b["q_buf"],  b["q_rope"], self.num_q_heads,  f"{p}.self_attn.q_norm.weight"),
                        (b["k_buf"],  b["k_rope"], self.num_kv_heads, f"{p}.self_attn.k_norm.weight"),
                    ]:
                        nw = self.weights.get(wk)
                        if nw is not None:
                            self._dispatch("fused_per_head_norm_rope",
                                           [src, nw, pos_buf, dst, _freq_buf],
                                           {**_pfill_rope_base, "NUM_HEADS": n_h, "HAS_WEIGHT": 1},
                                           (n_h, T, 1))
                        else:
                            self._dispatch("rope", [src, pos_buf, dst, _freq_buf],
                                           {**_pfill_rope_base, "NUM_HEADS": n_h},
                                           (T, n_h, 1))

                    k_cache, v_cache = self.kv_pool[i]

                    # ── KV store: all T tokens at once ────────────────────────────
                    self._dispatch("kv_cache_store_both",
                                   [b["k_rope"], k_cache, b["v_buf"], v_cache, slot_map_buf],
                                   {"BLOCK_SIZE": self.block_size,
                                    "NUM_KV_HEADS": self.num_kv_heads,
                                    "HEAD_DIM": self.head_dim},
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
                    residual = b[h_names[(_hstate + 1) % 3]]
                    out_h    = b[h_names[(_hstate + 2) % 3]]
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
                                   ((T * inter // 4 + 255) // 256, 1, 1))
                    gemm_batch(b["ffn_act"], dw_k, b["ffn_out"], inter, hidden)

                    # ── Residual add (cross-layer fused if not last) ──────────────
                    if i < self.num_layers - 1:
                        next_w = self.weights[f"model.layers.{i+1}.input_layernorm.weight"]
                        self._dispatch("add_rms_norm",
                                       [residual, b["ffn_out"], next_w, out_h, b["normed"]],
                                       rms_base, (T, 1, 1))
                        normed_x = b["normed"]
                    else:
                        add_n = T * hidden
                        self._dispatch("add",
                                       [residual, b["ffn_out"], out_h],
                                       {"N": add_n},
                                       ((add_n // 4 + 255) // 256, 1, 1))

                    x_res   = out_h
                    _hstate = (_hstate + 2) % 3

        # ── Extract last token, apply final norm, run LM head ─────────────
        # Separate final encoder: copy_buffer_to_buffer is a GPU-side copy with
        # no CPU roundtrip. Recorded into this encoder alongside norm and LM head.
        with self._batched_dispatch():
            last_token_byte_offset = (T - 1) * hidden * 2  # f16 bytes
            if self._active_encoder is None:
                raise RuntimeError("_active_encoder is None inside _batched_dispatch")
            self._active_encoder.copy_buffer_to_buffer(
                x_res.buf, last_token_byte_offset,
                b["last_tok"].buf, 0,
                hidden * 2,
            )

            # Final norm on the single last-token vector — separate input/output buffers.
            self._dispatch("rms_norm",
                           [b["last_tok"], self.weights["model.norm.weight"], b["last_norm"]],
                           rms_base, (1, 1, 1))

            # LM head (SPLIT_K=0: row-per-thread for large vocab)
            lm_head_w = (self.weights.get("lm_head.weight") or
                         self.weights["model.embed_tokens.weight"])
            self._dispatch("matmul_quant",
                           [b["last_norm"], lm_head_w,
                            self.weights.get("lm_head.scales", _dummy),
                            b["logits"]],
                           {"K": hidden, "N": vocab, "USE_QUANT": 0, "SPLIT_K": 0},
                           ((vocab + 255) // 256, 1, 1))

            if getattr(self, "_greedy_decode", True):
                self._dispatch("argmax_f16", [b["logits"], self._ensure_sample_buf(vocab)],
                               {"N": vocab}, (1, 1, 1))
                self._copy_sample_to_staging()

        self._last_logit_buf = b["logits"]
        self._last_vocab     = vocab
        if getattr(self, "_greedy_decode", True):
            tok = self._read_sample_tok()
            return np.array([[tok]], dtype=np.int32)
        return self.logit_readback()  # shape (1, vocab) for non-greedy

    def _prefill_sequential_fallback(
        self,
        input_ids: "np.ndarray",
        positions: "np.ndarray",
        attn_metadata: object,
        T: int,
        hidden: int,
        ctx_len: int,
        vocab: int,
        rms_base: dict,
    ) -> "np.ndarray":
        """Process T prefill tokens one at a time through the decode-path infrastructure.

        Used when projection weights are quantized: matmul_quant_mr4 only supports
        USE_QUANT=0 and would silently reinterpret packed quantized bytes as f16.
        _transformer_layer dispatches matmul_quant with the correct USE_QUANT per key.

        KV entries are stored token-by-token so causal attention is satisfied at each step.
        Only the last token's logits are returned (prefill next-token prediction).
        """
        dev = self.wgpu_device.wgpu_device
        pre = self._pre
        sc  = self._sc

        bt_arr = self._bt_arr(attn_metadata)

        x_buf: "WebGPUBuffer" = pre["x"]

        for t in range(T):
            self._hstate = 0
            tok_pos   = int(positions[t])
            tok_ctx   = tok_pos + 1

            ids_t  = input_ids[t : t + 1]
            pos_t  = positions[t : t + 1]
            slot_t = np.array(attn_metadata.slot_mapping[t : t + 1], dtype=np.uint32)

            dev.queue.write_buffer(pre["ids"].buf,      0, ids_t.astype(np.uint32).tobytes())
            dev.queue.write_buffer(pre["pos"].buf,      0, pos_t.astype(np.uint32).tobytes())
            dev.queue.write_buffer(pre["slot_map"].buf, 0, slot_t.tobytes())
            dev.queue.write_buffer(pre["bt"].buf,       0, bt_arr.tobytes())

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
            lm_head_w = (self.weights.get("lm_head.weight") or
                         self.weights["model.embed_tokens.weight"])
            self._dispatch(
                "matmul_quant",
                [pre["norm_out"], lm_head_w,
                 self.weights.get("lm_head.scales", pre["norm_out"]),
                 pre["logits"]],
                {"K": hidden, "N": vocab, "USE_QUANT": 0, "SPLIT_K": 0},
                ((vocab + 255) // 256, 1, 1),
            )
            if getattr(self, "_greedy_decode", True):
                self._dispatch("argmax_f16", [pre["logits"], self._ensure_sample_buf(vocab)],
                               {"N": vocab}, (1, 1, 1))
                self._copy_sample_to_staging()

        self._last_logit_buf = pre["logits"]
        self._last_vocab     = vocab
        if getattr(self, "_greedy_decode", True):
            tok = self._read_sample_tok()
            return np.array([[tok]], dtype=np.int32)
        return self.logit_readback()

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
        q_dim = self.num_q_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        inter = self.intermediate_size
        ln_rope = self._ln_rope_theta
        _rms_c = self._rms_consts

        h_names = ["h0", "h1", "h2"]
        residual = sc[h_names[(self._hstate + 1) % 3]]
        out = sc[h_names[(self._hstate + 2) % 3]]
        add_n = num_tokens * hidden

        k_cache, v_cache = self.kv_pool[layer_idx]

        with self._batched_dispatch(label=f"L{layer_idx:02d}"):
            # normed_x is already the pre-normed input (no rms_norm dispatch here).

            # QKV projections: fused for f16 with per-head norm weights; separate otherwise.
            q_wk = f"{p}.self_attn.q_proj.weight"
            k_wk = f"{p}.self_attn.k_proj.weight"
            v_wk = f"{p}.self_attn.v_proj.weight"
            uq_q, uq_k, uq_v = self._uq_for_key(q_wk), self._uq_for_key(k_wk), self._uq_for_key(v_wk)
            _has_qnorm = self.weights.get(f"{p}.self_attn.q_norm.weight") is not None

            _use_fused_qkv = uq_q == 0 and uq_k == 0 and uq_v == 0 and _has_qnorm

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
                for out_buf, proj, dim, uq in [(sc["q_buf"], "q_proj", q_dim, uq_q),
                                               (sc["k_buf"], "k_proj", kv_dim, uq_k),
                                               (sc["v_buf"], "v_proj", kv_dim, uq_v)]:
                    w_key = f"{p}.self_attn.{proj}.weight"
                    qi = self._quant_extra(f"{p}.self_attn.{proj}", uq)
                    self._dispatch("matmul_quant",
                                   [normed_x, self.weights[w_key], self._scales_buf(w_key, uq, normed_x), out_buf],
                                   {"K": hidden, "N": dim, "USE_QUANT": uq,
                                    **self._split_k_extra(uq), **qi},
                                   _gemv_wg(dim, uq))
                _q_src = sc["q_buf"]
                _k_src = sc["k_buf"]
                _v_src = sc["v_buf"]
                _v_offset = 0

            # Per-head norm + RoPE for Q and K.
            # When using fused QKV (f16 + per-head norms): single fused_qk_norm_rope dispatch.
            # Otherwise: two separate fused_per_head_norm_rope (or plain rope) calls.
            q_norm_w = self.weights.get(f"{p}.self_attn.q_norm.weight")
            k_norm_w = self.weights.get(f"{p}.self_attn.k_norm.weight")
            _freq_buf = self._rope_freq_buf
            _rope_consts = {"HEAD_DIM": self.head_dim,
                            "ROPE_BASE": float(self.rope_theta),
                            "LN_ROPE_BASE": ln_rope,
                            "USE_FREQ_BUF": int(self._use_freq_buf),
                            "ATTN_SCALE": self._yarn_mscale}

            if _use_fused_qkv and k_norm_w is not None:
                # fused_qk_norm_rope: Q+K norm+rope in one dispatch.
                # Binding 6 (k_input): unused here (K_SEPARATE=0), bind qkv_buf as dummy.
                # Binding 7 (inv_freq_buf): always provided (wgpu requires all declared bindings).
                self._dispatch("fused_qk_norm_rope",
                               [sc["qkv_buf"], q_norm_w, k_norm_w, pos_buf,
                                sc["q_rope"], sc["k_rope"], sc["qkv_buf"], _freq_buf],
                               {**_rope_consts,
                                "NUM_Q_HEADS": self.num_q_heads,
                                "NUM_KV_HEADS": self.num_kv_heads,
                                "HAS_WEIGHT": 1,
                                "INPUT_OFFSET_K": q_dim},
                               (self.num_q_heads + self.num_kv_heads, num_tokens, 1))
            else:
                for src, dst, n_heads, norm_w, in_off in [
                    (_q_src, sc["q_rope"], self.num_q_heads,  q_norm_w, 0),
                    (_k_src, sc["k_rope"], self.num_kv_heads, k_norm_w, 0),
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
                                       {**_rope_consts, "NUM_HEADS": n_heads},
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
                            "CTX_LEN": self._effective_ctx_len(ctx_len)},
                           (self.num_q_heads, 1, 1))

            # Output projection
            w_key = f"{p}.self_attn.o_proj.weight"
            uq = self._uq_for_key(w_key)
            qi = self._quant_extra(f"{p}.self_attn.o_proj", uq)
            self._dispatch("matmul_quant", [sc["attn_out"], self.weights[w_key],
                                            self._scales_buf(w_key, uq, sc["attn_out"]), sc["o_proj_out"]],
                           {"K": q_dim, "N": hidden, "USE_QUANT": uq,
                            **self._split_k_extra(uq), **qi},
                           _gemv_wg(hidden, uq))

            # Fused post-attn residual-add + FFN pre-norm: saves 1 dispatch/layer.
            # residual = x_buf + o_proj_out; ffn_normed = rms_norm(residual, weight)
            self._dispatch("add_rms_norm",
                           [x_buf, sc["o_proj_out"],
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
                               {"N": add_n}, ((add_n // 4 + 255) // 256, 1, 1))
                normed_out = sc["normed"]  # stale; unused after last layer

        self._hstate = (self._hstate + 2) % 3
        return normed_out, out

    def _effective_ctx_len(self, ctx_len: int) -> int:
        """Effective context length for flash_attn_decode. Subclasses may cap (e.g. SWA)."""
        return ctx_len

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
        gelu_n = num_tokens * inter

        gw_k = f"{p}.mlp.gate_proj.weight"
        uw_k = f"{p}.mlp.up_proj.weight"
        uq_g = self._uq_for_key(gw_k)
        uq_u = self._uq_for_key(uw_k)
        if uq_g == 0 and uq_u == 0:
            # Single dispatch: GEMV for gate+up with inline SiLU -> ffn_act.
            self._dispatch("fused_gate_act",
                           [normed_x, self.weights[gw_k], self.weights[uw_k], sc["ffn_act"]],
                           {"K": hidden, "N": inter, "GELU": 0}, (inter, 1, 1))
        else:
            for out_b, w_k, uq2, mlp_proj in [
                    (sc["gate_buf"], gw_k, uq_g, "gate_proj"),
                    (sc["up_buf"],  uw_k, uq_u, "up_proj")]:
                qi2 = self._quant_extra(f"{p}.mlp.{mlp_proj}", uq2)
                self._dispatch("matmul_quant",
                               [normed_x, self.weights[w_k],
                                self._scales_buf(w_k, uq2, normed_x), out_b],
                               {"K": hidden, "N": inter, "USE_QUANT": uq2,
                                **self._split_k_extra(uq2), **qi2},
                               _gemv_wg(inter, uq2))
            self._dispatch("gelu_mul", [sc["gate_buf"], sc["up_buf"], sc["ffn_act"]],
                           {"N": gelu_n}, ((gelu_n // 4 + 255) // 256, 1, 1))

        # Down projection
        w_k = f"{p}.mlp.down_proj.weight"
        uq = self._uq_for_key(w_k)
        qi3 = self._quant_extra(f"{p}.mlp.down_proj", uq)
        self._dispatch("matmul_quant",
                       [sc["ffn_act"], self.weights[w_k],
                        self._scales_buf(w_k, uq, sc["ffn_act"]), sc["ffn_out"]],
                       {"K": inter, "N": hidden, "USE_QUANT": uq,
                        **self._split_k_extra(uq), **qi3},
                       _gemv_wg(hidden, uq))
        return sc["ffn_out"]
