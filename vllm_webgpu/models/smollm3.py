from __future__ import annotations
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm_webgpu.models.llama import LlamaWebGPUModel

if TYPE_CHECKING:
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.device import WebGPUDevice
    from vllm_webgpu.webgpu.pipeline import PipelineCache

logger = init_logger(__name__)


class SmolLM3WebGPUModel(LlamaWebGPUModel):
    """SmolLM3 WebGPU backend.

    SmolLM3 is a Llama-like model where a subset of attention layers use
    No Position Embeddings (NoPE): Q and K are not rotated by RoPE before
    being written to the KV cache or used in flash attention.

    NoPE layer indices are read from model_config in priority order:
      1. model_config.no_rope_layers  (explicit list of layer indices)
      2. model_config.nope_layers     (alternate attribute name)
      3. Default: every 4th layer starting at index 3 (i.e. 3, 7, 11, ...)

    For NoPE layers, _attn_block skips all RoPE dispatches. Q is copied from
    qkv_buf to q_buf (no rotation) and K from qkv_buf to k_buf, then the KV
    cache stores the unrotated K and flash attention uses the unrotated Q.
    Standard layers are unaffected.
    """

    def __init__(
        self,
        model_config,
        wgpu_device: "WebGPUDevice",
        pipeline_cache: "PipelineCache",
        block_size: int = 16,
    ) -> None:
        super().__init__(model_config, wgpu_device, pipeline_cache, block_size)

        no_rope = getattr(model_config, "no_rope_layers", None)
        if no_rope is None:
            no_rope = getattr(model_config, "nope_layers", None)
        if no_rope is not None:
            # no_rope_layers may be a per-layer flags array (0 = NoPE, 1 = RoPE)
            # rather than an explicit list of NoPE layer indices. SmolLM3 uses
            # the same flags format as Llama4 (config.no_rope_layers[i] == 0 means
            # the i-th layer has no positional encoding). Detect by length and
            # value range, then convert to indices.
            if (len(no_rope) == self.num_layers
                    and all(v in (0, 1) for v in no_rope)):
                no_rope = [i for i, v in enumerate(no_rope) if v == 0]
        else:
            no_rope = list(range(3, self.num_layers, 4))
        self._nope_layers: frozenset[int] = frozenset(no_rope)
        logger.info(
            "SmolLM3: %d NoPE layers (no RoPE): %s",
            len(self._nope_layers),
            sorted(self._nope_layers),
        )

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
        """Dispatch attention, skipping RoPE for NoPE layers."""
        if layer_idx not in self._nope_layers:
            return super()._attn_block(
                layer_idx, normed_x, pos_buf, slot_map, bt_buf, ctx_len, num_tokens)
        return self._attn_block_nope(
            layer_idx, normed_x, slot_map, bt_buf, ctx_len, num_tokens)

    def _attn_block_nope(
        self,
        layer_idx: int,
        normed_x: "WebGPUBuffer",
        slot_map: "WebGPUBuffer",
        bt_buf: "WebGPUBuffer",
        ctx_len: int,
        num_tokens: int,
    ) -> "WebGPUBuffer":
        """Attention block for NoPE layers (no rotary position embedding).

        Runs QKV projection, optionally applies per-head norm (without RoPE),
        then writes unrotated K to the KV cache and runs flash attention with
        unrotated Q. The KV cache therefore holds unrotated K/V, which is
        consistent with training for NoPE layers.

        For models without per-head q/k norms (standard SmolLM3), Q and K are
        copied verbatim from the fused qkv_buf or from the separate projection
        outputs. No approximation is introduced.
        """
        sc     = self._sc
        hidden = self.hidden_size
        p      = f"model.layers.{layer_idx}"
        q_dim  = self.q_dim
        kv_dim = self.kv_dim

        k_cache, v_cache = self.kv_pool[layer_idx]

        q_wk = f"{p}.self_attn.q_proj.weight"
        k_wk = f"{p}.self_attn.k_proj.weight"
        v_wk = f"{p}.self_attn.v_proj.weight"
        uq_q, uq_k, uq_v = self._uq_for_key(q_wk), self._uq_for_key(k_wk), self._uq_for_key(v_wk)
        q_norm_w = self.weights.get(f"{p}.self_attn.q_norm.weight")
        k_norm_w = self.weights.get(f"{p}.self_attn.k_norm.weight")
        _use_fused_qkv = uq_q == 0 and uq_k == 0 and uq_v == 0

        if _use_fused_qkv:
            self._dispatch("fused_qkv",
                           [normed_x, self.weights[q_wk], self.weights[k_wk], self.weights[v_wk],
                            sc["qkv_buf"]],
                           {"K": hidden, "Q_DIM": q_dim, "KV_DIM": kv_dim},
                           (q_dim + 2 * kv_dim, 1, 1))
            # Extract Q and K from qkv_buf into dedicated scratch buffers.
            # We cannot pass qkv_buf with an offset directly to flash_attn_decode,
            # so a GPU copy is the cleanest solution.
            enc = self._active_encoder
            enc.copy_buffer_to_buffer(sc["qkv_buf"].buf, 0,         sc["q_buf"].buf, 0, q_dim  * 2)
            enc.copy_buffer_to_buffer(sc["qkv_buf"].buf, q_dim * 2, sc["k_buf"].buf, 0, kv_dim * 2)
            _v_src    = sc["qkv_buf"]
            _v_offset = q_dim + kv_dim
        else:
            self._qkv_proj(normed_x, layer_idx, uq_q, uq_k, uq_v)
            # sc["q_buf"] and sc["k_buf"] are already populated by _qkv_proj.
            _v_src    = sc["v_buf"]
            _v_offset = 0

        # Apply per-head norm WITHOUT RoPE when norm weights exist.
        # SmolLM3 standard layers have no q_norm/k_norm, so this branch is
        # typically skipped. If per-head norms are present, we apply rms_norm
        # over the full q/k vector (per-head approximation) without rotation.
        if q_norm_w is not None:
            # Apply per-head norm to Q, storing result back in q_buf.
            # fused_per_head_norm_rope with pos_buf pointing to a zeroed position
            # would apply no-op rotation (cos=1, sin=0). Instead we just apply
            # rms_norm to the full Q vector using HIDDEN_DIM = q_dim.
            from vllm_webgpu.models.base import _vals_per_thread
            q_rms_c = {"HIDDEN_DIM": q_dim, "VALS_PER_THREAD": _vals_per_thread(q_dim)}
            self._dispatch("rms_norm",
                           [sc["q_buf"], q_norm_w, sc["q_rope"]],
                           q_rms_c, (num_tokens, 1, 1))
            # Copy normed Q back to q_buf so flash_attn_decode receives it there.
            enc = self._active_encoder
            enc.copy_buffer_to_buffer(sc["q_rope"].buf, 0, sc["q_buf"].buf, 0, q_dim * 2)
        if k_norm_w is not None:
            from vllm_webgpu.models.base import _vals_per_thread
            k_rms_c = {"HIDDEN_DIM": kv_dim, "VALS_PER_THREAD": _vals_per_thread(kv_dim)}
            self._dispatch("rms_norm",
                           [sc["k_buf"], k_norm_w, sc["k_rope"]],
                           k_rms_c, (num_tokens, 1, 1))
            enc = self._active_encoder
            enc.copy_buffer_to_buffer(sc["k_rope"].buf, 0, sc["k_buf"].buf, 0, kv_dim * 2)

        # KV cache store using unrotated K (q_buf, k_buf, no rope applied).
        self._dispatch("kv_cache_store_both",
                       [sc["k_buf"], k_cache, _v_src, v_cache, slot_map],
                       {"BLOCK_SIZE": self.block_size, "NUM_KV_HEADS": self.num_kv_heads,
                        "HEAD_DIM": self.head_dim, "V_IN_OFFSET": _v_offset},
                       (num_tokens, self.num_kv_heads, 1))

        # Flash attention with unrotated Q.
        _start_block, _eff_ctx_len = self._ctx_window(ctx_len)
        self._dispatch("flash_attn_decode",
                       [sc["q_buf"], k_cache, v_cache, bt_buf, sc["attn_out"]],
                       {"BLOCK_SIZE": self.block_size,
                        "NUM_Q_HEADS": self.num_q_heads,
                        "NUM_KV_HEADS": self.num_kv_heads,
                        "HEAD_DIM": self.head_dim,
                        "CTX_LEN": _eff_ctx_len,
                        "START_BLOCK": _start_block,
                        "SCALE": self._attn_scale},
                       (self.num_q_heads, 1, 1))

        # Output projection.
        w_key = f"{p}.self_attn.o_proj.weight"
        uq = self._uq_for_key(w_key)
        qi = self._quant_extra(f"{p}.self_attn.o_proj", uq)
        self._dispatch("matmul_quant",
                       [sc["attn_out"], self.weights[w_key],
                        self._scales_buf(w_key, uq, self._dummy_buf), sc["o_proj_out"]],
                       {"K": q_dim, "N": hidden, "USE_QUANT": uq, **qi},
                       (hidden, 1, 1))
        return sc["o_proj_out"]

    def _prefill_batch_forward(
        self,
        input_ids,
        positions,
        attn_metadata: object,
        T: int,
    ):
        """SmolLM3 prefill: route to sequential when NoPE layers are present.

        The inherited batch prefill from LlamaWebGPUModel dispatches RoPE for
        every layer unconditionally (it does not call _attn_block, so the
        _attn_block_nope override is bypassed). NoPE layers must store
        unrotated K in the KV cache and use unrotated Q for attention; applying
        RoPE to them silently produces wrong keys and queries.

        When all layers use RoPE (_nope_layers is empty), fall through to the
        Llama batch prefill path, which is correct and faster.
        """
        if self._nope_layers:
            return self._prefill_sequential_fallback(input_ids, positions, attn_metadata, T)
        return super()._prefill_batch_forward(input_ids, positions, attn_metadata, T)
