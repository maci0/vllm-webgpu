from __future__ import annotations
import logging
import time
from abc import abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np

from vllm_webgpu.webgpu.pipeline import PipelineKey

if TYPE_CHECKING:
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.device import WebGPUDevice
    from vllm_webgpu.webgpu.pipeline import PipelineCache

logger = logging.getLogger(__name__)

# Shader names + expected binding count for RoPE shaders that need a dummy
# inv_freq_buf appended when USE_FREQ_BUF=0. Defined at module level to avoid
# allocating a new set on every _dispatch() call.
_ROPE_SHADERS_BY_LEN = {
    ("rope", 3), ("fused_per_head_norm_rope", 4), ("fused_qk_norm_rope", 7),
}


def compute_yarn_freqs(head_dim: int, rope_theta: float, rope_scaling: dict) -> tuple[np.ndarray, float]:
    """Compute YaRN-scaled inverse frequencies for RoPE.

    Returns:
        freqs:  [head_dim // 2] float32 array of scaled inv_freq values.
        mscale: attention output scale factor (0.1 * ln(factor) + 1.0).
                Must be applied to the output of cos/sin in the shader, NOT
                folded into the frequencies — cos(pos * freq * mscale) is wrong.
    """
    factor    = float(rope_scaling.get("factor", 1.0))
    beta_fast = float(rope_scaling.get("beta_fast", 32.0))
    beta_slow = float(rope_scaling.get("beta_slow", 1.0))
    orig_ctx  = int(rope_scaling.get("original_max_position_embeddings", 4096))

    inv_freq = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))

    low_freq_len  = orig_ctx / beta_slow
    high_freq_len = orig_ctx / beta_fast
    wavelengths   = 2.0 * np.pi / inv_freq

    # Three regions by wavelength:
    #   short (< high_freq_len): high-frequency dimensions, no scaling
    #   long  (> low_freq_len):  low-frequency dimensions, scale by factor
    #   middle: smooth linear interpolation between the two extremes
    interp_scale = (orig_ctx / wavelengths - beta_fast) / (beta_slow - beta_fast)
    blended = inv_freq * (1.0 - interp_scale * (1.0 - 1.0 / factor))

    scaled_inv_freq = np.where(
        wavelengths < high_freq_len,
        inv_freq,
        np.where(wavelengths > low_freq_len, inv_freq / factor, blended),
    )

    # YaRN attention scale: mscale = 0.1 * ln(factor) + 1.0
    # Must be applied AFTER cos/sin in the shader (mscale * cos(pos * freq)),
    # not folded into inv_freq (which would compute cos(pos * freq * mscale) instead).
    mscale = 0.1 * np.log(factor) + 1.0
    return scaled_inv_freq.astype(np.float32), float(mscale)



class BaseWebGPUModel:
    # Declare whether forward() can return a (1, 1) int32 token ID instead of
    # full (1, vocab) float32 logits. Subclasses that implement logit_readback()
    # and GPU argmax set this to True so the runner can rely on a stable contract
    # rather than testing for the existence of the logit_readback method.
    logit_returns_token_id: bool = False

    def __init__(self, model_config, wgpu_device: "WebGPUDevice", pipeline_cache: "PipelineCache") -> None:
        self.model_config = model_config
        self.wgpu_device = wgpu_device
        self.pipeline_cache = pipeline_cache
        self.weights: dict[str, "WebGPUBuffer"] = {}
        self.kv_pool: list[tuple["WebGPUBuffer", "WebGPUBuffer"]] = []
        self._active_encoder = None  # set when inside a _batched_dispatch() context
        # Profiling
        self.profiling: bool = False
        # GPU sampler: pre-allocated buffers for GPU argmax.
        # Allocated lazily on first call (need vocab_size from subclass).
        self._gpu_sample_tok: "WebGPUBuffer | None" = None    # [1] u32 next token (STORAGE)
        self._gpu_sample_vocab: int = 0
        self._gpu_sample_staging = None   # MAP_READ staging buffer for zero-sync readback
        self._gpu_sample_tok_cpu: int = 0  # cached CPU result after readback
        self._prof_stats: dict[str, list[float]] = defaultdict(list)  # shader -> [ms, ...]
        self._prof_current_label: str = ""  # set per _batched_dispatch block
        # Dummy bias buffer for matmul_quant binding 4 (allocated on first use).
        # The shader always declares binding 4; callers that don't use HAS_BIAS
        # must still provide a buffer so the bind group layout matches.
        self._dummy_bias_buf: "WebGPUBuffer | None" = None
        # Precomputed RoPE inverse frequencies for USE_FREQ_BUF=1 (YaRN and similar).
        # All rope/fused-rope shaders declare an inv_freq_buf binding unconditionally
        # (wgpu-native does not eliminate dead bindings even at USE_FREQ_BUF=0), so
        # every dispatch must provide a buffer at the slot. Initialised here as a
        # 1-element dummy; LlamaWebGPUModel._init_rope_freq_buf() replaces it with
        # actual YaRN frequencies when rope_scaling.rope_type == "yarn".
        import wgpu as _wgpu
        from vllm_webgpu.webgpu.buffer import WebGPUBuffer as _WGPUBuf
        _rw = _wgpu.BufferUsage.STORAGE | _wgpu.BufferUsage.COPY_SRC | _wgpu.BufferUsage.COPY_DST
        self._rope_freq_buf: "WebGPUBuffer" = _WGPUBuf.empty(
            wgpu_device.wgpu_device, 4, usage=_rw)  # 1-element f32 placeholder
        self._use_freq_buf: bool = False
        self._yarn_mscale: float = 1.0  # set to mscale when rope_type='yarn'

    @contextmanager
    def _batched_dispatch(self, label: str = ""):
        """Record dispatch calls into a single CommandEncoder and submit once at exit.

        Re-entrant when not profiling: if an outer _batched_dispatch is already active,
        inner calls simply record into the existing encoder (no extra submit).
        When profiling=True, each named block gets its own encoder + GPU sync for timing.
        """
        dev = self.wgpu_device.wgpu_device

        if not self.profiling and self._active_encoder is not None:
            # Re-entrant: record into the outer encoder, no additional submit.
            yield
            return

        # Create a new encoder; save/restore the outer encoder for profiling re-entrancy.
        encoder = dev.create_command_encoder()
        saved_encoder = self._active_encoder
        self._active_encoder = encoder
        self._prof_current_label = label
        try:
            yield
            t0 = time.perf_counter() if (self.profiling and label) else 0.0
            dev.queue.submit([encoder.finish()])
            if self.profiling and label:
                dev.queue.on_submitted_work_done_sync()
                self._prof_stats[label].append((time.perf_counter() - t0) * 1000.0)
        finally:
            self._active_encoder = saved_encoder
            self._prof_current_label = ""

    def profile_report(self) -> str:
        """Return a formatted profiling report. Call after forward() with profiling=True."""
        if not self._prof_stats:
            return "No profiling data. Set model.profiling=True before forward()."
        lines = ["Kernel timing (ms per call, averaged):"]
        total = 0.0
        rows = []
        for label, times in sorted(self._prof_stats.items(), key=lambda x: -sum(x[1])):
            avg = sum(times) / len(times)
            total += avg
            rows.append((label, avg, len(times)))
        for label, avg, n in rows:
            pct = 100.0 * avg / total if total else 0
            lines.append(f"  {label:<40s} {avg:7.3f} ms  {pct:5.1f}%  (n={n})")
        lines.append(f"  {'TOTAL':<40s} {total:7.3f} ms")
        return "\n".join(lines)

    def profile_reset(self) -> None:
        self._prof_stats.clear()

    @staticmethod
    def _resolve_model_path(path: str) -> str:
        """Resolve a HuggingFace model ID or local path to an actual directory."""
        from pathlib import Path
        if Path(path).exists():
            return path
        try:
            from huggingface_hub import snapshot_download
            return snapshot_download(path, local_files_only=True)
        except Exception:
            pass
        return path  # let the caller fail with a meaningful error

    def load_weights(self, path: str) -> None:
        """Load model weights from a HuggingFace safetensors directory.

        Supports single-file (model.safetensors) and sharded (model.safetensors.index.json)
        safetensors formats. GGUF loading not supported — use the vllm-gguf plugin.
        MLX affine-int4 (Qwen3.5-9B MLX community format) is supported as a special case.
        """
        from vllm_webgpu.quant.weight_loader import (
            detect_weight_format, load_safetensors_weights,
            load_safetensors_weights_sharded, load_mlx_weights,
        )
        path = self._resolve_model_path(path)
        fmt = detect_weight_format(path)
        if fmt == "safetensors":
            # If path is a directory, the actual file is model.safetensors inside it.
            from pathlib import Path as _Path
            actual = str(_Path(path) / "model.safetensors") if _Path(path).is_dir() else path
            self.weights = load_safetensors_weights(actual, self.wgpu_device.wgpu_device)
        elif fmt == "safetensors_sharded":
            self.weights = load_safetensors_weights_sharded(path, self.wgpu_device.wgpu_device)
        elif fmt == "mlx_int4":
            # MLX community format (Qwen3.5-9B): affine int4 with bf16 scales/biases
            self.weights = load_mlx_weights(path, self.wgpu_device.wgpu_device)
        elif fmt == "gguf":
            raise ValueError(
                f"GGUF format not supported by this plugin — use the vllm-gguf plugin: {path}"
            )
        else:
            raise ValueError(f"Unknown weight format for {path}")
        logger.info("Loaded %d weight tensors (%s format)", len(self.weights), fmt)


    def _quant_info(self, base_key: str) -> dict:
        """Return quantization metadata for a weight base key, or empty dict."""
        meta = self.weights.get("__quant_meta__", {})
        return meta.get(base_key, {})

    # ── GPU sampler helpers ───────────────────────────────────────────────────

    def _ensure_gpu_sampler(self, vocab: int) -> None:
        """Lazily allocate GPU sampler buffers sized for vocab."""
        if self._gpu_sample_tok is not None and self._gpu_sample_vocab == vocab:
            return
        import wgpu as wgpu_lib
        from vllm_webgpu.webgpu.buffer import WebGPUBuffer
        dev = self.wgpu_device.wgpu_device
        rw = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST
        self._gpu_sample_tok   = WebGPUBuffer.empty(dev, 4, usage=rw)      # 1 × u32
        self._gpu_sample_vocab = vocab
        # MAP_READ staging buffer: copy argmax result here inside the MAIN command encoder,
        # then map after the single main sync — eliminates the second GPU sync per token.
        self._gpu_sample_staging = dev.create_buffer(
            size=4,
            usage=wgpu_lib.BufferUsage.COPY_DST | wgpu_lib.BufferUsage.MAP_READ)

    def _copy_sample_to_staging(self) -> None:
        """Copy the 4-byte argmax result to the MAP_READ staging buffer.

        Called INSIDE _batched_dispatch() so the copy is part of the main command
        encoder. After the main on_submitted_work_done_sync(), the result is ready
        to map without a second submit/sync cycle.
        """
        if self._gpu_sample_staging is None or self._gpu_sample_tok is None:
            return
        if self._active_encoder is not None:
            self._active_encoder.copy_buffer_to_buffer(
                self._gpu_sample_tok.buf, 0, self._gpu_sample_staging, 0, 4)

    def _read_sample_tok(self) -> int:
        """Map and read the staging buffer (no submit/sync — already done by main batch)."""
        import wgpu as wgpu_lib
        if self._gpu_sample_staging is None:
            return 0
        self._gpu_sample_staging.map_sync(mode=wgpu_lib.MapMode.READ)
        val = int(np.frombuffer(self._gpu_sample_staging.read_mapped(), dtype=np.uint32)[0])
        self._gpu_sample_staging.unmap()
        return int(val)

    def _scales_buf(self, w_key: str, uq: int, fallback: "object") -> "object":
        """Return the GPU scales buffer for any quant format.

        For GPU quants (USE_QUANT 3-8): scales live at w_key + '.scales'
            e.g. 'model.layers.0.self_attn.q_proj.weight.scales'
        For simple Q4 (USE_QUANT 1): scales live at w_key[:-7] + '.scales'
            e.g. 'model.layers.0.self_attn.q_proj.scales'
        """
        if uq in (3, 4, 5, 6, 7, 8):
            return self.weights.get(w_key + ".scales", fallback)
        return self.weights.get(w_key[:-7] + ".scales", fallback)

    def _quant_extra(self, base_key: str, uq: int) -> dict:
        """Return additional shader override constants for quantized dispatch."""
        if uq in (3, 4):
            return {"GROUP_K": self._quant_info(base_key).get("group_size", 128)}
        if uq in (5, 6):
            meta = self._quant_info(base_key)
            d: dict = {"GLOBAL_SCALE": float(meta.get("global_scale", 1.0))}
            gs = meta.get("group_size")
            if uq == 6:
                d["GROUP_K"] = gs if gs is not None else 16
            elif gs == 1:
                # Per-channel FP8: GROUP_K=1 signals the shader to read scales[row].
                d["GROUP_K"] = 1
            return d
        if uq == 7:
            # Int8 per-channel: no group size, no global scale — just USE_QUANT=7.
            return {}
        if uq == 8:
            # NF4: GROUP_K = absmax block size (BnB default 64).
            return {"GROUP_K": self._quant_info(base_key).get("group_size", 64)}
        return {}

    def _uq_for_key(self, key: str) -> int:
        """Return USE_QUANT for a weight key (closure-free helper)."""
        quant_types = self.weights.get("__quant_types__", {})
        _qt = quant_types if isinstance(quant_types, dict) else {}
        tt = _qt.get(key, 0)
        if tt == 12:
            return 2
        w = self.weights.get(key)
        if w is not None:
            dtype = getattr(w, "dtype", "f16")
            qmeta = self.weights.get("__quant_meta__", {})
            meta = qmeta.get(key[:-7], {}) if isinstance(qmeta, dict) else {}
            fmt = meta.get("fmt", "")
            if dtype == "i32":
                return 4 if fmt == "awq_sym" else 3
            if dtype == "u8":
                if fmt == "nvfp4_gpu": return 6
                if fmt == "int8_gpu":  return 7
                if fmt == "fp8_gpu":   return 5
                if fmt == "nf4_gpu":   return 8
        if self.weights.get(key[:-7] + ".scales") is not None:
            return 1
        return 0

    def _dispatch(
        self,
        shader_name: str,
        bindings: "list[WebGPUBuffer]",
        constants: dict[str, int | float],
        workgroups: tuple[int, int, int],
        shader_subdir: str = "generic",
    ) -> None:
        import wgpu as wgpu_lib

        # matmul_quant always declares binding 4 (bias). Callers that don't set
        # HAS_BIAS=1 still need to provide a buffer so the bind group layout matches.
        if shader_name == "matmul_quant" and len(bindings) == 4:
            if self._dummy_bias_buf is None:
                from vllm_webgpu.webgpu.buffer import WebGPUBuffer
                dev = self.wgpu_device.wgpu_device
                self._dummy_bias_buf = WebGPUBuffer.empty(
                    dev, 4,
                    usage=wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC,
                )
            bindings = list(bindings) + [self._dummy_bias_buf]

        # rope/fused RoPE shaders always declare an inv_freq_buf binding for USE_FREQ_BUF=1.
        # Callers that keep USE_FREQ_BUF=0 (no YaRN) must still satisfy the layout.
        #   rope:                    3 bindings + dummy at slot 3
        #   fused_per_head_norm_rope: 4 bindings + dummy at slot 4
        #   fused_qk_norm_rope:      7 bindings + dummy at slot 7
        if (shader_name, len(bindings)) in _ROPE_SHADERS_BY_LEN:
            # Use the pre-allocated rope_freq_buf placeholder (initialized in __init__).
            # When USE_FREQ_BUF=1 (YaRN), _init_rope_freq_buf() replaces it with real data.
            bindings = list(bindings) + [self._rope_freq_buf]

        key = PipelineKey(
            shader_name=f"{shader_subdir}/{shader_name}",
            defines=tuple(sorted(constants.items())),
        )
        pipeline = self.pipeline_cache.get_or_create(key)

        dev = self.wgpu_device.wgpu_device
        bg_layout = pipeline.get_bind_group_layout(0)
        entries = [
            {"binding": i, "resource": {"buffer": buf.buf}}
            for i, buf in enumerate(bindings)
        ]
        bg = dev.create_bind_group(layout=bg_layout, entries=entries)

        if self._active_encoder is not None:
            # Batch mode: record into the shared encoder; submit happens at context manager exit.
            encoder = self._active_encoder
        else:
            # Standalone mode: create a fresh encoder and submit immediately.
            encoder = dev.create_command_encoder()

        cp = encoder.begin_compute_pass()
        cp.set_pipeline(pipeline)
        cp.set_bind_group(0, bg)
        cp.dispatch_workgroups(*workgroups)
        cp.end()

        if self._active_encoder is None:
            dev.queue.submit([encoder.finish()])

    @abstractmethod
    def forward(
        self,
        input_ids: np.ndarray,
        positions: np.ndarray,
        attn_metadata: object,
    ) -> np.ndarray:
        """Returns logits as float32 numpy array [num_tokens, vocab_size]."""
        ...

    def warmup(self) -> None:
        """Pre-compile shader pipelines by running dummy decode and prefill steps.

        Covers both the decode hot path (T=1) and the prefill path (T=4) so
        that first-inference latency from shader JIT compilation is eliminated.
        Any error is logged and swallowed so warmup never blocks inference from
        starting.

        Subclasses may override to exercise additional shader combinations.
        """
        if not self.weights:
            logger.info("Skipping warmup: weights not loaded")
            return
        if not self.kv_pool:
            logger.info("Skipping warmup: KV pool not allocated")
            return
        logger.info("Warming up shader pipelines (decode + prefill)...")
        try:
            # Decode warmup: compiles all decode-path shaders.
            class _Dec:
                slot_mapping = [0]
                block_tables = [np.array([0], dtype=np.uint32)]
                max_decode_seq_len = 1
            self.forward(np.array([0], dtype=np.uint32),
                         np.array([0], dtype=np.uint32), _Dec())

            # Prefill warmup: compiles matmul_quant_mr4, flash_attn_prefill, etc.
            T = 4
            bt = np.zeros(max(len(self.kv_pool[0][0].shape) if self.kv_pool else 1, T), dtype=np.uint32)
            for i in range(T):
                bt[i // 16] = i // 16
            class _Pre:
                slot_mapping = list(range(T))
                block_tables = [bt.copy()]
                max_decode_seq_len = T
            self.forward(np.zeros(T, dtype=np.uint32),
                         np.arange(T, dtype=np.uint32), _Pre())

            logger.info("Warmup complete (decode + prefill)")
        except Exception as exc:
            logger.warning("Warmup failed (non-fatal): %s", exc)
