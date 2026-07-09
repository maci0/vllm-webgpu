from __future__ import annotations
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from vllm_webgpu.webgpu.pipeline import PipelineKey

if TYPE_CHECKING:
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.device import WebGPUDevice
    from vllm_webgpu.webgpu.pipeline import PipelineCache

# Scratch buffer rotation names shared across models.
_H_NAMES: tuple[str, str, str] = ("h0", "h1", "h2")


def _gemv_wg(N: int) -> tuple:
    """Workgroup count for matmul_quant dispatch.

    SPLIT_K=1 (one workgroup per output row): always (N, 1, 1).
    """
    return (N, 1, 1)


def _vec4_wg(N: int) -> tuple:
    """Workgroup count for element-wise vec4 dispatches (gelu_mul, add, ...).

    Each thread handles 4 elements packed as vec4<f16>.  The formula rounds N
    up to the nearest multiple of 4 then divides by the workgroup size (256).
    """
    return ((N // 4 + 255) // 256, 1, 1)

logger = logging.getLogger(__name__)



def compute_yarn_freqs(
    head_dim: int,
    rope_theta: float,
    rope_scaling: dict,
    rotary_dim: int | None = None,
) -> tuple[np.ndarray, float]:
    """Compute YaRN-scaled inverse frequencies for RoPE.

    Calls the same vLLM helpers used internally by YaRNScalingRotaryEmbedding
    without constructing the module, avoiding the 512 MB cos/sin cache that
    __init__ allocates via torch.einsum over (orig_ctx * factor) positions.

    Mirrors YaRNScalingRotaryEmbedding.__init__ (mscale) and
    _compute_inv_freq (freq array) exactly. Cross-check this function whenever
    vLLM bumps the YaRN formula in case the helpers or their composition change.

    Args:
        head_dim:    Full attention head dimension.
        rope_theta:  RoPE base frequency (e.g. 10000.0).
        rope_scaling: rope_scaling config dict from the model config.
        rotary_dim:  Number of head dimensions that receive RoPE. Defaults to
                     head_dim (full rotation). Pass rope_scaling.get('rotary_dim',
                     head_dim) for models that use partial RoPE; omitting it for
                     those models would produce wrong frequencies.

    Returns:
        freqs:  [rotary_dim // 2] float32 array of scaled inv_freq values.
        mscale: attention output scale factor (0.1 * ln(factor) + 1.0).
                Must be applied to the output of cos/sin in the shader, NOT
                folded into the frequencies (cos(pos * freq * mscale) is wrong).
    """
    import torch
    from vllm.model_executor.layers.rotary_embedding.common import (
        yarn_find_correction_range,
        yarn_get_mscale,
        yarn_linear_ramp_mask,
    )

    if rotary_dim is None:
        rotary_dim = head_dim

    factor               = float(rope_scaling.get("factor", 1.0))
    beta_fast            = int(rope_scaling.get("beta_fast", 32))
    beta_slow            = int(rope_scaling.get("beta_slow", 1))
    orig_ctx             = int(rope_scaling.get("original_max_position_embeddings", 4096))
    extrapolation_factor = float(rope_scaling.get("extrapolation_factor", 1.0))
    attn_factor          = float(rope_scaling.get("attn_factor", 1.0))
    apply_yarn_scaling   = bool(rope_scaling.get("apply_yarn_scaling", True))
    truncate             = bool(rope_scaling.get("truncate", True))

    # Mirrors YaRNScalingRotaryEmbedding.__init__ (yarn_scaling_rope.py:40-43)
    mscale = (
        float(yarn_get_mscale(factor) * attn_factor)
        if apply_yarn_scaling
        else float(attn_factor)
    )

    # Mirrors YaRNScalingRotaryEmbedding._compute_inv_freq (yarn_scaling_rope.py:49-73)
    pos_freqs = rope_theta ** (
        torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim
    )
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    low, high = yarn_find_correction_range(
        beta_fast, beta_slow, rotary_dim, rope_theta, orig_ctx, truncate
    )
    inv_freq_mask = (
        1 - yarn_linear_ramp_mask(low, high, rotary_dim // 2, dtype=torch.float32)
    ) * extrapolation_factor
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_mask)
        + inv_freq_extrapolation * inv_freq_mask
    )

    return inv_freq.numpy().astype(np.float32), mscale



class BaseWebGPUModel(ABC):
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
        # Logit readback: set by subclasses before returning from forward().
        self._last_logit_buf: "WebGPUBuffer | None" = None
        self._last_vocab: int = 0
        self._prof_stats: dict[str, list[float]] = defaultdict(list)  # shader -> [ms, ...]
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
        from vllm_webgpu.webgpu.buffer import WebGPUBuffer as _WGPUBuf
        self._rope_freq_buf: "WebGPUBuffer" = _WGPUBuf.empty(
            wgpu_device.wgpu_device, 4)  # 1-element f32 placeholder
        self._use_freq_buf: bool = False
        self._yarn_mscale: float = 1.0  # set to mscale when rope_type='yarn'
        # Greedy-decode flag: True means forward() returns a (1,1) int32 token
        # ID via GPU argmax; False means it returns (1, vocab) float32 logits
        # for temperature sampling. Initialized True so hasattr() returns True,
        # allowing the model runner to flip it to False for non-greedy requests.
        self._greedy_decode: bool = True

    @staticmethod
    def _vals_per_thread(hidden_size: int) -> int:
        """Return the VALS_PER_THREAD constant for rms_norm shaders.

        Each workgroup covers 256 threads. When hidden_size fits within
        256*16 elements, each thread handles ceil(hidden/256) values (capped
        at 16). Larger hidden sizes require a different shader path (0 signals
        the caller to fall back).
        """
        if hidden_size <= 256 * 16:
            return min((hidden_size + 255) // 256, 16)
        return 0

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
        try:
            yield
            if self.profiling and label:
                t0 = time.perf_counter()
            dev.queue.submit([encoder.finish()])
            if self.profiling and label:
                dev.queue.on_submitted_work_done_sync()
                self._prof_stats[label].append((time.perf_counter() - t0) * 1000.0)
        finally:
            self._active_encoder = saved_encoder

    def profile_report(self) -> str:
        """Return a formatted profiling report. Call after forward() with profiling=True."""
        if not self._prof_stats:
            return "No profiling data. Set model.profiling=True before forward()."
        lines = ["Kernel timing (ms per call, averaged):"]
        rows = [(lbl, sum(t) / len(t), len(t)) for lbl, t in self._prof_stats.items()]
        rows.sort(key=lambda r: r[1] * r[2], reverse=True)
        total = sum(r[1] * r[2] for r in rows)
        for label, avg, n in rows:
            label_total = avg * n
            pct = 100.0 * label_total / total if total else 0
            lines.append(f"  {label:<40s} {avg:7.3f} ms  x{n:4d}  {label_total:8.3f} ms  {pct:5.1f}%")
        lines.append(f"  {'TOTAL':<40s} {'':7s}       {'':6s}  {total:8.3f} ms")
        return "\n".join(lines)

    def profile_reset(self) -> None:
        self._prof_stats.clear()

    def _bt_arr(self, attn_metadata: object) -> "np.ndarray":
        """Return the block-table as a uint32 numpy array.

        Uses block_tables[0] when present, falling back to a single-element [0]
        placeholder for warmup or metadata objects that lack a block table.
        """
        return np.array(getattr(attn_metadata, "block_tables", [[0]])[0], dtype=np.uint32)

    def load_weights(
        self, path: str, f32_keys: "frozenset[str] | None" = None
    ) -> None:
        """Load model weights from a HuggingFace safetensors directory.

        Supports single-file (model.safetensors) and sharded (model.safetensors.index.json)
        safetensors formats. GGUF loading not supported — use the vllm-gguf plugin.
        MLX affine-int4 (Qwen3.5-9B MLX community format) is supported as a special case.

        Args:
            f32_keys: Optional set of checkpoint key names that must be uploaded as float32
                      instead of the default float16. Passed through to the safetensors loader.
        """
        from vllm.transformers_utils.repo_utils import get_model_path
        from vllm_webgpu.quant.weight_loader import (
            detect_weight_format, load_safetensors_weights,
            load_safetensors_weights_sharded, load_mlx_weights,
        )
        path = str(get_model_path(path))
        fmt = detect_weight_format(path)
        if fmt == "safetensors":
            # If path is a directory, the actual file is model.safetensors inside it.
            p = Path(path)
            actual = str(p / "model.safetensors") if p.is_dir() else path
            self.weights = load_safetensors_weights(
                actual, self.wgpu_device.wgpu_device, f32_keys=f32_keys)
        elif fmt == "safetensors_sharded":
            self.weights = load_safetensors_weights_sharded(
                path, self.wgpu_device.wgpu_device, f32_keys=f32_keys)
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
        return self.weights.get("__quant_meta__", {}).get(base_key, {})

    # ── GPU sampler helpers ───────────────────────────────────────────────────


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
        return val

    def _ensure_sample_buf(self, vocab: int) -> "WebGPUBuffer":
        """Lazily allocate GPU sampler buffers sized for vocab and return the token output buffer."""
        if self._gpu_sample_tok is None or self._gpu_sample_vocab != vocab:
            import wgpu as wgpu_lib
            from vllm_webgpu.webgpu.buffer import WebGPUBuffer
            dev = self.wgpu_device.wgpu_device
            self._gpu_sample_tok   = WebGPUBuffer.empty(dev, 4)      # 1 × u32
            self._gpu_sample_vocab = vocab
            # MAP_READ staging buffer: copy argmax result here inside the MAIN command encoder,
            # then map after the single main sync — eliminates the second GPU sync per token.
            self._gpu_sample_staging = dev.create_buffer(
                size=4,
                usage=wgpu_lib.BufferUsage.COPY_DST | wgpu_lib.BufferUsage.MAP_READ)
        return self._gpu_sample_tok  # type: ignore[return-value]

    def logit_readback(self) -> "np.ndarray":
        """Full vocab logits GPU->CPU (only for temperature sampling or analysis)."""
        if self._last_logit_buf is None:
            raise RuntimeError("logit_readback() called before forward()")
        return (
            self._last_logit_buf.to_numpy()
            .view(np.float16)
            .reshape(1, self._last_vocab)
            .astype(np.float32)
        )

    def _scales_buf(self, w_key: str, uq: int, fallback: "object") -> "object":
        """Return the GPU scales buffer for any quant format.

        For GPU quants (USE_QUANT 3-8): scales live at w_key + '.scales'
            e.g. 'model.layers.0.self_attn.q_proj.weight.scales'
        For plain weights (USE_QUANT 0): no scales exist; return fallback.
        """
        if uq in (3, 4, 5, 6, 7, 8):
            return self.weights.get(w_key + ".scales", fallback)
        return fallback

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
            elif gs is not None:
                raise ValueError(
                    f"fp8_gpu with group_size={gs} > 1 is not supported by this shader path"
                )
            return d
        if uq == 8:
            # NF4: GROUP_K = absmax block size (BnB default 64).
            return {"GROUP_K": self._quant_info(base_key).get("group_size", 64)}
        return {}

    def _uq_for_key(self, key: str) -> int:
        """Return USE_QUANT for a weight key (closure-free helper)."""
        w = self.weights.get(key)
        if w is not None:
            dtype = getattr(w, "dtype", "f16")
            meta = self._quant_info(key.removesuffix(".weight"))
            fmt = meta.get("fmt", "")
            if dtype == "i32":
                return 4 if fmt == "awq_sym" else 3
            if dtype == "u8":
                if fmt == "nvfp4_gpu": return 6
                if fmt == "int8_gpu":  return 7
                if fmt == "fp8_gpu":   return 5
                if fmt == "nf4_gpu":   return 8
        return 0

    def _dispatch(
        self,
        shader_name: str,
        bindings: "list[WebGPUBuffer]",
        constants: dict[str, int | float],
        workgroups: tuple[int, int, int],
        shader_subdir: str = "generic",
    ) -> None:
        # matmul_quant always declares binding 4 (bias). Callers that don't set
        # HAS_BIAS=1 still need to provide a buffer so the bind group layout matches.
        if shader_name == "matmul_quant" and len(bindings) == 4:
            if self._dummy_bias_buf is None:
                import wgpu as wgpu_lib
                from vllm_webgpu.webgpu.buffer import WebGPUBuffer
                dev = self.wgpu_device.wgpu_device
                self._dummy_bias_buf = WebGPUBuffer.empty(
                    dev, 4,
                    usage=wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC,
                )
            bindings = list(bindings) + [self._dummy_bias_buf]

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
        """Run one forward pass and return output as a numpy array.

        When logit_returns_token_id is False: returns float32 [num_tokens, vocab_size] logits.
        When logit_returns_token_id is True:  returns int32 [1, 1] with the GPU-argmax token id.
        """
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
            _Dec = SimpleNamespace(
                slot_mapping=[0],
                block_tables=[np.array([0], dtype=np.uint32)],
                max_decode_seq_len=1,
            )
            self.forward(np.array([0], dtype=np.uint32),
                         np.array([0], dtype=np.uint32), _Dec)

            # Prefill warmup: compiles matmul_quant_mr4, flash_attn_prefill, etc.
            T = 4
            bt = np.array([0], dtype=np.uint32)
            _Pre = SimpleNamespace(
                slot_mapping=list(range(T)),
                block_tables=[bt.copy()],
                max_decode_seq_len=T,
            )
            self.forward(np.zeros(T, dtype=np.uint32),
                         np.arange(T, dtype=np.uint32), _Pre)

            logger.info("Warmup complete (decode + prefill)")
        except Exception as exc:
            logger.warning("Warmup failed (non-fatal): %s", exc)
