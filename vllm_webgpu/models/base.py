from __future__ import annotations
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from typing import TYPE_CHECKING

import numpy as np

from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE as _SAFE_WEIGHTS_NAME
from vllm.logger import init_logger
from vllm_webgpu.webgpu.buffer import WebGPUBuffer
from vllm_webgpu.webgpu.pipeline import PipelineKey

if TYPE_CHECKING:
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

    Each thread handles 4 elements packed as vec4<f16>. The formula rounds the
    thread count up to fill complete workgroups of 256.
    """
    return (((N + 3) // 4 + 255) // 256, 1, 1)


def _rows_wg(N: int) -> tuple:
    """Workgroup count for SPLIT_K=0 row-parallel dispatches (LM head, matmul rows).

    Each workgroup covers 256 output rows. Used for lm_head and other matmuls
    where SPLIT_K=0 assigns one workgroup per output tile of 256 rows.
    """
    return ((N + 255) // 256, 1, 1)

logger = init_logger(__name__)



def compute_yarn_freqs(
    head_dim: int,
    rope_theta: float,
    rope_scaling: dict,
    rotary_dim: int | None = None,
) -> tuple[np.ndarray, float]:
    """Compute YaRN-scaled inverse frequencies for RoPE.

    Implements the YaRN inv_freq formula directly using
    yarn_find_correction_range and yarn_linear_ramp_mask from vLLM, avoiding
    the object.__new__ bypass that was fragile across vLLM version bumps.

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
    orig_ctx             = int(rope_scaling["original_max_position_embeddings"])
    extrapolation_factor = float(rope_scaling.get("extrapolation_factor", 1.0))
    attn_factor          = float(rope_scaling.get("attn_factor", 1.0))
    apply_yarn_scaling   = bool(rope_scaling.get("apply_yarn_scaling", True))
    truncate             = bool(rope_scaling.get("truncate", True))

    mscale = (
        float(yarn_get_mscale(factor) * attn_factor)
        if apply_yarn_scaling
        else float(attn_factor)
    )

    pos_freqs = rope_theta ** (
        torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim
    )
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    low, high = yarn_find_correction_range(
        beta_fast, beta_slow, rotary_dim, rope_theta, orig_ctx, truncate
    )
    inv_freq_mask = (
        1 - yarn_linear_ramp_mask(low, high, rotary_dim // 2, dtype=torch.float)
    ) * extrapolation_factor
    # The five lines below mirror YaRNScalingRotaryEmbedding._compute_inv_freq
    # (vllm/model_executor/layers/rotary_embedding/yarn_scaling_rope.py, lines 50-72).
    # Instantiating that class would trigger a full cos/sin cache build (expensive),
    # so the blend is reproduced here using the same three utility functions imported above.
    # If vLLM changes the blend formula, update this block to match.
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_mask)
        + inv_freq_extrapolation * inv_freq_mask
    )
    return inv_freq.numpy(), mscale



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
        self._gpu_sample_staging = None   # MAP_READ staging buffer for zero-sync readback
        # Logit readback: set by subclasses before returning from forward().
        self._last_logit_buf: "WebGPUBuffer | None" = None
        self._last_vocab: int = 0
        self._prof_stats: dict[str, list[float]] = defaultdict(list)  # shader -> [ms, ...]
        # Dummy bias buffer for matmul_quant binding 4.
        # The shader always declares binding 4; callers that don't use HAS_BIAS
        # must still provide a buffer so the bind group layout matches.
        self._dummy_bias_buf: "WebGPUBuffer" = WebGPUBuffer.empty(wgpu_device.wgpu_device, 4)
        # Dummy scales buffer for the scales slot on USE_QUANT=0 dispatches.
        # Subclasses that do not override this must still bind something at the
        # scales slot so the bind group layout matches.
        self._dummy_scales_buf: "WebGPUBuffer" = WebGPUBuffer.empty(wgpu_device.wgpu_device, 4)
        # Precomputed RoPE inverse frequencies for USE_FREQ_BUF=1 (YaRN and similar).
        # All rope/fused-rope shaders declare an inv_freq_buf binding unconditionally
        # (wgpu-native does not eliminate dead bindings even at USE_FREQ_BUF=0), so
        # every dispatch must provide a buffer at the slot. Initialised here as a
        # 1-element dummy; LlamaWebGPUModel._init_rope_freq_buf() replaces it with
        # actual YaRN frequencies when rope_scaling.rope_type == "yarn".
        self._rope_freq_buf: "WebGPUBuffer" = WebGPUBuffer.empty(
            wgpu_device.wgpu_device, 4)  # 1-element f32 placeholder
        self._use_freq_buf: bool = False
        self._yarn_mscale: float = 1.0  # set to mscale when rope_type='yarn'
        # Greedy-decode flag: True means forward() returns a (1,1) int32 token
        # ID via GPU argmax; False means it returns (1, vocab) float32 logits
        # for temperature sampling. Initialized True so hasattr() returns True,
        # allowing the model runner to flip it to False for non-greedy requests.
        self._greedy_decode: bool = True
        # Per-key weight transforms applied during load_weights before GPU upload.
        # Keys are checkpoint key names; values are callables (np.ndarray) -> np.ndarray.
        # Populated by subclasses (e.g. LlamaWebGPUModel tiles shared norm weights)
        # to avoid a GPU roundtrip (to_numpy → tile → re-upload) in _postprocess_weights.
        self._weight_transforms: dict = {}

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

        Nested profiling and data-dependency ordering
        ---------------------------------------------
        When profiling=True, forward() wraps the entire pass in an unlabeled outer
        _batched_dispatch() and each transformer layer in a labeled inner one.  Without
        special handling the GPU would receive all inner-layer commands before the
        embedding and initial-norm commands that produce their inputs, because the outer
        encoder is only submitted after all inner blocks finish.

        Fix: when a labeled inner block is entered while an outer encoder is already
        active, we submit the outer encoder's accumulated commands immediately (so the
        GPU executes them first), then create a fresh replacement encoder.  That
        replacement becomes the new "outer" encoder: commands recorded between inner
        blocks (and after the last one) are captured there and submitted when the outer
        context manager exits.  At exit, each level submits self._active_encoder rather
        than the original encoder local, so the outer CM picks up the replacement
        encoder that holds the post-layer commands (final norm, LM head).
        """
        dev = self.wgpu_device.wgpu_device

        if not self.profiling and self._active_encoder is not None:
            # Re-entrant: record into the outer encoder, no additional submit.
            yield
            return

        saved_encoder = self._active_encoder

        if self.profiling and saved_encoder is not None:
            # Flush the outer encoder's pending commands (e.g. embedding lookup,
            # initial RMSNorm) before this inner block starts, so the GPU
            # executes them in dependency order.  A replacement encoder is created
            # so that commands recorded after this inner block exits are captured
            # and submitted when the outer context manager exits.
            dev.queue.submit([saved_encoder.finish()])
            saved_encoder = dev.create_command_encoder()

        encoder = dev.create_command_encoder()
        self._active_encoder = encoder
        try:
            yield
            # Submit the currently active encoder for this level.  For inner CMs,
            # self._active_encoder is still `encoder`.  For the outer CM, inner
            # blocks may have replaced self._active_encoder with a fresh
            # replacement encoder that holds post-layer commands; submit that one.
            t0 = time.perf_counter() if (self.profiling and label) else 0.0
            dev.queue.submit([self._active_encoder.finish()])
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
        rows = sorted([(lbl, sum(t), len(t)) for lbl, t in self._prof_stats.items()],
                      key=lambda r: r[1], reverse=True)
        total = sum(r[1] for r in rows)
        for label, sum_t, n in rows:
            avg = sum_t / n
            pct = 100.0 * sum_t / total if total else 0
            lines.append(f"  {label:<40s} {avg:7.3f} ms  x{n:4d}  {sum_t:8.3f} ms  {pct:5.1f}%")
        lines.append(f"  {'TOTAL':<40s} {'':7s}       {'':6s}  {total:8.3f} ms")
        return "\n".join(lines)

    def profile_reset(self) -> None:
        self._prof_stats.clear()

    def _check_single_sequence(self, attn_metadata: object) -> None:
        """Raise RuntimeError if more than one sequence is present in attn_metadata.

        Each forward() call handles exactly one sequence. Batching N sequences
        requires N separate pre-allocated buffer sets and per-sequence attention
        dispatch, which is not currently implemented.
        """
        if hasattr(attn_metadata, "block_tables") and len(attn_metadata.block_tables) > 1:
            raise RuntimeError(
                f"multi-sequence batching not supported: got {len(attn_metadata.block_tables)} "
                "block tables; call forward() once per decode request"
            )

    def _compute_ctx_len(self, attn_metadata: object, positions: "np.ndarray") -> int:
        """Derive the decode context length from attn_metadata, falling back to position."""
        v = attn_metadata.max_decode_seq_len
        return int(v if v is not None else positions[-1] + 1)

    def _bt_arr(self, attn_metadata: object) -> "np.ndarray":
        """Return the block-table as a uint32 numpy array.

        Uses block_tables[0] when present, falling back to a single-element [0]
        placeholder for warmup or metadata objects that lack a block table.
        """
        return np.asarray(getattr(attn_metadata, "block_tables", [[0]])[0], dtype=np.uint32)

    def load_weights(
        self, path: str, f32_keys: "frozenset[str] | None" = None,
        skip_prefixes: "frozenset[str] | None" = None,
    ) -> None:
        """Load model weights from a HuggingFace safetensors directory.

        Supports single-file (model.safetensors) and sharded (model.safetensors.index.json)
        safetensors formats. GGUF loading not supported — use the vllm-gguf plugin.
        MLX affine-int4 (Qwen3.5-9B MLX community format) is supported as a special case.

        Args:
            f32_keys: Optional set of checkpoint key names that must be uploaded as float32
                      instead of the default float16. Passed through to the safetensors loader.
            skip_prefixes: Optional set of key prefixes to skip entirely. Keys whose names
                           start with any of these prefixes are excluded before any GPU buffer
                           allocation, keeping them out of VRAM for the lifetime of the load.
        """
        from vllm_webgpu.quant.weight_loader import (
            _check_unsupported_quant, _load_quant_cfg,
            detect_weight_format, load_safetensors_weights,
            load_safetensors_weights_sharded,
        )
        fmt = detect_weight_format(path)
        transforms = self._weight_transforms
        if fmt == "safetensors":
            # If path is a directory, the actual file is model.safetensors inside it.
            p = Path(path)
            actual = str(p / _SAFE_WEIGHTS_NAME) if p.is_dir() else path
            # Read config.json once and share between the unsupported-quant check and
            # the loader, eliminating the redundant second parse in load_safetensors_weights.
            _cfg_json = (p if p.is_dir() else p.parent) / "config.json"
            _quant_cfg = _load_quant_cfg(_cfg_json) if _cfg_json.exists() else {}
            if p.is_dir():
                _check_unsupported_quant(p, quant_cfg=_quant_cfg)
            self.weights = load_safetensors_weights(
                actual, self.wgpu_device.wgpu_device, f32_keys=f32_keys,
                weight_transforms=transforms, skip_prefixes=skip_prefixes,
                quant_cfg=_quant_cfg)
        elif fmt == "safetensors_sharded":
            # MLX affine int4 directories also return "safetensors_sharded" from
            # detect_weight_format; load_safetensors_weights_sharded detects the
            # .biases keys in the already-loaded index and dispatches accordingly.
            # Read config.json once and share with both the unsupported-quant check
            # and the loader, eliminating the redundant second parse inside
            # load_safetensors_weights_sharded (mirrors the single-file branch above).
            _cfg_json = Path(path) / "config.json"
            _quant_cfg = _load_quant_cfg(_cfg_json) if _cfg_json.exists() else {}
            _check_unsupported_quant(Path(path), quant_cfg=_quant_cfg)
            self.weights = load_safetensors_weights_sharded(
                path, self.wgpu_device.wgpu_device, f32_keys=f32_keys,
                weight_transforms=transforms, skip_prefixes=skip_prefixes,
                quant_cfg=_quant_cfg)
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

    def _readback_recurrent_states(
        self, bufs: "list[tuple[str, int, object]]"
    ) -> "dict[str, dict]":
        """Copy an iterable of (kind, layer_idx, WebGPUBuffer) triples to CPU.

        All buffers are batched into a single staging buffer and submitted in
        one command encoder, avoiding N separate GPU-to-CPU round trips.

        Returns {"conv": {layer_idx: bytes}, "ssm": {layer_idx: bytes}}.
        Subclasses build the `bufs` list from their own buffer collections
        (dict.items() for NemotronH, enumerate() with None-guard for Qwen35).
        """
        import wgpu
        dev = self.wgpu_device.wgpu_device

        if not bufs:
            return {"conv": {}, "ssm": {}}

        offsets: list[int] = []
        total = 0
        for _, _, buf in bufs:
            offsets.append(total)
            total += buf.nbytes

        staging = dev.create_buffer(
            size=total,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        enc = dev.create_command_encoder()
        for (_, _, buf), off in zip(bufs, offsets):
            enc.copy_buffer_to_buffer(buf.buf, 0, staging, off, buf.nbytes)
        dev.queue.submit([enc.finish()])

        staging.map_sync(mode=wgpu.MapMode.READ)
        raw = bytes(staging.read_mapped())
        staging.unmap()

        result: dict[str, dict] = {"conv": {}, "ssm": {}}
        for (kind, i, buf), off in zip(bufs, offsets):
            result[kind][i] = raw[off : off + buf.nbytes]
        return result

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
        if self._gpu_sample_staging is None:
            return 0
        import wgpu as wgpu_lib
        self._gpu_sample_staging.map_sync(mode=wgpu_lib.MapMode.READ)
        val = np.frombuffer(self._gpu_sample_staging.read_mapped(), dtype=np.uint32).item()
        self._gpu_sample_staging.unmap()
        return val

    def _ensure_sample_buf(self) -> "WebGPUBuffer":
        """Lazily allocate GPU sampler buffers and return the token output buffer."""
        if self._gpu_sample_tok is None:
            import wgpu as wgpu_lib
            dev = self.wgpu_device.wgpu_device
            self._gpu_sample_tok   = WebGPUBuffer.empty(dev, 4)      # 1 × u32
            # MAP_READ staging buffer: copy argmax result here inside the MAIN command encoder,
            # then map after the single main sync — eliminates the second GPU sync per token.
            self._gpu_sample_staging = dev.create_buffer(
                size=4,
                usage=wgpu_lib.BufferUsage.COPY_DST | wgpu_lib.BufferUsage.MAP_READ)
        return self._gpu_sample_tok

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
        return self.weights.get(w_key + ".scales", fallback) if uq else fallback

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
            elif gs is None:
                # Per-tensor FP8: GROUP_K != 1 routes to the per-tensor branch.
                # Explicit value guards against a future shader default change.
                d["GROUP_K"] = 128
            else:
                raise ValueError(
                    f"fp8_gpu with group_size={gs!r} is not supported "
                    f"(expected None for per-tensor or 1 for per-channel)"
                )
            return d
        if uq == 8:
            # NF4: GROUP_K = absmax block size (BnB default 64).
            return {"GROUP_K": self._quant_info(base_key).get("group_size", 64)}
        return {}

    def _first_weight_key(self, *candidates: str) -> str:
        """Return the first candidate key present in self.weights, or the last as fallback."""
        return next((k for k in candidates if k in self.weights), candidates[-1])

    def _lm_head_key(self) -> str:
        """Return the weight key for the LM head.

        Falls back to the embedding key for models with tied weights that have
        no separate lm_head.weight tensor in the checkpoint.
        """
        return self._first_weight_key("lm_head.weight", "model.embed_tokens.weight")

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
        # The auto-append is an invariant of the dispatch protocol for this shader:
        # all call sites pass exactly 4 bindings (HAS_BIAS=0 is always the default).
        # The assertion guards against silent misuse if the binding count ever changes.
        if shader_name == "matmul_quant":
            assert len(bindings) in (4, 5), (
                f"matmul_quant expects 4 or 5 bindings, got {len(bindings)}"
            )
            if len(bindings) == 4:
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

        standalone = self._active_encoder is None
        # Batch mode: record into the shared encoder; submit happens at context manager exit.
        # Standalone mode: create a fresh encoder and submit immediately.
        encoder = dev.create_command_encoder() if standalone else self._active_encoder

        cp = encoder.begin_compute_pass()
        cp.set_pipeline(pipeline)
        cp.set_bind_group(0, bg)
        cp.dispatch_workgroups(*workgroups)
        cp.end()

        if standalone:
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
