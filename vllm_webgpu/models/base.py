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


class BaseWebGPUModel:
    def __init__(self, model_config, wgpu_device: "WebGPUDevice", pipeline_cache: "PipelineCache") -> None:
        self.model_config = model_config
        self.wgpu_device = wgpu_device
        self.pipeline_cache = pipeline_cache
        self.weights: dict[str, "WebGPUBuffer"] = {}
        self.kv_pool: list[tuple["WebGPUBuffer", "WebGPUBuffer"]] = []
        self._active_encoder = None  # set when inside a _batched_dispatch() context
        # Profiling
        self.profiling: bool = False
        # GPU sampler: pre-allocated buffers for GPU argmax / Gumbel sampling.
        # Allocated lazily on first call (need vocab_size from subclass).
        self._gpu_sample_tok: "WebGPUBuffer | None" = None    # [1] u32 next token (STORAGE)
        self._gpu_sample_noise: "WebGPUBuffer | None" = None  # [vocab] f32 Gumbel noise
        self._gpu_sample_vocab: int = 0
        self._gpu_sample_staging = None   # MAP_READ staging buffer for zero-sync readback
        self._gpu_sample_tok_cpu: int = 0  # cached CPU result after readback
        self._prof_stats: dict[str, list[float]] = defaultdict(list)  # shader -> [ms, ...]
        self._prof_current_label: str = ""  # set per _batched_dispatch block

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
        p = Path(path)
        if p.exists():
            return str(p)
        # Try HuggingFace cache
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        safe_id = path.replace("/", "--")
        model_cache = hf_cache / f"models--{safe_id}"
        if model_cache.exists():
            snapshots = sorted((model_cache / "snapshots").iterdir())
            if snapshots:
                return str(snapshots[-1])
        # Fall back: maybe huggingface_hub can download/locate it
        try:
            from huggingface_hub import snapshot_download
            return snapshot_download(path, local_files_only=True)
        except Exception:
            pass
        return path  # let the caller fail with a meaningful error

    def load_weights(self, path: str) -> None:
        """Load model weights from a HuggingFace safetensors directory.

        Supports single-file (model.safetensors) and sharded (model.safetensors.index.json)
        safetensors formats. GGUF is not supported here — use the vllm-gguf plugin instead.
        MLX affine-int4 (Qwen3.5-9B MLX community format) is supported as a special case.
        """
        from vllm_webgpu.quant.gguf_loader import (
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

    def _is_quantized(self, weight_key: str) -> bool:
        """Return True if the weight is stored quantized (INT32) for GPU dequant."""
        buf = self.weights.get(weight_key)
        return buf is not None and getattr(buf, "dtype", "f16") == "i32"

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
        self._gpu_sample_noise = WebGPUBuffer.empty(dev, vocab * 4, usage=rw)  # [vocab] f32
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
        import struct
        val = struct.unpack('<I', bytes(self._gpu_sample_staging.read_mapped()))[0]
        self._gpu_sample_staging.unmap()
        return int(val)

    def gpu_next_token(self, logit_buf: "WebGPUBuffer", vocab: int,
                       temperature: float = 0.0) -> int:
        """Sample next token on GPU. Returns token id (int).

        temperature=0  → argmax (greedy, no logit readback).
        temperature>0  → Gumbel-max sampling (reads only 4 bytes back).

        Either way only 4 bytes are transferred GPU→CPU vs vocab*2 bytes.
        """
        self._ensure_gpu_sampler(vocab)
        dev = self.wgpu_device.wgpu_device

        with self._batched_dispatch(label="sample"):
            if temperature <= 0.0:
                # Pure argmax: no logit transformation needed.
                self._dispatch("argmax_f16",
                               [logit_buf, self._gpu_sample_tok],
                               {"N": vocab}, (1, 1, 1))
            else:
                # Gumbel-max: update noise buffer with fresh uniform random floats.
                noise_np = np.random.uniform(0, 1, vocab).astype(np.float32)
                dev.queue.write_buffer(self._gpu_sample_noise.buf, 0, noise_np.tobytes())
                inv_t = float(1.0 / temperature)
                self._dispatch("gumbel_sample",
                               [logit_buf, self._gpu_sample_noise, self._gpu_sample_tok],
                               {"N": vocab, "TEMPERATURE": float(temperature), "INV_TEMP": inv_t},
                               (1, 1, 1))

        # Only 4 bytes GPU→CPU.
        return int(self._gpu_sample_tok.to_numpy().view(np.uint32)[0])

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
            if uq == 6:
                d["GROUP_K"] = meta.get("group_size", 16)
            return d
        if uq == 7:
            # Int8 per-channel: no group size, no global scale — just USE_QUANT=7.
            return {}
        if uq == 8:
            # NF4: GROUP_K = absmax block size (BnB default 64).
            return {"GROUP_K": self._quant_info(base_key).get("group_size", 64)}
        return {}

    def _gemv_consts_and_wg(self, weight_key: str, K: int, N: int,
                            base_key: str = "") -> tuple:
        """Return (constants_dict, workgroup_tuple) for a matmul_quant dispatch.

        Chooses GPU dequant (USE_QUANT=3) for quantized weights, otherwise f16
        (USE_QUANT=0) with split-K coalesced reads.  LM-head (vocab > 65535)
        falls back to row-per-thread (SPLIT_K=0).
        """
        if self._is_quantized(weight_key):
            qi = self._quant_info(base_key or weight_key[:-len(".weight")])
            gk = qi.get("group_size", 128)
            return ({"K": K, "N": N, "USE_QUANT": 3, "SPLIT_K": 1, "GROUP_K": gk},
                    (N, 1, 1))
        if N > 65535:
            return ({"K": K, "N": N, "USE_QUANT": 0, "SPLIT_K": 0},
                    ((N + 255) // 256, 1, 1))
        return ({"K": K, "N": N, "USE_QUANT": 0, "SPLIT_K": 1},
                (N, 1, 1))

    def _dispatch(
        self,
        shader_name: str,
        bindings: "list[WebGPUBuffer]",
        constants: dict[str, int | float],
        workgroups: tuple[int, int, int],
        shader_subdir: str = "generic",
    ) -> None:
        import wgpu as wgpu_lib

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
        """Compile all pipelines upfront to avoid first-inference latency."""
        logger.info("Warming up shader pipelines...")
        # Subclasses override to trigger get_or_create for all shaders they use.
