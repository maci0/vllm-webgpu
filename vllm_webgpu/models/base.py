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
                f"GGUF format not supported directly. Use the vllm-gguf plugin instead: {path}"
            )
        else:
            raise ValueError(f"Unknown weight format for {path}")
        logger.info("Loaded %d weight tensors (%s format)", len(self.weights), fmt)

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
