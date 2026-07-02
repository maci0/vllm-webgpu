from __future__ import annotations
import logging
from abc import abstractmethod
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

    @contextmanager
    def _batched_dispatch(self):
        """Record all _dispatch() calls inside this block into a single CommandEncoder.

        Reduces queue.submit() calls from N (one per dispatch) to 1 per block exit.
        Semantically identical to separate submits since the queue serializes in order.
        """
        dev = self.wgpu_device.wgpu_device
        encoder = dev.create_command_encoder()
        self._active_encoder = encoder
        try:
            yield
            # Submit only on clean exit: an exception mid-layer would submit partial
            # GPU work and corrupt the KV cache even though Python raised an error.
            dev.queue.submit([encoder.finish()])
        finally:
            self._active_encoder = None

    def load_weights(self, path: str) -> None:
        from vllm_webgpu.quant.gguf_loader import (
            detect_weight_format, load_safetensors_weights, load_gguf_weights,
        )
        fmt = detect_weight_format(path)
        if fmt == "safetensors":
            self.weights = load_safetensors_weights(path, self.wgpu_device.wgpu_device)
        elif fmt == "gguf":
            self.weights = load_gguf_weights(path, self.wgpu_device.wgpu_device)
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
