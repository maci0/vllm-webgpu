from __future__ import annotations
import logging
from typing import TYPE_CHECKING

from vllm_webgpu.utils import _OVERHEAD_BYTES

if TYPE_CHECKING:
    from vllm_webgpu.v1.worker import WebGPUWorker

logger = logging.getLogger(__name__)


class WebGPUCachePlanner:
    def __init__(self, worker: "WebGPUWorker") -> None:
        self._worker = worker

    @classmethod
    def from_runner(cls, wgpu_device: object, model_runner: object) -> "WebGPUCachePlanner":
        """Construct a planner from a device and model_runner without a full worker."""
        inst = object.__new__(cls)
        inst._worker = type("_W", (), {"wgpu_device": wgpu_device, "model_runner": model_runner})()
        return inst

    def get_model_memory_usage(self) -> int:
        """Sum of all weight buffer sizes in bytes."""
        model = getattr(self._worker.model_runner, "model", None)
        if model is None:
            return 0
        return sum(buf.nbytes for buf in model.weights.values())

    def determine_available_memory(self) -> int:
        """
        Available memory for KV cache = GPU memory limit - model weights - overhead.
        Falls back to reporting one max-length sequence if memory config is auto.
        """
        from vllm_webgpu.config import get_config

        config = get_config()
        limits = self._worker.wgpu_device.limits
        total = limits.get("max-buffer-size", 4 * 1024 ** 3)  # 4GB default cap; wgpu uses hyphenated keys
        model_mem = self.get_model_memory_usage()

        if config.is_auto_memory:
            available = total - model_mem - _OVERHEAD_BYTES
            logger.info(
                "WebGPU memory: total=%dMB, model=%dMB, available=%dMB",
                total // 2**20, model_mem // 2**20, available // 2**20,
            )
            return max(available, 0)
        return max(int(total * config.memory_fraction) - model_mem, 0)

    def allocate_kv_pool(
        self,
        num_blocks: int,
        num_layers: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        """Pre-allocate all K/V cache buffers for all layers at startup."""
        import wgpu as wgpu_lib
        from vllm_webgpu.webgpu.buffer import WebGPUBuffer

        dev = self._worker.wgpu_device.wgpu_device
        rw = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST
        bytes_per_layer = num_blocks * block_size * num_kv_heads * head_dim * 2  # f16

        model = self._worker.model_runner.model
        model.kv_pool.clear()

        for layer_i in range(num_layers):
            k_buf = WebGPUBuffer.empty(dev, bytes_per_layer, usage=rw)
            v_buf = WebGPUBuffer.empty(dev, bytes_per_layer, usage=rw)
            model.kv_pool.append((k_buf, v_buf))

        total_mb = (bytes_per_layer * num_layers * 2) // 2**20
        logger.info(
            "KV cache: %d blocks × %d layers × %d KV heads × %d head_dim = %dMB",
            num_blocks, num_layers, num_kv_heads, head_dim, total_mb,
        )
