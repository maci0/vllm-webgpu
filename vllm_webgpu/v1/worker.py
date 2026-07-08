from __future__ import annotations
import gc
import logging
import time
from typing import TYPE_CHECKING, Any

try:
    # All imports below are vLLM v1 internals verified against vllm>=0.24,<0.25.
    # These paths have no stability guarantees; a patch release may move or rename
    # them. Pin vllm in pyproject.toml and run CI against the exact pinned version.
    # Update this comment and pyproject.toml when bumping the vLLM version.
    from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
    from vllm.utils.torch_utils import set_random_seed           # vllm>=0.24
    from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase   # vllm>=0.24
except ImportError:
    from collections import namedtuple as _namedtuple
    CompilationTimes = _namedtuple("CompilationTimes", ["language_model", "encoder"])  # type: ignore[assignment,misc]

    def set_random_seed(seed: int) -> None:  # type: ignore[misc]
        pass

    def init_distributed_environment(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        pass

    def ensure_model_parallel_initialized(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        pass

    class WorkerBase:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)
            # Unpack vllm_config sub-attributes to match the real WorkerBase contract.
            # Keep in sync with vllm.v1.worker.worker_base.WorkerBase.__init__ (lines 64-79).
            vc = kwargs.get("vllm_config")
            if vc is not None:
                self.cache_config = getattr(vc, "cache_config", None)
                self.model_config = getattr(vc, "model_config", None)
                self.lora_config = getattr(vc, "lora_config", None)
                self.load_config = getattr(vc, "load_config", None)
                self.parallel_config = getattr(vc, "parallel_config", None)
                self.scheduler_config = getattr(vc, "scheduler_config", None)
                self.device_config = getattr(vc, "device_config", None)
                self.speculative_config = getattr(vc, "speculative_config", None)
                self.observability_config = getattr(vc, "observability_config", None)
                self.kv_transfer_config = getattr(vc, "kv_transfer_config", None)
                self.compilation_config = getattr(vc, "compilation_config", None)
                try:
                    from vllm.platforms import current_platform as _cp
                    self.current_platform = _cp
                except ImportError:
                    self.current_platform = None
                rank = kwargs.get("rank")
                if rank is not None and self.parallel_config is not None:
                    self.parallel_config.rank = rank

from vllm_webgpu.config import get_config
from vllm_webgpu.v1.cache_policy import WebGPUCachePlanner

# Import WebGPUDevice at module level so it can be patched in tests.
# The actual wgpu library is optional; if unavailable, a sentinel is set.
try:
    from vllm_webgpu.webgpu.device import WebGPUDevice
except ImportError:
    WebGPUDevice = None  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from vllm.tasks import SupportedTask
    from vllm.v1.kv_cache_interface import KVCacheSpec
    from vllm_webgpu.v1.model_runner import WebGPUModelRunner

logger = logging.getLogger(__name__)


class WebGPUWorker(WorkerBase):
    model_runner: "WebGPUModelRunner"

    def __init__(
        self,
        vllm_config: Any,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
        self.webgpu_config = get_config()
        if getattr(self, "parallel_config", None) is not None:
            self.parallel_config.disable_custom_all_reduce = True
        self.wgpu_device: "WebGPUDevice | None" = None

    def init_device(self) -> None:
        from vllm_webgpu.v1.model_runner import WebGPUModelRunner

        self.wgpu_device = WebGPUDevice.initialize(self.webgpu_config.power_preference)

        try:
            import torch
            self.device = torch.device("cpu")
        except ImportError:
            pass

        pc = self.vllm_config.parallel_config
        init_distributed_environment(pc.world_size, self.rank, self.distributed_init_method, self.local_rank, backend="gloo")
        ensure_model_parallel_initialized(pc.tensor_parallel_size, pc.pipeline_parallel_size, pc.prefill_context_parallel_size, pc.decode_context_parallel_size)
        if hasattr(self, "model_config"):
            set_random_seed(self.model_config.seed)

        self.model_runner = WebGPUModelRunner(self.vllm_config, self.wgpu_device)

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        if load_dummy_weights:
            logger.warning(
                "load_dummy_weights=True requested but WebGPU backend does not "
                "support dummy weight loading; loading real weights instead."
            )
        self.model_runner.load_model()

    def determine_available_memory(self) -> int:
        return WebGPUCachePlanner(self).determine_available_memory()

    def get_kv_cache_spec(self) -> "dict[str, KVCacheSpec]":
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: Any) -> None:
        if hasattr(self, "cache_config") and self.cache_config is not None:
            self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def compile_or_warm_up_model(self) -> CompilationTimes:
        if hasattr(self, "model_config"):
            set_random_seed(self.model_config.seed)
        start = time.perf_counter()
        self.model_runner.warm_up()
        elapsed = time.perf_counter() - start
        return CompilationTimes(language_model=elapsed, encoder=0.0)

    def execute_model(self, scheduler_output: Any) -> Any:
        return self.model_runner.execute_model(scheduler_output)

    def sample_tokens(self, grammar_output: Any) -> Any:
        return self.model_runner.sample_tokens(grammar_output)

    def get_model(self) -> Any:
        return self.model_runner.model

    def update_max_model_len(self, max_model_len: int) -> None:
        if hasattr(self, "model_config"):
            self.model_config.max_model_len = max_model_len
        if hasattr(self.model_runner, "update_max_model_len"):
            self.model_runner.update_max_model_len(max_model_len)

    def get_cache_block_size_bytes(self) -> int:
        return self.model_runner.get_cache_block_size_bytes()

    def get_supported_tasks(self) -> "tuple[SupportedTask, ...]":
        return self.model_runner.get_supported_tasks()

    def add_lora(self, lora_request: Any) -> bool:
        logger.warning("LoRA not supported on WebGPU")
        return False

    def remove_lora(self, lora_id: int) -> bool:
        return False

    def pin_lora(self, lora_id: int) -> bool:
        return False

    def list_loras(self) -> set[int]:
        return set()

    def sleep(self, level: int = 1) -> None:
        logger.warning("Sleep mode not supported on WebGPU")

    def wake_up(self, tags: list[str] | None = None) -> None:
        logger.warning("Wake mode not supported on WebGPU")

    def check_health(self) -> None:
        """Verify the WebGPU device is alive by submitting a no-op compute pass.

        Reading Python-side `limits` only proves the Python object exists, not
        that the GPU is responsive. A submitted no-op forces the command queue
        to acknowledge the device.
        """
        if self.wgpu_device is None:
            raise RuntimeError("WebGPU device not initialized")
        try:
            dev = self.wgpu_device.wgpu_device
            # Empty command encoder — flush forces the queue to process.
            encoder = dev.create_command_encoder()
            dev.queue.submit([encoder.finish()])
        except Exception as e:
            raise RuntimeError(f"WebGPU device health check failed: {e}") from e

    def reset_encoder_cache(self) -> None:
        self.model_runner.reset_encoder_cache()

    def shutdown(self) -> None:
        if hasattr(self, "model_runner") and self.model_runner is not None:
            del self.model_runner
        self.wgpu_device = None
        gc.collect()
        logger.info("WebGPU worker shutdown complete")
