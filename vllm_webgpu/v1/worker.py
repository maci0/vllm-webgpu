from __future__ import annotations
import gc
import logging
import time
from typing import TYPE_CHECKING, Any

try:
    from vllm.config import VllmConfig
    from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
    from vllm.tasks import SupportedTask
    from vllm.utils.torch_utils import set_random_seed
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase
except ImportError:
    VllmConfig = Any  # type: ignore[assignment,misc]
    SupportedTask = Any  # type: ignore[assignment,misc]
    GrammarOutput = Any  # type: ignore[assignment,misc]
    SchedulerOutput = Any  # type: ignore[assignment,misc]
    KVCacheConfig = Any  # type: ignore[assignment,misc]
    KVCacheSpec = Any  # type: ignore[assignment,misc]
    ModelRunnerOutput = Any  # type: ignore[assignment,misc]
    CompilationTimes = Any  # type: ignore[assignment,misc]

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

from vllm_webgpu.config import get_config
from vllm_webgpu.v1.cache_policy import WebGPUCachePlanner

# Import WebGPUDevice at module level so it can be patched in tests.
# The actual wgpu library is optional; if unavailable, a sentinel is set.
try:
    from vllm_webgpu.webgpu.device import WebGPUDevice
except ImportError:
    WebGPUDevice = None  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from vllm_webgpu.v1.model_runner import WebGPUModelRunner

logger = logging.getLogger(__name__)


def _init_distributed(vllm_config: Any, rank: int, init_method: str, local_rank: int) -> None:
    pc = vllm_config.parallel_config
    init_distributed_environment(pc.world_size, rank, init_method, local_rank, backend="gloo")
    ensure_model_parallel_initialized(pc.tensor_parallel_size, pc.pipeline_parallel_size)


class WebGPUWorker(WorkerBase):
    model_runner: "WebGPUModelRunner"

    def __init__(
        self,
        vllm_config: Any,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
        self.webgpu_config = get_config()
        if hasattr(self, "parallel_config"):
            self.parallel_config.disable_custom_all_reduce = True
        self.wgpu_device: "WebGPUDevice | None" = None

    def init_device(self) -> None:
        from vllm_webgpu.v1.model_runner import WebGPUModelRunner  # noqa: F401

        self.wgpu_device = WebGPUDevice.initialize(self.webgpu_config.power_preference)

        try:
            import torch
            self.device = torch.device("cpu")
        except ImportError:
            pass

        _init_distributed(self.vllm_config, self.rank, self.distributed_init_method, self.local_rank)
        if hasattr(self, "model_config"):
            set_random_seed(self.model_config.seed)

        self.model_runner = WebGPUModelRunner(self.vllm_config, self.wgpu_device)

    def load_model(self) -> None:
        self.model_runner.load_model()

    def determine_available_memory(self) -> int:
        return WebGPUCachePlanner(self).determine_available_memory()

    def get_kv_cache_spec(self) -> dict:
        return self.model_runner.get_kv_cache_spec()

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        if hasattr(self, "cache_config"):
            self.cache_config.num_gpu_blocks = num_gpu_blocks
            self.cache_config.num_cpu_blocks = num_cpu_blocks

    def initialize_from_config(self, kv_cache_config: Any) -> None:
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def compile_or_warm_up_model(self) -> Any:
        if hasattr(self, "model_config"):
            set_random_seed(self.model_config.seed)
        start = time.perf_counter()
        self.model_runner.warm_up()
        elapsed = time.perf_counter() - start
        if CompilationTimes is not Any:
            return CompilationTimes(language_model=elapsed, encoder=0.0)
        return elapsed

    def execute_model(self, scheduler_output: Any) -> Any:
        return self.model_runner.execute_model(scheduler_output)

    def sample_tokens(self, grammar_output: Any) -> Any:
        return self.model_runner.sample_tokens(grammar_output)

    def get_model(self) -> Any:
        return self.model_runner.model

    def update_max_model_len(self, max_model_len: int) -> None:
        if hasattr(self, "model_config"):
            self.model_config.max_model_len = max_model_len

    def get_cache_block_size_bytes(self) -> int:
        return self.model_runner.get_cache_block_size_bytes()

    def get_supported_tasks(self) -> tuple:
        return self.model_runner.supported_worker_tasks()

    def add_lora(self, lora_request: Any) -> bool:
        logger.warning("LoRA not supported on WebGPU")
        return False

    def remove_lora(self, lora_id: int) -> bool:
        return False

    def pin_lora(self, lora_id: int) -> bool:
        return False

    def list_loras(self) -> set:
        return set()

    def sleep(self, level: int = 1) -> None:
        logger.warning("Sleep mode not supported on WebGPU")

    def wake_up(self, tags: list | None = None) -> None:
        logger.warning("Wake mode not supported on WebGPU")

    def check_health(self) -> None:
        """Verify the WebGPU device is alive by querying device limits."""
        if self.wgpu_device is None:
            raise RuntimeError("WebGPU device not initialized")
        try:
            _ = self.wgpu_device.limits
        except Exception as e:
            raise RuntimeError(f"WebGPU device health check failed: {e}") from e

    def shutdown(self) -> None:
        if hasattr(self, "model_runner") and self.model_runner is not None:
            del self.model_runner
        gc.collect()
        logger.info("WebGPU worker shutdown complete")
