from __future__ import annotations
import time
from typing import TYPE_CHECKING

import torch

# vLLM v1 internals verified against vllm>=0.24,<0.25.
# These paths have no stability guarantees; a patch release may move or rename
# them. Pin vllm in pyproject.toml and run CI against the exact pinned version.
# Update this comment and pyproject.toml when bumping the vLLM version.
from vllm.distributed.ec_transfer import ensure_ec_transfer_shutdown
from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized, ensure_kv_transfer_shutdown
from vllm.v1.worker.gpu_worker import init_worker_distributed_environment
from vllm.logger import init_logger
from vllm.utils.torch_utils import set_random_seed           # vllm>=0.24
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase   # vllm>=0.24

from vllm_webgpu.config import get_config
from vllm_webgpu.v1.cache_policy import determine_available_memory

# Import WebGPUDevice at module level so it can be patched in tests.
from vllm_webgpu.webgpu.device import WebGPUDevice

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.lora.request import LoRARequest
    from vllm.tasks import SupportedTask
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
    from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
    from vllm_webgpu.v1.model_runner import WebGPUModelRunner

logger = init_logger(__name__)


class WebGPUWorker(WorkerBase):
    # WorkerBase declares `self.model_runner: nn.Module | None = None` (worker_base.py:89).
    # WebGPUModelRunner is not an nn.Module subclass, so this annotation intentionally
    # narrows the slot type to the concrete runner used here. WorkerBase does not call
    # any nn.Module methods (parameters(), state_dict(), eval()) on model_runner today,
    # so this is runtime-safe. If vLLM ever exposes a ModelRunnerBase protocol for this
    # slot, switch to that. Until then, keep this annotation so type checkers within this
    # package see the correct concrete type.
    model_runner: "WebGPUModelRunner | None"  # type: ignore[assignment]

    def __init__(
        self,
        vllm_config: VllmConfig,
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
        self.wgpu_device: "WebGPUDevice | None" = None

    def init_device(self) -> None:
        from vllm_webgpu.v1.model_runner import WebGPUModelRunner

        self.wgpu_device = WebGPUDevice.initialize(self.webgpu_config.power_preference)

        self.device = torch.device("cpu")

        init_worker_distributed_environment(self.vllm_config, self.rank, self.distributed_init_method, self.local_rank, backend="gloo")
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
        return determine_available_memory(self)

    def get_kv_cache_spec(self) -> "dict[str, KVCacheSpec]":
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: "KVCacheConfig") -> None:
        # Set num_gpu_blocks first (matches GPU worker initialization order) so
        # that any downstream code inspecting cache_config after an exception
        # sees the correct block count rather than a stale pre-call value.
        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)
        if self.model_config.enable_return_routed_experts:
            raise NotImplementedError(
                "enable_return_routed_experts is not supported on the WebGPU backend"
            )
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def compile_or_warm_up_model(self) -> CompilationTimes:
        start = time.perf_counter()
        self.model_runner.warm_up()
        elapsed = time.perf_counter() - start
        logger.info("WebGPU warm-up completed in %.2fs", elapsed)
        set_random_seed(self.model_config.seed)
        return CompilationTimes(language_model=elapsed, encoder=0.0)

    def execute_dummy_batch(self) -> None:
        # WebGPU has no CUDA streams to keep warm, so this is intentionally a
        # no-op. The vLLM abstract executor calls this when the scheduler loop
        # runs with no batch to execute but unfinished requests remain.
        logger.debug("execute_dummy_batch: no-op on WebGPU backend")

    @torch.inference_mode()
    def execute_model(
        self, scheduler_output: "SchedulerOutput"
    ) -> "ModelRunnerOutput | AsyncModelRunnerOutput | None":
        return self.model_runner.execute_model(scheduler_output)

    @torch.inference_mode()
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> "ModelRunnerOutput | AsyncModelRunnerOutput":
        return self.model_runner.sample_tokens(grammar_output)

    def get_model(self) -> "torch.nn.Module":
        raise NotImplementedError(
            "WebGPU models are not nn.Module instances; apply_model and model "
            "inspection are not supported. Access the model via "
            "model_runner.model directly."
        )

    def update_max_model_len(self, max_model_len: int) -> None:
        self.model_config.max_model_len = max_model_len
        # WebGPUModelRunner reads max_model_len via the shared vllm_config.model_config
        # reference, so the update propagates automatically without a separate call.
        # gpu_worker.py explicitly calls model_runner.update_max_model_len() because
        # the GPU model runner may cache the value locally for block-table sizing;
        # WebGPUModelRunner has no such local cache, so the call is intentionally omitted.
        logger.debug("Updated max_model_len to %d", max_model_len)

    def get_cache_block_size_bytes(self) -> int:
        return self.model_runner.get_cache_block_size_bytes()

    def get_supported_tasks(self) -> "tuple[SupportedTask, ...]":
        return self.model_runner.get_supported_tasks()

    def add_lora(self, lora_request: "LoRARequest") -> bool:
        logger.warning("LoRA not supported on WebGPU")
        return False

    def remove_lora(self, lora_id: int) -> bool:
        return False

    def pin_lora(self, lora_id: int) -> bool:
        return False

    def list_loras(self) -> set[int]:
        return set()

    def sleep(self, level: int = 1) -> None:
        logger.warning(
            "Sleep requested but WebGPU backend cannot offload weights; "
            "available memory is unchanged. If loading a second model fails "
            "with OOM, this is the cause."
        )

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
        # No encoder cache to reset; no-op. Defined because the vLLM executor
        # calls this on all workers via RPC (see gpu_worker.py:785).
        pass

    def shutdown(self) -> None:
        # Guard against interpreter teardown: module-level globals are set to
        # None before __del__ / atexit callbacks run. gpu_worker.py uses the
        # same guards (lines 1173-1176) with the comment 'has_kv_transfer_group
        # can be None during interpreter shutdown'.
        if ensure_kv_transfer_shutdown is not None:
            ensure_kv_transfer_shutdown()
        if ensure_ec_transfer_shutdown is not None:
            ensure_ec_transfer_shutdown()
        # Release the GPU device. Both the worker and model_runner hold a
        # reference to wgpu_device; clearing only one leaves the object alive.
        # Do not null model_runner itself: any post-shutdown delegate call (e.g.
        # get_supported_tasks) would raise AttributeError on NoneType instead of
        # a clear error.
        if self.model_runner is not None:
            self.model_runner.wgpu_device = None
        self.wgpu_device = None
        logger.info("WebGPU worker shutdown complete")
