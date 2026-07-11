from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.utils.mem_utils import get_cpu_memory

from vllm.platforms import Platform, PlatformEnum

from vllm.v1.attention.backends.registry import AttentionBackendEnum

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig
    from vllm.v1.attention.selector import AttentionSelectorConfig

logger = init_logger(__name__)


def _get_wgpu_adapter():
    """Return the wgpu adapter, probing on each call.

    Not cached intentionally: get_config.cache_clear() can change
    power_preference between calls (e.g. in tests), and re-probing is cheap
    on failure. In production vLLM calls is_available() once and
    get_device_name() once at startup, so the double-probe is a bounded cost
    of two adapter requests total, not per-request.
    Returns None if wgpu is unavailable or the probe failed.
    """
    try:
        from vllm_webgpu.config import get_config
        cfg = get_config()
        import wgpu
        return wgpu.gpu.request_adapter_sync(power_preference=cfg.power_preference)
    except Exception:
        return None



class WebGPUPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name: str = "webgpu"
    device_type: str = "cpu"

    @classmethod
    def import_kernels(cls) -> None:
        # No CUDA or C extensions exist for WebGPU. This override suppresses
        # the base-class attempt to import vllm._C, which is CUDA-only and
        # produces a spurious warning on every process start.
        pass

    @classmethod
    def import_ir_kernels(cls) -> None:
        # Intentionally empty: WebGPU shaders are loaded lazily by PipelineCache.
        # Suppresses vllm.kernels CUDA/ROCm side-effects.
        pass

    @classmethod
    def is_available(cls) -> bool:
        adapter = _get_wgpu_adapter()
        if adapter is None:
            return False
        try:
            info = adapter.info
            if info.is_fallback_adapter:
                logger.debug(
                    "WebGPU adapter is a CPU/software renderer, not selecting WebGPU platform",
                )
                return False
        except Exception as exc:
            logger.debug("WebGPU adapter probe failed: %s", exc)
            return False
        return True

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        try:
            adapter = _get_wgpu_adapter()
            if adapter:
                info = adapter.info
                return f"WebGPU ({info.device or 'unknown'})"
        except Exception:
            pass
        return "WebGPU"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        # System RAM is the correct budget on all supported platforms.
        # On Apple Silicon, all memory is unified so this is exact.
        # On discrete-GPU Linux/Windows, the actual KV-cache budget is determined
        # by determine_available_memory() in cache_policy.py, which already calls
        # get_cpu_memory() directly; this value only affects vLLM scheduling
        # heuristics. max_buffer_size (the per-buffer driver limit, often ≤4 GB)
        # is not VRAM and must not be used here.
        return get_cpu_memory()

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        # macOS (Darwin): wgpu-native uses Metal, which is not fork-safe.
        # The parent process probes the wgpu adapter during is_available(),
        # and after fork the child cannot call request_device_sync() on Metal.
        # Force spawn so the EngineCore subprocess starts with a clean state.
        if sys.platform == "darwin":
            if (existing := os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")) not in (None, "spawn"):
                logger.warning(
                    "VLLM_WORKER_MULTIPROC_METHOD is set to %r but Metal requires "
                    "'spawn'. Overriding to 'spawn' for Metal safety.",
                    existing,
                )
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm_webgpu.v1.worker.WebGPUWorker"
        existing_backend = parallel_config.distributed_executor_backend
        if existing_backend not in (None, "uni"):
            logger.warning(
                "WebGPU platform only supports the 'uni' executor backend, "
                "but distributed_executor_backend was explicitly set to %r. "
                "Overriding to 'uni'; multi-node backends are not supported.",
                existing_backend,
            )
        parallel_config.distributed_executor_backend = "uni"
        vllm_config.scheduler_config.enable_chunked_prefill = False
        if vllm_config.model_config is not None:
            vllm_config.scheduler_config.verify_max_model_len(
                vllm_config.model_config.max_model_len
            )
        # WebGPU compute shaders complete synchronously before execute_model
        # returns, so there is no GPU/CPU overlap to pipeline. The batch queue
        # introduced by async_scheduling adds one extra engine-loop iteration of
        # latency per request with zero throughput benefit. Disable it here,
        # after vLLM's auto-detection (VllmConfig.__post_init__) runs, so the
        # detection result is overridden. Mirrors the CPU platform's approach.
        vllm_config.scheduler_config.async_scheduling = False

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: AttentionBackendEnum,
        attn_selector_config: AttentionSelectorConfig,
        num_heads: int | None = None,
    ) -> str:
        if attn_selector_config.use_mla:
            raise NotImplementedError("MLA is not supported on WebGPU.")
        if attn_selector_config.use_sparse:
            raise NotImplementedError("Sparse attention is not supported on WebGPU.")
        if selected_backend is not None and selected_backend != AttentionBackendEnum.CPU_ATTN:
            logger.info(
                "WebGPU platform only supports CPU_ATTN backend, "
                "but selected_backend is %r. Overriding to CPU_ATTN.",
                selected_backend,
            )
        return AttentionBackendEnum.CPU_ATTN.get_path()

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        return False

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        if device.index not in (None, 0):
            raise ValueError(f"WebGPU only supports device 0, got {device}")
        # wgpu manages its own device context independently of torch.cpu device
        # state, so torch.cpu.set_device() has no effect on WebGPU dispatch.
        # The call is intentionally omitted; the CPU platform's contract is
        # satisfied structurally (index validated above) but not via torch.cpu.

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        return f"webgpu:{device_id}"

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        # WebGPU has no global RNG; Python/numpy RNG is seeded by vLLM's set_random_seed.
        pass
