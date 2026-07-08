from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum as _ABE
    from vllm.v1.attention.selector import AttentionSelectorConfig

logger = logging.getLogger(__name__)

try:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum as _ABE_RT
    _CPU_ATTN_PATH = _ABE_RT.CPU_ATTN.get_path()
except ImportError:
    _CPU_ATTN_PATH = ""


@functools.cache
def _get_wgpu_adapter():
    """Return the wgpu adapter, probing once and caching the result.

    Returns None if wgpu is unavailable or the probe failed.
    """
    try:
        import wgpu
        from vllm_webgpu.config import get_config
        return wgpu.gpu.request_adapter_sync(power_preference=get_config().power_preference)
    except Exception:
        return None


try:
    from vllm.platforms.interface import Platform as _Platform, PlatformEnum as _PlatformEnum
except ImportError:
    class _Platform: pass                                   # noqa: E701
    class _PlatformEnum: OOT = "OOT"                       # noqa: E701


class WebGPUPlatform(_Platform):
    _enum = _PlatformEnum.OOT
    device_name: str = "cpu"
    device_type: str = "cpu"
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
        except Exception:
            return False
        return True

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        try:
            adapter = _get_wgpu_adapter()
            if adapter:
                info = adapter.info
                return f"WebGPU ({info.get('device') or 'unknown'})"
        except Exception:
            pass
        return "WebGPU"

    @classmethod
    def get_device_count(cls) -> int:
        return 1

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        import psutil
        return psutil.virtual_memory().total

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        import os
        import sys
        # macOS (Darwin): wgpu-native uses Metal, which is not fork-safe.
        # The parent process probes the wgpu adapter during is_available(),
        # and after fork the child cannot call request_device_sync() on Metal.
        # Force spawn so the EngineCore subprocess starts with a clean state.
        if sys.platform == "darwin":
            os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm_webgpu.v1.worker.WebGPUWorker"
        existing_backend = parallel_config.distributed_executor_backend
        if existing_backend not in (None, "auto", "uni"):
            logger.warning(
                "WebGPU platform only supports the 'uni' executor backend, "
                "but distributed_executor_backend was explicitly set to %r. "
                "Overriding to 'uni'; multi-node backends are not supported.",
                existing_backend,
            )
        parallel_config.distributed_executor_backend = "uni"
        parallel_config.disable_custom_all_reduce = True
        vllm_config.scheduler_config.enable_chunked_prefill = False

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: _ABE,
        attn_selector_config: AttentionSelectorConfig,
        num_heads: int | None = None,
    ) -> str:
        return _CPU_ATTN_PATH

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        return False

    @classmethod
    def set_device(cls, device) -> None:
        idx = device.index if device.index is not None else 0
        if idx != 0:
            raise ValueError(f"WebGPU only supports device 0, got {device}")

    @classmethod
    def current_device(cls) -> int:
        return 0

    @classmethod
    def synchronize(cls, device_id: int = 0) -> None:
        pass

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        return f"webgpu:{device_id}"

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        # WebGPU has no global RNG; Python/numpy RNG is seeded by vLLM's set_random_seed.
        pass
