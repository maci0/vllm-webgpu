from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum as _ABE
    from vllm.v1.attention.selector import AttentionSelectorConfig

logger = logging.getLogger(__name__)

# Sentinel used to distinguish "not yet probed" from None (probe failed).
_ADAPTER_NOT_PROBED = object()
_wgpu_adapter = _ADAPTER_NOT_PROBED


def _get_wgpu_adapter():
    """Return the wgpu adapter, probing once and caching the result.

    Returns None if wgpu is unavailable or the probe failed.
    """
    global _wgpu_adapter
    if _wgpu_adapter is _ADAPTER_NOT_PROBED:
        try:
            import wgpu
            _wgpu_adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        except Exception:
            _wgpu_adapter = None
    return _wgpu_adapter


try:
    from vllm.platforms.interface import Platform as _Platform, PlatformEnum as _PlatformEnum, DeviceCapability as _DeviceCapability
except ImportError:
    class _Platform: pass                                   # noqa: E701
    class _PlatformEnum: OOT = "OOT"                       # noqa: E701
    class _DeviceCapability:                               # fallback missing .to_int() is intentional
        def __init__(self, major=0, minor=0):
            self.major, self.minor = major, minor


class WebGPUPlatform(_Platform):
    _enum = _PlatformEnum.OOT if hasattr(_PlatformEnum, 'OOT') else None
    device_name: str = "cpu"
    device_type: str = "cpu"
    dispatch_key: str = "CPU"

    @classmethod
    def is_available(cls) -> bool:
        adapter = _get_wgpu_adapter()
        if adapter is None:
            return False
        try:
            info = adapter.info
            adapter_type = info.get("adapter_type", "")
            if isinstance(adapter_type, str) and adapter_type.lower() in ("cpu", "software"):
                logger.debug(
                    "WebGPU adapter is a CPU/software renderer (%s), not selecting WebGPU platform",
                    adapter_type,
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
                return f"WebGPU ({info.get('device', 'unknown')})"
        except Exception:
            pass
        return "WebGPU"

    @classmethod
    def get_device_count(cls) -> int:
        return 1

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> "_DeviceCapability | None":
        return None

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        try:
            import psutil
            return psutil.virtual_memory().total
        except Exception:
            return 0

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
        if existing_backend in (None, "auto"):
            parallel_config.distributed_executor_backend = "uni"
        elif existing_backend != "uni":
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
        try:
            from vllm.v1.attention.backends.registry import AttentionBackendEnum
            return AttentionBackendEnum.CPU_ATTN.get_path()
        except Exception:
            return "cpu_attn"

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
    def get_all_gpu_pci_bus_ids(cls) -> dict[int, str]:
        # WebGPU does not expose PCI bus IDs; return empty rather than raising.
        return {}

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        # WebGPU has no global RNG; Python/numpy RNG is seeded by vLLM's set_random_seed.
        pass
