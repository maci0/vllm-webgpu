from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum as _ABE
    from vllm.v1.attention.selector import AttentionSelectorConfig

logger = logging.getLogger(__name__)


def _get_platform_base_class():
    """Get the Platform base class, returning a fallback if vllm is unavailable."""
    try:
        from vllm.platforms.interface import Platform
        return Platform
    except ImportError:
        # Fallback: create a minimal base class for testing
        return object


def _get_device_capability_class():
    """Get the DeviceCapability class, returning a fallback if vllm is unavailable."""
    try:
        from vllm.platforms.interface import DeviceCapability
        return DeviceCapability
    except ImportError:
        # Fallback for testing
        class DeviceCapability:
            def __init__(self, major=0, minor=0):
                self.major = major
                self.minor = minor
        return DeviceCapability


def _get_platform_enum_class():
    """Get the PlatformEnum class, returning a fallback if vllm is unavailable."""
    try:
        from vllm.platforms.interface import PlatformEnum
        return PlatformEnum
    except ImportError:
        # Fallback for testing
        class PlatformEnum:
            OOT = "OOT"
        return PlatformEnum


_Platform = _get_platform_base_class()
_DeviceCapability = _get_device_capability_class()
_PlatformEnum = _get_platform_enum_class()


class WebGPUPlatform(_Platform):
    _enum = _PlatformEnum.OOT if hasattr(_PlatformEnum, 'OOT') else None
    device_name: str = "cpu"
    device_type: str = "cpu"
    dispatch_key: str = "CPU"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import wgpu
            adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
            return adapter is not None
        except Exception:
            return False

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        try:
            import wgpu
            adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
            if adapter:
                info = adapter.request_adapter_info()
                return f"WebGPU ({info.get('device', 'unknown')})"
        except Exception:
            pass
        return "WebGPU"

    @classmethod
    def get_device_count(cls) -> int:
        return 1

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> object:
        return _DeviceCapability(major=8, minor=0)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        try:
            import psutil
            return psutil.virtual_memory().total
        except Exception:
            return 0

    @classmethod
    def get_device_available_memory(cls, device_id: int = 0) -> int:
        try:
            import psutil
            return psutil.virtual_memory().available
        except Exception:
            return 0

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm_webgpu.v1.worker.WebGPUWorker"
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
    def set_device(cls, device_id: int) -> None:
        if device_id != 0:
            raise ValueError(f"WebGPU only supports device 0, got {device_id}")

    @classmethod
    def current_device(cls) -> int:
        return 0

    @classmethod
    def synchronize(cls, device_id: int = 0) -> None:
        pass

    @classmethod
    def verify_quantization(cls, quant: str) -> None:
        pass
