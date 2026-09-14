from __future__ import annotations

from vllm.logger import init_logger

logger = init_logger(__name__)

_REQUIRED_LIMITS: dict[str, int] = {}

_F16_FEATURE = "shader-f16"


class WebGPUDevice:
    def __init__(
        self,
        wgpu_device,
        supports_f16: bool,
        adapter_info: dict | None = None,
    ) -> None:
        self.wgpu_device = wgpu_device
        self.supports_f16 = supports_f16
        # Kept because the adapter object itself is dropped after initialize().
        # `adapter_type` in particular decides whether the KV cache budget may
        # be drawn from system RAM -- see cache_policy.determine_available_memory.
        self.adapter_info: dict = dict(adapter_info or {})

    @property
    def is_discrete_gpu(self) -> bool:
        """True when GPU memory is a separate pool from system RAM.

        Anything else -- an integrated GPU, Apple Silicon's UMA, a software
        adapter -- shares the host's memory, so system RAM is the right pool to
        size against.
        """
        return str(self.adapter_info.get("adapter_type", "")).lower() == "discretegpu"

    @classmethod
    def initialize(cls, power_preference: str = "high-performance") -> "WebGPUDevice":
        import wgpu

        adapter = wgpu.gpu.request_adapter_sync(power_preference=power_preference)
        if adapter is None:
            raise RuntimeError(
                "No WebGPU adapter found. Ensure a GPU driver with WebGPU support is installed."
            )

        features = list(adapter.features)
        supports_f16 = _F16_FEATURE in features
        required_features = [_F16_FEATURE] if supports_f16 else []

        device = adapter.request_device_sync(
            required_features=required_features,
            required_limits=_REQUIRED_LIMITS,
        )

        info = adapter.info
        logger.info(
            "WebGPU adapter: %s, backend: %s, f16=%s",
            info.get("device", "unknown"),
            info.get("backend_type", "unknown"),
            supports_f16,
        )

        return cls(
            wgpu_device=device,
            supports_f16=supports_f16,
            adapter_info=dict(info),
        )

    @property
    def limits(self) -> dict:
        return dict(self.wgpu_device.limits)
