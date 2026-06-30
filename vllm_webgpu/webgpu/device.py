from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

_REQUIRED_LIMITS: dict[str, int] = {}

_F16_FEATURE = "shader-f16"


class WebGPUDevice:
    def __init__(
        self,
        adapter,
        wgpu_device,
        supports_f16: bool,
    ) -> None:
        self.adapter = adapter
        self.wgpu_device = wgpu_device
        self.supports_f16 = supports_f16

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

        return cls(adapter=adapter, wgpu_device=device, supports_f16=supports_f16)

    @property
    def limits(self) -> dict:
        return dict(self.wgpu_device.limits)
