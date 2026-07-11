from __future__ import annotations

from functools import cache
from dataclasses import dataclass

import vllm_webgpu.envs as envs

try:
    from wgpu.enums import PowerPreference
    VALID_POWER_PREFERENCES = frozenset(PowerPreference)
except ImportError:
    VALID_POWER_PREFERENCES = frozenset({"low-power", "high-performance"})


@dataclass(frozen=True)
class WebGPUConfig:
    memory_fraction: float | None
    power_preference: str

    def __post_init__(self) -> None:
        if self.memory_fraction is not None and not (0 < self.memory_fraction <= 1):
            raise ValueError(
                f"VLLM_WEBGPU_MEMORY_FRACTION={self.memory_fraction!r} must be "
                "'auto' or a value in (0, 1]."
            )
        if self.power_preference not in VALID_POWER_PREFERENCES:
            raise ValueError(
                f"VLLM_WEBGPU_POWER_PREFERENCE={self.power_preference!r}. "
                f"Valid: {sorted(VALID_POWER_PREFERENCES)}"
            )

    @classmethod
    def from_env(cls) -> "WebGPUConfig":
        return cls(
            memory_fraction=envs.VLLM_WEBGPU_MEMORY_FRACTION,
            power_preference=envs.VLLM_WEBGPU_POWER_PREFERENCE,
        )


@cache
def get_config() -> WebGPUConfig:
    return WebGPUConfig.from_env()
