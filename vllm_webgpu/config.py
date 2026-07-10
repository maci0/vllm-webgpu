from __future__ import annotations

import functools
from dataclasses import dataclass

import vllm_webgpu.envs as envs

try:
    from wgpu.enums import PowerPreference as _WP
    VALID_POWER_PREFERENCES = frozenset(_WP)
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

    @property
    def is_auto_memory(self) -> bool:
        return self.memory_fraction is None

    @classmethod
    def from_env(cls) -> "WebGPUConfig":
        raw = envs.VLLM_WEBGPU_MEMORY_FRACTION
        if raw.lower() == "auto":
            memory_fraction = None
        else:
            try:
                memory_fraction = float(raw)
            except ValueError as e:
                raise ValueError(
                    f"VLLM_WEBGPU_MEMORY_FRACTION={raw!r} must be 'auto' or float in (0,1]."
                ) from e
        return cls(
            memory_fraction=memory_fraction,
            power_preference=envs.VLLM_WEBGPU_POWER_PREFERENCE,
        )


@functools.cache
def get_config() -> WebGPUConfig:
    return WebGPUConfig.from_env()
