from __future__ import annotations

from functools import cache
from dataclasses import dataclass

import vllm_webgpu.envs as envs

VALID_POWER_PREFERENCES: frozenset[str] = frozenset({"low-power", "high-performance"})


@dataclass(frozen=True)
class WebGPUConfig:
    power_preference: str

    def __post_init__(self) -> None:
        if self.power_preference not in VALID_POWER_PREFERENCES:
            raise ValueError(
                f"VLLM_WEBGPU_POWER_PREFERENCE={self.power_preference!r}. "
                f"Valid: {sorted(VALID_POWER_PREFERENCES)}"
            )

    @classmethod
    def from_env(cls) -> "WebGPUConfig":
        return cls(
            power_preference=envs.VLLM_WEBGPU_POWER_PREFERENCE,
        )


@cache
def get_config() -> WebGPUConfig:
    return WebGPUConfig.from_env()
