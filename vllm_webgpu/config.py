from __future__ import annotations

import os
from dataclasses import dataclass

import vllm_webgpu.envs as envs

AUTO_MEMORY_FRACTION = -1.0
VALID_POWER_PREFERENCES = frozenset({"high-performance", "low-power"})
VALID_QUANTIZATIONS = frozenset({"q4_k_m", "f16", "auto"})


@dataclass
class WebGPUConfig:
    memory_fraction: float
    power_preference: str
    quantization: str
    block_size: int
    debug: bool

    def __post_init__(self) -> None:
        if not self.is_auto_memory and not (0 < self.memory_fraction <= 1):
            raise ValueError(
                f"VLLM_WEBGPU_MEMORY_FRACTION={self.memory_fraction!r} must be "
                "'auto' or a value in (0, 1]."
            )
        if self.power_preference not in VALID_POWER_PREFERENCES:
            raise ValueError(
                f"VLLM_WEBGPU_POWER_PREFERENCE={self.power_preference!r}. "
                f"Valid: {sorted(VALID_POWER_PREFERENCES)}"
            )
        if self.quantization not in VALID_QUANTIZATIONS:
            raise ValueError(
                f"VLLM_WEBGPU_QUANTIZATION={self.quantization!r}. "
                f"Valid: {sorted(VALID_QUANTIZATIONS)}"
            )

    @property
    def is_auto_memory(self) -> bool:
        return self.memory_fraction == AUTO_MEMORY_FRACTION

    @classmethod
    def from_env(cls) -> "WebGPUConfig":
        raw = envs.VLLM_WEBGPU_MEMORY_FRACTION
        if raw.lower() == "auto":
            memory_fraction = AUTO_MEMORY_FRACTION
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
            quantization=envs.VLLM_WEBGPU_QUANTIZATION,
            block_size=envs.VLLM_WEBGPU_BLOCK_SIZE,
            debug=envs.VLLM_WEBGPU_DEBUG,
        )


_config: WebGPUConfig | None = None


def get_config() -> WebGPUConfig:
    global _config
    if _config is None:
        _config = WebGPUConfig.from_env()
    return _config


def reset_config() -> None:
    global _config
    _config = None
