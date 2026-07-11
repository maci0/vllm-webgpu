import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

def _parse_memory_fraction(raw: str) -> "float | None":
    if raw.lower() == "auto":
        return None
    try:
        return float(raw)
    except ValueError:
        raise ValueError(
            f"VLLM_WEBGPU_MEMORY_FRACTION={raw!r} must be 'auto' or a float."
        )

if TYPE_CHECKING:
    VLLM_WEBGPU_MEMORY_FRACTION: "float | None" = None
    VLLM_WEBGPU_POWER_PREFERENCE: str = "high-performance"
    VLLM_WEBGPU_BLOCK_SIZE: int = 16

environment_variables: dict[str, Callable[[], Any]] = {
    "VLLM_WEBGPU_MEMORY_FRACTION": lambda: _parse_memory_fraction(os.getenv("VLLM_WEBGPU_MEMORY_FRACTION", "auto")),
    "VLLM_WEBGPU_POWER_PREFERENCE": lambda: os.getenv("VLLM_WEBGPU_POWER_PREFERENCE", "high-performance"),
    "VLLM_WEBGPU_BLOCK_SIZE": lambda: int(os.getenv("VLLM_WEBGPU_BLOCK_SIZE", "16")),
    "GDN_BF16": lambda: os.getenv("GDN_BF16", "0") == "1",
}


def __getattr__(name: str) -> Any:
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
