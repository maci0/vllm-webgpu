import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    VLLM_WEBGPU_MEMORY_FRACTION: str = "auto"
    VLLM_WEBGPU_POWER_PREFERENCE: str = "high-performance"

environment_variables: dict[str, Callable[[], Any]] = {
    "VLLM_WEBGPU_MEMORY_FRACTION": lambda: os.getenv("VLLM_WEBGPU_MEMORY_FRACTION", "auto"),
    "VLLM_WEBGPU_POWER_PREFERENCE": lambda: os.getenv("VLLM_WEBGPU_POWER_PREFERENCE", "high-performance"),
    "GDN_BF16": lambda: os.getenv("GDN_BF16", "0") == "1",
}


def __getattr__(name: str) -> Any:
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
