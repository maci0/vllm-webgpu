import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    VLLM_WEBGPU_MEMORY_FRACTION: str = "auto"
    VLLM_WEBGPU_POWER_PREFERENCE: str = "high-performance"
    VLLM_WEBGPU_QUANTIZATION: str = "auto"
    VLLM_WEBGPU_BLOCK_SIZE: int = 16
    VLLM_WEBGPU_DEBUG: bool = False

environment_variables: dict[str, Callable[[], Any]] = {
    "VLLM_WEBGPU_MEMORY_FRACTION": lambda: os.getenv("VLLM_WEBGPU_MEMORY_FRACTION", "auto"),
    "VLLM_WEBGPU_POWER_PREFERENCE": lambda: os.getenv("VLLM_WEBGPU_POWER_PREFERENCE", "high-performance"),
    "VLLM_WEBGPU_QUANTIZATION": lambda: os.getenv("VLLM_WEBGPU_QUANTIZATION", "auto"),
    "VLLM_WEBGPU_BLOCK_SIZE": lambda: int(os.getenv("VLLM_WEBGPU_BLOCK_SIZE", "16")),
    "VLLM_WEBGPU_DEBUG": lambda: os.getenv("VLLM_WEBGPU_DEBUG", "0") == "1",
}


def __getattr__(name: str) -> Any:
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
