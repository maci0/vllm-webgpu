import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    VLLM_WEBGPU_POWER_PREFERENCE: str = "high-performance"
    VLLM_WEBGPU_BLOCK_SIZE: int = 16
    GDN_BF16: bool = False

environment_variables: dict[str, Callable[[], Any]] = {
    "VLLM_WEBGPU_POWER_PREFERENCE": lambda: os.getenv("VLLM_WEBGPU_POWER_PREFERENCE", "high-performance"),
    "VLLM_WEBGPU_BLOCK_SIZE": lambda: int(os.getenv("VLLM_WEBGPU_BLOCK_SIZE", "16")),
    "GDN_BF16": lambda: os.getenv("GDN_BF16", "0") == "1",
}


def __getattr__(name: str) -> Any:
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
