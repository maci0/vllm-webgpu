import logging
import os

__version__ = "0.1.0"

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    try:
        from vllm.envs import VLLM_LOGGING_LEVEL
        vllm_logger = logging.getLogger("vllm")
        webgpu_logger = logging.getLogger("vllm_webgpu")
        webgpu_logger.setLevel(logging.getLevelName(VLLM_LOGGING_LEVEL))
        if vllm_logger.handlers and not webgpu_logger.handlers:
            for handler in vllm_logger.handlers:
                webgpu_logger.addHandler(handler)
            webgpu_logger.propagate = False
    except Exception:
        pass


def _register() -> str | None:
    _configure_logging()

    try:
        import vllm.envs
        from vllm_webgpu.envs import environment_variables
        vllm.envs.environment_variables.update(environment_variables)
    except ImportError:
        pass

    try:
        from vllm_webgpu.compat import apply_compat_patches
        apply_compat_patches()
    except ImportError:
        pass

    from vllm_webgpu.platform import WebGPUPlatform
    if WebGPUPlatform.is_available():
        return "vllm_webgpu.platform.WebGPUPlatform"
    return None


def __getattr__(name: str):
    if name == "register":
        return _register
    if name == "WebGPUPlatform":
        from vllm_webgpu.platform import WebGPUPlatform
        return WebGPUPlatform
    if name == "WebGPUConfig":
        from vllm_webgpu.config import WebGPUConfig
        return WebGPUConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["WebGPUConfig", "WebGPUPlatform", "register"]
