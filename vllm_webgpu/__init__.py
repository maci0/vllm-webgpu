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
        target = vllm.envs.environment_variables
        for key, factory in environment_variables.items():
            if key in target:
                existing = target[key]
                if existing is not factory:
                    logger.warning(
                        "vllm_webgpu: env var %r already registered with a "
                        "different factory; skipping to avoid overwrite",
                        key,
                    )
            else:
                try:
                    target[key] = factory
                except TypeError:
                    logger.warning(
                        "vllm_webgpu: vllm.envs.environment_variables is "
                        "read-only; could not register %r",
                        key,
                    )
                    break
    except ImportError:
        pass

    from vllm_webgpu.platform import WebGPUPlatform
    if not WebGPUPlatform.is_available():
        return None

    # Do not preempt native GPU platforms unless the user explicitly requests it.
    # On a CUDA/ROCm machine where wgpu is also installed, wgpu can reach the
    # GPU via Vulkan and is_available() returns True. Because OOT plugins take
    # priority over all built-in plugins in vLLM's platform resolution, this
    # would silently redirect inference to the WebGPU CPU-path backend.
    if not os.environ.get("VLLM_WEBGPU_FORCE", "").strip():
        for plugin_name in ("cuda_platform_plugin", "rocm_platform_plugin"):
            try:
                mod = __import__(
                    "vllm.platforms",
                    fromlist=[plugin_name],
                )
                plugin_fn = getattr(mod, plugin_name, None)
                if plugin_fn is not None and plugin_fn():
                    logger.info(
                        "vllm_webgpu: native GPU platform (%s) detected; "
                        "yielding to it. Set VLLM_WEBGPU_FORCE=1 to override.",
                        plugin_name,
                    )
                    return None
            except Exception:
                pass

    return "vllm_webgpu.platform.WebGPUPlatform"


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
