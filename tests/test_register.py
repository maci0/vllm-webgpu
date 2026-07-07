"""Tests for vllm_webgpu._register() OOT plugin entry point.

Focuses on the native-GPU-platform guard introduced to prevent WebGPU from
preempting CUDA/ROCm on machines where wgpu can reach a GPU via Vulkan.
"""
import importlib
import sys
from unittest.mock import MagicMock, patch


def _make_vllm_platforms_mock(cuda_returns=False, rocm_returns=False):
    """Build a fake vllm.platforms module with controllable plugin functions."""
    mod = MagicMock()
    mod.cuda_platform_plugin = MagicMock(return_value=cuda_returns)
    mod.rocm_platform_plugin = MagicMock(return_value=rocm_returns)
    return mod


def _reload_init():
    """Force a fresh import of vllm_webgpu so module-level state is reset."""
    for key in list(sys.modules.keys()):
        if key == "vllm_webgpu" or key.startswith("vllm_webgpu."):
            del sys.modules[key]
    import vllm_webgpu
    return vllm_webgpu


def _call_register_with_available(monkeypatch, cuda=False, rocm=False, force_env=""):
    """Call _register() with WebGPUPlatform.is_available() == True and the given
    native-GPU-platform responses, returning the result."""
    monkeypatch.setenv("VLLM_WEBGPU_FORCE", force_env)

    vllm_platforms_mock = _make_vllm_platforms_mock(cuda_returns=cuda, rocm_returns=rocm)

    pkg = _reload_init()

    with patch.object(
        sys.modules.get("vllm_webgpu.platform", MagicMock()),
        "WebGPUPlatform",
        create=True,
    ):
        # Patch is_available on the actual class that _register imports
        with patch("vllm_webgpu.platform.WebGPUPlatform.is_available", return_value=True):
            with patch.dict("sys.modules", {"vllm.platforms": vllm_platforms_mock}):
                # Also ensure the env/compat imports don't fail
                with patch("vllm_webgpu.envs.environment_variables", {}, create=True):
                    with patch("vllm.envs.environment_variables", {}, create=True):
                        result = pkg._register()

    return result


# ---------------------------------------------------------------------------
# Core guard behaviour
# ---------------------------------------------------------------------------

def test_register_yields_to_cuda_when_detected(monkeypatch):
    """_register() must return None when CUDA is available and FORCE is unset."""
    result = _call_register_with_available(monkeypatch, cuda=True, force_env="")
    assert result is None, f"Expected None (yield to CUDA), got {result!r}"


def test_register_yields_to_rocm_when_detected(monkeypatch):
    """_register() must return None when ROCm is available and FORCE is unset."""
    result = _call_register_with_available(monkeypatch, rocm=True, force_env="")
    assert result is None, f"Expected None (yield to ROCm), got {result!r}"


def test_register_activates_without_native_gpu(monkeypatch):
    """_register() returns the WebGPU class path when no native GPU is detected."""
    result = _call_register_with_available(monkeypatch, cuda=False, rocm=False, force_env="")
    assert result == "vllm_webgpu.platform.WebGPUPlatform"


def test_register_force_overrides_cuda(monkeypatch):
    """VLLM_WEBGPU_FORCE=1 bypasses the native-GPU guard even when CUDA is present."""
    result = _call_register_with_available(monkeypatch, cuda=True, force_env="1")
    assert result == "vllm_webgpu.platform.WebGPUPlatform"


def test_register_force_overrides_rocm(monkeypatch):
    """VLLM_WEBGPU_FORCE=1 bypasses the native-GPU guard even when ROCm is present."""
    result = _call_register_with_available(monkeypatch, rocm=True, force_env="1")
    assert result == "vllm_webgpu.platform.WebGPUPlatform"


def test_register_platform_import_error_does_not_block(monkeypatch):
    """If vllm.platforms cannot be imported at all, _register() still activates."""
    monkeypatch.setenv("VLLM_WEBGPU_FORCE", "")

    pkg = _reload_init()

    with patch("vllm_webgpu.platform.WebGPUPlatform.is_available", return_value=True):
        # Remove vllm.platforms so the import inside _register() raises ImportError
        cleaned = {k: v for k, v in sys.modules.items() if k != "vllm.platforms"}
        with patch.dict("sys.modules", cleaned, clear=True):
            sys.modules.pop("vllm.platforms", None)
            with patch("vllm_webgpu.envs.environment_variables", {}, create=True):
                with patch("vllm.envs.environment_variables", {}, create=True):
                    result = pkg._register()

    assert result == "vllm_webgpu.platform.WebGPUPlatform"


def test_register_returns_none_when_not_available(monkeypatch):
    """_register() returns None when WebGPUPlatform.is_available() is False."""
    monkeypatch.setenv("VLLM_WEBGPU_FORCE", "")

    pkg = _reload_init()

    with patch("vllm_webgpu.platform.WebGPUPlatform.is_available", return_value=False):
        with patch("vllm_webgpu.envs.environment_variables", {}, create=True):
            with patch("vllm.envs.environment_variables", {}, create=True):
                result = pkg._register()

    assert result is None
