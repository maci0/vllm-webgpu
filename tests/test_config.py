import os
import pytest
from vllm_webgpu.config import WebGPUConfig, get_config, reset_config


def setup_function():
    reset_config()


def teardown_function():
    reset_config()
    for key in ["VLLM_WEBGPU_MEMORY_FRACTION", "VLLM_WEBGPU_POWER_PREFERENCE",
                "VLLM_WEBGPU_QUANTIZATION", "VLLM_WEBGPU_BLOCK_SIZE", "VLLM_WEBGPU_DEBUG"]:
        os.environ.pop(key, None)


def test_defaults():
    cfg = WebGPUConfig.from_env()
    assert cfg.is_auto_memory
    assert cfg.power_preference == "high-performance"
    assert cfg.quantization == "auto"
    assert cfg.block_size == 16
    assert cfg.debug is False


def test_memory_fraction_float(monkeypatch):
    monkeypatch.setenv("VLLM_WEBGPU_MEMORY_FRACTION", "0.8")
    reset_config()
    cfg = WebGPUConfig.from_env()
    assert cfg.memory_fraction == pytest.approx(0.8)
    assert not cfg.is_auto_memory


def test_invalid_memory_fraction(monkeypatch):
    monkeypatch.setenv("VLLM_WEBGPU_MEMORY_FRACTION", "bad")
    reset_config()
    with pytest.raises(ValueError, match="VLLM_WEBGPU_MEMORY_FRACTION"):
        WebGPUConfig.from_env()


def test_get_config_singleton():
    a = get_config()
    b = get_config()
    assert a is b


def test_reset_config():
    a = get_config()
    reset_config()
    b = get_config()
    assert a is not b


def test_register_returns_class_path_when_available():
    from unittest.mock import patch, MagicMock
    import sys
    import vllm_webgpu

    # Mock vllm.envs if not available
    if "vllm" not in sys.modules:
        sys.modules["vllm"] = MagicMock()
        sys.modules["vllm.envs"] = MagicMock()

    with patch("vllm_webgpu.platform.WebGPUPlatform.is_available", return_value=True):
        with patch("vllm.envs.environment_variables", {}):
            result = vllm_webgpu.register()
    assert result == "vllm_webgpu.platform.WebGPUPlatform"


def test_register_returns_none_when_unavailable():
    from unittest.mock import patch, MagicMock
    import sys
    import vllm_webgpu

    # Mock vllm.envs if not available
    if "vllm" not in sys.modules:
        sys.modules["vllm"] = MagicMock()
        sys.modules["vllm.envs"] = MagicMock()

    with patch("vllm_webgpu.platform.WebGPUPlatform.is_available", return_value=False):
        with patch("vllm.envs.environment_variables", {}):
            result = vllm_webgpu.register()
    assert result is None
