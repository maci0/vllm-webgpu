import pytest
from unittest.mock import MagicMock, patch


def test_is_available_no_wgpu():
    with patch.dict("sys.modules", {"wgpu": None}):
        from vllm_webgpu.platform import WebGPUPlatform
        assert WebGPUPlatform.is_available() is False


def test_is_available_no_adapter():
    mock_wgpu = MagicMock()
    mock_wgpu.gpu.request_adapter_sync.return_value = None
    with patch.dict("sys.modules", {"wgpu": mock_wgpu}):
        from vllm_webgpu import platform as plat
        import importlib
        importlib.reload(plat)
        assert plat.WebGPUPlatform.is_available() is False


def test_is_available_with_adapter():
    mock_wgpu = MagicMock()
    mock_wgpu.gpu.request_adapter_sync.return_value = MagicMock()
    with patch.dict("sys.modules", {"wgpu": mock_wgpu}):
        from vllm_webgpu import platform as plat
        import importlib
        importlib.reload(plat)
        assert plat.WebGPUPlatform.is_available() is True


def test_check_and_update_config_sets_worker():
    from vllm_webgpu.platform import WebGPUPlatform
    vllm_config = MagicMock()
    vllm_config.parallel_config.worker_cls = "auto"
    vllm_config.scheduler_config.enable_chunked_prefill = True
    WebGPUPlatform.check_and_update_config(vllm_config)
    assert vllm_config.parallel_config.worker_cls == "vllm_webgpu.v1.worker.WebGPUWorker"
    assert vllm_config.parallel_config.distributed_executor_backend == "uni"
    assert vllm_config.parallel_config.disable_custom_all_reduce is True
    assert vllm_config.scheduler_config.enable_chunked_prefill is False


def test_check_and_update_config_preserves_worker_cls():
    from vllm_webgpu.platform import WebGPUPlatform
    vllm_config = MagicMock()
    vllm_config.parallel_config.worker_cls = "my.custom.Worker"
    vllm_config.scheduler_config.enable_chunked_prefill = False
    WebGPUPlatform.check_and_update_config(vllm_config)
    assert vllm_config.parallel_config.worker_cls == "my.custom.Worker"


def test_is_pin_memory_available():
    from vllm_webgpu.platform import WebGPUPlatform
    assert WebGPUPlatform.is_pin_memory_available() is False


def test_get_device_count():
    from vllm_webgpu.platform import WebGPUPlatform
    assert WebGPUPlatform.get_device_count() == 1


def test_webgpu_device_initialize():
    """Requires a real WebGPU adapter. Skip if unavailable."""
    pytest.importorskip("wgpu")
    import wgpu
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        pytest.skip("No WebGPU adapter available")
    from vllm_webgpu.webgpu.device import WebGPUDevice
    dev = WebGPUDevice.initialize("high-performance")
    assert dev.wgpu_device is not None
    assert isinstance(dev.supports_f16, bool)
    assert "max-buffer-size" in dev.limits
