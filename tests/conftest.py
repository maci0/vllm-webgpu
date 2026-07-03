import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_gpu: mark test as requiring a real WebGPU adapter")
    config.addinivalue_line("markers", "integration: mark test as an end-to-end integration test requiring a real GPU")

    # Pre-warm the vLLM GDN op cache before any test runs. test_config.py
    # permanently replaces sys.modules["vllm"] with a MagicMock which would
    # prevent later lazy imports from reaching the real package.
    try:
        from vllm_webgpu.models.qwen35 import _get_vllm_gdn_ops
        _get_vllm_gdn_ops()
    except Exception:
        pass  # vllm or GDN ops unavailable; affected tests will fail clearly


@pytest.fixture(scope="session")
def wgpu_device():
    """Session-scoped real WebGPU device. Skips if unavailable."""
    wgpu = pytest.importorskip("wgpu")
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        pytest.skip("No WebGPU adapter available")
    from vllm_webgpu.webgpu.device import WebGPUDevice
    return WebGPUDevice.initialize("high-performance")
