import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_gpu: mark test as requiring a real WebGPU adapter")


@pytest.fixture(scope="session")
def wgpu_device():
    """Session-scoped real WebGPU device. Skips if unavailable."""
    wgpu = pytest.importorskip("wgpu")
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        pytest.skip("No WebGPU adapter available")
    from vllm_webgpu.webgpu.device import WebGPUDevice
    return WebGPUDevice.initialize("high-performance")
