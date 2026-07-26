import sys
from pathlib import Path

import pytest

# Allow `import ref_transformer` from tests/ helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent))


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_gpu: mark test as requiring a real WebGPU adapter")
    config.addinivalue_line("markers", "integration: mark test as an end-to-end integration test requiring a real GPU")

    # Pre-import vllm before any test runs. test_config.py has a guard
    # `if "vllm" not in sys.modules` that installs a permanent MagicMock
    # when vllm is absent. By importing vllm here, the guard is never
    # triggered and the real package stays in sys.modules throughout the
    # session, preventing ModuleNotFoundError in later tests (e.g.
    # test_integration.py importing vllm.transformers_utils).
    try:
        import vllm  # noqa: F401
    except ImportError:
        pass  # vllm not installed — test_config.py mock path still works


@pytest.fixture(scope="session")
def wgpu_device():
    """Session-scoped real WebGPU device. Skips if unavailable.

    On headless CI runners without a Vulkan/Metal ICD, request_adapter_sync
    may raise rather than return None; treat that as skip, not suite failure.
    """
    wgpu = pytest.importorskip("wgpu")
    try:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    except Exception as exc:
        pytest.skip(f"No WebGPU adapter available ({exc})")
    if adapter is None:
        pytest.skip("No WebGPU adapter available")
    from vllm_webgpu.webgpu.device import WebGPUDevice
    return WebGPUDevice.initialize("high-performance")
