import pytest
from unittest.mock import MagicMock, patch


def test_worker_instantiates():
    """Worker can be instantiated without a real GPU."""
    with patch("vllm_webgpu.v1.worker.WebGPUDevice"):
        from vllm_webgpu.v1.worker import WebGPUWorker
        vllm_config = MagicMock()
        vllm_config.parallel_config.world_size = 1
        vllm_config.parallel_config.tensor_parallel_size = 1
        vllm_config.parallel_config.pipeline_parallel_size = 1
        worker = WebGPUWorker(
            vllm_config=vllm_config,
            local_rank=0,
            rank=0,
            distributed_init_method="env://",
        )
        assert worker is not None


def test_worker_check_health_calls_dispatch(wgpu_device):
    """check_health submits a no-op dispatch — just verifies device is alive."""
    from vllm_webgpu.v1.worker import WebGPUWorker
    import numpy as np

    vllm_config = MagicMock()
    vllm_config.parallel_config.world_size = 1
    vllm_config.parallel_config.tensor_parallel_size = 1
    vllm_config.parallel_config.pipeline_parallel_size = 1

    worker = MagicMock(spec=WebGPUWorker)
    worker.wgpu_device = wgpu_device
    WebGPUWorker.check_health(worker)   # should not raise
