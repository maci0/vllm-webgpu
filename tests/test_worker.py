import numpy as np
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


def test_compute_request_logprobs():
    """_compute_request_logprobs returns top-N tokens sorted by log-prob."""
    from vllm_webgpu.v1.model_runner import WebGPUModelRunner

    vocab = 32
    logits = np.zeros(vocab, dtype=np.float32)
    logits[5] = 10.0   # highest logit — should be rank 0
    logits[3] = 5.0    # second highest
    logits[7] = 2.0    # third

    top_ids, top_lp, rank = WebGPUModelRunner._compute_request_logprobs(logits, sampled_tok=5, num_logprobs=3)

    assert len(top_ids) == 3
    assert len(top_lp) == 3
    assert top_ids[0] == 5, "highest logit token should be first"
    assert top_ids[1] == 3
    assert top_ids[2] == 7
    assert (top_lp <= 0).all(), "log-probs must be non-positive"
    assert rank == 0, "sampled token 5 has the highest logit so rank should be 0"


def test_make_model_output_with_logprobs():
    """_make_model_output builds a non-None LogprobsLists when logprob data is supplied."""
    from vllm_webgpu.v1.model_runner import WebGPUModelRunner, LogprobsLists, ModelRunnerOutput

    if ModelRunnerOutput is None or LogprobsLists is None:
        pytest.skip("vllm not available (vllm mock installed by test_config.py)")

    runner = MagicMock(spec=WebGPUModelRunner)
    runner._last_model_output = None

    vocab = 16
    logits = np.zeros(vocab, dtype=np.float32)
    logits[2] = 8.0
    logits[9] = 4.0
    logits[1] = 1.0

    lp_data = WebGPUModelRunner._compute_request_logprobs(logits, sampled_tok=2, num_logprobs=2)

    out = WebGPUModelRunner._make_model_output(runner, ["req-1"], [2], [lp_data])
    assert out is not None
    assert out.logprobs is not None, "logprobs should be populated, not None"
    assert out.logprobs.logprob_token_ids.shape == (1, 2)
    assert out.logprobs.logprob_token_ids[0, 0] == 2
    assert out.logprobs.sampled_token_ranks[0] == 0


def test_make_model_output_no_logprobs():
    """_make_model_output sets logprobs=None when no request supplies logprob data."""
    from vllm_webgpu.v1.model_runner import WebGPUModelRunner, ModelRunnerOutput

    if ModelRunnerOutput is None:
        pytest.skip("vllm not available (vllm mock installed by test_config.py)")

    runner = MagicMock(spec=WebGPUModelRunner)
    runner._last_model_output = None

    out = WebGPUModelRunner._make_model_output(runner, ["req-1"], [7], [None])
    assert out is not None
    assert out.logprobs is None


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
