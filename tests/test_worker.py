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
    logits[5] = 10.0   # highest logit — should be rank 1 (1-based, vLLM convention)
    logits[3] = 5.0    # second highest
    logits[7] = 2.0    # third

    top_ids, top_lp, rank = WebGPUModelRunner._compute_request_logprobs(logits, sampled_tok=5, num_logprobs=3)

    assert len(top_ids) == 3
    assert len(top_lp) == 3
    assert top_ids[0] == 5, "highest logit token should be first"
    assert top_ids[1] == 3
    assert top_ids[2] == 7
    assert (top_lp <= 0).all(), "log-probs must be non-positive"
    assert rank == 1, "sampled token 5 has the highest logit so rank should be 1 (1-based)"


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
    assert out.logprobs.sampled_token_ranks[0] == 1


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


def test_compute_prompt_logprobs():
    """_compute_prompt_logprobs returns LogprobsTensors with correct shape."""
    from vllm_webgpu.v1.model_runner import WebGPUModelRunner, LogprobsTensors

    if LogprobsTensors is None:
        pytest.skip("vllm not available")

    vocab = 32
    T = 5
    # Synthetic logits: position i has token i+1 as the highest logit
    full_logits = np.zeros((T, vocab), dtype=np.float32)
    for i in range(T):
        full_logits[i, i + 1] = 10.0  # best prediction at pos i is token i+1

    tok_ids = list(range(T + 1))  # prompt tokens [0..T]
    num_prompt_logprobs = 2

    result = WebGPUModelRunner._compute_prompt_logprobs(full_logits, tok_ids[:T], num_prompt_logprobs)

    assert result is not None
    # Shape: [T-1, num_prompt_logprobs+1]
    assert result.logprob_token_ids.shape == (T - 1, num_prompt_logprobs + 1)
    assert result.logprobs.shape == (T - 1, num_prompt_logprobs + 1)
    assert result.selected_token_ranks.shape == (T - 1,)
    # Each position predicts tok_ids[i+1] correctly, so rank should be 1 (1-based, vLLM convention)
    assert (result.selected_token_ranks == 1).all()
    # Log-probs must be non-positive
    assert (result.logprobs <= 0).all()


def test_compute_prompt_logprobs_short_sequence():
    """_compute_prompt_logprobs returns None for sequences shorter than 2 tokens."""
    from vllm_webgpu.v1.model_runner import WebGPUModelRunner, LogprobsTensors

    if LogprobsTensors is None:
        pytest.skip("vllm not available")

    # Single-token prompt: no valid position to compute prompt logprobs
    result = WebGPUModelRunner._compute_prompt_logprobs(
        np.zeros((1, 32), dtype=np.float32), [5], 2
    )
    assert result is None


def test_make_model_output_with_prompt_logprobs():
    """_make_model_output passes prompt_logprobs_dict through to ModelRunnerOutput."""
    from vllm_webgpu.v1.model_runner import WebGPUModelRunner, LogprobsTensors, ModelRunnerOutput

    if ModelRunnerOutput is None or LogprobsTensors is None:
        pytest.skip("vllm not available")

    import torch

    runner = MagicMock(spec=WebGPUModelRunner)
    runner._last_model_output = None

    fake_tensors = LogprobsTensors(
        logprob_token_ids=torch.zeros((3, 3), dtype=torch.int32),
        logprobs=torch.full((3, 3), -1.0),
        selected_token_ranks=torch.zeros(3, dtype=torch.int32),
    )
    pld = {"req-1": fake_tensors}

    out = WebGPUModelRunner._make_model_output(runner, ["req-1"], [7], [None], prompt_logprobs_dict=pld)
    assert out is not None
    assert out.prompt_logprobs_dict == pld


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


def _make_gemma4_runner_pre_load(layer_types, default_hd=256, default_kv=8,
                                  global_hd=512, global_kv=1,
                                  hidden_size=4096, num_q_heads=16,
                                  num_hidden_layers=None):
    """Build a WebGPUModelRunner mock with model=None and a Gemma4-style hf_config."""
    from vllm_webgpu.v1.model_runner import WebGPUModelRunner
    from vllm_webgpu.config import WebGPUConfig

    if num_hidden_layers is None:
        num_hidden_layers = len(layer_types)

    hf_config = MagicMock()
    hf_config.num_hidden_layers = num_hidden_layers
    hf_config.hidden_size = hidden_size
    hf_config.num_attention_heads = num_q_heads
    hf_config.num_key_value_heads = default_kv
    hf_config.head_dim = default_hd
    hf_config.global_head_dim = global_hd
    hf_config.num_global_key_value_heads = global_kv
    hf_config.layer_types = layer_types
    # Simulate safetensors: no _layer_attention_params on hf_config
    del hf_config._layer_attention_params

    vllm_config = MagicMock()
    vllm_config.model_config.hf_config = hf_config

    runner = MagicMock(spec=WebGPUModelRunner)
    runner.model = None  # not yet loaded
    runner.vllm_config = vllm_config
    runner.webgpu_config = WebGPUConfig.from_env()
    return runner


def test_get_kv_cache_spec_pre_load_gemma4_heterogeneous():
    """get_kv_cache_spec derives correct per-layer specs from layer_types before load_model()."""
    from vllm_webgpu.v1.model_runner import WebGPUModelRunner, FullAttentionSpec

    if FullAttentionSpec is None:
        pytest.skip("vllm not available")

    # 6-layer model: every 6th (index 5) is full_attention (global), others are local
    layer_types = ["sliding_attention"] * 5 + ["full_attention"]
    runner = _make_gemma4_runner_pre_load(
        layer_types=layer_types,
        default_hd=256, default_kv=8,
        global_hd=512, global_kv=1,
    )

    spec = WebGPUModelRunner.get_kv_cache_spec(runner)

    assert len(spec) == 6, f"expected 6 specs, got {len(spec)}"

    # Local layers (0-4): num_kv_heads=8, head_size=256
    for i in range(5):
        key = f"model.layers.{i}.self_attn"
        assert key in spec, f"missing key {key}"
        s = spec[key]
        assert s.num_kv_heads == 8, f"layer {i}: expected num_kv_heads=8, got {s.num_kv_heads}"
        assert s.head_size == 256, f"layer {i}: expected head_size=256, got {s.head_size}"

    # Global layer (5): num_kv_heads=1, head_size=512
    key = "model.layers.5.self_attn"
    assert key in spec, f"missing key {key}"
    s = spec[key]
    assert s.num_kv_heads == 1, f"layer 5: expected num_kv_heads=1, got {s.num_kv_heads}"
    assert s.head_size == 512, f"layer 5: expected head_size=512, got {s.head_size}"


def test_get_kv_cache_spec_pre_load_gemma4_uniform_fallback():
    """get_kv_cache_spec falls back to uniform specs when no layer_types and model=None."""
    from vllm_webgpu.v1.model_runner import WebGPUModelRunner, FullAttentionSpec

    if FullAttentionSpec is None:
        pytest.skip("vllm not available")

    # No layer_types: uniform Llama/Gemma3 config
    hf_config = MagicMock()
    hf_config.num_hidden_layers = 4
    hf_config.hidden_size = 2048
    hf_config.num_attention_heads = 16
    hf_config.num_key_value_heads = 4
    hf_config.head_dim = 128
    # Simulate no layer_types and no _layer_attention_params
    del hf_config.layer_types
    del hf_config._layer_attention_params

    from vllm_webgpu.config import WebGPUConfig
    vllm_config = MagicMock()
    vllm_config.model_config.hf_config = hf_config
    vllm_config.model_config.get_head_size.return_value = 128
    vllm_config.model_config.get_total_num_kv_heads.return_value = 4

    from vllm_webgpu.v1.model_runner import WebGPUModelRunner
    runner = MagicMock(spec=WebGPUModelRunner)
    runner.model = None
    runner.vllm_config = vllm_config
    runner.webgpu_config = WebGPUConfig.from_env()

    spec = WebGPUModelRunner.get_kv_cache_spec(runner)

    assert len(spec) == 4
    for i in range(4):
        s = spec[f"model.layers.{i}.self_attn"]
        assert s.num_kv_heads == 4
        assert s.head_size == 128
