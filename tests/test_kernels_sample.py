import numpy as np
import pytest
from pathlib import Path

SHADERS_DIR = Path(__file__).parent.parent / "vllm_webgpu" / "shaders"


def test_argmax(wgpu_device):
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey

    vocab = 128
    logits = np.random.randn(vocab).astype(np.float16)
    logits[77] = 100.0   # known maximum
    expected_idx = np.uint32(77)

    dev = wgpu_device.wgpu_device
    logits_buf = WebGPUBuffer.from_numpy(dev, logits)
    out_buf = WebGPUBuffer.empty(dev, 4,
                                 usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)

    cache = PipelineCache(dev, SHADERS_DIR / "generic")
    key = PipelineKey("argmax", (("VOCAB_SIZE", vocab),))
    pipeline = cache.get_or_create(key)

    bg = dev.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": logits_buf.buf}},
            {"binding": 1, "resource": {"buffer": out_buf.buf}},
        ],
    )
    encoder = dev.create_command_encoder()
    cp = encoder.begin_compute_pass()
    cp.set_pipeline(pipeline)
    cp.set_bind_group(0, bg)
    cp.dispatch_workgroups(1, 1, 1)
    cp.end()
    dev.queue.submit([encoder.finish()])

    result = out_buf.to_numpy().view(np.uint32)[0]
    assert result == expected_idx


def test_topk256(wgpu_device):
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey

    vocab = 128
    k = 5
    logits = np.random.randn(vocab).astype(np.float16)
    # Plant known top-k values at specific indices
    top_indices = [77, 3, 100, 50, 22]
    top_values = [10.0, 9.0, 8.0, 7.0, 6.0]
    for idx, val in zip(top_indices, top_values):
        logits[idx] = np.float16(val)

    dev = wgpu_device.wgpu_device
    logits_buf = WebGPUBuffer.from_numpy(dev, logits)
    idx_buf = WebGPUBuffer.empty(dev, k * 4,
                                 usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
    val_buf = WebGPUBuffer.empty(dev, k * 4,
                                 usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)

    cache = PipelineCache(dev, SHADERS_DIR / "generic")
    key = PipelineKey("topk256", (("VOCAB_SIZE", vocab), ("K", k)))
    pipeline = cache.get_or_create(key)

    bg = dev.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": logits_buf.buf}},
            {"binding": 1, "resource": {"buffer": idx_buf.buf}},
            {"binding": 2, "resource": {"buffer": val_buf.buf}},
        ],
    )
    encoder = dev.create_command_encoder()
    cp = encoder.begin_compute_pass()
    cp.set_pipeline(pipeline)
    cp.set_bind_group(0, bg)
    cp.dispatch_workgroups(1, 1, 1)
    cp.end()
    dev.queue.submit([encoder.finish()])

    result_idx = idx_buf.to_numpy().view(np.uint32)
    assert list(result_idx) == top_indices, f"top-k indices mismatch: {list(result_idx)} != {top_indices}"
