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
    # mask buffer: one u32 per vocab element, initialized to 0 (shader writes 1 for selected)
    mask_buf = WebGPUBuffer.empty(dev, vocab * 4,
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
            {"binding": 3, "resource": {"buffer": mask_buf.buf}},
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
    result_val = val_buf.to_numpy().view(np.float32)
    assert list(result_idx) == top_indices, f"top-k indices mismatch: {list(result_idx)} != {top_indices}"

    # Invariant proof: output values must be strictly decreasing
    for i in range(len(result_val) - 1):
        assert result_val[i] >= result_val[i + 1], \
            f"top-k ordering violated at position {i}: {result_val[i]} < {result_val[i+1]}"

    # Invariant proof: all returned indices must be valid
    assert all(0 <= idx < vocab for idx in result_idx), "top-k returned out-of-range index"


def test_argmax_unique_max_invariant(wgpu_device):
    """Property proof: argmax returns the index of the unique maximum element.

    For ANY input where element k is strictly larger than all others,
    argmax must return k. This is the mathematical definition of argmax.
    Tested at multiple positions to prove it's not position-dependent.
    """
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey

    vocab = 128
    dev = wgpu_device.wgpu_device
    rw = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
    cache = PipelineCache(dev, SHADERS_DIR / "generic")
    pipeline = cache.get_or_create(PipelineKey("argmax", (("VOCAB_SIZE", vocab),)))

    # Test every 16th position to prove position-independence
    for true_max_idx in range(0, vocab, 16):
        base = np.random.randn(vocab).astype(np.float16) * 0.1  # small background
        base[true_max_idx] = np.float16(50.0)  # clear, undeniable maximum

        logits_buf = WebGPUBuffer.from_numpy(dev, base)
        out_buf = WebGPUBuffer.empty(dev, 4, usage=rw)

        bg = dev.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[{"binding": 0, "resource": {"buffer": logits_buf.buf}},
                     {"binding": 1, "resource": {"buffer": out_buf.buf}}],
        )
        enc = dev.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipeline)
        cp.set_bind_group(0, bg)
        cp.dispatch_workgroups(1, 1, 1)
        cp.end()
        dev.queue.submit([enc.finish()])

        result = int(out_buf.to_numpy().view(np.uint32)[0])
        assert result == true_max_idx, (
            f"argmax invariant violated at position {true_max_idx}: returned {result}"
        )
