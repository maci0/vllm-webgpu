import pytest
from pathlib import Path
import textwrap


SIMPLE_WGSL = textwrap.dedent("""\
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;

    @compute @workgroup_size(1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        output[gid.x] = input[gid.x] * 2.0;
    }
""")


def test_pipeline_cache_hit(wgpu_device, tmp_path):
    from vllm_webgpu.webgpu.pipeline import PipelineKey, PipelineCache
    shader_file = tmp_path / "test.wgsl"
    shader_file.write_text(SIMPLE_WGSL)

    cache = PipelineCache(wgpu_device.wgpu_device, tmp_path)
    key = PipelineKey("test", ())
    p1 = cache.get_or_create(key)
    p2 = cache.get_or_create(key)
    assert p1 is p2


def test_pipeline_cache_miss_different_defines(wgpu_device, tmp_path):
    from vllm_webgpu.webgpu.pipeline import PipelineKey, PipelineCache
    shader_file = tmp_path / "test.wgsl"
    shader_file.write_text(SIMPLE_WGSL)

    cache = PipelineCache(wgpu_device.wgpu_device, tmp_path)
    k1 = PipelineKey("test", (("N", 4),))
    k2 = PipelineKey("test", (("N", 8),))
    p1 = cache.get_or_create(k1)
    p2 = cache.get_or_create(k2)
    assert p1 is not p2


def test_pipeline_executes(wgpu_device, tmp_path):
    import numpy as np
    from vllm_webgpu.webgpu.pipeline import PipelineKey, PipelineCache
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    import wgpu

    shader_file = tmp_path / "test.wgsl"
    shader_file.write_text(SIMPLE_WGSL)

    dev = wgpu_device.wgpu_device
    inp = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    inp_buf = WebGPUBuffer.from_numpy(dev, inp)
    out_buf = WebGPUBuffer.empty(dev, inp.nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)

    cache = PipelineCache(dev, tmp_path)
    key = PipelineKey("test", ())
    pipeline = cache.get_or_create(key)

    bg_layout = pipeline.get_bind_group_layout(0)
    bg = dev.create_bind_group(
        layout=bg_layout,
        entries=[
            {"binding": 0, "resource": {"buffer": inp_buf.buf}},
            {"binding": 1, "resource": {"buffer": out_buf.buf}},
        ],
    )
    encoder = dev.create_command_encoder()
    cp = encoder.begin_compute_pass()
    cp.set_pipeline(pipeline)
    cp.set_bind_group(0, bg)
    cp.dispatch_workgroups(4, 1, 1)
    cp.end()
    dev.queue.submit([encoder.finish()])

    result = out_buf.to_numpy().view(np.float32)
    np.testing.assert_allclose(result, inp * 2.0)
