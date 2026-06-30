import numpy as np
import pytest


def test_from_numpy_f16(wgpu_device):
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    arr = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float16)
    buf = WebGPUBuffer.from_numpy(wgpu_device.wgpu_device, arr)
    assert buf.shape == (1, 4)
    assert buf.dtype == "f16"
    assert buf.nbytes == arr.nbytes


def test_round_trip_f32(wgpu_device):
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    arr = np.random.randn(4, 8).astype(np.float32)
    buf = WebGPUBuffer.from_numpy(wgpu_device.wgpu_device, arr)
    result = buf.to_numpy().reshape(arr.shape)
    np.testing.assert_array_equal(result, arr)


def test_round_trip_f16(wgpu_device):
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    arr = np.random.randn(16, 64).astype(np.float16)
    buf = WebGPUBuffer.from_numpy(wgpu_device.wgpu_device, arr)
    result = buf.to_numpy().view(np.float16).reshape(arr.shape)
    np.testing.assert_array_equal(result, arr)


def test_empty_buffer(wgpu_device):
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    buf = WebGPUBuffer.empty(wgpu_device.wgpu_device, nbytes=1024)
    assert buf.nbytes == 1024
    assert buf.shape == (1024,)
    assert buf.dtype == "u8"


def test_dtype_detection(wgpu_device):
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    f16 = WebGPUBuffer.from_numpy(wgpu_device.wgpu_device, np.zeros(4, dtype=np.float16))
    f32 = WebGPUBuffer.from_numpy(wgpu_device.wgpu_device, np.zeros(4, dtype=np.float32))
    u8 = WebGPUBuffer.from_numpy(wgpu_device.wgpu_device, np.zeros(4, dtype=np.uint8))
    assert f16.dtype == "f16"
    assert f32.dtype == "f32"
    assert u8.dtype == "u8"
