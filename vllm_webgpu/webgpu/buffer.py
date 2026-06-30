from __future__ import annotations
import numpy as np

_DTYPE_MAP = {
    np.float16: "f16",
    np.float32: "f32",
    np.uint8: "u8",
    np.int32: "i32",
    np.uint32: "u32",
}

_STORAGE_COPY_DST = None  # set lazily from wgpu.BufferUsage
_STORAGE_COPY_DST_SRC = None


def _usage_storage_copy_dst():
    import wgpu
    return wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC


def _usage_storage_rw():
    import wgpu
    return wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC


class WebGPUBuffer:
    def __init__(self, buf, device, shape: tuple[int, ...], dtype: str) -> None:
        self.buf = buf
        self.device = device
        self.shape = shape
        self.dtype = dtype

    @property
    def nbytes(self) -> int:
        return self.buf.size

    @staticmethod
    def from_numpy(wgpu_device, arr: np.ndarray, usage: int | None = None) -> "WebGPUBuffer":
        if usage is None:
            usage = _usage_storage_copy_dst()
        data = np.ascontiguousarray(arr)
        buf = wgpu_device.create_buffer_with_data(data=data.tobytes(), usage=usage)
        dtype = _DTYPE_MAP.get(arr.dtype.type, "u8")
        return WebGPUBuffer(buf=buf, device=wgpu_device, shape=tuple(arr.shape), dtype=dtype)

    @staticmethod
    def empty(wgpu_device, nbytes: int, usage: int | None = None) -> "WebGPUBuffer":
        if usage is None:
            usage = _usage_storage_rw()
        buf = wgpu_device.create_buffer(size=nbytes, usage=usage)
        return WebGPUBuffer(buf=buf, device=wgpu_device, shape=(nbytes,), dtype="u8")

    def to_numpy(self) -> np.ndarray:
        """Read buffer back to CPU. Slow — debug and logit readback only."""
        import wgpu
        staging = self.device.create_buffer(
            size=self.buf.size,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        encoder = self.device.create_command_encoder()
        encoder.copy_buffer_to_buffer(self.buf, 0, staging, 0, self.buf.size)
        self.device.queue.submit([encoder.finish()])
        staging.map_sync(mode=wgpu.MapMode.READ)
        data = bytes(staging.read_mapped())
        staging.unmap()

        # Return raw uint8 bytes
        result = np.frombuffer(data, dtype=np.uint8).copy()

        # For dtypes other than u8, view the bytes back as the original dtype
        if self.dtype != "u8":
            dtype_map = {
                "f16": np.float16,
                "f32": np.float32,
                "i32": np.int32,
                "u32": np.uint32,
            }
            result = result.view(dtype_map.get(self.dtype, np.uint8))

        return result
