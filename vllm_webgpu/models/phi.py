from __future__ import annotations
from typing import TYPE_CHECKING

import wgpu as wgpu_lib

from vllm.logger import init_logger
from vllm_webgpu.models.base import _STAGING_USAGE
from vllm_webgpu.models.llama import LlamaWebGPUModel
from vllm_webgpu.webgpu.buffer import WebGPUBuffer

if TYPE_CHECKING:
    from vllm_webgpu.webgpu.device import WebGPUDevice
    from vllm_webgpu.webgpu.pipeline import PipelineCache

logger = init_logger(__name__)


class PhiWebGPUModel(LlamaWebGPUModel):
    """Phi-4 / Phi-4-mini WebGPU backend.

    Phi-3/4 checkpoints store fused qkv_proj.weight and gate_up_proj.weight.
    load_weights splits them into separate q/k/v and gate/up tensors so the
    inherited LlamaWebGPUModel dispatch can proceed without modification.
    """

    def load_weights(
        self,
        path: str,
        f32_keys=None,
        skip_prefixes=None,
        scale_transforms=None,
    ) -> None:
        super().load_weights(
            path,
            f32_keys=f32_keys,
            skip_prefixes=skip_prefixes,
            scale_transforms=scale_transforms,
        )
        self._split_fused_phi_weights()

    def _split_fused_phi_weights(self) -> None:
        """Split fused qkv_proj and gate_up_proj into separate projection buffers.

        Phi-3/4 checkpoints pre-fuse Q, K, V into qkv_proj.weight and the gate
        and up projections into gate_up_proj.weight. LlamaWebGPUModel._attn_block
        and _ffn_dispatch expect separate tensors, so this method splits them
        using GPU-side copy_buffer_to_buffer (no CPU roundtrip for f16 and GPTQ).

        Supported quantization formats: f16 (USE_QUANT=0), GPTQ int4 (USE_QUANT=3).
        AWQ (USE_QUANT=4) stores weights K-major and cannot be split by row bytes.
        """
        dev = self.wgpu_device.wgpu_device
        q_dim  = self.q_dim
        kv_dim = self.kv_dim
        hidden = self.hidden_size
        inter  = self.intermediate_size

        for i in range(self.num_layers):
            p = f"model.layers.{i}"

            # Split qkv_proj.weight -> q_proj.weight + k_proj.weight + v_proj.weight
            qkv_key = f"{p}.self_attn.qkv_proj.weight"
            if qkv_key in self.weights:
                self._split_row_major(
                    dev, qkv_key,
                    [(f"{p}.self_attn.q_proj.weight",  q_dim),
                     (f"{p}.self_attn.k_proj.weight", kv_dim),
                     (f"{p}.self_attn.v_proj.weight", kv_dim)],
                    K=hidden,
                )

            # Split gate_up_proj.weight -> gate_proj.weight + up_proj.weight
            gu_key = f"{p}.mlp.gate_up_proj.weight"
            if gu_key in self.weights:
                self._split_row_major(
                    dev, gu_key,
                    [(f"{p}.mlp.gate_proj.weight", inter),
                     (f"{p}.mlp.up_proj.weight",   inter)],
                    K=hidden,
                )

    def _split_row_major(
        self,
        dev,
        src_key: str,
        slices: "list[tuple[str, int]]",
        K: int,
    ) -> None:
        """Split a row-major weight buffer into non-overlapping row slices.

        Args:
            src_key:  Key of the source (fused) buffer in self.weights.
            slices:   List of (dst_key, n_rows) tuples defining each split piece.
            K:        Input feature dimension; determines bytes per row.

        Only f16 (USE_QUANT=0) and GPTQ int4 (USE_QUANT=3) are supported.
        AWQ stores weights K-major (transposed) and cannot be split by row bytes.
        """
        src_buf = self.weights[src_key]
        uq = self._uq_for_key(src_key)
        if uq == 0:
            row_bytes = K * 2          # f16: 2 bytes per element
            col_dim   = K
        elif uq == 3:
            row_bytes = (K // 8) * 4   # GPTQ int4: 8 int4 packed per i32 (4 bytes)
            col_dim   = K // 8
        else:
            raise NotImplementedError(
                f"Phi weight split for USE_QUANT={uq} (key={src_key!r}) is not supported. "
                "Only f16 (0) and GPTQ int4 (3) implement row-major layout. "
                "AWQ (4) stores weights K-major and cannot be sliced by row-byte offset."
            )

        enc    = dev.create_command_encoder()
        offset = 0
        dst_bufs: list[tuple[str, "WebGPUBuffer"]] = []
        for dst_key, n_rows in slices:
            nb      = n_rows * row_bytes
            dst_buf = self._make_buf(nb)
            dst_buf.dtype = src_buf.dtype
            dst_buf.shape = (n_rows, col_dim)
            enc.copy_buffer_to_buffer(src_buf.buf, offset, dst_buf.buf, 0, nb)
            dst_bufs.append((dst_key, dst_buf))
            offset += nb
        dev.queue.submit([enc.finish()])

        for dst_key, dst_buf in dst_bufs:
            self.weights[dst_key] = dst_buf
        del self.weights[src_key]

        # GPTQ scale split: scale tensors have shape [K//group_size, N_total].
        # Slicing along the N axis requires a CPU roundtrip (non-contiguous in memory).
        if uq == 3:
            scales_key = f"{src_key}.scales"
            if scales_key in self.weights:
                self._split_gptq_scales(dev, scales_key, slices)

    def _split_gptq_scales(
        self,
        dev,
        src_scales_key: str,
        slices: "list[tuple[str, int]]",
    ) -> None:
        """Split a GPTQ scale buffer [K//gs, N_total] along the N (output) axis.

        GPTQ scales have layout [G, N] where G = K // group_size and N = total
        output neurons.  Splitting along N requires a CPU roundtrip because the
        slices are not contiguous in GPU memory (column-major strides).
        """
        import numpy as np

        src_buf = self.weights[src_scales_key]
        G, N_total = src_buf.shape
        nb = src_buf.nbytes

        staging = dev.create_buffer(size=nb, usage=_STAGING_USAGE)
        enc = dev.create_command_encoder()
        enc.copy_buffer_to_buffer(src_buf.buf, 0, staging, 0, nb)
        dev.queue.submit([enc.finish()])
        staging.map_sync(mode=wgpu_lib.MapMode.READ)
        raw = bytes(staging.read_mapped())
        staging.unmap()

        arr = np.frombuffer(raw, dtype=np.float32).reshape(G, N_total)

        col = 0
        for dst_key, n_rows in slices:
            chunk = np.ascontiguousarray(arr[:, col:col + n_rows])
            dst_scales_key = f"{dst_key}.scales"
            self.weights[dst_scales_key] = WebGPUBuffer.from_numpy(dev, chunk)
            col += n_rows
        del self.weights[src_scales_key]

        # Propagate quant_meta from the fused key to the split keys so that
        # _uq_for_key and _quant_extra find the correct group_size and fmt.
        src_base = src_scales_key.removesuffix(".weight.scales")
        if src_base in self.weight_meta:
            meta_entry = self.weight_meta[src_base]
            for dst_key, _ in slices:
                dst_base = dst_key.removesuffix(".weight")
                self.weight_meta[dst_base] = dict(meta_entry)
            del self.weight_meta[src_base]
