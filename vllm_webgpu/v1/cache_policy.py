from __future__ import annotations
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.mem_constants import MiB_bytes
from vllm.utils.mem_utils import get_cpu_memory
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (AttentionSpec,
                                         FullAttentionSpec,
                                         MLAAttentionSpec,
                                         SlidingWindowMLASpec,
                                         TQFullAttentionSpec)

OVERHEAD_BYTES = 512 * MiB_bytes  # driver overhead + activations
MIN_WEBGPU_BUFFER_BYTES: int = 16  # WebGPU spec forbids zero-size buffers

if TYPE_CHECKING:
    from vllm_webgpu.v1.worker import WebGPUWorker

from vllm_webgpu.webgpu.buffer import WebGPUBuffer

logger = init_logger(__name__)

# Layer type strings that carry KV state and require cache allocation.
# Must stay in sync with get_kv_cache_spec in model_runner.py, which imports
# this constant and uses it as the authoritative set.
KV_ATTN_TYPES: frozenset[str] = frozenset(
    {"attention", "full_attention", "sliding_attention"}
)


def allocate_kv_from_tensors(
    wgpu_device,
    model,
    kv_cache_tensors,
    num_blocks: int,
    num_total_layers: int,
    kv_cache_groups: list,
) -> None:
    """Allocate KV cache buffers from vLLM's authoritative KVCacheTensor list.

    Per-buffer bytes are derived from the spec's real_page_size_bytes (excludes
    per-token-head quantization scale overhead) when kv_cache_groups is provided.
    Dividing tensor.size by 2 would silently over-allocate when the KV cache
    dtype uses per-token-head scales, because page_size_bytes (and therefore
    tensor.size) includes those scale bytes but the WebGPU K/V shaders do not
    read them. Non-KV layers receive 16-byte placeholder buffers.
    """
    if model is None:
        raise RuntimeError("model must not be None during KV cache allocation")

    # Build layer_name -> KVCacheSpec map so we can use real_page_size_bytes,
    # which excludes the per-token-head scale overhead that page_size_bytes adds.
    layer_spec_map = {name: group.kv_cache_spec for group in kv_cache_groups for name in group.layer_names}

    # Build layer_index -> (k_bytes, v_bytes) from the tensors vLLM already computed.
    # shared_by holds names like "model.layers.{i}.self_attn" or "model.layers.{i}.mixer".
    layer_kv_bytes: dict[int, tuple[int, int]] = {}
    for tensor in kv_cache_tensors:
        if tensor.block_stride > 0:
            raise NotImplementedError(
                f"Packed KV cache layout (block_stride={tensor.block_stride}) is not supported by the WebGPU backend"
            )
        if len(tensor.shared_by) > 1:
            raise NotImplementedError(
                f"Shared-block-table KV cache (shared_by={tensor.shared_by}) is not supported by the WebGPU backend"
            )
        # Prefer spec fields over tensor.size // 2. The latter includes
        # per-token-head scale bytes that inflate the allocation beyond what
        # the K or V data actually occupies, and also averages head_size and
        # head_size_v instead of allocating each buffer at its correct size.
        first_name = tensor.shared_by[0] if tensor.shared_by else None
        spec = layer_spec_map.get(first_name)
        if isinstance(spec, MLAAttentionSpec):
            raise NotImplementedError(
                f"MLA KV cache ({type(spec).__name__}) is not supported by the WebGPU backend. "
                "MLAAttentionSpec uses a compressed latent layout that differs from the standard "
                "per-head K/V formula and cannot be sized with storage_block_size * head_size * dtype_bytes."
            )
        elif isinstance(spec, TQFullAttentionSpec):
            raise NotImplementedError(
                f"TQFullAttentionSpec KV cache is not supported by the WebGPU backend. "
                "TQFullAttentionSpec overrides real_page_size_bytes with a tq_slot_size-based formula "
                "that differs from the standard block_size * num_kv_heads * (head_size + head_size_v) * dtype_bytes. "
                "Allocating with head_size/head_size_v would produce wrong buffer sizes."
            )
        elif isinstance(spec, FullAttentionSpec):
            if spec.kv_quant_mode.is_nvfp4:
                raise NotImplementedError(
                    "NVFP4 KV cache quantization is not supported by the WebGPU backend. "
                    "The packed NVFP4 layout (64 fp4 data + 8 fp8 scale per head position) "
                    "cannot be expressed using the head_size * dtype_bytes formula."
                )
            # Compute K and V sizes independently so that asymmetric head
            # dimensions (e.g. MLA-style models where head_size != head_size_v)
            # get correctly sized buffers instead of an averaged size.
            dtype_bytes = get_dtype_size(spec.dtype)
            storage_bs = spec.storage_block_size
            k_bytes = num_blocks * storage_bs * spec.num_kv_heads * spec.head_size * dtype_bytes
            v_bytes = num_blocks * storage_bs * spec.num_kv_heads * spec.head_size_v * dtype_bytes
            # k_bytes + v_bytes == real_page_size_bytes * num_blocks by construction:
            # real_page_size_bytes = block_size * num_kv_heads * (head_size + head_size_v) * dtype_bytes,
            # and storage_bs == block_size for FullAttentionSpec (storage_block_size returns self.block_size).
        elif isinstance(spec, SlidingWindowMLASpec):
            raise NotImplementedError(
                f"SlidingWindowMLASpec KV cache is not supported by the WebGPU backend. "
                "SlidingWindowMLASpec stores a single MLA latent per position, so "
                "real_page_size_bytes is the full per-position size, not a K+V pair. "
                "Halving it would silently corrupt both cache buffers."
            )
        else:
            assert spec is not None, (
                f"No spec in layer_spec_map for {first_name!r}; "
                "vLLM planner contract violated (shared_by name missing from kv_cache_groups)"
            )
            assert isinstance(spec, AttentionSpec), (
                f"Unexpected spec type {type(spec).__name__} for {first_name!r}; "
                "only AttentionSpec subclasses are expected in layer_spec_map"
            )
        if first_name is not None:
            try:
                idx = extract_layer_index(first_name)
                layer_kv_bytes[idx] = (k_bytes, v_bytes)
            except AssertionError as exc:
                logger.error(
                    "Cannot parse layer index from KVCacheTensor.shared_by entry %r "
                    "(spec=%s, k=%d, v=%d bytes lost): %s",
                    first_name,
                    type(spec).__name__,
                    k_bytes,
                    v_bytes,
                    exc,
                )
                raise

    model.kv_pool.clear()
    total_bytes = 0
    for i in range(num_total_layers):
        if i in layer_kv_bytes:
            k_bytes, v_bytes = layer_kv_bytes[i]
            model.kv_pool.append((
                WebGPUBuffer.empty(wgpu_device, k_bytes),
                WebGPUBuffer.empty(wgpu_device, v_bytes),
            ))
            total_bytes += k_bytes + v_bytes
        else:
            model.kv_pool.append((
                WebGPUBuffer.empty(wgpu_device, MIN_WEBGPU_BUFFER_BYTES),
                WebGPUBuffer.empty(wgpu_device, MIN_WEBGPU_BUFFER_BYTES),
            ))

    logger.info(
        "KV cache: %d blocks, %d kv-attn layers, total=%dMiB",
        num_blocks, len(layer_kv_bytes), total_bytes // MiB_bytes,
    )


def get_layer_types(model, hf_config) -> list | None:
    """Return the layer-type list for a model, using a canonical three-way fallback.

    Priority: model._layer_types (set at load time) > hf_config.layer_types
    (Gemma4 and similar) > hf_config.layers_block_type (NemotronH/Falcon).
    Returns None when none of the three attributes is present.

    Uses explicit `is not None` guards (not `or`) so that an empty list, which
    is a valid value distinct from "attribute absent", is not silently skipped.
    vLLM's ModelConfig.get_num_layers_by_block_type uses the same pattern.

    Note: the model._layer_types probe is only exercised by scripts/kv_utils.py,
    which passes a fully loaded model object. The vLLM engine path always calls
    this function with model=None (before weight loading), so the first probe is
    a permanent no-op in the engine code path.
    """
    for obj, attr in [
        (model, "_layer_types"),
        (hf_config, "layer_types"),
        (hf_config, "layers_block_type"),
    ]:
        val = getattr(obj, attr, None)
        if val is not None:
            return val
    return None


def _get_weight_memory_usage(worker: "WebGPUWorker") -> int:
    """Sum of weight buffer sizes in bytes (excludes scratch/dummy/rope buffers)."""
    model = worker.model_runner.model if worker.model_runner is not None else None
    if model is None:
        return 0
    return sum(buf.nbytes for buf in model.weights.values() if isinstance(buf, WebGPUBuffer))


def determine_available_memory(worker: "WebGPUWorker") -> int:
    """
    Available memory for KV cache = device total - model weights - overhead.

    Uses vLLM's get_cpu_memory to query real system memory, which is correct for
    unified-memory platforms (Apple Silicon) and avoids confusing wgpu's per-buffer
    maxBufferSize limit with total device memory. On a 7B f16 model (~14 GB weights)
    with a 4 GB maxBufferSize, subtracting from maxBufferSize yields negative available
    memory and clamps to 0 KV blocks.

    """
    config = worker.webgpu_config
    model_mem = _get_weight_memory_usage(worker)

    total: int = get_cpu_memory()

    base = total - model_mem - OVERHEAD_BYTES
    fraction = config.memory_fraction if config.memory_fraction is not None else 1.0
    available = max(int(base * fraction), 0)
    logger.info(
        "WebGPU memory: total=%dMiB model=%dMiB available=%dMiB",
        total // MiB_bytes, model_mem // MiB_bytes, available // MiB_bytes,
    )
    return available

