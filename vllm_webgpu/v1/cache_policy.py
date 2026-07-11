from __future__ import annotations
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.mem_constants import MiB_bytes
from vllm.utils.mem_utils import get_cpu_memory
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (FullAttentionSpec,
                                         KVQuantMode,
                                         MLAAttentionSpec,
                                         SlidingWindowMLASpec,
                                         SlidingWindowSpec,
                                         TQFullAttentionSpec,
                                         UniformTypeKVCacheSpecs)

OVERHEAD_BYTES = 512 * MiB_bytes  # driver overhead + activations
MIN_WEBGPU_BUFFER_BYTES: int = 16  # WebGPU spec forbids zero-size buffers

if TYPE_CHECKING:
    import wgpu
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheSpec, KVCacheTensor
    from vllm_webgpu.models.base import BaseWebGPUModel
    from vllm_webgpu.v1.worker import WebGPUWorker

from vllm_webgpu.webgpu.buffer import WebGPUBuffer

logger = init_logger(__name__)

# Layer type strings that carry KV state and require cache allocation.
# Must stay in sync with get_kv_cache_spec in model_runner.py, which imports
# this constant and uses it as the authoritative set.
#
# Note: Minimax models encode layer type as an integer where 1 = attention and
# 0 = non-attention. That integer sentinel is NOT in this frozenset. Use
# is_attn_layer() rather than direct `in KV_ATTN_TYPES` checks to handle both
# string and integer encodings at every call site.
KV_ATTN_TYPES: frozenset[str] = frozenset(
    {"attention", "full_attention", "sliding_attention"}
)


def is_attn_layer(lt: "str | int") -> bool:
    """Return True when a layer-type value represents an attention layer.

    Handles both string layer types (in KV_ATTN_TYPES) and the Minimax integer
    encoding where 1 means attention and 0 means non-attention (Mamba/MLP).
    Use this instead of bare `lt in KV_ATTN_TYPES` everywhere so that the
    integer sentinel never needs to be repeated at individual call sites.
    """
    return lt in KV_ATTN_TYPES or lt == 1


def allocate_kv_from_tensors(
    wgpu_device: "wgpu.GPUDevice",
    model: "BaseWebGPUModel | None",
    kv_cache_tensors: list[KVCacheTensor],
    num_blocks: int,
    num_total_layers: int,
    kv_cache_groups: list[KVCacheGroupSpec],
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
    # UniformTypeKVCacheSpecs wraps per-layer specs (e.g. heterogeneous-but-same-type
    # attention layers like Gemma4's 4-head local vs 8-head global); unpack it so
    # each layer name resolves to its individual spec rather than the umbrella object.
    layer_spec_map: dict[str, KVCacheSpec] = {}
    for group in kv_cache_groups:
        gs = group.kv_cache_spec
        if isinstance(gs, UniformTypeKVCacheSpecs):
            layer_spec_map.update(gs.kv_cache_specs)
        else:
            for name in group.layer_names:
                layer_spec_map[name] = gs

    # Build layer_index -> (k_bytes, v_bytes) from the tensors vLLM already computed.
    # shared_by holds names like "model.layers.{i}.self_attn" or "model.layers.{i}.mixer".
    layer_kv_bytes: dict[int, tuple[int, int]] = {}
    for tensor in kv_cache_tensors:
        if tensor.block_stride > 0:
            raise NotImplementedError(
                f"Packed KV cache layout (block_stride={tensor.block_stride}) is not supported by the WebGPU backend"
            )
        if not tensor.shared_by:
            raise NotImplementedError(
                "KVCacheTensor with empty shared_by is not supported by the WebGPU backend"
            )
        # Multi-group models (e.g. two kv_cache_groups with equal page sizes)
        # produce tensors where shared_by contains one layer name per group.
        # Each name is a distinct layer that needs its own wgpu buffers.
        # Prefer spec fields over tensor.size // 2. The latter includes
        # per-token-head scale bytes that inflate the allocation beyond what
        # the K or V data actually occupies, and also averages head_size and
        # head_size_v instead of allocating each buffer at its correct size.
        for layer_name in tensor.shared_by:
            spec = layer_spec_map.get(layer_name)
            if spec is None:
                raise RuntimeError(
                    f"Layer {layer_name!r} appears in kv_cache_tensors.shared_by but is absent "
                    "from every kv_cache_group.layer_names. This is a vLLM integration bug."
                )
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
            elif type(spec) is FullAttentionSpec:
                if spec.kv_quant_mode != KVQuantMode.NONE:
                    raise NotImplementedError(
                        f"Quantized KV cache (kv_quant_mode={spec.kv_quant_mode!r}) is not supported by the WebGPU backend; KV shaders expect float16 data."
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
            elif isinstance(spec, SlidingWindowSpec):
                raise NotImplementedError(
                    "SlidingWindowSpec KV cache is not supported by the WebGPU backend."
                )
            else:
                raise NotImplementedError(
                    f"Unsupported KV cache spec type {type(spec).__name__} for {layer_name!r}; "
                    "add an explicit branch to handle it."
                )
            try:
                idx = extract_layer_index(layer_name)
                layer_kv_bytes[idx] = (k_bytes, v_bytes)
            except AssertionError as exc:
                logger.error(
                    "Cannot parse layer index from KVCacheTensor.shared_by entry %r "
                    "(spec=%s, k=%d, v=%d bytes lost): %s",
                    layer_name,
                    type(spec).__name__,
                    k_bytes,
                    v_bytes,
                    exc,
                )
                raise

    # Verify that every attention layer in hybrid models got a real KV buffer.
    # Models with _layer_types (e.g. NemotronH) index kv_pool unconditionally in
    # _attn_layer; a 16-byte placeholder there silently corrupts kv_cache_store_both
    # and flash_attn_decode without any GPU-side error.
    # Pure-attention models lack _layer_types, so the check is safely skipped.
    _model_layer_types = getattr(model, "_layer_types", None)
    if _model_layer_types is not None and len(_model_layer_types) == num_total_layers:
        _missing_attn = [
            i for i, lt in enumerate(_model_layer_types)
            if is_attn_layer(lt) and i not in layer_kv_bytes
        ]
        if _missing_attn:
            raise RuntimeError(
                f"Attention layer(s) {_missing_attn} were not assigned real KV "
                f"buffers (kv_cache_tensors covered layers "
                f"{sorted(layer_kv_bytes.keys())}). These layers would receive "
                f"16-byte placeholder buffers, silently corrupting "
                f"kv_cache_store_both and flash_attn_decode dispatches. "
                f"Verify that kv_cache_config.kv_cache_tensors includes entries "
                f"for all attention layers declared in the model's layer type list."
            )

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


def get_layer_types(model, hf_config, hf_outer_config=None) -> list | None:
    """Return the layer-type list for a model, using a canonical three-way fallback.

    Priority: model._layer_types (set at load time) > hf_config.layers_block_type
    (NemotronH/Falcon) > hf_config.layer_types (Gemma4 and similar) >
    hf_outer_config.attn_type_list (Minimax).

    Returns None when none of the attributes is present.

    Uses explicit `is not None` guards (not `or`) so that an empty list, which
    is a valid value distinct from "attribute absent", is not silently skipped.

    hf_outer_config: optional outer ModelConfig.hf_config, used only for the
    attn_type_list probe. For multimodal models (e.g. Minimax) where
    hf_text_config != hf_config, attn_type_list lives on the outer config.
    Defaults to hf_config when not provided (single-config models).

    Note: the model._layer_types probe is only exercised by scripts/kv_utils.py,
    which passes a fully loaded model object. The vLLM engine path always calls
    this function with model=None (before weight loading), so the first probe is
    a permanent no-op in the engine code path.
    """
    if model is not None:
        v = getattr(model, "_layer_types", None)
        if v is not None:
            return v
    v = getattr(hf_config, "layers_block_type", None)
    if v is None:
        v = getattr(hf_config, "layer_types", None)
    if v is not None:
        return v
    # Minimax-style: integer list where 1 = attention, 0 = non-attention.
    # model_runner.py filters these with `lt != 1` rather than `lt not in KV_ATTN_TYPES`.
    # attn_type_list lives on the outer hf_config for multimodal models where
    # hf_text_config (passed as hf_config here) differs from the outer config.
    _outer = hf_outer_config if hf_outer_config is not None else hf_config
    return getattr(_outer, "attn_type_list", None)


def _get_weight_memory_usage(worker: "WebGPUWorker") -> int:
    """Sum of weight buffer sizes in bytes. The isinstance guard excludes the
    "__quant_meta__" dict entry that the quantized weight loader stores in
    model.weights alongside real buffers."""
    model = getattr(worker.model_runner, "model", None)
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
    if explicit := worker.cache_config.kv_cache_memory_bytes:
        return explicit

    model_mem = _get_weight_memory_usage(worker)

    total: int = get_cpu_memory()

    base = total - model_mem - OVERHEAD_BYTES
    fraction = worker.cache_config.gpu_memory_utilization
    available = max(int(base * fraction), 0)
    logger.info(
        "WebGPU memory: total=%dMiB model=%dMiB available=%dMiB",
        total // MiB_bytes, model_mem // MiB_bytes, available // MiB_bytes,
    )
    return available

