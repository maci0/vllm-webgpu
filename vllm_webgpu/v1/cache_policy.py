from __future__ import annotations
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.mem_constants import MiB_bytes
from vllm.utils.mem_utils import get_cpu_memory
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (AttentionSpec, FullAttentionSpec,
                                         MLAAttentionSpec,
                                         SlidingWindowMLASpec)

OVERHEAD_BYTES = 512 * MiB_bytes  # driver overhead + activations
_MIN_WEBGPU_BUFFER_BYTES: int = 16  # WebGPU spec forbids zero-size buffers

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
    kv_cache_groups=None,
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
    layer_spec_map: dict[str, object] = {}
    if kv_cache_groups is not None:
        for group in kv_cache_groups:
            for name in group.layer_names:
                layer_spec_map[name] = group.kv_cache_spec

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
        if spec is not None and isinstance(spec, MLAAttentionSpec):
            raise NotImplementedError(
                f"MLA KV cache ({type(spec).__name__}) is not supported by the WebGPU backend. "
                "MLAAttentionSpec uses a compressed latent layout that differs from the standard "
                "per-head K/V formula and cannot be sized with storage_block_size * head_size * dtype_bytes."
            )
        elif spec is not None and isinstance(spec, FullAttentionSpec):
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
            if k_bytes + v_bytes != spec.real_page_size_bytes * num_blocks:
                logger.warning(
                    "Spec-derived K+V (%d B) does not match spec.real_page_size_bytes*num_blocks (%d B) "
                    "(layer %r); allocating separate K (%d B) and V (%d B) buffers "
                    "-- overhead bytes (per-token-head scales) are not "
                    "accessible to WebGPU shaders.",
                    k_bytes + v_bytes, spec.real_page_size_bytes * num_blocks, first_name, k_bytes, v_bytes,
                )
        elif isinstance(spec, SlidingWindowMLASpec):
            raise NotImplementedError(
                f"SlidingWindowMLASpec KV cache is not supported by the WebGPU backend. "
                "SlidingWindowMLASpec stores a single MLA latent per position, so "
                "real_page_size_bytes is the full per-position size, not a K+V pair. "
                "Halving it would silently corrupt both cache buffers."
            )
        elif isinstance(spec, AttentionSpec):
            if spec.kv_quant_mode.is_nvfp4:
                raise NotImplementedError(
                    "NVFP4 KV cache is not supported by the WebGPU backend"
                )
            # Non-FullAttentionSpec (e.g. SlidingWindowSpec): use spec-derived
            # page size, splitting correctly for potentially asymmetric head dims.
            if hasattr(spec, "head_size_v") and spec.head_size_v is not None and spec.head_size != spec.head_size_v:
                k_bytes = num_blocks * spec.storage_block_size * spec.num_kv_heads * spec.head_size * get_dtype_size(spec.dtype)
                v_bytes = num_blocks * spec.storage_block_size * spec.num_kv_heads * spec.head_size_v * get_dtype_size(spec.dtype)
            else:
                naive = tensor.size // 2
                half = spec.real_page_size_bytes * num_blocks // 2
                k_bytes = half
                v_bytes = half
                if half != naive:
                    logger.warning(
                        "KV tensor size contains non-data bytes (per-token-head scale "
                        "overhead): spec-derived per_buf=%d B, tensor.size//2=%d B "
                        "(layer %r). Using spec-derived value; scale bytes are not "
                        "accessible to WebGPU shaders.",
                        half, naive, first_name,
                    )
        elif spec is None:
            if first_name is None:
                # shared_by was empty; cannot identify layer — use 16-byte placeholder below.
                continue
            half = tensor.size // 2
            k_bytes = half
            v_bytes = half
            logger.warning(
                "No spec found for layer %r; falling back to tensor.size // 2. "
                "May over-allocate for quantized KV cache with scale bytes.",
                first_name,
            )
        else:
            k_bytes = _MIN_WEBGPU_BUFFER_BYTES
            v_bytes = _MIN_WEBGPU_BUFFER_BYTES
            logger.warning(
                "Spec for layer %r (%s) is not an attention spec; using %d-byte placeholder.",
                first_name or "<unknown>",
                type(spec).__name__,
                _MIN_WEBGPU_BUFFER_BYTES,
            )
        if first_name is not None:
            try:
                idx = extract_layer_index(first_name)
                layer_kv_bytes[idx] = (k_bytes, v_bytes)
            except Exception:
                logger.warning("Cannot parse layer index from KVCacheTensor.shared_by entry %r", first_name)

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
                WebGPUBuffer.empty(wgpu_device, _MIN_WEBGPU_BUFFER_BYTES),
                WebGPUBuffer.empty(wgpu_device, _MIN_WEBGPU_BUFFER_BYTES),
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


def _make_convertor(hf_cfg):
    """Instantiate the vLLM model-arch convertor for hf_cfg.

    Single point of dispatch for MODEL_ARCH_CONFIG_CONVERTORS so callers avoid
    repeating the getattr/get/instantiate pattern.

    This function is only reached from standalone scripts (run_inference.py,
    profile_kernels.py) that lack a VllmConfig. In the vLLM engine path,
    vllm_config.model_config already exposes get_total_num_kv_heads() and
    get_head_size() directly, and allocate_kv_from_tensors is used instead of
    allocate_kv_from_hf_config, so _make_convertor is never called there.
    """
    from vllm.transformers_utils.model_arch_config_convertor import (
        MODEL_ARCH_CONFIG_CONVERTORS,
        ModelArchConfigConvertorBase,
    )
    from vllm.transformers_utils.config import get_hf_text_config
    hf_text = get_hf_text_config(hf_cfg)
    return MODEL_ARCH_CONFIG_CONVERTORS.get(
        getattr(hf_cfg, "model_type", ""), ModelArchConfigConvertorBase
    )(hf_cfg, hf_text)



def get_kv_dims_from_hf_config(hf_cfg) -> tuple[int, int]:
    """Return (num_kv_heads, head_size) from a raw HuggingFace config object.

    Intended for standalone scripts (e.g. profile_kernels.py) that do not have a
    VllmConfig available. In contexts where a VllmConfig is present, prefer
    model_config.get_total_num_kv_heads() / model_config.get_head_size() directly.
    """
    conv = _make_convertor(hf_cfg)
    return conv.get_total_num_kv_heads(), conv.get_head_size()


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
    fraction = 1.0 if config.is_auto_memory else config.memory_fraction
    available = max(int(base * fraction), 0)
    logger.info(
        "WebGPU memory: total=%dMiB model=%dMiB available=%dMiB",
        total // MiB_bytes, model_mem // MiB_bytes, available // MiB_bytes,
    )
    return available

