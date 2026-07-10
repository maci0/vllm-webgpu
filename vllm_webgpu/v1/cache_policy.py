from __future__ import annotations
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.mem_constants import MiB_bytes
from vllm.utils.mem_utils import get_cpu_memory
import torch
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import AttentionSpec, FullAttentionSpec

OVERHEAD_BYTES = 512 * MiB_bytes  # driver overhead + activations

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


def _allocate_kv_pool_hybrid(
    dev,
    model,
    num_blocks: int,
    num_layers: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    layer_types: list | None = None,
    dtype: torch.dtype = torch.float16,
) -> None:
    """Allocate KV pool for all layers.

    When layer_types is None, or all entries are in KV_ATTN_TYPES, every layer
    gets a full KV cache buffer. When layer_types is provided, layers whose type
    is not in KV_ATTN_TYPES get 16-byte placeholder buffers.
    """
    if model is None:
        raise RuntimeError("model must not be None during KV cache allocation")
    bytes_per_layer = num_blocks * block_size * num_kv_heads * head_dim * get_dtype_size(dtype)

    if layer_types is not None and len(layer_types) != num_layers:
        raise ValueError(
            f"layer_types length {len(layer_types)} != num_layers {num_layers}"
        )

    model.kv_pool.clear()

    kv_layer_count = 0
    for i in range(num_layers):
        needs_kv_cache = layer_types is None or layer_types[i] in KV_ATTN_TYPES
        if needs_kv_cache:
            k_buf = WebGPUBuffer.empty(dev, bytes_per_layer)
            v_buf = WebGPUBuffer.empty(dev, bytes_per_layer)
            kv_layer_count += 1
        else:
            k_buf = WebGPUBuffer.empty(dev, 16)
            v_buf = WebGPUBuffer.empty(dev, 16)
        model.kv_pool.append((k_buf, v_buf))

    if layer_types is None:
        total_mb = (bytes_per_layer * num_layers * 2) // MiB_bytes
        logger.info(
            "KV cache: %d blocks × %d tokens/block × %d layers × %d KV heads × %d head_dim (%s, K+V) = %dMB",
            num_blocks, block_size, num_layers, num_kv_heads, head_dim, dtype, total_mb,
        )
    else:
        total_mb = (bytes_per_layer * kv_layer_count * 2) // MiB_bytes
        logger.info(
            "KV cache (hybrid): %d kv-attn × %d blocks × %d tokens/block × %d KV heads × %d head_dim (%s, K+V) = %dMB",
            kv_layer_count, num_blocks, block_size, num_kv_heads, head_dim, dtype, total_mb,
        )


def _allocate_kv_pool_per_layer(
    dev,
    model,
    num_blocks: int,
    block_size: int,
    layer_params: list,
    dtype: torch.dtype = torch.float16,
) -> None:
    """Allocate KV pool for models with per-layer KV dims (e.g. Gemma4 with mixed local/global attention dims).

    layer_params is a list of dicts (one per layer) each with keys:
      num_kv_heads, head_dim
    """
    if model is None:
        raise RuntimeError("model must not be None during KV cache allocation")
    model.kv_pool.clear()
    logger.info("KV cache (per-layer): %d layers, mixed dims", len(layer_params))
    for lp in layer_params:
        if lp["num_kv_heads"] == 0:
            # Non-attention layer (num_kv_heads == 0). A zero-byte buffer
            # violates the WebGPU spec (size must be > 0), so use the same
            # 16-byte placeholder that _allocate_kv_pool_hybrid uses.
            model.kv_pool.append((
                WebGPUBuffer.empty(dev, 16),
                WebGPUBuffer.empty(dev, 16),
            ))
            continue
        if lp["head_dim"] == 0:
            raise ValueError(
                f"num_kv_heads={lp['num_kv_heads']} but head_dim=0; invalid KV spec"
            )
        kv_bytes = num_blocks * block_size * lp["num_kv_heads"] * lp["head_dim"] * get_dtype_size(dtype)
        model.kv_pool.append((
            WebGPUBuffer.empty(dev, kv_bytes),
            WebGPUBuffer.empty(dev, kv_bytes),
        ))


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
        spec = layer_spec_map.get(first_name) if first_name is not None else None
        if spec is not None and isinstance(spec, FullAttentionSpec):
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
            if k_bytes + v_bytes != tensor.size:
                logger.warning(
                    "Spec-derived total (%d B) does not match tensor.size (%d B) "
                    "(layer %r); allocating separate K (%d B) and V (%d B) buffers "
                    "— overhead bytes (per-token-head scales, padding) are not "
                    "accessible to WebGPU shaders.",
                    k_bytes + v_bytes, tensor.size, first_name, k_bytes, v_bytes,
                )
        elif isinstance(spec, AttentionSpec):
            if spec.kv_quant_mode.is_nvfp4:
                raise NotImplementedError(
                    "NVFP4 KV cache is not supported by the WebGPU backend"
                )
            # Non-FullAttentionSpec (e.g. SlidingWindowSpec): use spec-derived
            # page size, splitting correctly for potentially asymmetric head dims.
            naive = tensor.size // 2
            if hasattr(spec, "head_size_v") and spec.head_size != spec.head_size_v:
                k_bytes = num_blocks * spec.block_size * spec.num_kv_heads * spec.head_size * get_dtype_size(spec.dtype)
                v_bytes = num_blocks * spec.block_size * spec.num_kv_heads * spec.head_size_v * get_dtype_size(spec.dtype)
            else:
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
            k_bytes = 16
            v_bytes = 16
            logger.warning(
                "Spec for layer %r (%s) is not an attention spec; using 16-byte placeholder.",
                first_name or "<unknown>",
                type(spec).__name__,
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
                WebGPUBuffer.empty(wgpu_device, 16),
                WebGPUBuffer.empty(wgpu_device, 16),
            ))

    logger.info(
        "KV cache: %d blocks, %d kv-attn layers, total=%dMB",
        num_blocks, len(layer_kv_bytes), total_bytes // MiB_bytes,
    )


def allocate_kv_from_hf_config(
    wgpu_device,
    model,
    hf_config,
    num_blocks: int,
    block_size: int,
) -> None:
    """Allocate KV cache from a HuggingFace config object.

    Used only by standalone scripts (run_inference.py, profile_kernels.py).
    The vLLM engine path uses allocate_kv_from_tensors, which derives buffer
    sizes directly from vLLM KVCacheTensor objects and requires no
    per-architecture changes here.

    Priority order:
      1. model._lp (populated at load time for heterogeneous-dim models)
      2. hf_config._layer_attention_params (absent for safetensors checkpoints)
      3. Raw hf_config scalar fields via the vLLM model-arch convertor
    """
    kv_dtype = torch.float16

    lp_list = getattr(model, "_lp", None)
    if lp_list is None:
        lp_list = getattr(hf_config, "_layer_attention_params", None)
    if lp_list:
        _allocate_kv_pool_per_layer(
            wgpu_device, model,
            num_blocks=num_blocks,
            block_size=block_size,
            layer_params=lp_list,
            dtype=kv_dtype,
        )
        return

    # Standalone scripts (run_inference.py, profile_kernels.py) call this without
    # a vLLM ModelConfig. Use the convertor for canonical KV head/dim values that
    # handle non-standard attribute names across architectures.
    _conv = _make_convertor(hf_config)
    num_kv_heads = _conv.get_total_num_kv_heads()
    head_dim = _conv.get_head_size()
    # For multimodal configs, hf_config.num_hidden_layers is the outer
    # wrapper's count, which may differ from the text model. Use the
    # convertor (which reads from _hf_text) to get the correct value.
    _num_layers = _conv.get_num_hidden_layers()
    # model._layer_types wins; fall back to hf_config fields used by different
    # architectures (Gemma4 uses "layer_types", Falcon uses "layers_block_type").
    layer_types = get_layer_types(model, hf_config)
    # Treat uniform full-attention lists the same as None (avoids tiny buffers).
    if layer_types and KV_ATTN_TYPES.issuperset(layer_types):
        layer_types = None

    _allocate_kv_pool_hybrid(
        wgpu_device,
        model,
        num_blocks=num_blocks,
        num_layers=_num_layers,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        layer_types=layer_types,
        dtype=kv_dtype,
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


# Backward-compatible alias; get_kv_dims_from_hf_config is the canonical name.
# get_kv_dims_from_config is deprecated.
get_kv_dims_from_config = get_kv_dims_from_hf_config


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
        "WebGPU memory: total=%dMB model=%dMB available=%dMB",
        total // MiB_bytes, model_mem // MiB_bytes, available // MiB_bytes,
    )
    return available

