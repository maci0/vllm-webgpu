"""KV cache allocation helpers for standalone scripts (run_inference.py, profile_kernels.py).

These functions are NOT part of the vLLM engine integration path. The engine
calls allocate_kv_from_tensors (in vllm_webgpu.v1.cache_policy) directly from
KVCacheTensor objects produced by vLLM's KV cache planner. The functions here
derive KV cache parameters from a raw HuggingFace config, which is only
appropriate for scripts that do not have a VllmConfig or KVCacheConfig
available.
"""
from __future__ import annotations

import torch
from vllm.logger import init_logger
from vllm.utils.mem_constants import MiB_bytes
from vllm.utils.torch_utils import get_dtype_size

from vllm_webgpu.v1.cache_policy import (
    KV_ATTN_TYPES,
    MIN_WEBGPU_BUFFER_BYTES,
    get_layer_types,
)
from vllm_webgpu.webgpu.buffer import WebGPUBuffer

logger = init_logger(__name__)


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
    # vLLM internal: MODEL_ARCH_CONFIG_CONVERTORS is a registry dict mapping
    # model_type strings to arch-specific KV head/size convertor classes.
    # Revisit when vLLM exposes a public factory for standalone (no VllmConfig)
    # KV-head queries — at that point replace this block with that call.
    from vllm.transformers_utils.model_arch_config_convertor import (
        MODEL_ARCH_CONFIG_CONVERTORS,
        ModelArchConfigConvertorBase,
    )
    from vllm.transformers_utils.config import get_hf_text_config
    hf_text = get_hf_text_config(hf_cfg)
    return MODEL_ARCH_CONFIG_CONVERTORS.get(
        getattr(hf_cfg, "model_type", ""), ModelArchConfigConvertorBase
    )(hf_cfg, hf_text)



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

    bytes_per_layer is the per-buffer (K or V) allocation:
      num_blocks * block_size * num_kv_heads * head_dim * dtype_bytes
    K and V are allocated separately at this size each.
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
            k_buf = WebGPUBuffer.empty(dev, MIN_WEBGPU_BUFFER_BYTES)
            v_buf = WebGPUBuffer.empty(dev, MIN_WEBGPU_BUFFER_BYTES)
        model.kv_pool.append((k_buf, v_buf))

    if layer_types is None:
        total_mb = (bytes_per_layer * num_layers * 2) // MiB_bytes
        logger.info(
            "KV cache: %d blocks × %d tokens/block × %d layers × %d KV heads × %d head_dim (%s, K+V) = %dMiB",
            num_blocks, block_size, num_layers, num_kv_heads, head_dim, dtype, total_mb,
        )
    else:
        total_mb = (bytes_per_layer * kv_layer_count * 2) // MiB_bytes
        logger.info(
            "KV cache (hybrid): %d kv-attn × %d blocks × %d tokens/block × %d KV heads × %d head_dim (%s, K+V) = %dMiB",
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
            # placeholder that _allocate_kv_pool_hybrid uses.
            model.kv_pool.append((
                WebGPUBuffer.empty(dev, MIN_WEBGPU_BUFFER_BYTES),
                WebGPUBuffer.empty(dev, MIN_WEBGPU_BUFFER_BYTES),
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


def allocate_kv_from_hf_config(
    wgpu_device,
    model,
    hf_config,
    num_blocks: int,
    block_size: int,
) -> None:
    """Allocate KV cache from a HuggingFace config object.

    Used only by standalone scripts (run_inference.py, profile_kernels.py).
    The vLLM engine path uses allocate_kv_from_tensors (in cache_policy), which
    derives buffer sizes directly from vLLM KVCacheTensor objects and requires no
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
