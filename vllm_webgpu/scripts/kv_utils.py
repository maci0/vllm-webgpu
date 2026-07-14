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
    MIN_WEBGPU_BUFFER_BYTES,
    get_layer_types,
    is_attn_layer,
)
from vllm_webgpu.webgpu.buffer import WebGPUBuffer

logger = init_logger(__name__)

# Import-time guard: verify that MODEL_ARCH_CONFIG_CONVERTORS and
# ModelArchConfigConvertorBase still exist at the expected module path and
# that ModelArchConfigConvertorBase still exposes the three methods called in
# _make_convertor's callers (get_total_num_kv_heads, get_head_size,
# get_num_hidden_layers). Any vLLM patch that renames, moves, or splits this
# internal registry will fail here rather than silently returning wrong KV
# dimensions at runtime.
#
# hasattr checks work on compiled/stripped wheels where inspect.getsource
# raises OSError. The except clause logs a warning instead of silently passing
# so that a missing module is visible in logs even when it is not fatal.
try:
    from vllm.transformers_utils.model_arch_config_convertor import (
        MODEL_ARCH_CONFIG_CONVERTORS as _CONVERTORS,
        ModelArchConfigConvertorBase as _ConvertorBase,
    )
    assert isinstance(_CONVERTORS, dict), (
        "MODEL_ARCH_CONFIG_CONVERTORS is no longer a dict at "
        "vllm.transformers_utils.model_arch_config_convertor. "
        "_make_convertor uses .get() on this registry to resolve "
        "per-architecture KV head/size classes. Review _make_convertor "
        "and update to match the new upstream structure."
    )
    # Anchor 1: get_total_num_kv_heads (vLLM 0.9+, base class).
    assert hasattr(_ConvertorBase, "get_total_num_kv_heads"), (
        "ModelArchConfigConvertorBase.get_total_num_kv_heads no longer "
        "exists at vllm.transformers_utils.model_arch_config_convertor. "
        "_make_convertor callers call .get_total_num_kv_heads() on the "
        "returned convertor instance (allocate_kv_from_hf_config L189). "
        "Find the replacement API and update the call site."
    )
    # Anchor 2: get_head_size (vLLM 0.9+, base class).
    assert hasattr(_ConvertorBase, "get_head_size"), (
        "ModelArchConfigConvertorBase.get_head_size no longer exists at "
        "vllm.transformers_utils.model_arch_config_convertor. "
        "_make_convertor callers call .get_head_size() on the returned "
        "convertor instance (allocate_kv_from_hf_config L190). "
        "Find the replacement API and update the call site."
    )
    # Anchor 3: get_num_hidden_layers (vLLM 0.9+, base class).
    assert hasattr(_ConvertorBase, "get_num_hidden_layers"), (
        "ModelArchConfigConvertorBase.get_num_hidden_layers no longer "
        "exists at vllm.transformers_utils.model_arch_config_convertor. "
        "_make_convertor callers call .get_num_hidden_layers() on the "
        "returned convertor instance (allocate_kv_from_hf_config L194). "
        "Find the replacement API and update the call site."
    )
    del _CONVERTORS, _ConvertorBase
except ImportError as _e:
    logger.warning(
        "kv_utils: could not import MODEL_ARCH_CONFIG_CONVERTORS from "
        "vllm.transformers_utils.model_arch_config_convertor (%s). "
        "API-compatibility guards are disabled; a vLLM version mismatch "
        "may not be caught until runtime.",
        _e,
    )
    del _e


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
    head_dim_v: int | None = None,
) -> None:
    """Allocate KV pool for all layers.

    When layer_types is None, or all entries are attention-bearing, every layer
    gets a full KV cache buffer. When layer_types is provided, non-attention
    layers get 16-byte placeholder buffers; is_attn_layer() from cache_policy.py
    is the authoritative check for both string and integer layer-type encodings.

    K and V buffers are allocated separately. When head_dim_v is provided, the V
    buffer uses that head dim instead of head_dim (for architectures with asymmetric
    K/V head sizes). This mirrors the formula used by allocate_kv_from_tensors.
    """
    if model is None:
        raise RuntimeError("model must not be None during KV cache allocation")
    if dtype != torch.float16:
        raise ValueError(
            f"_allocate_kv_pool_hybrid only supports float16; got {dtype}. "
            "KV shaders only support float16; other dtypes produce incorrect results at runtime."
        )
    k_bytes_per_layer = num_blocks * block_size * num_kv_heads * head_dim * get_dtype_size(dtype)
    v_bytes_per_layer = num_blocks * block_size * num_kv_heads * (head_dim_v or head_dim) * get_dtype_size(dtype)

    if layer_types is not None and len(layer_types) != num_layers:
        raise ValueError(
            f"layer_types length {len(layer_types)} != num_layers {num_layers}"
        )

    model.kv_pool.clear()

    kv_layer_count = 0
    for i in range(num_layers):
        needs_kv_cache = layer_types is None or is_attn_layer(layer_types[i])
        if needs_kv_cache:
            k_buf = WebGPUBuffer.empty(dev, k_bytes_per_layer)
            v_buf = WebGPUBuffer.empty(dev, v_bytes_per_layer)
            kv_layer_count += 1
        else:
            k_buf = WebGPUBuffer.empty(dev, MIN_WEBGPU_BUFFER_BYTES)
            v_buf = WebGPUBuffer.empty(dev, MIN_WEBGPU_BUFFER_BYTES)
        model.kv_pool.append((k_buf, v_buf))

    if layer_types is None:
        total_mb = ((k_bytes_per_layer + v_bytes_per_layer) * num_layers) // MiB_bytes
        logger.info(
            "KV cache: %d blocks × %d tokens/block × %d layers × %d KV heads × %d head_dim (%s, K+V) = %dMiB",
            num_blocks, block_size, num_layers, num_kv_heads, head_dim, dtype, total_mb,
        )
    else:
        total_mb = ((k_bytes_per_layer + v_bytes_per_layer) * kv_layer_count) // MiB_bytes
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
    if dtype != torch.float16:
        raise ValueError(
            f"_allocate_kv_pool_per_layer only supports float16; got {dtype}."
        )
    model.kv_pool.clear()
    logger.info("KV cache (per-layer): %d layers, mixed dims (%s)", len(layer_params), dtype)
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
        k_bytes = num_blocks * block_size * lp["num_kv_heads"] * lp["head_dim"] * get_dtype_size(dtype)
        v_bytes = num_blocks * block_size * lp["num_kv_heads"] * lp.get("head_dim_v", lp["head_dim"]) * get_dtype_size(dtype)
        model.kv_pool.append((
            WebGPUBuffer.empty(dev, k_bytes),
            WebGPUBuffer.empty(dev, v_bytes),
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
    # model._layer_types wins when set (scripts path, model is fully loaded);
    # fall back to hf_config fields used by different architectures.
    # Use is not None rather than truthiness: an explicit empty list [] is a valid
    # signal that the model has no hybrid layers and should not fall through to the
    # hf_config probe (which might incorrectly return non-None for some architectures).
    layer_types = getattr(model, "_layer_types", None)
    if layer_types is None:
        from vllm.transformers_utils.config import get_hf_text_config as _get_hf_text_config
        layer_types = get_layer_types(_get_hf_text_config(hf_config), hf_outer_config=hf_config)
    # Treat uniform full-attention lists the same as None (avoids tiny buffers).
    if layer_types and all(is_attn_layer(lt) for lt in layer_types):
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
