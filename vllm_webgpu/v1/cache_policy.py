from __future__ import annotations
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.utils.mem_utils import get_cpu_memory
from vllm_webgpu.utils import OVERHEAD_BYTES

if TYPE_CHECKING:
    from vllm_webgpu.v1.worker import WebGPUWorker

logger = init_logger(__name__)

# Layer type strings that carry KV state and require cache allocation.
# Must stay in sync with get_kv_cache_spec in model_runner.py, which imports
# this constant and uses it as the authoritative set.
KV_ATTN_TYPES: frozenset[str] = frozenset(
    {"attention", "full_attention", "sliding_attention"}
)


def _alloc_rw_buffer(dev, size: int):
    """Allocate a STORAGE|COPY_SRC|COPY_DST WebGPU buffer of `size` bytes."""
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    return WebGPUBuffer.empty(dev, size)


def allocate_kv_pool_hybrid(
    dev,
    model,
    num_blocks: int,
    num_layers: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    layer_types: list | None = None,
) -> None:
    """Allocate KV pool for all layers.

    When layer_types is None (or all entries are "full_attention"), every layer
    gets a full KV cache buffer.  When layer_types is provided, non-full-attention
    layers get 16-byte placeholder buffers (never accessed during inference).
    """
    bytes_per_layer = num_blocks * block_size * num_kv_heads * head_dim * 2

    model.kv_pool.clear()

    full_attn_count = 0
    for i in range(num_layers):
        is_full = layer_types is None or layer_types[i] in KV_ATTN_TYPES
        if is_full:
            k_buf = _alloc_rw_buffer(dev, bytes_per_layer)
            v_buf = _alloc_rw_buffer(dev, bytes_per_layer)
            full_attn_count += 1
        else:
            k_buf = _alloc_rw_buffer(dev, 16)
            v_buf = _alloc_rw_buffer(dev, 16)
        model.kv_pool.append((k_buf, v_buf))

    if layer_types is None:
        total_mb = (bytes_per_layer * num_layers * 2) // 2**20
        logger.info(
            "KV cache: %d blocks × %d layers × %d KV heads × %d head_dim = %dMB",
            num_blocks, num_layers, num_kv_heads, head_dim, total_mb,
        )
    else:
        total_mb = (bytes_per_layer * full_attn_count * 2) // 2**20
        logger.info(
            "KV cache (hybrid): %d full-attn × %d blocks × %d KV heads × %d head_dim = %dMB",
            full_attn_count, num_blocks, num_kv_heads, head_dim, total_mb,
        )


def allocate_kv_pool_per_layer(
    dev,
    model,
    num_blocks: int,
    block_size: int,
    layer_params: list,
) -> None:
    """Allocate KV pool for models with per-layer KV dims (e.g. Nemotron-H).

    layer_params is a list of dicts (one per layer) each with keys:
      num_kv_heads, head_dim
    """
    if model is None:
        raise RuntimeError("model must not be None during KV cache allocation")
    model.kv_pool.clear()
    logger.info("KV cache (per-layer): %d layers, mixed dims", len(layer_params))
    for lp in layer_params:
        kv_bytes = num_blocks * block_size * lp["num_kv_heads"] * lp["head_dim"] * 2  # f16
        model.kv_pool.append((
            _alloc_rw_buffer(dev, kv_bytes),
            _alloc_rw_buffer(dev, kv_bytes),
        ))


def _get_model_memory_usage(worker: "WebGPUWorker") -> int:
    """Sum of all weight buffer sizes in bytes."""
    model = getattr(getattr(worker, "model_runner", None), "model", None)
    if model is None:
        return 0
    return sum(buf.nbytes for buf in model.weights.values() if hasattr(buf, "nbytes"))


def determine_available_memory(worker: "WebGPUWorker") -> int:
    """
    Available memory for KV cache = device total - model weights - overhead.

    Uses vLLM's get_cpu_memory to query real system memory, which is correct for
    unified-memory platforms (Apple Silicon) and avoids confusing wgpu's per-buffer
    maxBufferSize limit with total device memory. On a 7B f16 model (~14 GB weights)
    with a 4 GB maxBufferSize, subtracting from maxBufferSize yields negative available
    memory and clamps to 0 KV blocks.

    Falls back to a model-ratio heuristic when get_cpu_memory is unavailable.
    """
    config = worker.webgpu_config
    model_mem = _get_model_memory_usage(worker)

    total: int = get_cpu_memory()

    if config.is_auto_memory:
        available = max(total - model_mem - OVERHEAD_BYTES, 0)
        logger.info(
            "WebGPU memory: total=%dMB model=%dMB available=%dMB",
            total // 2**20, model_mem // 2**20, available // 2**20,
        )
        return available

    # Explicit fraction: user asked for `memory_fraction` of device total for KV.
    available = int(total * config.memory_fraction) - model_mem - OVERHEAD_BYTES
    return max(available, 0)

