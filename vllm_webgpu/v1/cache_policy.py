from __future__ import annotations
from typing import TYPE_CHECKING

from vllm.config.utils import getattr_iter
from vllm.logger import init_logger
from vllm.utils.mem_utils import get_cpu_memory
from vllm_webgpu.utils import OVERHEAD_BYTES

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
) -> None:
    """Allocate KV pool for all layers.

    When layer_types is None, or all entries are in KV_ATTN_TYPES, every layer
    gets a full KV cache buffer. When layer_types is provided, layers whose type
    is not in KV_ATTN_TYPES get 16-byte placeholder buffers.
    """
    if model is None:
        raise RuntimeError("model must not be None during KV cache allocation")
    bytes_per_layer = num_blocks * block_size * num_kv_heads * head_dim * 2

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
        total_mb = (bytes_per_layer * num_layers * 2) // 2**20
        logger.info(
            "KV cache: %d blocks × %d tokens/block × %d layers × %d KV heads × %d head_dim (f16, K+V) = %dMB",
            num_blocks, block_size, num_layers, num_kv_heads, head_dim, total_mb,
        )
    else:
        total_mb = (bytes_per_layer * kv_layer_count * 2) // 2**20
        logger.info(
            "KV cache (hybrid): %d kv-attn × %d blocks × %d tokens/block × %d KV heads × %d head_dim (f16, K+V) = %dMB",
            kv_layer_count, num_blocks, block_size, num_kv_heads, head_dim, total_mb,
        )


def _allocate_kv_pool_per_layer(
    dev,
    model,
    num_blocks: int,
    block_size: int,
    layer_params: list,
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
        kv_bytes = num_blocks * block_size * lp["num_kv_heads"] * lp["head_dim"] * 2  # f16
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
    model_config=None,
) -> None:
    """Allocate KV cache from a HuggingFace config object.

    Single source of truth used by WebGPUModelRunner.initialize_kv_cache,
    run_inference.run, and profile_kernels. Add new model support here only.

    Priority order:
      1. model._lp (populated at load time for heterogeneous-dim models)
      2. hf_config._layer_attention_params (absent for safetensors checkpoints)
      3. Uniform allocation via model_config canonical accessors (when provided)
         or raw hf_config scalar fields (standalone scripts without vLLM engine)

    Pass model_config (a vLLM ModelConfig) whenever the vLLM engine is running.
    Its get_head_size() and get_total_num_kv_heads() handle non-standard attribute
    names across architectures (PLaMo2.1, Falcon, DeepSeek-MLA, etc.), keeping
    this path consistent with get_kv_cache_spec().
    """
    lp_list = getattr(model, "_lp", None)
    if lp_list is None:
        lp_list = getattr(hf_config, "_layer_attention_params", None)
    if lp_list:
        _allocate_kv_pool_per_layer(
            wgpu_device, model,
            num_blocks=num_blocks,
            block_size=block_size,
            layer_params=lp_list,
        )
        return

    if model_config is not None:
        num_kv_heads = model_config.get_total_num_kv_heads()
        head_dim = model_config.get_head_size()
    else:
        # Fallback for standalone scripts (run_inference.py, profile_kernels.py)
        # that call allocate_kv_from_hf_config without a vLLM ModelConfig.
        # hf_config.num_key_value_heads may diverge from what ModelConfig
        # reports for architectures with TP-remapped or MLA-style heads.
        # Pass model_config when possible to get the canonical values.
        num_kv_heads = get_num_kv_heads(hf_config)
        head_dim = getattr_iter(hf_config, ('head_dim', 'hidden_size_per_head'), default_factory=lambda: hf_config.hidden_size // hf_config.num_attention_heads)
    # model._layer_types wins; fall back to hf_config fields used by different
    # architectures (Gemma4 uses "layer_types", Falcon uses "layers_block_type").
    layer_types = (
        getattr(model, "_layer_types", None)
        or getattr(hf_config, "layer_types", None)
        or getattr(hf_config, "layers_block_type", None)
    )
    # Treat uniform full-attention lists the same as None (avoids tiny buffers).
    if layer_types and KV_ATTN_TYPES.issuperset(layer_types):
        layer_types = None

    _allocate_kv_pool_hybrid(
        wgpu_device,
        model,
        num_blocks=num_blocks,
        num_layers=hf_config.num_hidden_layers,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        layer_types=layer_types,
    )


def get_num_kv_heads(hf_cfg) -> int:
    """Read num_kv_heads from an hf_config, handling architecture-specific attribute names.

    Attribute order mirrors DefaultArchConfig.get_total_num_kv_heads() in
    vllm.transformers_utils.model_arch_config_convertor, which is the
    authoritative source. Align with that list when bumping vLLM to catch
    newly added architecture aliases early.
    """
    return getattr_iter(
        hf_cfg,
        ('n_head_kv', 'num_kv_heads', 'num_key_value_heads',
         'multi_query_group_num', 'num_attention_groups'),
        default_factory=lambda: hf_cfg.num_attention_heads,
    )


def _get_model_memory_usage(worker: "WebGPUWorker") -> int:
    """Sum of all weight buffer sizes in bytes."""
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
    model_mem = _get_model_memory_usage(worker)

    total: int = get_cpu_memory()

    base = total - model_mem - OVERHEAD_BYTES
    fraction = 1.0 if config.is_auto_memory else config.memory_fraction
    available = max(int(base * fraction), 0)
    logger.info(
        "WebGPU memory: total=%dMB model=%dMB available=%dMB",
        total // 2**20, model_mem // 2**20, available // 2**20,
    )
    return available

