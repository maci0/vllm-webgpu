from __future__ import annotations
import os
from typing import TYPE_CHECKING, NamedTuple

from vllm.utils.cpu_resource_utils import get_memory_node_info, get_visible_memory_node

from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.mem_constants import MiB_bytes
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (ChunkedLocalAttentionSpec,
                                         FullAttentionSpec,
                                         KVQuantMode,
                                         MLAAttentionSpec,
                                         SinkFullAttentionSpec,
                                         SlidingWindowMLASpec,
                                         SlidingWindowSpec,
                                         TQFullAttentionSpec,
                                         UniformTypeKVCacheSpecs)

# Minimum overhead budget: driver + runtime allocations for small models.
# For large models activations scale with parameter count; see determine_available_memory.
_OVERHEAD_BYTES = 512 * MiB_bytes
# Fraction of model weight bytes reserved as activation/overhead budget.
# A 7B f16 model (~14 GiB) with a 2048-token batch generates 2-4 GiB of
# intermediate activations; 15% of 14 GiB ~ 2.1 GiB covers that range.
# This kicks in only when it exceeds the _OVERHEAD_BYTES floor.
_ACTIVATION_OVERHEAD_FRACTION = 0.15
MIN_WEBGPU_BUFFER_BYTES: int = 16  # WebGPU spec forbids zero-size buffers

if TYPE_CHECKING:
    import wgpu
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
    from vllm_webgpu.models.base import BaseWebGPUModel
    from vllm_webgpu.v1.worker import WebGPUWorker

from vllm_webgpu.webgpu.buffer import WebGPUBuffer

logger = init_logger(__name__)

_ATTN_LAYER_TYPES: frozenset = frozenset({"attention", "full_attention", "sliding_attention", "hybrid", 1})


class _LayerKV(NamedTuple):
    k_bytes: int
    v_bytes: int
    layer_name: str


def is_attn_layer(lt: "str | int") -> bool:
    """Return True when a layer-type value represents an attention layer.

    Handles both string layer types and the Minimax integer encoding where
    1 means attention and 0 means non-attention (Mamba/MLP). The "hybrid"
    value is used by Zamba2-style models (see vLLM model.py:1331-1337 where
    get_num_layers_by_block_type counts "hybrid" as an attention layer only
    when text_model_type == 'zamba2').
    Use this instead of bare string-set membership checks everywhere so that
    the integer sentinel never needs to be repeated at individual call sites.

    NOTE: 'hybrid' is treated as an attention layer unconditionally here. This
    is correct for Zamba2-family configs (the only current user of 'hybrid').
    If a future model uses 'hybrid' to mean a non-attention variant (e.g. a
    linear-attention or SSM layer), this function must be updated to accept a
    model_type argument and return True only when model_type == 'zamba2'. Until
    then, adding such a model to this backend will over-allocate KV buffers for
    its hybrid layers. A CI test should verify the 'hybrid' invariant when new
    hybrid-layer models are registered.
    """
    # "linear_attention" (Qwen3.5 / CPU platform) is intentionally absent: it
    # carries no KV cache state and must not be treated as an attention layer.
    # "sliding_attention" is an extension beyond what vLLM's get_num_layers_by_block_type
    # counts: that API groups sliding-window layers separately from "attention"
    # (see vllm/config/model.py:1348-1357). Including it here is correct because
    # sliding-window attention layers do require KV cache buffers; the count API
    # just exposes them under a different label.
    return lt in _ATTN_LAYER_TYPES


def allocate_kv_from_tensors(
    wgpu_device: "wgpu.GPUDevice",
    model: "BaseWebGPUModel",
    kv_cache_config: "KVCacheConfig",
    num_total_layers: int,
) -> None:
    """Allocate KV cache buffers from vLLM's authoritative KVCacheTensor list.

    K and V buffer sizes are computed directly from spec fields (num_blocks,
    block_size, num_kv_heads, head_size / head_size_v, dtype) rather than
    splitting real_page_size_bytes (which introduces float division for asymmetric
    heads) or halving tensor.size (which includes per-token-head scale overhead).
    Non-KV layers receive 16-byte placeholder buffers.
    """
    if model is None:
        raise RuntimeError("model must not be None during KV cache allocation")

    kv_cache_tensors = kv_cache_config.kv_cache_tensors
    num_blocks = kv_cache_config.num_blocks
    kv_cache_groups = kv_cache_config.kv_cache_groups

    # Build layer_name -> KVCacheSpec map so each layer's head_size and head_size_v
    # fields are accessible for independent K/V byte calculation, avoiding the combined
    # page_size_bytes which includes per-token-head scale overhead and cannot be split
    # correctly for asymmetric head dimensions.
    # UniformTypeKVCacheSpecs wraps per-layer specs (e.g. heterogeneous-but-same-type
    # attention layers like Gemma4's 4-head local vs 8-head global); unpack it so
    # each layer name resolves to its individual spec rather than the umbrella object.
    layer_spec_map: dict[str, KVCacheSpec] = {}
    for group in kv_cache_groups:
        gs = group.kv_cache_spec
        if isinstance(gs, UniformTypeKVCacheSpecs):
            if gs.kv_cache_specs.keys() != set(group.layer_names):
                raise RuntimeError(
                    f"UniformTypeKVCacheSpecs keys {set(gs.kv_cache_specs.keys())} "
                    f"do not match group.layer_names {set(group.layer_names)}"
                )
            layer_spec_map.update(gs.kv_cache_specs)
        else:
            layer_spec_map.update({name: gs for name in group.layer_names})

    # Build layer_index -> (k_bytes, v_bytes) from the tensors vLLM already computed.
    # Keyed by layer index (int) so each entry can be written directly into model.kv_pool.
    # Two distinct layer names that resolve to the same integer would silently overwrite
    # each other; the explicit collision check at the end of the loop catches that.
    # The name-to-index resolution uses extract_layer_index once per tensor entry.
    # shared_by holds names like "model.layers.{i}.self_attn" or "model.layers.{i}.mixer".
    layer_idx_kv: dict[int, _LayerKV] = {}
    for tensor in kv_cache_tensors:
        if tensor.block_stride > 0:
            # block_stride > 0 means K and V data for multiple layers share one
            # contiguous buffer, with each layer's slice separated by block_stride
            # bytes (vLLM's packed/interleaved layout for small or shared pages).
            # Supporting this would require reading tensor.size and tensor.offset
            # to compute per-layer byte ranges instead of deriving sizes from
            # spec fields (num_kv_heads, head_size, block_size, dtype_bytes).
            raise NotImplementedError(
                f"Packed KV cache layout (block_stride={tensor.block_stride}) is not supported by the WebGPU backend. "
                "To add support: use tensor.size and tensor.offset to compute per-layer byte ranges "
                "rather than deriving buffer sizes from spec fields."
            )
        # vLLM 0.24 never produces offset != 0 without block_stride > 0 (the only
        # non-zero-offset construction site, _get_kv_cache_config_packed, always sets
        # block_stride = total_num_bytes_per_block > 0, which is caught above).
        # This guard fires only if a future vLLM version introduces a non-zero offset
        # without a packed stride, breaking that invariant.
        if tensor.offset != 0:
            raise NotImplementedError(
                f"KVCacheTensor with non-zero offset ({tensor.offset}) is not supported by the WebGPU backend"
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
            k_bytes = v_bytes = 0
            if isinstance(spec, MLAAttentionSpec):
                raise NotImplementedError(
                    f"MLA KV cache ({type(spec).__name__}) is not supported by the WebGPU backend. "
                    "MLAAttentionSpec uses a compressed latent layout that differs from the standard "
                    "per-head K/V formula and cannot be sized with storage_block_size * head_size * dtype_bytes."
                )
            elif isinstance(spec, TQFullAttentionSpec):
                raise NotImplementedError(
                    "TQFullAttentionSpec KV cache is not supported by the WebGPU backend. "
                    "TQFullAttentionSpec overrides real_page_size_bytes with a tq_slot_size-based formula "
                    "that differs from the standard block_size * num_kv_heads * (head_size + head_size_v) * dtype_bytes. "
                    "Allocating with head_size/head_size_v would produce wrong buffer sizes."
                )
            elif isinstance(spec, SinkFullAttentionSpec):
                raise NotImplementedError(
                    "SinkFullAttentionSpec KV cache is not supported by the WebGPU backend. "
                    "SinkFullAttentionSpec is used by StaticSinkAttention models that require "
                    "sink-token pinning during attention computation. Buffer sizes would be "
                    "allocated correctly, but the WebGPU attention kernel does not implement "
                    "sink-token logic, producing silently wrong output."
                )
            # SlidingWindowMLASpec and SlidingWindowSpec are NOT subclasses of
            # FullAttentionSpec, so they do not interact with the check below.
            # They are placed here (before FullAttentionSpec) to keep all
            # non-FullAttentionSpec rejection paths grouped together.
            elif isinstance(spec, SlidingWindowMLASpec):
                raise NotImplementedError(
                    "SlidingWindowMLASpec KV cache is not supported by the WebGPU backend. "
                    "SlidingWindowMLASpec stores a single MLA latent per position, so "
                    "real_page_size_bytes is the full per-position size, not a K+V pair. "
                    "Halving it would silently corrupt both cache buffers."
                )
            elif isinstance(spec, SlidingWindowSpec):
                raise NotImplementedError(
                    "SlidingWindowSpec KV cache is not supported by the WebGPU backend."
                )
            elif isinstance(spec, ChunkedLocalAttentionSpec):
                raise NotImplementedError(
                    "ChunkedLocalAttentionSpec KV cache is not supported by the WebGPU backend. "
                    "ChunkedLocalAttentionSpec is a direct AttentionSpec subclass (not FullAttentionSpec) "
                    "used by hybrid KV cache managers for chunked local attention layers. "
                    "The WebGPU flash_attn_decode kernel does not implement chunked local attention masking."
                )
            elif isinstance(spec, FullAttentionSpec):
                if spec.sliding_window is not None:
                    raise NotImplementedError(
                        f"FullAttentionSpec with sliding_window={spec.sliding_window!r} is not supported by the WebGPU backend. "
                        "Buffer sizes would be correct but flash_attn_decode does not implement "
                        "sliding window masking, producing silently wrong output."
                    )
                if spec.attention_chunk_size is not None:
                    raise NotImplementedError(
                        f"FullAttentionSpec with attention_chunk_size={spec.attention_chunk_size!r} is not supported by the WebGPU backend. "
                        "Chunked local attention layers are converted to FullAttentionSpec when the hybrid KV cache manager is disabled; "
                        "flash_attn_decode does not implement chunked local attention masking."
                    )
                if spec.kv_quant_mode != KVQuantMode.NONE:
                    raise NotImplementedError(
                        f"Quantized KV cache (kv_quant_mode={spec.kv_quant_mode!r}) is not supported by the WebGPU backend; KV shaders expect float16 data."
                    )
                if spec.non_causal:
                    raise NotImplementedError(
                        "FullAttentionSpec with non_causal=True is not supported by the WebGPU backend; "
                        "flash_attn_decode implements causal masking only."
                    )
                if type(spec) is not FullAttentionSpec:
                    raise NotImplementedError(
                        f"FullAttentionSpec subclass {type(spec).__name__} overrides real_page_size_bytes; "
                        "the head_size ratio split formula may be wrong. Add an explicit branch to handle it."
                    )
                # Compute K and V buffer sizes directly from per-dimension fields.
                # Using the direct formula avoids float division (head_size /
                # (head_size + head_size_v) is non-integer for odd head sizes) and
                # makes the separate K and V buffer sizes explicit. Ratio-splitting
                # from real_page_size_bytes is mathematically equivalent for plain
                # (non-quantized, non-NVFP4) FullAttentionSpec but adds unnecessary
                # indirection. The real danger is tensor.size // 2 (which inflates
                # by per-token-head scale overhead), not real_page_size_bytes.
                dtype_size = get_dtype_size(spec.dtype)
                k_bytes = num_blocks * spec.block_size * spec.num_kv_heads * spec.head_size * dtype_size
                v_bytes = num_blocks * spec.block_size * spec.num_kv_heads * spec.head_size_v * dtype_size
            else:
                raise NotImplementedError(
                    f"Unsupported KV cache spec type {type(spec).__name__} for {layer_name!r}; "
                    "add an explicit branch to handle it."
                )
            try:
                # extract_layer_index is called with the default num_attn_module=1.
                # This is intentional: the WebGPU backend does not support models
                # whose layer names contain two numeric segments (multi-attn-module
                # naming). All currently supported architectures use single-integer
                # layer names (e.g. "model.layers.3.self_attn"). If a future model
                # uses two integers in its layer names, extract_layer_index would
                # raise AssertionError here and the error block below would surface it.
                _idx = extract_layer_index(layer_name)
            except (AssertionError, ValueError, IndexError) as exc:
                # extract_layer_index uses bare assert statements; IndexError fires
                # when -O disables asserts and int_vals ends up empty (bare [0] access
                # on an empty list). ValueError caught in case vLLM converts asserts.
                # AssertionError fires when num_attn_module=1 but the name has two
                # integers (multi-attn-module model) — unsupported by this backend.
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
            # Two distinct layer names that map to the same integer (e.g.
            # "model.layers.3.self_attn" and "model.layers.3.cross_attn" in a future
            # multi-attention-per-layer model) would silently overwrite each other's
            # buffer sizes, producing wrong K/V allocations with no error at runtime.
            if _idx in layer_idx_kv:
                raise RuntimeError(
                    f"Two layer names resolve to the same index {_idx}: "
                    f"{layer_name!r} and {layer_idx_kv[_idx].layer_name!r}. This is a model configuration bug."
                )
            layer_idx_kv[_idx] = _LayerKV(k_bytes, v_bytes, layer_name)

    # Sliding-attention layers in supported models always receive FullAttentionSpec(sliding_window=None)
    # from get_kv_cache_spec; the SlidingWindowSpec/FullAttentionSpec(sliding_window!=None) rejections
    # above fire before this point for unsupported variants.

    # Verify that every attention layer in hybrid models got a real KV buffer.
    # Models with _layer_types (e.g. NemotronH) index kv_pool unconditionally in
    # _attn_layer; a 16-byte placeholder there silently corrupts kv_cache_store_both
    # and flash_attn_decode without any GPU-side error.
    # Pure-attention models lack _layer_types, so the check is safely skipped.
    _model_layer_types = getattr(model, "_layer_types", None)
    if _model_layer_types is not None and len(_model_layer_types) == num_total_layers:
        _missing_attn = [
            i for i, lt in enumerate(_model_layer_types)
            if is_attn_layer(lt) and i not in layer_idx_kv
        ]
        if _missing_attn:
            raise RuntimeError(
                f"Attention layer(s) {_missing_attn} were not assigned real KV "
                f"buffers (kv_cache_tensors covered layers "
                f"{sorted(layer_idx_kv.keys())}). These layers would receive "
                f"16-byte placeholder buffers, silently corrupting "
                f"kv_cache_store_both and flash_attn_decode dispatches. "
                f"Verify that kv_cache_config.kv_cache_tensors includes entries "
                f"for all attention layers declared in the model's layer type list."
            )

    model.kv_pool.clear()
    total_bytes = 0
    for i in range(num_total_layers):
        if i in layer_idx_kv:
            entry = layer_idx_kv[i]
            k_bytes, v_bytes = entry.k_bytes, entry.v_bytes
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
        num_blocks, len(layer_idx_kv), total_bytes // MiB_bytes,
    )


def get_layer_types(hf_text_config, hf_outer_config=None) -> list | None:
    """Return the layer-type list from hf_text_config, using a canonical fallback chain.

    Priority: hf_text_config.layers_block_type (NemotronH/Falcon) >
    hf_outer_config.attn_type_list (Minimax) >
    hf_text_config.layer_types (Gemma4/Qwen3.5).

    Returns None when none of the attributes is present.

    Uses explicit `is not None` guards (not `or`) so that an empty list, which
    is a valid value distinct from "attribute absent", is not silently skipped.

    hf_outer_config: optional outer ModelConfig.hf_config, used only for the
    attn_type_list probe. For multimodal models (e.g. Minimax) where
    hf_text_config != hf_config, attn_type_list lives on the outer config.
    Defaults to hf_text_config when not provided (single-config models).

    Callers with a fully-loaded model object should check model._layer_types before
    calling this function and use that value if present. This function only inspects
    hf_text_config; the vLLM engine always calls it before weight loading (model=None).
    """
    # Minimax-style: integer list where 1 = attention, 0 = non-attention.
    # model_runner.py handles integer-encoded layer types via is_attn_layer(lt), which returns True when lt == 1 (Minimax attention).
    # attn_type_list lives on the outer hf_config for multimodal models where
    # hf_text_config differs from the outer config.
    _outer = hf_outer_config if hf_outer_config is not None else hf_text_config
    # Priority order mirrors vLLM's ModelConfig.get_num_layers_by_block_type
    # (vllm/config/model.py:1327-1369 as of the installed vLLM):
    #   probe 1 (L1327): layers_block_type  -- NemotronH / Falcon
    #   probe 2 (L1341): attn_type_list     -- Minimax (truthiness, not is-not-None)
    #   probe 3 (L1346): layer_types        -- Gemma4 / Qwen3.5
    #
    # NOTE: Jamba (has_noops / block_configs) is intentionally NOT supported here.
    # vLLM's has_noops path (vllm/config/model.py:1322-1324) uses hf_config.block_configs,
    # a list of structured config objects (bc.attention.no_op), not a flat list of
    # type strings.  This function returns a flat list, so the block_configs format
    # is incompatible with that contract.  Jamba models will fall through to returning
    # None, which causes all layers to be treated as attention layers -- incorrect for
    # Jamba but safe to fail loudly rather than silently misclassify.  If Jamba support
    # is needed: convert block_configs to a flat string list and return it here.
    #
    # NOTE: Zamba2 special case (vllm/config/model.py:L1331) within probe 1 maps
    # "hybrid" entries to attention when attn_block_type=True. This function returns
    # the raw list unchanged; callers use is_attn_layer() which treats
    # "hybrid" as an attention type unconditionally (matching Zamba2-family semantics). Zamba2
    # support is therefore already handled without any remapping step here.
    #
    # VERSION SYNC: last verified against vLLM 0.24.0.
    # On each vLLM bump, diff ModelConfig.get_num_layers_by_block_type
    # (vllm/config/model.py:1327-1369) against the probe sequence below and
    # update the version number above.
    # The block_configs / has_noops path is the known gap; check whether vLLM
    # has added any further probes beyond the three mirrored here.
    #
    # Upstream request: vLLM does not expose a public ModelConfig.get_layer_types()
    # that returns the type list rather than a count. If it did, probes 1-3 below
    # could be replaced with a direct call, eliminating this fragile copy.
    # Until then, this VERSION SYNC comment is the correct mitigation.
    #
    # Probe 1 uses is-not-None (empty list is a valid value meaning all-non-attention).
    # Probe 2 uses truthiness matching vLLM model.py:1342 (empty attn_type_list falls
    # through to layer_types). The asymmetry is intentional: it mirrors vLLM's own
    # inconsistency at model.py:1341-1342 and is not a bug.
    # Probe 4 (outer-config layer_types) is not present in vLLM; it is a local
    # extension for multimodal models where layer_types lives only on the outer config.
    v = getattr(hf_text_config, "layers_block_type", None)  # vllm/config/model.py:1330
    if v is not None:
        return v
    v = getattr(_outer, "attn_type_list", None)             # vllm/config/model.py:1341 (truthiness)
    if v:
        return v
    v = getattr(hf_text_config, "layer_types", None)        # vllm/config/model.py:1346
    if v is not None:
        return v
    if _outer is not hf_text_config:                        # local extension: outer-only configs
        v = getattr(_outer, "layer_types", None)
        if v is not None:
            return v
    return None


def determine_available_memory(worker: "WebGPUWorker") -> int:
    """
    Available memory for KV cache = OS-available RAM - overhead.

    Uses psutil.virtual_memory().available rather than the system total so that
    memory already consumed by other processes (browsers, other inference servers,
    OS page cache that cannot be reclaimed quickly) is excluded from the budget.
    On Apple Silicon this is still correct: the UMA pool is reflected in
    virtual_memory() just as on x86, and available already excludes the model
    weights this process has loaded.

    Overhead budget: max(_OVERHEAD_BYTES, model_mem * _ACTIVATION_OVERHEAD_FRACTION).
    The fraction-based term accounts for activation memory scaling with model size.
    For models up to ~3.4 GiB weights the 512 MiB floor applies; above that the
    fraction dominates. A 7B f16 model (~14 GiB) gets ~2.1 GiB reserved, which
    covers typical prefill activation peaks at batch sizes up to ~2048 tokens.

    Safe range for the flat 512 MiB floor: models whose weights fit in <= 3.4 GiB
    (e.g. 1B-2B f16 models). For anything larger the fraction term is used instead.

    """
    # NOTE: mirrors gpu_worker.py walrus+truthiness check (`if kv_cache_memory_bytes := ...`).
    # Treats 0 as not-set and falls through to the profiling path rather than returning
    # 0 bytes (which would produce 0 KV blocks and an unrecoverable engine startup failure).
    # The GPU worker calls profile_run() even when an explicit value is present, to compile
    # CUDA graphs. WebGPU intentionally omits that step: there are no CUDA graphs, and
    # warm_up() in compile_or_warm_up_model() is sufficient.
    if explicit := worker.cache_config.kv_cache_memory_bytes:
        return explicit

    _model = worker.model_runner.model if worker.model_runner is not None else None
    model_mem = (
        sum(buf.nbytes for buf in _model.weights.values())
        if _model is not None else 0
    )

    # get_visible_memory_node() always returns [0] on macOS (hardcoded in vLLM's
    # cpu_resource_utils.py). The assert below is unreachable there but keeps
    # the guard valid on Linux where /proc/{pid}/status may lack Mems_allowed_list.
    _nodes = get_visible_memory_node()
    if not _nodes:
        raise RuntimeError(
            "get_visible_memory_node() returned an empty list — "
            "/proc/{}/status may lack Mems_allowed_list or "
            "CPU_VISIBLE_MEMORY_NODES is misconfigured".format(os.getpid())
        )
    node_infos = [get_memory_node_info(n) for n in _nodes]
    total_memory = sum(i.total_memory for i in node_infos)
    total_available = sum(i.available_memory for i in node_infos)
    overhead = max(_OVERHEAD_BYTES, int(model_mem * _ACTIVATION_OVERHEAD_FRACTION))
    # total_available excludes memory held by other processes as well as by this
    # process (including model weights already uploaded), so there is no need to
    # subtract model_mem explicitly.
    base = total_available - overhead
    fraction = worker.cache_config.gpu_memory_utilization
    available = max(int(base * fraction), 0)
    logger.info(
        "WebGPU memory: total=%dMiB available_os=%dMiB model=%dMiB "
        "overhead=%dMiB kv_budget=%dMiB",
        total_memory // MiB_bytes, total_available // MiB_bytes,
        model_mem // MiB_bytes, overhead // MiB_bytes, available // MiB_bytes,
    )
    return available

