from __future__ import annotations
from functools import cached_property
from itertools import chain
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from torch.nn.functional import pad

from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.outputs import ModelRunnerOutput, LogprobsTensors, EMPTY_MODEL_RUNNER_OUTPUT
from vllm.v1.sample.sampler import Sampler
from vllm.sampling_params import SamplingType

from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm_webgpu.utils import sample_token as _sample_token, zero_bytes
from vllm_webgpu.v1.cache_policy import MIN_WEBGPU_BUFFER_BYTES, allocate_kv_from_tensors, get_layer_types, is_attn_layer
from vllm_webgpu.webgpu.pipeline import PipelineCache



if TYPE_CHECKING:
    from typing import Any, Sequence
    from vllm.tasks import SupportedTask
    from vllm_webgpu.models.base import BaseWebGPUModel
    from vllm_webgpu.webgpu.device import WebGPUDevice
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheSpec
    from vllm.v1.outputs import AsyncModelRunnerOutput, LogprobsLists

logger = init_logger(__name__)


def _resolve_num_logprobs(sp, rid: str) -> "int | None":
    """Validate and return num_logprobs from a SamplingParams object.

    Raises NotImplementedError for unsupported logprob modes that the WebGPU
    backend cannot handle. Called from both the prefill and decode loops to
    avoid duplicating the validation block in each.
    """
    if sp is None:
        return None
    if sp.logprob_token_ids is not None:
        raise NotImplementedError(
            f"req {rid}: logprob_token_ids (fixed-token-set logprobs) is not supported on the WebGPU backend; "
            "only top-k logprobs by probability rank are available, not for arbitrary token ID sets"
        )
    num_logprobs = sp.logprobs
    if num_logprobs == -1:
        raise NotImplementedError(
            f"req {rid}: logprobs=-1 (full-vocab) is not supported on the WebGPU backend; "
            "use a positive integer instead"
        )
    return num_logprobs


# KV cache dtype used by all WebGPU attention layers. Referenced in both
# get_kv_cache_spec and get_cache_block_size_bytes so that changing it
# keeps both methods consistent.
_KV_DTYPE = torch.float16

ARCH_MAP = {
    "LlamaForCausalLM": "llama",
    "MistralForCausalLM": "mixtral",
    "MixtralForCausalLM": "mixtral",
    "Qwen2ForCausalLM": "llama",
    "Qwen3ForCausalLM": "llama",
    "Gemma3ForCausalLM": "gemma4",
    "Gemma3ForConditionalGeneration": "gemma4",
    "Gemma4ForCausalLM": "gemma4",
    "Gemma4UnifiedForConditionalGeneration": "gemma4",
    "Qwen3_5ForConditionalGeneration": "qwen35",
    "Qwen3_5MoeForConditionalGeneration": "qwen35",  # MoE variant; FFN routing on GPU via topk_sort
    "DiffusionGemmaForBlockDiffusion": "diffusion_gemma",
    "GptOssForCausalLM": "gpt_oss",
    "NemotronHForCausalLM": "nemotron_h",
}


_FAMILY_TO_CLASS: "dict[str, str]" = {
    "llama":           "vllm_webgpu.models.llama.LlamaWebGPUModel",
    "mixtral":         "vllm_webgpu.models.mixtral.MixtralWebGPUModel",
    "gemma4":          "vllm_webgpu.models.gemma4.Gemma4WebGPUModel",
    "qwen35":          "vllm_webgpu.models.qwen35.Qwen35WebGPUModel",
    "diffusion_gemma": "vllm_webgpu.models.diffusion_gemma.DiffusionGemmaWebGPUModel",
    "gpt_oss":         "vllm_webgpu.models.gpt_oss.GptOssWebGPUModel",
    "nemotron_h":      "vllm_webgpu.models.nemotron_h.NemotronHWebGPUModel",
}


def _build_model(arch: str, family: "str | None", model_config: Any, wgpu_device: Any, pipeline_cache: Any, block_size: int) -> "BaseWebGPUModel":
    qualname = _FAMILY_TO_CLASS.get(family or "")
    if qualname is None:
        raise NotImplementedError(
            f"Architecture {arch!r} is not supported. "
            f"Supported: {sorted(ARCH_MAP)}"
        )
    cls = resolve_obj_by_qualname(qualname)
    return cls(model_config, wgpu_device, pipeline_cache, block_size=block_size)


def _compute_prompt_logprobs(
    full_logits: "np.ndarray",
    tok_ids: "list[int]",
    num_prompt_logprobs: int,
) -> "LogprobsTensors | None":
    """Compute per-position prompt logprobs for a prefill pass.

    For T prompt tokens, produces T-1 rows: row i uses full_logits[i]
    to evaluate the probability of tok_ids[i+1].  Returns a
    LogprobsTensors of shape [T-1, min(num_prompt_logprobs, vocab_size)+1].
    k is capped at vocab_size. Returns None when T < 2 or the logits
    buffer has fewer rows than prompt positions need.
    """
    T = len(tok_ids)
    if T < 2:
        return None

    # Guard against argmax-only output (single-element vocabulary dimension),
    # matching the equivalent check in _extract_logprob_data.
    if full_logits.shape[-1] <= 1:
        return None

    num_positions = T - 1

    # Guard: the model may return only the last token's logits (shape
    # [1, vocab]) even during a multi-token prefill.  In that case we
    # cannot reconstruct per-position distributions and must bail out
    # rather than letting the subsequent row-index into a 1-row array
    # raise IndexError.  At least 2 logit rows are required to compute
    # even a single prompt-logprob position (T >= 2 means num_positions
    # >= 1), so max(num_positions, 2) expresses both constraints at once.
    if full_logits.shape[0] < max(num_positions, 2):
        logger.warning(
            "prompt_logprobs: logits buffer has %d rows but %d prompt "
            "positions need coverage; skipping (model returns "
            "last-token-only logits for this prefill length)",
            full_logits.shape[0],
            num_positions,
        )
        return None

    if num_prompt_logprobs < 0:
        num_prompt_logprobs = full_logits.shape[-1]
    k = min(num_prompt_logprobs, full_logits.shape[-1])

    # Sampler.compute_logprobs and gather_logprobs are @staticmethod in vLLM >= 0.24; class-level calls are intentional.
    lp_t = Sampler.compute_logprobs(torch.from_numpy(full_logits[:num_positions]))
    lp = Sampler.gather_logprobs(lp_t, k, torch.tensor(tok_ids[1:], dtype=torch.int64))
    return lp


def _stack_logprobs(items: "Sequence[LogprobsTensors]") -> "LogprobsLists":
    """Cat a list of LogprobsTensors along the batch dimension and convert to lists.

    cu_num_generated_tokens defaults to None: WebGPU produces exactly one output
    row per request, so LogprobsLists.slice_request(i, n) uses i directly as the
    row index when cu_num_generated_tokens is None (see vllm/v1/outputs.py:41-42).
    WebGPU tensors are already on CPU, so .cpu() inside tolists() is a no-op.
    selected_token_ranks (from LogprobsTensors) is cast to int32 before
    storing it as LogprobsLists.sampled_token_ranks: gather_logprobs returns
    int64 for selected_token_ranks (via batched_count_greater_than) but int32
    for logprob_token_ids (indices). The cast normalises selected_token_ranks.
    The padded path does NOT pre-cast each piece; the single cast here covers
    both paths uniformly.
    If a future vLLM release casts token_ranks to int32 in gather_logprobs,
    remove this cast.
    """
    combined = LogprobsTensors(
        torch.cat([x.logprob_token_ids for x in items]),
        torch.cat([x.logprobs for x in items]),
        torch.cat([x.selected_token_ranks for x in items]).to(torch.int32),
    )
    return combined.tolists()


def _make_model_output(
    req_ids: list[str],
    sampled: list[int],
    logprobs_data: "Sequence[LogprobsTensors | None]" = (),
    prompt_logprobs_dict: "dict[str, LogprobsTensors | None] | None" = None,
) -> "ModelRunnerOutput":
    if prompt_logprobs_dict is None:
        prompt_logprobs_dict = {}
    if not req_ids:
        return EMPTY_MODEL_RUNNER_OUTPUT

    # Build LogprobsLists for top-k sampled-token logprob entries.
    # One row per request in the batch (matching req_id_to_index), so that
    # LogprobsLists.slice_request(i, n) works with cu_num_generated_tokens=None
    # and uses i directly as the row index, matching the vLLM API contract.
    # Requests that have no logprobs get a dummy row (zeros / -inf) that is
    # never exposed to callers because the scheduler guards slice_request on
    # num_logprobs.
    built_logprobs = None
    widths = [d.logprob_token_ids.shape[1] for d in logprobs_data if d is not None]
    if widths:
        max_k = max(widths)
        # Short-circuit when all entries are present and share the same width.
        all_present = all(d is not None for d in logprobs_data)
        if all_present and len(set(widths)) == 1:
            built_logprobs = _stack_logprobs(cast("list[LogprobsTensors]", logprobs_data))
        else:
            # Pad entries to max_k width and collect sentinel rows for requests
            # without logprob data. empty_cpu() produces int32 for selected_token_ranks
            # while gather_logprobs returns int64; torch.cat on CPU promotes int32 to
            # int64 automatically, and _stack_logprobs normalises the result to int32.
            pieces = []
            for d in logprobs_data:
                if d is not None:
                    n_pad = max_k - d.logprob_token_ids.shape[1]
                    pieces.append(LogprobsTensors(
                        pad(d.logprob_token_ids, (0, n_pad), value=0) if n_pad else d.logprob_token_ids,
                        pad(d.logprobs, (0, n_pad), value=-float("inf")) if n_pad else d.logprobs,
                        d.selected_token_ranks,
                    ))
                else:
                    # Sentinel row for requests with no logprob data. Sentinel
                    # rows are never sliced by the caller (the scheduler only
                    # calls slice_request for requests where num_logprobs > 0),
                    # so uninitialized memory from empty_cpu is safe here.
                    pieces.append(LogprobsTensors.empty_cpu(1, max_k))
            built_logprobs = _stack_logprobs(pieces)

    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        sampled_token_ids=[[t] for t in sampled],
        logprobs=built_logprobs,
        prompt_logprobs_dict=prompt_logprobs_dict,
    )


def _extract_logprob_data(
    logits: "np.ndarray",
    row_idx: int,
    tok: int,
    num_logprobs: "int | None",
    rid: str,
) -> "LogprobsTensors | None":
    """Extract logprob data for one request from a logits array.

    Args:
        logits: The full logit array returned by forward().
        row_idx: Which row to use (-1 for last token, 0 for single-token decode).
        tok: The sampled token id.
        num_logprobs: Number of top logprobs requested, or None to skip.
        rid: Request id (used only for the warning message).

    Returns a LogprobsTensors of shape [1, k+1] or None when logprobs cannot be computed.
    """
    if num_logprobs is None:
        return None
    if logits.shape[-1] <= 1:
        logger.warning("req %s: logprobs requested but model returned argmax-only output", rid)
        return None
    lp_t = Sampler.compute_logprobs(torch.from_numpy(logits[row_idx]).unsqueeze(0))
    k = min(num_logprobs, logits.shape[-1])
    # selected_token_ranks is cast to int32 in _stack so no explicit cast is needed
    # here. Prompt logprobs bypass _stack and are consumed as LogprobsTensors by
    # the engine (ranks.tolist() produces plain Python ints regardless of dtype).
    return Sampler.gather_logprobs(lp_t, k, torch.tensor([tok], dtype=torch.int64))


class WebGPUModelRunner:
    def __init__(self, vllm_config: Any, wgpu_device: "WebGPUDevice") -> None:
        self.vllm_config = vllm_config
        self.wgpu_device = wgpu_device
        self.pipeline_cache = PipelineCache(wgpu_device.wgpu_device)
        self.model: "BaseWebGPUModel | None" = None
        self._req_state: dict[str, Any] = {}  # per-request decode state {req_id: {pos, block_ids}}
        self._num_kv_blocks: int = 0  # set by initialize_kv_cache; used by _zero_kv_blocks
        self._block_size: int = vllm_config.cache_config.block_size
        self._use_fp64_gumbel: bool = vllm_config.model_config.use_fp64_gumbel
        # Capability flags: set False here so the attribute set is complete from
        # construction time. load_model() overwrites these with correct values once
        # the model is available. Without these defaults, any code path that reads
        # a flag before load_model() completes (e.g. a unit test calling
        # _execute_model_v2 directly) would raise AttributeError.
        self._has_reset = self._has_save = self._has_replay = self._has_restore = False

    def load_model(self) -> None:
        self.__dict__.pop("kv_cache_spec", None)
        # Reset capability flags before model is assigned; a reload clears stale values.
        self._has_reset = False
        self._has_save = False
        self._has_replay = False
        self._has_restore = False
        mc = self.vllm_config.model_config
        arch = mc.architecture
        hf_config = mc.hf_config

        block_size = self._block_size

        family = ARCH_MAP.get(arch)
        spec_config = self.vllm_config.speculative_config
        if spec_config is not None:
            # The decode path in execute_model forwards exactly 1 token per
            # request regardless of scheduler_output.num_scheduled_tokens[rid].
            # With speculative decoding the scheduler sets num_scheduled_tokens > 1
            # for decode steps; the engine then expects N sampled tokens back and
            # will either crash or silently corrupt output when it receives 1.
            # Fail fast here rather than produce wrong results at runtime.
            #
            # Note: vllm_config.num_speculative_tokens is not used here because
            # it also returns diffusion_config.canvas_length for diffusion models,
            # which would cause a false-positive error for architectures like
            # DiffusionGemmaForBlockDiffusion that are registered in ARCH_MAP.
            num_spec = spec_config.num_speculative_tokens
            raise NotImplementedError(
                f"{arch} with speculative decoding "
                f"(num_speculative_tokens={num_spec}) is not supported on the "
                f"WebGPU backend. The decode path forwards exactly 1 token per "
                f"step; the engine expects num_speculative_tokens+1 tokens back. "
                f"Implement multi-token decode in execute_model before enabling."
            )

        self.model = _build_model(arch, family, hf_config, self.wgpu_device, self.pipeline_cache, block_size=block_size)
        if family == "nemotron_h":
            # spec_config is None here: the raise above blocks any non-None value.
            # Pass 0 so NemotronHWebGPUModel.load_weights can accept num_spec when
            # speculative decoding is eventually implemented in execute_model.
            self.model.load_weights(mc.model, num_spec=0)
        else:
            self.model.load_weights(mc.model)
        # Cache model capability flags once here; self.model is fixed after load_model()
        # and these attributes never change between inference steps.
        self._has_reset = hasattr(self.model, "reset_recurrent_states")
        self._has_save = hasattr(self.model, "save_recurrent_states")
        self._has_replay = hasattr(self.model, "replay_prefix_for_ssm")
        self._has_restore = hasattr(self.model, "restore_recurrent_states")
        logger.info("Model loaded: arch=%s", arch)

    def initialize_kv_cache(self, kv_cache_config: Any) -> None:
        self._num_kv_blocks = kv_cache_config.num_blocks

        allocate_kv_from_tensors(
            self.wgpu_device.wgpu_device,
            self.model,
            kv_cache_config,
            num_total_layers=self.vllm_config.model_config.get_total_num_hidden_layers(),
        )

    def get_kv_cache_spec(self) -> "dict[str, KVCacheSpec]":
        return self.kv_cache_spec

    @cached_property
    def kv_cache_spec(self) -> "dict[str, KVCacheSpec]":
        num_hidden_layers = self.vllm_config.model_config.get_total_num_hidden_layers()
        block_size = self._block_size
        spec: dict[str, Any] = {}

        # Use per-layer params if available (Gemma4 heterogeneous layers).
        # Prefer the model object's _lp list (populated from layer_types config)
        # over the raw HF config attribute, which may not be set for safetensors.
        # Returns None when model is not yet loaded (vLLM calls get_kv_cache_spec
        # before load_model), letting kv_cache_spec fall through to get_layer_types.
        lp_list = getattr(self.model, "_lp", None)

        # Fallback: derive per-layer KV spec from layer_types + global_head_dim when
        # the model has not been loaded yet and hf_config lacks _layer_attention_params.
        # This covers Gemma4 safetensors where vLLM may call get_kv_cache_spec()
        # before load_model(), so self.model is still None. Mirrors the derivation
        # in Gemma4WebGPUModel.__init__() to ensure uniform and per-layer specs agree.
        #
        # Only layers whose type appears in ATTN_TYPES get a KV cache entry.
        # Mamba, MLP, and linear-attention layers carry no KV state and must be
        # excluded; emitting a FullAttentionSpec for them over-reports KV memory.
        # NemotronH attention layers live under .mixer, not .self_attn.
        _is_nemotron_h = ARCH_MAP.get(self.vllm_config.model_config.architecture) == "nemotron_h"
        _attn_suffix = ".mixer" if _is_nemotron_h else ".self_attn"
        # In the standard UniProc executor, load_model() runs before get_kv_cache_spec():
        # _init_executor() calls init_device() then load_model(), and only after that
        # does the engine call _initialize_kv_caches() which invokes get_kv_cache_spec().
        # The lp_list branch (lines below) is therefore the primary execution path.
        # _layer_types is the fallback for edge cases: tests that call get_kv_cache_spec
        # directly, OOT executors with different ordering, or model reload mid-session.
        tc = self.vllm_config.model_config.hf_text_config
        _layer_types = get_layer_types(
            tc,
            hf_outer_config=self.vllm_config.model_config.hf_config,
        )

        if lp_list and len(lp_list) == num_hidden_layers:
            for i, lp in enumerate(lp_list):
                spec[f"model.layers.{i}{_attn_suffix}"] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=lp["num_kv_heads"],
                    head_size=lp["head_dim"],
                    head_size_v=lp.get("head_dim_v"),
                    dtype=_KV_DTYPE,
                )
        elif _layer_types and len(_layer_types) == num_hidden_layers:
            default_hd = self.vllm_config.model_config.get_head_size()
            default_kv = self.vllm_config.model_config.get_total_num_kv_heads()
            global_hd = getattr(tc, "global_head_dim", default_hd)
            global_kv = getattr(tc, "num_global_key_value_heads", default_kv)
            k_eq_v = getattr(tc, "attention_k_eq_v", False)
            for i, lt in enumerate(_layer_types):
                if not is_attn_layer(lt):
                    continue
                if lt == "full_attention":
                    full_kv = global_kv if k_eq_v else default_kv
                    _hd_v = getattr(tc, "head_size_v", None)
                    spec[f"model.layers.{i}{_attn_suffix}"] = FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=full_kv,
                        head_size=global_hd,
                        head_size_v=_hd_v,
                        dtype=_KV_DTYPE,
                    )
                else:
                    # Treat all non-full-attention types (including sliding_attention) as
                    # full-attention: SlidingWindowSpec is not supported by
                    # allocate_kv_from_tensors, so we allocate for the full context window.
                    _hd_v_local = getattr(tc, "head_size_v", None)
                    spec[f"model.layers.{i}{_attn_suffix}"] = FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=default_kv,
                        head_size=default_hd,
                        head_size_v=_hd_v_local,
                        dtype=_KV_DTYPE,
                    )
        else:
            head_size = self.vllm_config.model_config.get_head_size()
            num_kv_heads = self.vllm_config.model_config.get_total_num_kv_heads()
            # NemotronH must never fall through to the uniform path: every layer
            # gets a .mixer suffix, so non-attention layers (Mamba, MLP) would
            # receive spurious KV cache entries.  A mismatched layer_types list
            # is a configuration error, not a safe fallback.
            if _is_nemotron_h:
                msg = (
                    "layer_types is missing for NemotronH"
                    if _layer_types is None
                    else (
                        f"layer_types length ({len(_layer_types)}) does not match "
                        f"num_hidden_layers ({num_hidden_layers}) for NemotronH"
                    )
                )
                raise ValueError(msg + ": KV spec cannot be determined safely")
            # Only trust layer_types when it covers every layer; a partial or
            # mismatched list (including a stray MagicMock in tests) falls back
            # to the uniform path so all layers get a spec entry.
            for i in range(num_hidden_layers):
                spec[f"model.layers.{i}{_attn_suffix}"] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=num_kv_heads,
                    head_size=head_size,
                    head_size_v=getattr(tc, "head_size_v", None),
                    dtype=_KV_DTYPE,
                )
        return spec

    def get_cache_block_size_bytes(self) -> int:
        return sum(s.page_size_bytes for s in self.kv_cache_spec.values())

    def warm_up(self) -> None:
        if self.model is not None:
            self.model.warmup()

    def _zero_kv_blocks(self, block_ids: list[int]) -> None:
        """Zero the KV cache entries for recycled block IDs.

        When the block pool reuses a block from a completed request, stale K/V
        values remain in the buffer until the new request's forward pass writes
        to those positions. Attention over positions beyond the current write
        cursor would read that garbage, producing incorrect outputs.

        This mirrors gpu_model_runner.py _zero_block_ids (lines 1155-1156).
        Called before any forward pass in the step so the zeroing is committed
        to the GPU before the first attention dispatch.

        Each layer buffer has layout [num_blocks * block_size * num_kv_heads * head_dim]
        in f16. The byte range for block_id is
          [block_id * bytes_per_block, (block_id + 1) * bytes_per_block).
        Placeholder buffers (16 bytes, used for non-attention layers) are skipped.
        """
        if self.model is None or self._num_kv_blocks == 0:
            return
        queue = self.wgpu_device.wgpu_device.queue

        for k_buf, v_buf in self.model.kv_pool:
            if k_buf.nbytes <= MIN_WEBGPU_BUFFER_BYTES:
                # 16-byte placeholder for non-attention layers (Mamba, MLP-only, etc.)
                continue
            bpb = k_buf.nbytes // self._num_kv_blocks
            zeros = zero_bytes(bpb)
            bpb_v = v_buf.nbytes // self._num_kv_blocks
            zeros_v = zeros if bpb_v == bpb else zero_bytes(bpb_v)
            for block_id in block_ids:
                queue.write_buffer(k_buf.buf, block_id * bpb, zeros)
                queue.write_buffer(v_buf.buf, block_id * bpb_v, zeros_v)

    def execute_model(self, scheduler_output: "SchedulerOutput") -> "ModelRunnerOutput | AsyncModelRunnerOutput | None":
        if scheduler_output.has_structured_output_requests:
            raise NotImplementedError(
                "Guided/constrained decoding is not supported on the WebGPU backend. "
                "The WebGPU argmax path discards the logit distribution required for "
                "grammar token masks. Use unconstrained sampling or a CPU/CUDA backend."
            )
        if self.model is None:
            return EMPTY_MODEL_RUNNER_OUTPUT
        return self._execute_model_v2(scheduler_output)

    def _execute_model_v2(self, scheduler_output: "SchedulerOutput") -> "ModelRunnerOutput":
        """vLLM >= 0.24 SchedulerOutput format."""
        # Zero recycled KV blocks before any forward pass. The block pool may
        # reuse blocks from completed requests; without zeroing, attention over
        # positions beyond the current write cursor reads stale K/V values and
        # produces incorrect outputs. The GPU runner does the same via
        # _zero_block_ids (gpu_model_runner.py lines 1155-1156).
        if scheduler_output.new_block_ids_to_zero:
            self._zero_kv_blocks(scheduler_output.new_block_ids_to_zero)

        # Prune state only for fully finished requests. Preempted requests may be
        # rescheduled in a future step via scheduled_cached_reqs.resumed_req_ids
        # (when use_v2_model_runner=False the scheduler does not merge resumed
        # requests into scheduled_new_reqs, so eagerly deleting on preemption
        # causes a RuntimeError when the resumed-request decode path calls
        # _req_state.get(rid) and gets None).
        for rid in scheduler_output.finished_req_ids:
            self._req_state.pop(rid, None)

        cached = scheduler_output.scheduled_cached_reqs
        new_reqs = scheduler_output.scheduled_new_reqs
        block_size = self._block_size

        all_req_ids: list[str] = []
        all_sampled: list[int] = []
        all_logprobs_data: list[LogprobsTensors | None] = []  # per-request logprob data
        prompt_logprobs_dict: dict[str, LogprobsTensors | None] = {}  # req_id -> LogprobsTensors for prefill

        # ── Prefill: new requests ──────────────────────────────────────────────
        for req in new_reqs:
            rid = req.req_id
            tok_ids = req.prompt_token_ids
            if not tok_ids:
                raise NotImplementedError(
                    f"req {rid}: prompt_embeds (no token IDs) are not supported "
                    f"on the WebGPU backend"
                )
            if req.prompt_is_token_ids is not None and not all(req.prompt_is_token_ids):
                raise NotImplementedError(
                    f"req {rid}: mixed token/embedding prompts (prompt_is_token_ids) are not supported "
                    f"on the WebGPU backend"
                )

            # Multi-modal inputs (images, audio, video) are not implemented.
            # Conditional-generation architectures (Gemma3ForConditionalGeneration,
            # Gemma4UnifiedForConditionalGeneration, Qwen3_5ForConditionalGeneration,
            # etc.) are listed in ARCH_MAP only for text-only inference. If a request
            # carries mm_inputs, the image would be silently ignored and the model
            # would produce text as if no image was provided. Raise early instead.
            if req.mm_features:
                raise NotImplementedError(
                    f"req {rid}: multi-modal inputs (images/audio/video) are not supported "
                    f"by the WebGPU backend. The conditional-generation architecture is "
                    f"registered for text-only inference only. Use the CausalLM variant "
                    f"of the model (e.g. Gemma3ForCausalLM) or wait for multi-modal support."
                )

            # Extract per-request logprob counts via the stable SamplingParams property.
            sp = req.sampling_params
            num_logprobs = _resolve_num_logprobs(sp, rid)
            num_prompt_logprobs = sp.prompt_logprobs if sp is not None else None

            # Warn early when full-vocab prompt logprobs are requested. The CPU
            # topk over the entire vocabulary (O(T * V log V)) can stall inference
            # for several seconds on large-vocab models. Surface the cost here at
            # request admission time rather than inside _compute_prompt_logprobs.
            if num_prompt_logprobs is not None and num_prompt_logprobs < 0:
                logger.warning(
                    "req %s: prompt_logprobs=%d requests full-vocabulary logprobs "
                    "at every prompt position. This runs on CPU and is O(T * V log V); "
                    "expect multi-second stalls for long prompts or large vocabularies. "
                    "Use a small positive value instead.",
                    rid,
                    num_prompt_logprobs,
                )

            raw_bids = req.block_ids
            if not raw_bids:
                raise RuntimeError(f"req {rid}: scheduler produced NewRequestData with empty block_ids")
            blk_ids = list(chain.from_iterable(raw_bids))

            bt = np.array(blk_ids, dtype=np.uint32)

            # Batch prefill: send the scheduled chunk of prompt tokens in a single
            # forward() call. With prefix caching, num_computed_tokens tokens are
            # already in the KV cache; only the uncached tail needs to be processed.
            num_computed = req.num_computed_tokens
            num_sched = scheduler_output.num_scheduled_tokens[rid]
            T = min(num_sched, len(tok_ids) - num_computed)
            chunk_toks = tok_ids[num_computed:num_computed + T]
            abs_idx = np.arange(num_computed, num_computed + T, dtype=np.uint32)
            blk_idx, within_block = np.divmod(abs_idx, block_size)
            oob = blk_idx >= len(blk_ids)
            if oob.any():
                bad = int(abs_idx[oob][0])
                raise RuntimeError(
                    f"block table too short for req {rid}: token {bad} needs block "
                    f"{bad // block_size} but only {len(blk_ids)} blocks allocated"
                )
            slots = (bt[blk_idx].astype(np.int64) * block_size + within_block).tolist()

            _batch_pm = SimpleNamespace(slot_mapping=slots, block_tables=[bt], max_decode_seq_len=num_computed + T)

            self.model._greedy_decode = (sp is None or sp.sampling_type == SamplingType.GREEDY) and num_logprobs is None and num_prompt_logprobs is None

            # Each prefill request starts from zero recurrent state. Reset here
            # (inside the loop) so that multiple new requests in the same step
            # each get a clean slate rather than inheriting the previous request's
            # post-prefill state.
            if self._has_reset:
                self.model.reset_recurrent_states()

            last_logits = self.model.forward(
                np.array(chunk_toks, dtype=np.uint32),
                abs_idx,
                _batch_pm,
            )

            # Save recurrent state so the decode path can restore it before this
            # request's first (and every subsequent) decode step.
            prefill_recurrent_states = None
            if self._has_save:
                prefill_recurrent_states = self.model.save_recurrent_states()

            # Use the last position's logits for the first generated token.
            # Apply sampling when SamplingParams request non-greedy decoding.
            # Seed a per-request generator once at prefill; the same object is
            # passed to every subsequent decode step so the RNG state advances
            # between steps rather than restarting from the same seed each time.
            # chunked prefill is disabled (platform.py:enable_chunked_prefill = False),
            # so every request here is truly new and must have no prior state.
            if rid in self._req_state:
                raise RuntimeError(
                    f"req {rid} already has saved state but chunked prefill is disabled; "
                    "update this path before re-enabling chunked prefill"
                )
            if sp is not None and sp.sampling_type == SamplingType.RANDOM_SEED:
                rng = torch.Generator()
                rng.manual_seed(sp.seed)
            else:
                rng = None
            if sp is None or last_logits.shape[-1] == 1:
                first_decode_tok = int(last_logits[-1, 0])
            else:
                first_decode_tok = _sample_token(
                    last_logits[-1], temperature=sp.temperature,
                    top_p=sp.top_p, top_k=sp.top_k, min_p=sp.min_p,
                    generator=rng,
                    use_fp64_gumbel=self._use_fp64_gumbel,
                )

            # Compute logprobs for this prefill token if the request asked for them.
            lp_data = _extract_logprob_data(last_logits, -1, first_decode_tok, num_logprobs, rid)

            # Compute prompt logprobs for each prompt position when full logits
            # are available.  Position i uses logits[i] to evaluate tok_ids[i+1],
            # producing T-1 rows of top-K logprob data.
            if num_prompt_logprobs is not None and T >= 2:
                # Pass only the token window so full_logits[i] and
                # tok_ids_param[i+1] stay aligned regardless of num_computed.
                pt = _compute_prompt_logprobs(
                    last_logits,
                    chunk_toks,
                    num_prompt_logprobs,
                )
                if pt is not None:
                    prompt_logprobs_dict[rid] = pt

            all_req_ids.append(rid)
            all_sampled.append(first_decode_tok)
            all_logprobs_data.append(lp_data)
            # Store last sampled token; decode path needs it (new_token_ids is empty without PP).
            # token_history stores the full token sequence for this request so that
            # replay_prefix_for_ssm can reconstruct Mamba SSM state after preemption.
            # Only allocated for models that implement replay_prefix_for_ssm; pure-attention
            # models never read it, so skip the allocation and per-step append entirely.
            self._req_state[rid] = {
                "pos": num_computed + T, "block_ids": blk_ids,
                "last_tok": first_decode_tok,
                "sampling_params": sp,
                "num_logprobs": num_logprobs,
                "recurrent_states": prefill_recurrent_states,
                "rng": rng,
                **({"token_history": list(tok_ids) + [first_decode_tok], "prefix_offset": num_computed} if self._has_replay else {}),
            }

        # ── Decode: cached requests ────────────────────────────────────────────
        # new_token_ids is empty without pipeline parallelism (vLLM design).
        # Use last_tok stored in _req_state from the previous step instead.
        #
        # Multi-sequence batching is not supported: one forward() call per request.
        # Why true batching can't be done without architecture changes:
        #   - queue.write_buffer() executes before submit, so all N writes to the
        #     same pre-allocated buffers would alias; only the last request's data
        #     would survive into the encoder.
        #   - Pre-allocated scratch buffers (_pre/_sc) are sized for T=1.
        #   - Attention shaders accept one block table, so cross-request KV attention
        #     would produce wrong results without per-sequence block-table dispatch.
        # To enable true batched decode we would need: N separate pre-alloc buffer
        # sets, N argmax result buffers, and attention shaders with a batched block
        # table (or a flash-attn style per-sequence loop inside the shader).
        if cached.req_ids:
            new_block_ids = cached.new_block_ids
            resumed_req_ids = cached.resumed_req_ids

            for i, rid in enumerate(cached.req_ids):
                state = self._req_state.get(rid)
                if state is None:
                    raise RuntimeError(
                        f"cached req {rid} missing from _req_state; internal state is "
                        "inconsistent (prefill may have returned None logits without "
                        "registering state)"
                    )
                pos = state["pos"]
                blk_ids = state["block_ids"]  # lazily copied below only when modified
                sp = state["sampling_params"]
                num_logprobs = state["num_logprobs"]
                is_resumed = rid in resumed_req_ids

                # Guard: a preempted request rescheduled with num_scheduled_tokens > 1
                # would cause a state desync. The decode loop processes exactly 1 token
                # and advances state["pos"] by 1, but _update_after_schedule will add
                # num_scheduled_tokens to num_computed_tokens. If those two values
                # diverge the scheduler believes the request is further ahead than the
                # KV cache actually is, and subsequent steps write to wrong KV slots
                # producing corrupt output. Raise here so the bug surfaces immediately
                # rather than silently corrupting generations.
                num_scheduled = scheduler_output.num_scheduled_tokens[rid]
                if num_scheduled > 1:
                    raise RuntimeError(
                        f"req {rid}: cached request has num_scheduled_tokens="
                        f"{num_scheduled} but the WebGPU decode loop processes exactly 1 "
                        f"token per step. This would cause a KV-cache slot desync between "
                        f"state['pos'] and num_computed_tokens. Disable preemption or set "
                        f"enable_prefix_caching=False to avoid this path."
                    )

                # Update block table: preempted/resumed requests replace their
                # block table entirely; others append newly allocated blocks.
                cur_new_bids = new_block_ids[i]
                if is_resumed and cur_new_bids is None:
                    raise RuntimeError(
                        f"resumed req {rid} has no new_block_ids from scheduler"
                    )
                if cur_new_bids is not None:
                    flat_new = list(chain.from_iterable(cur_new_bids))
                    if is_resumed:
                        blk_ids = flat_new  # already a fresh list from chain
                        # Realign pos with the scheduler's authoritative view.
                        # After preemption num_computed_tokens is often 0 (full
                        # recompute); using the stale _req_state pos would write
                        # into the wrong block-table slot and skip uninitialized
                        # KV slots 0..old_pos-1 in the freshly allocated blocks.
                        pos = cached.num_computed_tokens[i]
                    else:
                        blk_ids = list(blk_ids)  # copy-on-write before mutating
                        blk_ids.extend(flat_new)

                # Decode step: forward one token at the current position.
                # Note: chunked prefill (context-phase continuation) is disabled
                # unconditionally by platform.py, so is_context_phase() is always
                # False. If chunked prefill is enabled in the future, restore the
                # context-phase branch here.
                #
                # For resumed preempted requests, pos has been rolled back to the
                # prefix-cache hit boundary (cached.num_computed_tokens[i]), which
                # is less than state["pos"] (the pre-preemption position). Using
                # state["last_tok"] would feed the token that was current at the
                # old position into the new (lower) KV slot, corrupting attention.
                # Read the authoritative token from all_token_ids instead, which
                # the scheduler populates for any request absent from the previous
                # step (exactly the resumed case).
                rolled_back_pos = is_resumed and pos < state["pos"]
                if rolled_back_pos:
                    _all_toks = cached.all_token_ids.get(rid)
                    tok = _all_toks[pos] if _all_toks and pos < len(_all_toks) else state["last_tok"]
                else:
                    tok = state["last_tok"]

                if pos // block_size >= len(blk_ids):
                    raise RuntimeError(
                        f"block table too short for req {rid}: pos={pos} needs block "
                        f"{pos // block_size} but only {len(blk_ids)} blocks allocated"
                    )
                slot = int(blk_ids[pos // block_size]) * block_size + pos % block_size

                _sm = SimpleNamespace(slot_mapping=[slot], block_tables=[np.array(blk_ids, dtype=np.uint32)], max_decode_seq_len=pos + 1)

                self.model._greedy_decode = (sp is None or sp.sampling_type == SamplingType.GREEDY) and num_logprobs is None

                # Restore this request's recurrent (Mamba/SSM) state before the
                # forward pass. Without this, each request in the batch reads the
                # in-place-updated state left by the previous request instead of
                # its own saved state, producing wrong recurrent outputs for every
                # request beyond the first in a multi-sequence decode batch.
                if self._has_restore:
                    saved_recurrent = state.get("recurrent_states")
                    if not rolled_back_pos and saved_recurrent is not None:
                        self.model.restore_recurrent_states(saved_recurrent)
                    elif self._has_reset:
                        # Reset Mamba conv/SSM states to zero before decoding.
                        self.model.reset_recurrent_states()
                        if rolled_back_pos and pos > 0:
                            # The request was preempted and resumed at position pos > 0.
                            # Prefix caching preserved pos tokens in the KV cache for
                            # attention layers, but the Mamba SSM state was lost. Replay
                            # tokens 0..pos-1 through the full model (skipping KV cache
                            # writes since the cache is already populated) to reconstruct
                            # the SSM state before the first resumed decode step.
                            token_history = state.get("token_history")
                            if not self._has_replay:
                                raise RuntimeError(
                                    f"req {rid}: preempted and resumed at pos={pos} with "
                                    f"prefix-cached KV but model does not implement "
                                    f"replay_prefix_for_ssm. Mamba SSM state cannot be "
                                    f"reconstructed. Aborting to prevent corrupt output."
                                )
                            if token_history is None or len(token_history) < pos:
                                raise RuntimeError(
                                    f"req {rid}: cannot reconstruct SSM state after "
                                    f"preemption at pos={pos}: token history has only "
                                    f"{len(token_history) if token_history else 0} tokens. "
                                    f"Aborting to prevent corrupt output."
                                )
                            prefix_offset = state.get("prefix_offset", 0)
                            self.model.replay_prefix_for_ssm(
                                np.array(token_history[prefix_offset:pos], dtype=np.uint32),
                                blk_ids,
                                start_pos=prefix_offset,
                            )

                logits = self.model.forward(
                    np.array([tok], dtype=np.uint32),
                    np.array([pos], dtype=np.uint32),
                    _sm,
                )

                # Save recurrent state immediately after the forward pass, before
                # any other request's forward can overwrite the shared GPU buffers.
                decode_recurrent_states = None
                if self._has_save:
                    decode_recurrent_states = self.model.save_recurrent_states()

                # Greedy path: model returns (1, 1) uint32 with the argmax index.
                # Non-greedy path: model returns (1, vocab) float32; sample here.
                # Use the persisted per-request generator so the RNG state
                # advances between steps (not reset to the same seed each step).
                rng = state["rng"]
                if sp is None or logits.shape[-1] == 1:
                    stok = int(logits[0, 0])
                else:
                    stok = _sample_token(
                        logits[0], temperature=sp.temperature,
                        top_p=sp.top_p, top_k=sp.top_k, min_p=sp.min_p,
                        generator=rng,
                        use_fp64_gumbel=self._use_fp64_gumbel,
                    )

                # Compute logprobs if requested for this request.
                lp_data = _extract_logprob_data(logits, 0, stok, num_logprobs, rid)

                # Commit state after a successful forward: don't mutate on failure.
                # rng is a stateful object; storing the same reference is sufficient.
                # Extend token_history with the newly generated token so that
                # replay_prefix_for_ssm has the full sequence if this request is
                # later preempted and resumed with prefix-cached KV.
                state["pos"] = pos + 1
                if blk_ids is not state["block_ids"]:
                    state["block_ids"] = blk_ids
                state["last_tok"] = stok
                state["recurrent_states"] = decode_recurrent_states
                if self._has_replay:
                    state.setdefault("token_history", []).append(stok)
                all_req_ids.append(rid)
                all_sampled.append(stok)
                all_logprobs_data.append(lp_data)

        # Return empty output rather than None when no requests scheduled.
        # vLLM's batch queue raises "unexpected error" on None from execute_model.
        return _make_model_output(
            all_req_ids, all_sampled, all_logprobs_data, prompt_logprobs_dict
        )

    def sample_tokens(self, grammar_output: "GrammarOutput") -> "ModelRunnerOutput | AsyncModelRunnerOutput":
        raise NotImplementedError(
            "Guided/constrained decoding (guided_json, guided_regex, guided_grammar) "
            "is not supported on the WebGPU backend. The GPU argmax path discards "
            "the full logit distribution required to apply grammar token masks. "
            "Use unconstrained sampling or switch to a CPU/CUDA backend."
        )

    def get_supported_tasks(self) -> "tuple[SupportedTask, ...]":
        return ("generate",)


