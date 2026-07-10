from __future__ import annotations
import itertools
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F

from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.outputs import ModelRunnerOutput, LogprobsTensors, EMPTY_MODEL_RUNNER_OUTPUT
from vllm.v1.sample.sampler import Sampler
from vllm.sampling_params import SamplingType

from vllm.logger import init_logger
from vllm_webgpu.utils import SHADERS_DIR, sample_token as _sample_token
from vllm_webgpu.v1.cache_policy import KV_ATTN_TYPES, allocate_kv_from_tensors, get_layer_types
from vllm_webgpu.webgpu.pipeline import PipelineCache



if TYPE_CHECKING:
    from collections.abc import Sequence
    from vllm.tasks import SupportedTask
    from vllm_webgpu.models.base import BaseWebGPUModel
    from vllm_webgpu.webgpu.device import WebGPUDevice
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheSpec

logger = init_logger(__name__)

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


def _build_model(arch: str, model_config: Any, wgpu_device: Any, pipeline_cache: Any, block_size: int) -> "BaseWebGPUModel":
    family = ARCH_MAP.get(arch)
    if family == "llama":
        from vllm_webgpu.models.llama import LlamaWebGPUModel
        return LlamaWebGPUModel(model_config, wgpu_device, pipeline_cache, block_size=block_size)
    if family == "mixtral":
        from vllm_webgpu.models.mixtral import MixtralWebGPUModel
        return MixtralWebGPUModel(model_config, wgpu_device, pipeline_cache, block_size=block_size)
    if family == "gemma4":
        from vllm_webgpu.models.gemma4 import Gemma4WebGPUModel
        return Gemma4WebGPUModel(model_config, wgpu_device, pipeline_cache, block_size=block_size)
    if family == "qwen35":
        from vllm_webgpu.models.qwen35 import Qwen35WebGPUModel
        return Qwen35WebGPUModel(model_config, wgpu_device, pipeline_cache, block_size=block_size)
    if family == "diffusion_gemma":
        from vllm_webgpu.models.diffusion_gemma import DiffusionGemmaWebGPUModel
        return DiffusionGemmaWebGPUModel(model_config, wgpu_device, pipeline_cache, block_size=block_size)
    if family == "gpt_oss":
        from vllm_webgpu.models.gpt_oss import GptOssWebGPUModel
        return GptOssWebGPUModel(model_config, wgpu_device, pipeline_cache, block_size=block_size)
    if family == "nemotron_h":
        from vllm_webgpu.models.nemotron_h import NemotronHWebGPUModel
        return NemotronHWebGPUModel(model_config, wgpu_device, pipeline_cache, block_size=block_size)
    raise NotImplementedError(
        f"Architecture {arch!r} is not supported. "
        f"Supported: {sorted(ARCH_MAP)}"
    )


class WebGPUModelRunner:
    def __init__(self, vllm_config: Any, wgpu_device: "WebGPUDevice") -> None:
        self.vllm_config = vllm_config
        self.wgpu_device = wgpu_device
        self.pipeline_cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
        self.model: "BaseWebGPUModel | None" = None
        self._last_model_output: Any = EMPTY_MODEL_RUNNER_OUTPUT  # cached for sample_tokens()
        self._req_state: dict[str, Any] = {}  # per-request decode state {req_id: {pos, block_ids}}
        self._num_kv_blocks: int = 0  # set by initialize_kv_cache; used by _zero_kv_blocks
        self._zeros_cache: dict[int, bytearray] = {}  # amortizes zero-byte alloc across scheduling steps
        self._block_size: int = vllm_config.cache_config.block_size

    def load_model(self) -> None:
        mc = self.vllm_config.model_config
        arch = (mc.architectures or ["LlamaForCausalLM"])[0]
        hf_config = mc.hf_config

        block_size = self._block_size

        # NemotronH Mamba conv state buffers are sized for num_spec=0:
        # each buffer holds (conv_kernel - 1) * conv_dim slots (f16), matching
        # the mamba2_causal_conv WGSL shader's ring-buffer layout. With
        # speculative decoding, vLLM core sizes conv states to
        # (conv_kernel - 1 + num_spec) slots. Loading such a state snapshot
        # into the undersized WebGPU buffer would produce a silent size mismatch
        # and corrupt the SSM state. Fail fast until the WebGPU conv state
        # allocation is updated to account for num_spec.
        if ARCH_MAP.get(arch) == "nemotron_h":
            num_spec = self.vllm_config.num_speculative_tokens
            if num_spec:
                raise NotImplementedError(
                    f"NemotronHWebGPUModel does not support speculative decoding "
                    f"(num_speculative_tokens={num_spec}). The Mamba conv state "
                    f"buffers are sized for num_spec=0 (conv_kernel - 1 slots). "
                    f"Update _init_mamba_states to pass num_spec to "
                    f"MambaStateShapeCalculator.mamba2_state_shape before enabling."
                )

        self.model = _build_model(arch, hf_config, self.wgpu_device, self.pipeline_cache, block_size=block_size)
        self.model.load_weights(mc.model)
        logger.info("Model loaded: arch=%s", arch)

    def initialize_kv_cache(self, kv_cache_config: Any) -> None:
        mc = self.vllm_config.model_config
        num_blocks = kv_cache_config.num_blocks
        self._num_kv_blocks = num_blocks

        allocate_kv_from_tensors(
            self.wgpu_device.wgpu_device,
            self.model,
            kv_cache_config.kv_cache_tensors,
            num_blocks=num_blocks,
            num_total_layers=mc.hf_config.num_hidden_layers,
            kv_cache_groups=kv_cache_config.kv_cache_groups,
        )

    def _get_lp_list(self) -> "list | None":
        """Return per-layer attention params, guarding against model=None.

        Returns None when no per-layer params exist. Falls back to the HF
        config attribute only when the model has not set _lp at all (None),
        not when it is explicitly set to [] — an empty list means the model
        has confirmed there are no heterogeneous layers, and that signal must
        not be overridden by a stale HF config attribute.
        """
        mc = self.vllm_config.model_config.hf_config
        lp = getattr(self.model, "_lp", None) if self.model is not None else None
        return lp if lp is not None else getattr(mc, "_layer_attention_params", None)

    def get_kv_cache_spec(self) -> "dict[str, KVCacheSpec]":
        mc = self.vllm_config.model_config.hf_config
        num_hidden_layers = self.vllm_config.model_config.get_total_num_hidden_layers()
        block_size = self._block_size
        spec: dict[str, Any] = {}

        def _make_spec(num_kv_heads: int, head_size: int) -> Any:
            return FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                dtype=_KV_DTYPE,
            )

        # Use per-layer params if available (Gemma4 heterogeneous layers).
        # Prefer the model object's _lp list (populated from layer_types config)
        # over the raw HF config attribute, which may not be set for safetensors.
        lp_list = self._get_lp_list()

        # Fallback: derive per-layer KV spec from layer_types + global_head_dim when
        # the model has not been loaded yet and hf_config lacks _layer_attention_params.
        # This covers Gemma4 safetensors where vLLM may call get_kv_cache_spec()
        # before load_model(), so self.model is still None. Mirrors the derivation
        # in Gemma4WebGPUModel.__init__() to ensure uniform and per-layer specs agree.
        #
        # Only layers whose type appears in ATTN_TYPES get a KV cache entry.
        # Mamba, MLP, and linear-attention layers carry no KV state and must be
        # excluded — emitting a FullAttentionSpec for them over-reports KV memory.
        # NemotronH attention layers live under .mixer, not .self_attn.
        _archs = getattr(mc, "architectures", None) or []
        _attn_suffix = ".mixer" if ARCH_MAP.get((_archs or [""])[0]) == "nemotron_h" else ".self_attn"
        _layer_types = get_layer_types(None, self.vllm_config.model_config.hf_text_config)

        if not lp_list:
            if _layer_types and len(_layer_types) == num_hidden_layers:
                default_hd = self.vllm_config.model_config.get_head_size()
                default_kv = self.vllm_config.model_config.get_total_num_kv_heads()
                global_hd = getattr(mc, "global_head_dim", default_hd)
                global_kv = getattr(mc, "num_global_key_value_heads", None) or default_kv
                for i, lt in enumerate(_layer_types):
                    if lt not in KV_ATTN_TYPES:
                        continue
                    if lt == "full_attention":
                        spec[f"model.layers.{i}{_attn_suffix}"] = _make_spec(global_kv, global_hd)
                    else:
                        spec[f"model.layers.{i}{_attn_suffix}"] = _make_spec(default_kv, default_hd)
                return spec

        if lp_list and len(lp_list) == num_hidden_layers:
            for i, lp in enumerate(lp_list):
                if _layer_types and len(_layer_types) == num_hidden_layers and _layer_types[i] not in KV_ATTN_TYPES:
                    continue
                if lp["num_kv_heads"] == 0:
                    # Non-attention layer: skip regardless of _layer_types to
                    # avoid emitting a zero-page-size FullAttentionSpec.
                    continue
                spec[f"model.layers.{i}{_attn_suffix}"] = _make_spec(
                    lp["num_kv_heads"], lp["head_dim"])
        else:
            head_size = self.vllm_config.model_config.get_head_size()
            num_kv_heads = self.vllm_config.model_config.get_total_num_kv_heads()
            # Only trust layer_types when it covers every layer; a partial or
            # mismatched list (including a stray MagicMock in tests) falls back
            # to the uniform path so all layers get a spec entry.
            lt_filtered = _layer_types if _layer_types and len(_layer_types) == num_hidden_layers else None
            for i in range(num_hidden_layers):
                if lt_filtered is not None and lt_filtered[i] not in KV_ATTN_TYPES:
                    continue
                spec[f"model.layers.{i}{_attn_suffix}"] = _make_spec(
                    num_kv_heads, head_size)
        return spec

    def get_cache_block_size_bytes(self) -> int:
        specs = self.get_kv_cache_spec()
        return sum(s.page_size_bytes for s in specs.values())

    def warm_up(self) -> None:
        if self.model is not None:
            self.model.warmup()

    def _zero_kv_blocks(self, block_ids: list[int]) -> None:
        """Zero the KV cache entries for recycled block IDs.

        When the block pool reuses a block from a completed request, stale K/V
        values remain in the buffer until the new request's forward pass writes
        to those positions. Attention over positions beyond the current write
        cursor would read that garbage, producing incorrect outputs.

        This mirrors gpu_model_runner.py _zero_block_ids (lines 1105-1108).
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
        _zeros_cache = self._zeros_cache
        for k_buf, v_buf in self.model.kv_pool:
            if k_buf.nbytes <= 16:
                # 16-byte placeholder for non-attention layers (Mamba, MLP-only, etc.)
                continue
            bytes_per_block = k_buf.nbytes // self._num_kv_blocks
            if bytes_per_block not in _zeros_cache:
                _zeros_cache[bytes_per_block] = bytearray(bytes_per_block)
            zeros = _zeros_cache[bytes_per_block]
            bytes_per_block_v = v_buf.nbytes // self._num_kv_blocks
            if bytes_per_block_v not in _zeros_cache:
                _zeros_cache[bytes_per_block_v] = bytearray(bytes_per_block_v)
            zeros_v = _zeros_cache[bytes_per_block_v]
            for block_id in block_ids:
                offset = block_id * bytes_per_block
                queue.write_buffer(k_buf.buf, offset, zeros)
                offset_v = block_id * bytes_per_block_v
                queue.write_buffer(v_buf.buf, offset_v, zeros_v)

    def execute_model(self, scheduler_output: "SchedulerOutput") -> None:
        if scheduler_output.has_structured_output_requests:
            raise NotImplementedError(
                "Guided/constrained decoding is not supported on the WebGPU backend. "
                "The WebGPU argmax path discards the logit distribution required for "
                "grammar token masks. Use unconstrained sampling or a CPU/CUDA backend."
            )
        if self.model is None:
            self._last_model_output = EMPTY_MODEL_RUNNER_OUTPUT
            return None
        self._last_model_output = self._execute_model_v2(scheduler_output)
        return None

    @staticmethod
    def _compute_request_logprobs(
        logits_1d: "np.ndarray", sampled_tok: int, num_logprobs: int
    ) -> "LogprobsTensors":
        """Compute top-N logprobs from a 1-D float32 logits vector.

        Returns a LogprobsTensors of shape [1, min(num_logprobs, vocab_size)+1]
        for top-k requests (slot 0 is always the sampled token; slots 1..k are
        the top-k tokens by log probability, matching the layout expected by
        LogprobsLists). k is capped at vocab_size so the shape may be smaller
        than num_logprobs+1 for small-vocabulary models.
        """
        k = min(num_logprobs, logits_1d.shape[0])
        lp_t = Sampler.compute_logprobs(torch.from_numpy(logits_1d).unsqueeze(0))
        lp = Sampler.gather_logprobs(lp_t, k, torch.tensor([sampled_tok], dtype=torch.int64))
        return lp

    @staticmethod
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

        num_positions = T - 1

        # Guard: the model may return only the last token's logits (shape
        # [1, vocab]) even during a multi-token prefill.  In that case we
        # cannot reconstruct per-position distributions and must bail out
        # rather than letting the subsequent row-index into a 1-row array
        # raise IndexError.
        if full_logits.shape[0] < num_positions:
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

        lp_t = Sampler.compute_logprobs(torch.from_numpy(full_logits[:num_positions]))
        lp = Sampler.gather_logprobs(lp_t, k, torch.tensor(tok_ids[1:], dtype=torch.int64))
        return lp

    def _make_model_output(
        self,
        req_ids: list[str],
        sampled: list[int],
        logprobs_data: "Sequence[LogprobsTensors | None]" = (),
        prompt_logprobs_dict: "dict[str, LogprobsTensors] | None" = None,
    ) -> Any:
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
        merged_prompt_logprobs = prompt_logprobs_dict or {}
        has_topk = any(d is not None for d in logprobs_data)
        if has_topk:
            _widths = {d.logprob_token_ids.shape[1] for d in logprobs_data if d is not None}
            max_k = max(_widths)
            # Short-circuit when all real entries have the same width: skip padding.
            if len(_widths) == 1 and all(d is not None for d in logprobs_data):
                stacked = LogprobsTensors(
                    torch.cat([d.logprob_token_ids for d in logprobs_data]),
                    torch.cat([d.logprobs for d in logprobs_data]),
                    torch.cat([d.selected_token_ranks for d in logprobs_data]),
                )
                built_logprobs = stacked.tolists()
            else:
                pieces = []
                for d in logprobs_data:
                    if d is not None:
                        pad = max_k - d.logprob_token_ids.shape[1]
                        pieces.append(LogprobsTensors(
                            F.pad(d.logprob_token_ids, (0, pad), value=0),
                            F.pad(d.logprobs, (0, pad), value=-float("inf")),
                            d.selected_token_ranks,
                        ))
                    else:
                        pieces.append(LogprobsTensors(
                            torch.zeros(1, max_k, dtype=torch.int32),
                            torch.full((1, max_k), -float("inf")),
                            torch.zeros(1, dtype=torch.int64),
                        ))
                stacked = LogprobsTensors(
                    torch.cat([p.logprob_token_ids for p in pieces]),
                    torch.cat([p.logprobs for p in pieces]),
                    torch.cat([p.selected_token_ranks for p in pieces]),
                )
                built_logprobs = stacked.tolists()

        out = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=[[t] for t in sampled],
            logprobs=built_logprobs,
            prompt_logprobs_dict=merged_prompt_logprobs,
        )
        return out

    def _extract_logprob_data(
        self,
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
        if logits.shape[-1] == 1 and self.model.logit_returns_token_id:
            full = self.model.logit_readback()
        elif logits.shape[-1] > 1:
            full = logits
        else:
            full = None
            logger.warning("req %s: logprobs requested but model does not support logit readback", rid)
        if full is None:
            return None
        return self._compute_request_logprobs(full[row_idx], tok, num_logprobs)

    def _execute_model_v2(self, scheduler_output: "SchedulerOutput") -> Any:
        """vLLM >= 0.24 SchedulerOutput format."""
        # Zero recycled KV blocks before any forward pass. The block pool may
        # reuse blocks from completed requests; without zeroing, attention over
        # positions beyond the current write cursor reads stale K/V values and
        # produces incorrect outputs. The GPU runner does the same via
        # _zero_block_ids (gpu_model_runner.py lines 1155-1156).
        if scheduler_output.new_block_ids_to_zero:
            self._zero_kv_blocks(scheduler_output.new_block_ids_to_zero)

        # Prune state for requests that completed in the previous step.
        for rid in scheduler_output.finished_req_ids:
            self._req_state.pop(rid, None)

        cached = scheduler_output.scheduled_cached_reqs
        new_reqs = scheduler_output.scheduled_new_reqs
        block_size = self._block_size

        all_req_ids: list[str] = []
        all_sampled: list[int] = []
        all_logprobs_data: list = []  # per-request logprob tuples or None
        prompt_logprobs_dict: dict[str, Any] = {}  # req_id -> LogprobsTensors for prefill

        # ── Prefill: new requests ──────────────────────────────────────────────
        for req in new_reqs:
            rid = req.req_id
            tok_ids = req.prompt_token_ids or []
            if not tok_ids:
                continue

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

            # Extract per-request logprob counts from SamplingParams.
            sp = req.sampling_params
            if sp is not None and sp.logprob_token_ids:
                raise NotImplementedError(
                    f"req {rid}: logprob_token_ids is not supported on the WebGPU backend; "
                    "use logprobs=N instead"
                )
            num_logprobs = sp.num_logprobs if sp is not None else None
            if num_logprobs == -1:
                raise NotImplementedError(
                    f"req {rid}: logprobs=-1 (full-vocab) is not supported on the WebGPU backend; "
                    "use a positive integer instead"
                )
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
            assert raw_bids, f"req {rid}: scheduler produced NewRequestData with empty block_ids"
            blk_ids = list(itertools.chain.from_iterable(raw_bids))

            bt = np.array(blk_ids, dtype=np.uint32)

            # Batch prefill: send the scheduled chunk of prompt tokens in a single
            # forward() call. With prefix caching, num_computed_tokens tokens are
            # already in the KV cache; only the uncached tail needs to be processed.
            num_computed = req.num_computed_tokens
            num_sched = scheduler_output.num_scheduled_tokens[rid]
            T = min(num_sched, len(tok_ids) - num_computed)
            chunk_toks = tok_ids[num_computed:num_computed + T]
            slots = []
            for idx in range(T):
                abs_idx = num_computed + idx  # absolute token position
                blk_idx = abs_idx // block_size
                if blk_idx >= len(blk_ids):
                    raise RuntimeError(
                        f"block table too short for req {rid}: token {abs_idx} needs block "
                        f"{blk_idx} but only {len(blk_ids)} blocks allocated"
                    )
                slots.append(int(blk_ids[blk_idx]) * block_size + abs_idx % block_size)

            _batch_pm = SimpleNamespace(slot_mapping=slots, block_tables=[bt], max_decode_seq_len=num_computed + T)

            self.model._greedy_decode = (sp is None or sp.sampling_type == SamplingType.GREEDY) and num_logprobs is None

            # Each prefill request starts from zero recurrent state. Reset here
            # (inside the loop) so that multiple new requests in the same step
            # each get a clean slate rather than inheriting the previous request's
            # post-prefill state.
            if hasattr(self.model, "reset_recurrent_states"):
                self.model.reset_recurrent_states()

            last_logits = self.model.forward(
                np.array(chunk_toks, dtype=np.uint32),
                np.arange(num_computed, num_computed + T, dtype=np.uint32),
                _batch_pm,
            )

            # Save recurrent state so the decode path can restore it before this
            # request's first (and every subsequent) decode step.
            prefill_recurrent_states = None
            if hasattr(self.model, "save_recurrent_states"):
                prefill_recurrent_states = self.model.save_recurrent_states()

            if last_logits is None:
                # Mid-prefill chunk: model produced no output logits yet (e.g.
                # chunked prefill where this is not the final chunk). Register
                # partial state so the decode loop does not crash with a missing
                # key if the scheduler promotes this request to cached_reqs
                # before full prefill completes. The last input token is used as
                # a sentinel; it will be overwritten when the final chunk runs.
                self._req_state[rid] = {
                    "pos": num_computed + T,
                    "block_ids": blk_ids,
                    "last_tok": int(chunk_toks[-1]) if len(chunk_toks) > 0 else 0,
                    "num_logprobs": num_logprobs,
                    "sampling_params": sp,
                    "recurrent_states": prefill_recurrent_states,
                }
                continue

            # Use the last position's logits for the first generated token.
            # Apply sampling when SamplingParams request non-greedy decoding.
            if last_logits.shape[-1] > 1:
                first_decode_tok = (
                    int(np.argmax(last_logits[-1])) if sp is None
                    else _sample_token(last_logits[-1], temperature=sp.temperature,
                                       top_p=sp.top_p, top_k=sp.top_k, seed=sp.seed)
                )
            else:
                first_decode_tok = int(last_logits[0, 0])

            # Compute logprobs for this prefill token if the request asked for them.
            lp_data = self._extract_logprob_data(last_logits, -1, first_decode_tok, num_logprobs, rid)

            # Compute prompt logprobs for each prompt position when full logits
            # are available.  Position i uses logits[i] to evaluate tok_ids[i+1],
            # producing T-1 rows of top-K logprob data.
            if num_prompt_logprobs is not None and T >= 1:
                if last_logits.shape[-1] > 1:  # full [T, vocab] logits
                    # Pass only the token window so full_logits[i] and
                    # tok_ids_param[i+1] stay aligned regardless of num_computed.
                    pt = self._compute_prompt_logprobs(
                        last_logits,
                        tok_ids[num_computed:num_computed + T + 1],
                        num_prompt_logprobs,
                    )
                    if pt is not None:
                        prompt_logprobs_dict[rid] = pt
                else:
                    logger.warning(
                        "req %s: prompt_logprobs requested but model returns argmax-only "
                        "logits; prompt logprobs cannot be computed",
                        rid,
                    )

            all_req_ids.append(rid)
            all_sampled.append(first_decode_tok)
            all_logprobs_data.append(lp_data)
            # Store last sampled token; decode path needs it (new_token_ids is empty without PP).
            self._req_state[rid] = {
                "pos": num_computed + T, "block_ids": blk_ids,
                "last_tok": first_decode_tok, "num_logprobs": num_logprobs,
                "sampling_params": sp,
                "recurrent_states": prefill_recurrent_states,
            }

        # ── Decode: cached requests ────────────────────────────────────────────
        # new_token_ids is empty without pipeline parallelism (vLLM design).
        # Use last_tok stored in _req_state from the previous step instead.
        #
        # Multi-sequence batching is not supported: one forward() call per request.
        # Why true batching can't be done without architecture changes:
        #   - queue.write_buffer() executes before submit, so all N writes to the
        #     same pre-allocated buffers would alias — only the last request's data
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
                blk_ids = list(state["block_ids"])
                num_logprobs = state.get("num_logprobs")

                # Update block table: preempted/resumed requests replace their
                # block table entirely; others append newly allocated blocks.
                cur_new_bids = new_block_ids[i]
                if cur_new_bids is not None:
                    flat_new = list(itertools.chain.from_iterable(cur_new_bids))
                    if rid in resumed_req_ids:
                        blk_ids = flat_new
                        # Realign pos with the scheduler's authoritative view.
                        # After preemption num_computed_tokens is often 0 (full
                        # recompute); using the stale _req_state pos would write
                        # into the wrong block-table slot and skip uninitialized
                        # KV slots 0..old_pos-1 in the freshly allocated blocks.
                        pos = cached.num_computed_tokens[i]
                    else:
                        blk_ids.extend(flat_new)

                # Decode step: forward one token at the current position.
                tok = state["last_tok"]

                if pos // block_size >= len(blk_ids):
                    raise RuntimeError(
                        f"block table too short for req {rid}: pos={pos} needs block "
                        f"{pos // block_size} but only {len(blk_ids)} blocks allocated"
                    )
                slot = int(blk_ids[pos // block_size]) * block_size + pos % block_size

                _sm = SimpleNamespace(slot_mapping=[slot], block_tables=[np.array(blk_ids, dtype=np.uint32)], max_decode_seq_len=pos + 1)

                sp = state.get("sampling_params")
                self.model._greedy_decode = (sp is None or sp.sampling_type == SamplingType.GREEDY) and num_logprobs is None

                # Restore this request's recurrent (Mamba/SSM) state before the
                # forward pass. Without this, each request in the batch reads the
                # in-place-updated state left by the previous request instead of
                # its own saved state, producing wrong recurrent outputs for every
                # request beyond the first in a multi-sequence decode batch.
                if hasattr(self.model, "restore_recurrent_states"):
                    saved_recurrent = state.get("recurrent_states")
                    rolled_back = rid in resumed_req_ids and pos < state["pos"]
                    if not rolled_back and saved_recurrent is not None:
                        self.model.restore_recurrent_states(saved_recurrent)
                    elif hasattr(self.model, "reset_recurrent_states"):
                        # Reset Mamba conv/SSM states to zero before decoding.
                        self.model.reset_recurrent_states()
                        if rolled_back and pos > 0:
                            # KNOWN WRONG-OUTPUT CONDITION: the request was preempted
                            # and resumed at position pos > 0 (prefix caching preserved
                            # pos tokens in the KV cache for attention layers). The SSM
                            # state has been reset to zero, but the attention KV cache
                            # already reflects pos tokens of history. All subsequent
                            # Mamba layer outputs are wrong: they decode from state=0
                            # while the residual stream is at sequence position pos.
                            #
                            # Correct fix: replay input tokens 0..pos-1 one at a time
                            # through the Mamba layers (without writing to the KV cache,
                            # which is already populated) to reconstruct the SSM state
                            # before the first decode step. This requires the full token
                            # sequence for positions 0..pos-1 to be stored in _req_state,
                            # which is not currently tracked. Until that is implemented,
                            # outputs for this request after resumption are silently
                            # corrupted for every token decoded past the rollback point.
                            logger.warning(
                                "req %s: preempted and resumed at pos=%d with prefix-cached "
                                "KV (attention sees %d tokens of history). Mamba SSM state "
                                "was reset to zero and cannot be reconstructed without the "
                                "full token history (not stored). All Mamba layer outputs "
                                "for this request are wrong until a new prefill runs.",
                                rid, pos, pos,
                            )

                logits = self.model.forward(
                    np.array([tok], dtype=np.uint32),
                    np.array([pos], dtype=np.uint32),
                    _sm,
                )

                # Save recurrent state immediately after the forward pass, before
                # any other request's forward can overwrite the shared GPU buffers.
                decode_recurrent_states = None
                if hasattr(self.model, "save_recurrent_states"):
                    decode_recurrent_states = self.model.save_recurrent_states()

                if logits is None:
                    continue

                # Greedy path: model returns (1, 1) int32 with the argmax index.
                # Non-greedy path: model returns (1, vocab) float32; sample here.
                if logits.shape[-1] == 1:
                    stok = int(logits[0, 0])
                else:
                    stok = (
                        int(np.argmax(logits[0])) if sp is None
                        else _sample_token(logits[0], temperature=sp.temperature,
                                           top_p=sp.top_p, top_k=sp.top_k, seed=sp.seed)
                    )

                # Compute logprobs if requested for this request.
                lp_data = self._extract_logprob_data(logits, 0, stok, num_logprobs, rid)

                # Commit state after a successful forward — don't mutate on failure.
                self._req_state[rid] = {
                    "pos": pos + 1, "block_ids": blk_ids,
                    "last_tok": stok, "num_logprobs": num_logprobs,
                    "sampling_params": sp,
                    "recurrent_states": decode_recurrent_states,
                }
                all_req_ids.append(rid)
                all_sampled.append(stok)
                all_logprobs_data.append(lp_data)

        # Return empty output rather than None when no requests scheduled.
        # vLLM's batch queue raises "unexpected error" on None from execute_model.
        return self._make_model_output(
            all_req_ids, all_sampled, all_logprobs_data, prompt_logprobs_dict
        )

    def sample_tokens(self, grammar_output: "GrammarOutput | None") -> Any:
        # In vLLM >= 0.24, the batch queue calls execute_model() then sample_tokens().
        # execute_model() caches its output; sample_tokens() returns it here.
        # Grammar/structured output (guided_json, guided_regex, guided_grammar) is not
        # supported: the WebGPU backend performs argmax on-GPU and does not preserve
        # the full logit distribution needed to apply token masks from the grammar FSM.
        # Silently returning unconstrained tokens would produce output that violates
        # the schema, so we raise early instead.
        if grammar_output is not None:
            raise NotImplementedError(
                "Guided/constrained decoding (guided_json, guided_regex, guided_grammar) "
                "is not supported on the WebGPU backend. The GPU argmax path discards "
                "the full logit distribution required to apply grammar token masks. "
                "Use unconstrained sampling or switch to a CPU/CUDA backend."
            )
        return self._last_model_output

    def get_supported_tasks(self) -> "tuple[SupportedTask, ...]":
        return ("generate",)


