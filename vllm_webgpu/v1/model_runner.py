from __future__ import annotations
import itertools
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.outputs import ModelRunnerOutput, LogprobsLists, LogprobsTensors, EMPTY_MODEL_RUNNER_OUTPUT
except ImportError:
    FullAttentionSpec = None  # type: ignore[assignment,misc]
    ModelRunnerOutput = None  # type: ignore[assignment,misc]
    LogprobsLists = None  # type: ignore[assignment,misc]
    LogprobsTensors = None  # type: ignore[assignment,misc]
    EMPTY_MODEL_RUNNER_OUTPUT = None  # type: ignore[assignment,misc]

try:
    from vllm.v1.sample.sampler import Sampler
except ImportError:
    Sampler = None  # type: ignore[assignment,misc]

from vllm.logger import init_logger
from vllm_webgpu.config import get_config
from vllm_webgpu.utils import SHADERS_DIR, sample_token as _sample_token
from vllm_webgpu.v1.cache_policy import KV_ATTN_TYPES, allocate_kv_from_hf_config
from vllm_webgpu.webgpu.pipeline import PipelineCache



if TYPE_CHECKING:
    from vllm.tasks import SupportedTask
    from vllm_webgpu.models.base import BaseWebGPUModel
    from vllm_webgpu.webgpu.device import WebGPUDevice
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheSpec

logger = init_logger(__name__)

def _is_greedy(sp) -> bool:
    """Return True when sampling params request greedy (argmax) decoding."""
    return sp is None or sp.temperature < 1e-5


def _sample_logits(logits_1d: "np.ndarray", sp) -> int:
    """Sample one token from a 1-D float32 logit vector using SamplingParams."""
    if _is_greedy(sp):
        return int(np.argmax(logits_1d))
    seed = getattr(sp, "seed", None)
    return _sample_token(
        logits_1d,
        temperature=sp.temperature,
        top_p=sp.top_p,
        top_k=sp.top_k,
        seed=seed,
    )


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


def _build_model(arch: str, model_config: Any, wgpu_device: Any, pipeline_cache: Any) -> "BaseWebGPUModel":
    family = ARCH_MAP.get(arch)
    if family == "llama":
        from vllm_webgpu.models.llama import LlamaWebGPUModel
        return LlamaWebGPUModel(model_config, wgpu_device, pipeline_cache)
    if family == "mixtral":
        from vllm_webgpu.models.mixtral import MixtralWebGPUModel
        return MixtralWebGPUModel(model_config, wgpu_device, pipeline_cache)
    if family == "gemma4":
        from vllm_webgpu.models.gemma4 import Gemma4WebGPUModel
        return Gemma4WebGPUModel(model_config, wgpu_device, pipeline_cache)
    if family == "qwen35":
        from vllm_webgpu.models.qwen35 import Qwen35WebGPUModel
        return Qwen35WebGPUModel(model_config, wgpu_device, pipeline_cache)
    if family == "diffusion_gemma":
        from vllm_webgpu.models.diffusion_gemma import DiffusionGemmaWebGPUModel
        return DiffusionGemmaWebGPUModel(model_config, wgpu_device, pipeline_cache)
    if family == "gpt_oss":
        from vllm_webgpu.models.gpt_oss import GptOssWebGPUModel
        return GptOssWebGPUModel(model_config, wgpu_device, pipeline_cache)
    if family == "nemotron_h":
        from vllm_webgpu.models.nemotron_h import NemotronHWebGPUModel
        return NemotronHWebGPUModel(model_config, wgpu_device, pipeline_cache)
    raise NotImplementedError(
        f"Architecture {arch!r} is not supported. "
        f"Supported: {sorted(ARCH_MAP)}"
    )


class WebGPUModelRunner:
    def __init__(self, vllm_config: Any, wgpu_device: "WebGPUDevice") -> None:
        self.vllm_config = vllm_config
        self.wgpu_device = wgpu_device
        self.webgpu_config = get_config()
        self.pipeline_cache = PipelineCache(wgpu_device.wgpu_device, SHADERS_DIR)
        self.model: "BaseWebGPUModel | None" = None
        self._last_model_output: Any = EMPTY_MODEL_RUNNER_OUTPUT  # cached for sample_tokens()
        self._req_state: dict[str, Any] = {}  # per-request decode state {req_id: {pos, block_ids}}

    def load_model(self) -> None:
        mc = self.vllm_config.model_config
        arch = (mc.architectures or ["LlamaForCausalLM"])[0]
        hf_config = mc.hf_config

        self.model = _build_model(arch, hf_config, self.wgpu_device, self.pipeline_cache)
        self.model.load_weights(mc.model)
        logger.info("Model loaded: arch=%s", arch)

    def initialize_kv_cache(self, kv_cache_config: Any) -> None:
        mc = self.vllm_config.model_config
        hf = mc.hf_config
        block_size = self.webgpu_config.block_size
        num_blocks = kv_cache_config.num_blocks

        allocate_kv_from_hf_config(
            self.wgpu_device.wgpu_device,
            self.model,
            hf,
            num_blocks=num_blocks,
            block_size=block_size,
            model_config=mc,
        )

    def _get_lp_list(self) -> "list | None":
        """Return per-layer attention params, guarding against model=None.

        Uses explicit None checks rather than `or` so that an empty list
        (a valid "no heterogeneous layers" signal) is not treated as falsy
        and silently replaced by the hf_config fallback.
        """
        mc = self.vllm_config.model_config.hf_config
        lp = getattr(self.model, "_lp", None) if self.model is not None else None
        return lp if lp is not None else getattr(mc, "_layer_attention_params", None)

    def get_kv_cache_spec(self) -> "dict[str, KVCacheSpec]":
        mc = self.vllm_config.model_config.hf_config
        block_size = self.webgpu_config.block_size
        spec: dict[str, Any] = {}
        if FullAttentionSpec is None:
            return spec

        _dtype = torch.float16

        def _make_spec(num_kv_heads: int, head_size: int) -> Any:
            return FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                dtype=_dtype,
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
        _attn_suffix = ".mixer" if "NemotronHForCausalLM" in _archs else ".self_attn"
        _layer_types = getattr(mc, "layer_types", None) or getattr(mc, "layers_block_type", None)

        if not lp_list:
            layer_types = _layer_types
            if layer_types and len(layer_types) == mc.num_hidden_layers:
                default_hd = getattr(mc, "head_dim", mc.hidden_size // mc.num_attention_heads)
                default_kv = getattr(mc, "num_key_value_heads", 1)
                global_hd = getattr(mc, "global_head_dim", default_hd)
                global_kv = getattr(mc, "num_global_key_value_heads", None) or default_kv
                for i, lt in enumerate(layer_types):
                    if lt not in KV_ATTN_TYPES:
                        continue
                    if lt == "full_attention":
                        spec[f"model.layers.{i}{_attn_suffix}"] = _make_spec(global_kv, global_hd)
                    else:
                        spec[f"model.layers.{i}{_attn_suffix}"] = _make_spec(default_kv, default_hd)
                return spec

        if lp_list and len(lp_list) == mc.num_hidden_layers:
            for i, lp in enumerate(lp_list):
                if _layer_types and _layer_types[i] not in KV_ATTN_TYPES:
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
            _lt2 = _layer_types
            # Only trust layer_types when it covers every layer; a partial or
            # mismatched list (including a stray MagicMock in tests) falls back
            # to the uniform path so all layers get a spec entry.
            if _lt2 and len(_lt2) != mc.num_hidden_layers:
                _lt2 = None
            for i in range(mc.num_hidden_layers):
                if _lt2 is not None and _lt2[i] not in KV_ATTN_TYPES:
                    continue
                spec[f"model.layers.{i}{_attn_suffix}"] = _make_spec(
                    num_kv_heads, head_size)
        return spec

    def get_cache_block_size_bytes(self) -> int:
        mc = self.vllm_config.model_config.hf_config
        block_size = self.webgpu_config.block_size
        head_dim = self.vllm_config.model_config.get_head_size()
        num_kv_heads = self.vllm_config.model_config.get_total_num_kv_heads()
        # Use the maximum per-layer values when heterogeneous layer params are available
        # (e.g. Gemma4 models with mixed local/global attention dimensions).
        lp_list = self._get_lp_list()
        if lp_list:
            head_dim = max((lp["head_dim"] for lp in lp_list), default=head_dim)
            num_kv_heads = max((lp["num_kv_heads"] for lp in lp_list), default=num_kv_heads)
        if FullAttentionSpec is not None:
            return FullAttentionSpec(
                block_size=block_size, num_kv_heads=num_kv_heads,
                head_size=head_dim, dtype=torch.float16,
            ).page_size_bytes
        return block_size * num_kv_heads * head_dim * 2 * 2  # K + V, f16 fallback

    def warm_up(self) -> None:
        if self.model is not None:
            self.model.warmup()

    def execute_model(self, scheduler_output: "SchedulerOutput") -> None:
        if getattr(scheduler_output, "has_structured_output_requests", False):
            raise NotImplementedError(
                "Guided/constrained decoding is not supported on the WebGPU backend. "
                "The WebGPU argmax path discards the logit distribution required for "
                "grammar token masks. Use unconstrained sampling or a CPU/CUDA backend."
            )
        if self.model is None:
            self._last_model_output = EMPTY_MODEL_RUNNER_OUTPUT
            return None
        try:
            self._last_model_output = self._execute_model_v2(scheduler_output)
        except Exception as e:
            logger.exception("execute_model failed: %s", e)
            raise
        return None

    @staticmethod
    def _compute_request_logprobs(
        logits_1d: "np.ndarray", sampled_tok: int, num_logprobs: int
    ) -> "LogprobsTensors | None":
        """Compute top-N logprobs from a 1-D float32 logits vector.

        Returns a LogprobsTensors of shape [1, num_logprobs+1] for top-k
        requests (slot 0 is always the sampled token; slots 1..k are the
        top-k tokens by log probability, matching the layout expected by
        LogprobsLists), or None when logprobs cannot be computed.
        """
        if Sampler is None:
            return None
        vocab_size = logits_1d.shape[0]
        if num_logprobs == -1:
            # vLLM convention for -1: unsorted full-vocab distribution.
            # LogprobsLists has fixed width and cannot represent this; drop it
            # with a warning rather than silently returning wrong data.
            logger.warning(
                "full-vocab sampled logprobs (num_logprobs=-1) are not supported "
                "for decode steps and will be dropped; use a finite num_logprobs value"
            )
            return None
        k = min(num_logprobs, vocab_size)

        lp_t = Sampler.compute_logprobs(torch.from_numpy(logits_1d).unsqueeze(0))
        return Sampler.gather_logprobs(lp_t, k, torch.tensor([sampled_tok], dtype=torch.int64))

    @staticmethod
    def _compute_prompt_logprobs(
        full_logits: "np.ndarray",
        tok_ids: "list[int]",
        num_prompt_logprobs: int,
    ) -> "Any":
        """Compute per-position prompt logprobs for a prefill pass.

        For T prompt tokens, produces T-1 rows: row i uses full_logits[i]
        to evaluate the probability of tok_ids[i+1].  Returns a
        LogprobsTensors of shape [T-1, num_prompt_logprobs+1], or None when
        Sampler is unavailable or T < 2.
        """
        if Sampler is None:
            return None
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
        return Sampler.gather_logprobs(
            lp_t,
            k,
            torch.tensor(tok_ids[1:num_positions + 1], dtype=torch.int64),
        )

    def _make_model_output(
        self,
        req_ids: list[str],
        sampled: list[int],
        logprobs_data: "list | None" = None,
        prompt_logprobs_dict: "dict | None" = None,
    ) -> Any:
        if ModelRunnerOutput is None:
            return None

        if not req_ids:
            return EMPTY_MODEL_RUNNER_OUTPUT

        # Build LogprobsLists for top-k sampled-token logprob entries.
        # Each non-None entry in logprobs_data is a LogprobsTensors of shape
        # [1, k+1]. Stack them (preserving req alignment with placeholder rows
        # for requests that did not ask for logprobs), then call tolists() once.
        built_logprobs = None
        merged_prompt_logprobs = prompt_logprobs_dict or {}
        has_topk = logprobs_data and any(d is not None for d in logprobs_data)
        if LogprobsTensors is not None and has_topk:
            non_none = [d for d in logprobs_data if d is not None]
            max_k = max(d.logprob_token_ids.shape[1] for d in non_none)
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
                        torch.zeros((1, max_k), dtype=torch.int32),
                        torch.full((1, max_k), -float("inf"), dtype=torch.float32),
                        torch.zeros(1, dtype=torch.int32),
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
        if logits.shape[-1] == 1 and getattr(self.model, "logit_returns_token_id", False):
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
        if ModelRunnerOutput is None:
            return None

        # Prune state for requests that completed in the previous step.
        for rid in scheduler_output.finished_req_ids:
            self._req_state.pop(rid, None)

        cached = scheduler_output.scheduled_cached_reqs
        new_reqs = scheduler_output.scheduled_new_reqs
        block_size = self.webgpu_config.block_size

        all_req_ids: list[str] = []
        all_sampled: list[int] = []
        all_logprobs_data: list = []  # per-request logprob tuples or None
        prompt_logprobs_dict: dict[str, Any] = {}  # req_id -> LogprobsTensors for prefill

        # ── Prefill: new requests ──────────────────────────────────────────────
        # Reset recurrent state once before processing any new requests.
        # Doing this inside the loop would zero shared conv/SSM buffers after
        # the first request's prefill writes them, corrupting that request's
        # accumulated context when it enters decode on the next step.
        if new_reqs and hasattr(self.model, "reset_recurrent_states"):
            self.model.reset_recurrent_states()

        for req in new_reqs:
            rid = req.req_id
            tok_ids = list(req.prompt_token_ids or [])
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
            num_logprobs = getattr(sp, "num_logprobs", None) if sp is not None else None
            num_prompt_logprobs = getattr(sp, "prompt_logprobs", None) if sp is not None else None

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
            num_sched = scheduler_output.num_scheduled_tokens.get(rid, len(tok_ids))
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

            if hasattr(self.model, "_greedy_decode"):
                self.model._greedy_decode = _is_greedy(sp)

            last_logits = self.model.forward(
                np.array(chunk_toks, dtype=np.uint32),
                np.arange(num_computed, num_computed + T, dtype=np.uint32),
                _batch_pm,
            )

            if last_logits is None:
                continue

            # Use the last position's logits for the first generated token.
            # Apply sampling when SamplingParams request non-greedy decoding.
            if last_logits.shape[-1] > 1:
                first_decode_tok = _sample_logits(last_logits[-1], sp)
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
                state = self._req_state.get(rid, {"pos": 0, "block_ids": [], "last_tok": 0})
                pos = state["pos"]
                blk_ids = list(state.get("block_ids", []))
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
                if hasattr(self.model, "_greedy_decode"):
                    self.model._greedy_decode = _is_greedy(sp)

                logits = self.model.forward(
                    np.array([tok], dtype=np.uint32),
                    np.array([pos], dtype=np.uint32),
                    _sm,
                )

                if logits is None:
                    continue

                # Greedy path: model returns (1, 1) int32 with the argmax index.
                # Non-greedy path: model returns (1, vocab) float32; sample here.
                if logits.shape[-1] == 1:
                    stok = int(logits[0, 0])
                else:
                    stok = _sample_logits(logits[0], sp)

                # Compute logprobs if requested for this request.
                lp_data = self._extract_logprob_data(logits, 0, stok, num_logprobs, rid)

                # Commit state after a successful forward — don't mutate on failure.
                self._req_state[rid] = {
                    "pos": pos + 1, "block_ids": blk_ids,
                    "last_tok": stok, "num_logprobs": num_logprobs,
                    "sampling_params": state.get("sampling_params"),
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


