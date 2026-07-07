from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    from vllm.config import VllmConfig
    from vllm.tasks import SupportedTask
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec, FullAttentionSpec
    from vllm.v1.outputs import ModelRunnerOutput, SamplerOutput, LogprobsLists, LogprobsTensors
except ImportError:
    VllmConfig = Any  # type: ignore[assignment,misc]
    SupportedTask = Any  # type: ignore[assignment,misc]
    KVCacheConfig = Any  # type: ignore[assignment,misc]
    KVCacheSpec = Any  # type: ignore[assignment,misc]
    FullAttentionSpec = None  # type: ignore[assignment,misc]
    ModelRunnerOutput = None  # type: ignore[assignment,misc]
    SamplerOutput = None  # type: ignore[assignment,misc]
    LogprobsLists = None  # type: ignore[assignment,misc]
    LogprobsTensors = None  # type: ignore[assignment,misc]

from vllm_webgpu.config import get_config
from vllm_webgpu.utils import SHADERS_DIR
from vllm_webgpu.v1.cache_policy import WebGPUCachePlanner
from vllm_webgpu.webgpu.pipeline import PipelineCache

if TYPE_CHECKING:
    from vllm_webgpu.models.base import BaseWebGPUModel
    from vllm_webgpu.webgpu.device import WebGPUDevice
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput

logger = logging.getLogger(__name__)

def _is_greedy(sp) -> bool:
    """Return True when sampling params request greedy (argmax) decoding."""
    if sp is None:
        return True
    temp = float(getattr(sp, "temperature", 0.0) or 0.0)
    if temp > 1e-6:
        return False
    top_p = float(getattr(sp, "top_p", 1.0) or 1.0)
    top_k = int(getattr(sp, "top_k", -1) or -1)
    return top_p >= 1.0 and top_k <= 0


def _sample_logits(logits_1d: "np.ndarray", sp) -> int:
    """Sample one token from a 1-D float32 logit vector using SamplingParams.

    Applies temperature scaling, top-k, and top-p (nucleus) filtering in that
    order, then draws from the resulting categorical distribution.  Falls back
    to argmax when sp is None or the effective temperature is zero.
    """
    if sp is None or _is_greedy(sp):
        return int(np.argmax(logits_1d))

    temp = float(getattr(sp, "temperature", 1.0) or 1.0)
    top_p = float(getattr(sp, "top_p", 1.0) or 1.0)
    top_k = int(getattr(sp, "top_k", -1) or -1)

    # Temperature scaling with numerically stable softmax.
    scaled = logits_1d.astype(np.float32) / temp
    scaled -= scaled.max()
    probs = np.exp(scaled)
    probs /= probs.sum()

    # Top-k: zero out all tokens outside the top-k mass.
    if top_k > 0:
        k = min(top_k, len(probs))
        threshold = np.partition(probs, -k)[-k]
        probs = np.where(probs >= threshold, probs, 0.0)
        s = probs.sum()
        if s > 0:
            probs /= s

    # Top-p (nucleus): keep the smallest set of tokens whose cumulative
    # probability exceeds top_p.
    if 0.0 < top_p < 1.0:
        sorted_idx = np.argsort(probs)[::-1]
        cumsum = np.cumsum(probs[sorted_idx])
        # Include the first token that pushes cumsum over top_p.
        cutoff = max(1, int(np.searchsorted(cumsum, top_p, side="right")) + 1)
        keep = sorted_idx[:cutoff]
        mask = np.zeros_like(probs)
        mask[keep] = 1.0
        probs = probs * mask
        s = probs.sum()
        if s > 0:
            probs /= s

    return int(np.random.choice(len(probs), p=probs))


ARCH_MAP = {
    "LlamaForCausalLM": "llama",
    "MistralForCausalLM": "llama",
    "Qwen2ForCausalLM": "llama",
    "Qwen3ForCausalLM": "llama",
    "Gemma3ForCausalLM": "gemma4",
    "Gemma3ForConditionalGeneration": "gemma4",
    "Gemma4ForCausalLM": "gemma4",
    "Gemma4UnifiedForConditionalGeneration": "gemma4",
    "Qwen3_5ForConditionalGeneration": "qwen35",
    "Qwen3_5MoeForConditionalGeneration": "qwen35",  # MoE variant; FFN routing on GPU via topk_sort
    "DiffusionGemmaForBlockDiffusion": "diffusion_gemma",
}


def _build_model(arch: str, model_config: Any, wgpu_device: Any, pipeline_cache: Any) -> "BaseWebGPUModel":
    family = ARCH_MAP.get(arch)
    if family == "llama":
        from vllm_webgpu.models.llama import LlamaWebGPUModel
        return LlamaWebGPUModel(model_config, wgpu_device, pipeline_cache)
    if family == "gemma4":
        from vllm_webgpu.models.gemma4 import Gemma4WebGPUModel
        return Gemma4WebGPUModel(model_config, wgpu_device, pipeline_cache)
    if family == "qwen35":
        from vllm_webgpu.models.qwen35 import Qwen35WebGPUModel
        return Qwen35WebGPUModel(model_config, wgpu_device, pipeline_cache)
    if family == "diffusion_gemma":
        from vllm_webgpu.models.diffusion_gemma import DiffusionGemmaWebGPUModel
        return DiffusionGemmaWebGPUModel(model_config, wgpu_device, pipeline_cache)
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
        self._last_logits: np.ndarray | None = None
        self._last_model_output: Any = None  # cached for sample_tokens()
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
        cc = self.vllm_config.cache_config
        hf = mc.hf_config
        block_size = self.webgpu_config.block_size
        num_blocks = cc.num_gpu_blocks

        # Per-layer KV pool: Gemma4 has heterogeneous head_dim/num_kv_heads per layer.
        # Prefer model._lp (populated at load time) over hf._layer_attention_params, which
        # is absent for safetensors checkpoints. Matches the priority order in get_kv_cache_spec().
        lp_list = (
            (getattr(self.model, "_lp", None) if self.model is not None else None)
            or getattr(hf, "_layer_attention_params", None)
        )

        if lp_list:
            import wgpu as wgpu_lib
            from vllm_webgpu.webgpu.buffer import WebGPUBuffer
            rw = wgpu_lib.BufferUsage.STORAGE | wgpu_lib.BufferUsage.COPY_SRC | wgpu_lib.BufferUsage.COPY_DST
            dev = self.wgpu_device.wgpu_device
            pool = []
            for lp in lp_list:
                kv_bytes = num_blocks * block_size * lp["num_kv_heads"] * lp["head_dim"] * 2
                k_buf = WebGPUBuffer.empty(dev, kv_bytes, usage=rw)
                v_buf = WebGPUBuffer.empty(dev, kv_bytes, usage=rw)
                pool.append((k_buf, v_buf))
            if self.model is not None:
                self.model.kv_pool = pool
            logger.info("Per-layer KV pool: %d layers with mixed dims", len(pool))
        else:
            num_kv_heads = hf.num_key_value_heads
            head_dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
            planner = WebGPUCachePlanner.from_runner(self.wgpu_device, self)
            layer_types = getattr(hf, "layer_types", None)
            if layer_types and any(t != "full_attention" for t in layer_types):
                planner.allocate_kv_pool_hybrid(
                    num_blocks=num_blocks,
                    num_layers=hf.num_hidden_layers,
                    layer_types=layer_types,
                    block_size=block_size,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                )
            else:
                planner.allocate_kv_pool(
                    num_blocks=num_blocks,
                    num_layers=hf.num_hidden_layers,
                    block_size=block_size,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                )

    def get_kv_cache_spec(self) -> dict[str, Any]:
        mc = self.vllm_config.model_config.hf_config
        block_size = self.webgpu_config.block_size
        spec: dict[str, Any] = {}
        if FullAttentionSpec is None:
            return spec

        import inspect
        sig = inspect.signature(FullAttentionSpec.__init__)
        params = set(sig.parameters)
        try:
            import torch as _torch
            _dtype = _torch.float16
        except ImportError:
            _dtype = np.float16

        def _make_spec(num_kv_heads: int, head_size: int) -> Any:
            kw: dict[str, Any] = dict(
                block_size=block_size, num_kv_heads=num_kv_heads,
                head_size=head_size, dtype=_dtype,
            )
            if "use_mla" in params:
                kw["use_mla"] = False
            if "kv_quant_mode" in params:
                try:
                    from vllm.v1.kv_cache_interface import KVQuantMode
                    kw["kv_quant_mode"] = KVQuantMode.NONE
                except ImportError:
                    kw["kv_quant_mode"] = 0
            return FullAttentionSpec(**kw)

        # Use per-layer params if available (Gemma4 heterogeneous layers).
        # Prefer the model object's _lp list (populated from layer_types config)
        # over the raw HF config attribute, which may not be set for safetensors.
        lp_list = (getattr(self.model, "_lp", None) or
                   getattr(mc, "_layer_attention_params", None))

        # Fallback: derive lp_list from layer_types + global_head_dim when the
        # model has not been loaded yet and hf_config lacks _layer_attention_params.
        # This covers Gemma4 safetensors where vLLM may call get_kv_cache_spec()
        # before load_model(), so self.model is still None. Mirrors the derivation
        # in Gemma4WebGPUModel.__init__() to ensure uniform and per-layer specs agree.
        if not lp_list:
            layer_types = getattr(mc, "layer_types", None)
            if layer_types and len(layer_types) == mc.num_hidden_layers:
                default_hd = getattr(mc, "head_dim", mc.hidden_size // mc.num_attention_heads)
                default_kv = getattr(mc, "num_key_value_heads", 1)
                global_hd = getattr(mc, "global_head_dim", default_hd)
                global_kv = getattr(mc, "global_kv_heads",
                                    getattr(mc, "num_global_key_value_heads", 1))
                lp_list = []
                for lt in layer_types:
                    if lt == "full_attention":
                        lp_list.append({"num_kv_heads": global_kv, "head_dim": global_hd})
                    else:
                        lp_list.append({"num_kv_heads": default_kv, "head_dim": default_hd})

        if lp_list and len(lp_list) == mc.num_hidden_layers:
            for i, lp in enumerate(lp_list):
                spec[f"model.layers.{i}.self_attn"] = _make_spec(
                    lp["num_kv_heads"], lp["head_dim"])
        else:
            head_size = getattr(mc, "head_dim", mc.hidden_size // mc.num_attention_heads)
            for i in range(mc.num_hidden_layers):
                spec[f"model.layers.{i}.self_attn"] = _make_spec(
                    mc.num_key_value_heads, head_size)
        return spec

    def get_cache_block_size_bytes(self) -> int:
        mc = self.vllm_config.model_config.hf_config
        block_size = self.webgpu_config.block_size
        head_dim = getattr(mc, "head_dim", mc.hidden_size // mc.num_attention_heads)
        return block_size * mc.num_key_value_heads * head_dim * 2 * 2  # K + V, f16

    def warm_up(self) -> None:
        if self.model is not None:
            self.model.warmup()

    def execute_model(self, scheduler_output: "SchedulerOutput") -> Any:
        if self.model is None:
            return None

        try:
            return self._execute_model_v2(scheduler_output)
        except AttributeError:
            # vLLM < 0.24: old SchedulerOutput with scheduled_seq_groups
            return self._execute_model_v1(scheduler_output)
        except Exception as e:
            logger.exception("execute_model failed: %s", e)
            raise

    @staticmethod
    def _compute_request_logprobs(
        logits_1d: "np.ndarray", sampled_tok: int, num_logprobs: int
    ) -> "tuple[np.ndarray, np.ndarray, int]":
        """Compute top-N logprobs from a 1-D float32 logits vector.

        Returns (top_k_ids, top_k_log_probs, sampled_token_rank) where arrays
        have shape (num_logprobs,) with dtype int32 / float32 respectively.
        """
        lp = logits_1d.astype(np.float32)
        shifted = lp - lp.max()
        log_probs = shifted - np.log(np.exp(shifted).sum())
        k = min(num_logprobs, log_probs.size)
        top_ids = np.argpartition(log_probs, -k)[-k:]
        order = np.argsort(log_probs[top_ids])[::-1]
        top_ids = top_ids[order].astype(np.int32)
        top_lp = log_probs[top_ids].astype(np.float32)
        rank = int((log_probs > log_probs[sampled_tok]).sum())
        return top_ids, top_lp, rank

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
        torch or LogprobsTensors are unavailable or T < 2.
        """
        if LogprobsTensors is None:
            return None
        T = len(tok_ids)
        if T < 2:
            return None
        try:
            import torch as _torch
        except ImportError:
            return None

        num_positions = T - 1
        k = num_prompt_logprobs + 1  # +1: vLLM always has a slot for the sampled token

        tok_ids_arr = np.zeros((num_positions, k), dtype=np.int32)
        logprobs_arr = np.full((num_positions, k), -np.inf, dtype=np.float32)
        ranks_arr = np.zeros(num_positions, dtype=np.int32)

        for i in range(num_positions):
            top_ids, top_lp, rank = WebGPUModelRunner._compute_request_logprobs(
                full_logits[i], tok_ids[i + 1], k
            )
            actual_k = len(top_ids)
            tok_ids_arr[i, :actual_k] = top_ids
            logprobs_arr[i, :actual_k] = top_lp
            ranks_arr[i] = rank

        return LogprobsTensors(
            logprob_token_ids=_torch.from_numpy(tok_ids_arr),
            logprobs=_torch.from_numpy(logprobs_arr),
            selected_token_ranks=_torch.from_numpy(ranks_arr),
        )

    @staticmethod
    def _flat_block_ids(ids) -> list[int]:
        """Recursively flatten block IDs from vLLM's nested tuple/list/ndarray format."""
        if ids is None or (hasattr(ids, "__len__") and len(ids) == 0):
            return []
        result = []
        for x in ids:
            if isinstance(x, (list, tuple, np.ndarray)):
                result.extend(WebGPUModelRunner._flat_block_ids(x))
            else:
                result.append(int(x))
        return result

    def _make_model_output(
        self,
        req_ids: list[str],
        sampled: list[int],
        logprobs_data: "list | None" = None,
        prompt_logprobs_dict: "dict | None" = None,
    ) -> Any:
        if ModelRunnerOutput is None:
            return None

        # Build LogprobsLists when at least one request supplied logprob tuples.
        built_logprobs = None
        if (
            LogprobsLists is not None
            and logprobs_data
            and any(d is not None for d in logprobs_data)
        ):
            n = len(req_ids)
            max_k = max(len(d[0]) for d in logprobs_data if d is not None)
            tok_ids_arr = np.zeros((n, max_k), dtype=np.int32)
            logprobs_arr = np.full((n, max_k), -float("inf"), dtype=np.float32)
            ranks_arr = np.zeros(n, dtype=np.int32)
            for i, d in enumerate(logprobs_data):
                if d is not None:
                    ids, lp, rank = d
                    k = len(ids)
                    tok_ids_arr[i, :k] = ids
                    logprobs_arr[i, :k] = lp
                    ranks_arr[i] = rank
            built_logprobs = LogprobsLists(tok_ids_arr, logprobs_arr, ranks_arr)

        kw: dict[str, Any] = {
            "req_ids": req_ids,
            "req_id_to_index": {rid: i for i, rid in enumerate(req_ids)},
            "sampled_token_ids": [[t] for t in sampled],
            "logprobs": built_logprobs,
            "prompt_logprobs_dict": prompt_logprobs_dict or {},
        }
        # Fill any required fields introduced in newer vLLM versions with None so a
        # new required field causes a visible TypeError at import time rather than a
        # confusing runtime crash.  Hardcoded optional names are kept for fields that
        # need explicit None rather than MISSING.
        import dataclasses as _dc
        try:
            for _f in _dc.fields(ModelRunnerOutput):  # type: ignore[arg-type]
                if (
                    _f.default is _dc.MISSING
                    and _f.default_factory is _dc.MISSING  # type: ignore[misc]
                    and _f.name not in kw
                ):
                    kw[_f.name] = None
        except TypeError:
            # ModelRunnerOutput is not a dataclass in this vLLM version; fall back
            # to the hardcoded optional list.
            import inspect as _inspect
            out_params = set(_inspect.signature(ModelRunnerOutput.__init__).parameters)
            for opt in ("pooler_output", "kv_connector_output", "ec_connector_output",
                        "num_nans_in_logits", "cudagraph_stats", "routed_experts"):
                if opt in out_params:
                    kw[opt] = None
        out = ModelRunnerOutput(**kw)
        self._last_model_output = out
        return out

    def _execute_model_v2(self, scheduler_output: "SchedulerOutput") -> Any:
        """vLLM >= 0.24 SchedulerOutput format."""
        if ModelRunnerOutput is None or self.model is None:
            return None

        cached = scheduler_output.scheduled_cached_reqs
        new_reqs = scheduler_output.scheduled_new_reqs
        block_size = self.webgpu_config.block_size

        all_req_ids: list[str] = []
        all_sampled: list[int] = []
        all_logprobs_data: list = []  # per-request logprob tuples or None
        prompt_logprobs_dict: dict[str, Any] = {}  # req_id -> LogprobsTensors for prefill

        # ── Prefill: new requests ──────────────────────────────────────────────
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
            mm = getattr(req, "mm_inputs", None) or getattr(req, "multi_modal_inputs", None)
            if mm:
                raise NotImplementedError(
                    f"req {rid}: multi-modal inputs (images/audio/video) are not supported "
                    f"by the WebGPU backend. The conditional-generation architecture is "
                    f"registered for text-only inference only. Use the CausalLM variant "
                    f"of the model (e.g. Gemma3ForCausalLM) or wait for multi-modal support."
                )

            # Extract per-request logprob counts from SamplingParams.
            sp = getattr(req, "sampling_params", None)
            num_logprobs = getattr(sp, "num_logprobs", None) if sp is not None else None
            num_prompt_logprobs = getattr(sp, "prompt_logprobs", None) if sp is not None else None

            # Reset recurrent state for models with persistent state (Qwen3.5 GDN SSM).
            if hasattr(self.model, "reset_recurrent_states"):
                self.model.reset_recurrent_states()

            raw_bids = req.block_ids
            blk_ids = self._flat_block_ids(raw_bids) if raw_bids else list(range((len(tok_ids) + block_size - 1) // block_size))

            bt = np.array(blk_ids, dtype=np.uint32)

            # Batch prefill: send the scheduled chunk of prompt tokens in a single
            # forward() call. With chunked prefill enabled, num_scheduled_tokens
            # limits how many tokens to process per step; the rest are stored in
            # state and processed in subsequent steps via the cached-req path.
            num_sched = scheduler_output.num_scheduled_tokens.get(rid, len(tok_ids))
            T = min(num_sched, len(tok_ids))
            chunk_toks = tok_ids[:T]
            slots = []
            for idx in range(T):
                blk_idx = idx // block_size
                if blk_idx >= len(blk_ids):
                    raise RuntimeError(
                        f"block table too short for req {rid}: token {idx} needs block "
                        f"{blk_idx} but only {len(blk_ids)} blocks allocated"
                    )
                slots.append(blk_ids[blk_idx] * block_size + (idx % block_size))

            class _BatchPM:
                slot_mapping     = slots
                block_tables     = [bt]
                max_decode_seq_len = T

            if hasattr(self.model, "_greedy_decode"):
                self.model._greedy_decode = _is_greedy(sp)

            last_logits = self.model.forward(
                np.array(chunk_toks, dtype=np.uint32),
                np.arange(T, dtype=np.uint32),
                _BatchPM(),
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
            lp_data = None
            if num_logprobs is not None:
                if last_logits.shape[-1] == 1 and getattr(self.model, "logit_returns_token_id", False):
                    full = self.model.logit_readback()
                elif last_logits.shape[-1] > 1:
                    full = last_logits
                else:
                    full = None
                    logger.warning("req %s: logprobs requested but model does not support logit readback", rid)
                if full is not None:
                    lp_data = self._compute_request_logprobs(full[0], first_decode_tok, num_logprobs)

            # Compute prompt logprobs for each prompt position when full logits
            # are available.  Position i uses logits[i] to evaluate tok_ids[i+1],
            # producing T-1 rows of top-K logprob data.
            if num_prompt_logprobs is not None and T > 1:
                if last_logits.shape[-1] > 1:  # full [T, vocab] logits
                    pt = self._compute_prompt_logprobs(last_logits, tok_ids, num_prompt_logprobs)
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
            # When chunked prefill is active (T < len(tok_ids)), store the full prompt so
            # subsequent chunks can be processed correctly via the cached-req path.
            self._req_state[rid] = {
                "pos": T, "block_ids": blk_ids,
                "last_tok": first_decode_tok, "num_logprobs": num_logprobs,
                "sampling_params": sp,
                "all_prompt_tokens": tok_ids if T < len(tok_ids) else None,
            }

        # ── Decode / chunked-prefill continuation: cached requests ─────────────
        # new_token_ids is empty without pipeline parallelism (vLLM design).
        # Use last_tok stored in _req_state from the previous step instead.
        #
        # Chunked prefill: when a prompt spans multiple scheduling steps, vLLM
        # places the request in scheduled_cached_reqs for the second and later
        # chunks, with is_context_phase() == True. We detect this via
        # num_output_tokens == 0 and the presence of all_prompt_tokens in state,
        # then run a prefill forward pass for the next chunk rather than a decode.
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
        if hasattr(cached, "req_ids") and cached.req_ids:
            new_block_ids = getattr(cached, "new_block_ids", [])
            resumed_req_ids = getattr(cached, "resumed_req_ids", set())

            for i, rid in enumerate(cached.req_ids):
                state = self._req_state.get(rid, {"pos": 0, "block_ids": [], "last_tok": 0})
                pos = state["pos"]
                blk_ids = list(state.get("block_ids", []))
                num_logprobs = state.get("num_logprobs")
                all_prompt = state.get("all_prompt_tokens")

                # Update block table: preempted/resumed requests replace their
                # block table entirely; others append newly allocated blocks.
                cur_new_bids = (new_block_ids[i]
                                if new_block_ids and i < len(new_block_ids) and new_block_ids[i]
                                else None)
                if cur_new_bids is not None:
                    flat_new = self._flat_block_ids(cur_new_bids)
                    if rid in resumed_req_ids:
                        blk_ids = flat_new
                    else:
                        blk_ids.extend(flat_new)

                # Detect chunked-prefill continuation: request is still in the
                # context (prefill) phase and has remaining prompt tokens stored.
                is_context = (
                    all_prompt is not None
                    and pos < len(all_prompt)
                    and (
                        not hasattr(cached, "num_output_tokens")
                        or (i < len(cached.num_output_tokens)
                            and cached.num_output_tokens[i] == 0)
                    )
                )

                if is_context:
                    # Run a prefill forward pass for the next prompt chunk.
                    num_sched = scheduler_output.num_scheduled_tokens.get(rid, len(all_prompt) - pos)
                    chunk_end = min(pos + num_sched, len(all_prompt))
                    chunk_toks = all_prompt[pos:chunk_end]

                    if not chunk_toks:
                        logger.warning("req %s: context phase but no chunk tokens; skipping", rid)
                        continue

                    slots = []
                    for global_idx in range(pos, chunk_end):
                        blk_idx = global_idx // block_size
                        if blk_idx >= len(blk_ids):
                            raise RuntimeError(
                                f"block table too short for req {rid}: token {global_idx} "
                                f"needs block {blk_idx} but only {len(blk_ids)} allocated"
                            )
                        slots.append(blk_ids[blk_idx] * block_size + (global_idx % block_size))

                    class _ChunkPM:
                        slot_mapping      = slots
                        block_tables      = [np.array(blk_ids, dtype=np.uint32)]
                        max_decode_seq_len = chunk_end

                    sp = state.get("sampling_params")
                    if hasattr(self.model, "_greedy_decode"):
                        self.model._greedy_decode = _is_greedy(sp)

                    logits = self.model.forward(
                        np.array(chunk_toks, dtype=np.uint32),
                        np.arange(pos, chunk_end, dtype=np.uint32),
                        _ChunkPM(),
                    )

                    if logits is None:
                        continue

                    self._last_logits = logits

                    # Predict the next token; apply sampling for non-greedy requests.
                    if logits.shape[-1] > 1:
                        stok = _sample_logits(logits[-1], sp)
                    else:
                        stok = int(logits[0, 0])

                    lp_data = None
                    if num_logprobs is not None:
                        if logits.shape[-1] == 1 and getattr(self.model, "logit_returns_token_id", False):
                            full = self.model.logit_readback()
                        elif logits.shape[-1] > 1:
                            full = logits
                        else:
                            full = None
                            logger.warning(
                                "req %s: logprobs requested but model does not support logit readback", rid
                            )
                        if full is not None:
                            lp_data = self._compute_request_logprobs(full[-1], stok, num_logprobs)

                    self._req_state[rid] = {
                        "pos": chunk_end, "block_ids": blk_ids,
                        "last_tok": stok, "num_logprobs": num_logprobs,
                        "sampling_params": state.get("sampling_params"),
                        "all_prompt_tokens": all_prompt if chunk_end < len(all_prompt) else None,
                    }
                    all_req_ids.append(rid)
                    all_sampled.append(stok)
                    all_logprobs_data.append(lp_data)
                    continue

                # Decode step: forward one token at the current position.
                tok = state["last_tok"]

                if pos // block_size >= len(blk_ids):
                    raise RuntimeError(
                        f"block table too short for req {rid}: pos={pos} needs block "
                        f"{pos // block_size} but only {len(blk_ids)} blocks allocated"
                    )
                slot = blk_ids[pos // block_size] * block_size + (pos % block_size)

                class _SM:
                    slot_mapping      = [slot]
                    block_tables      = [np.array(blk_ids, dtype=np.uint32)]
                    max_decode_seq_len = pos + 1

                sp = state.get("sampling_params")
                if hasattr(self.model, "_greedy_decode"):
                    self.model._greedy_decode = _is_greedy(sp)

                logits = self.model.forward(
                    np.array([tok], dtype=np.uint32),
                    np.array([pos], dtype=np.uint32),
                    _SM(),
                )
                self._last_logits = logits

                # Greedy path: model returns (1, 1) int32 with the argmax index.
                # Non-greedy path: model returns (1, vocab) float32; sample here.
                if logits.shape[-1] == 1:
                    stok = int(logits[0, 0])
                else:
                    stok = _sample_logits(logits[0], sp)

                # Compute logprobs if requested for this request.
                lp_data = None
                if num_logprobs is not None:
                    if logits.shape[-1] == 1 and getattr(self.model, "logit_returns_token_id", False):
                        full = self.model.logit_readback()
                    elif logits.shape[-1] > 1:
                        full = logits
                    else:
                        full = None
                        logger.warning("req %s: logprobs requested but model does not support logit readback", rid)
                    if full is not None:
                        lp_data = self._compute_request_logprobs(full[0], stok, num_logprobs)

                # Commit state after a successful forward — don't mutate on failure.
                self._req_state[rid] = {
                    "pos": pos + 1, "block_ids": blk_ids,
                    "last_tok": stok, "num_logprobs": num_logprobs,
                    "sampling_params": state.get("sampling_params"),
                    "all_prompt_tokens": None,
                }
                all_req_ids.append(rid)
                all_sampled.append(stok)
                all_logprobs_data.append(lp_data)

        # Return empty output rather than None when no requests scheduled.
        # vLLM's batch queue raises "unexpected error" on None from execute_model.
        return self._make_model_output(
            all_req_ids, all_sampled, all_logprobs_data, prompt_logprobs_dict
        )

    def _execute_model_v1(self, scheduler_output: "SchedulerOutput") -> Any:
        """vLLM < 0.24 SchedulerOutput format (scheduled_seq_groups)."""
        seq_groups = scheduler_output.scheduled_seq_groups
        if not seq_groups:
            return None
        input_ids_list: list[int] = []
        positions_list: list[int] = []
        for sg in seq_groups:
            seq = sg.seq_group.seqs[0]
            tokens = seq.get_output_token_ids() or seq.get_prompt_token_ids()
            input_ids_list.extend(tokens[-1:])
            positions_list.append(seq.get_len() - 1)
        input_ids = np.array(input_ids_list, dtype=np.uint32)
        positions = np.array(positions_list, dtype=np.uint32)
        logits = self.model.forward(input_ids, positions, scheduler_output)
        self._last_logits = logits
        if SamplerOutput is None or ModelRunnerOutput is None:
            return None
        token_ids = logits.argmax(axis=-1).tolist()
        sampler_out = SamplerOutput(outputs=[], sampled_token_ids=token_ids, logprobs=None, prompt_logprobs=None)
        return ModelRunnerOutput(
            req_ids=[sg.seq_group.request_id for sg in seq_groups],
            req_id_to_index={sg.seq_group.request_id: i for i, sg in enumerate(seq_groups)},
            sampler_output=sampler_out, sampler_output_ready_event=None, pooler_output=[], finished_sending=None,
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

    def supported_worker_tasks(self) -> tuple[Any, ...]:
        # vLLM < 0.11: SupportedTask.GENERATE enum member
        # vLLM >= 0.24: SupportedTask is a type alias; tasks are plain strings
        try:
            return (SupportedTask.GENERATE,)
        except AttributeError:
            return ("generate",)

    def reset_mm_cache(self) -> None:
        pass

    def reset_encoder_cache(self) -> None:
        pass

