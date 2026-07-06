from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    from vllm.config import VllmConfig
    from vllm.tasks import SupportedTask
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec, FullAttentionSpec
    from vllm.v1.outputs import ModelRunnerOutput, SamplerOutput
except ImportError:
    VllmConfig = Any  # type: ignore[assignment,misc]
    SupportedTask = Any  # type: ignore[assignment,misc]
    KVCacheConfig = Any  # type: ignore[assignment,misc]
    KVCacheSpec = Any  # type: ignore[assignment,misc]
    FullAttentionSpec = None  # type: ignore[assignment,misc]
    ModelRunnerOutput = None  # type: ignore[assignment,misc]
    SamplerOutput = None  # type: ignore[assignment,misc]

from vllm_webgpu.config import get_config
from vllm_webgpu.utils import SHADERS_DIR
from vllm_webgpu.v1.cache_policy import WebGPUCachePlanner
from vllm_webgpu.webgpu.pipeline import PipelineCache

if TYPE_CHECKING:
    from vllm_webgpu.models.base import BaseWebGPUModel
    from vllm_webgpu.webgpu.device import WebGPUDevice
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput

logger = logging.getLogger(__name__)

ARCH_MAP = {
    "LlamaForCausalLM": "llama",
    "MistralForCausalLM": "llama",
    "Qwen2ForCausalLM": "llama",
    "Qwen3ForCausalLM": "llama",
    "Gemma3ForCausalLM": "gemma4",
    "Gemma3ForConditionalGeneration": "gemma4",
    "Gemma4ForCausalLM": "gemma4",
    "Qwen3_5ForConditionalGeneration": "qwen35",
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
        lp_list = getattr(hf, "_layer_attention_params", None)
        if lp_list and self.model is not None and hasattr(self.model, "_lp"):
            lp_list = self.model._lp

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
        lp_list = getattr(mc, "_layer_attention_params", None)
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
    def _flat_block_ids(ids) -> list[int]:
        """Recursively flatten block IDs from vLLM's nested tuple/list format."""
        if not ids:
            return []
        result = []
        for x in ids:
            if isinstance(x, (list, tuple)):
                result.extend(WebGPUModelRunner._flat_block_ids(x))
            else:
                result.append(int(x))
        return result

    def _make_model_output(self, req_ids: list[str], sampled: list[int]) -> Any:
        if ModelRunnerOutput is None:
            return None
        import inspect as _inspect
        out_params = set(_inspect.signature(ModelRunnerOutput.__init__).parameters)
        kw: dict[str, Any] = {
            "req_ids": req_ids,
            "req_id_to_index": {rid: i for i, rid in enumerate(req_ids)},
            "sampled_token_ids": [[t] for t in sampled],
            "logprobs": None,
            "prompt_logprobs_dict": {},
        }
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

        # ── Prefill: new requests ──────────────────────────────────────────────
        for req in new_reqs:
            rid = req.req_id
            tok_ids = list(req.prompt_token_ids or [])
            if not tok_ids:
                continue

            raw_bids = req.block_ids
            blk_ids = self._flat_block_ids(raw_bids) if raw_bids else list(range((len(tok_ids) + block_size - 1) // block_size))

            bt = np.array(blk_ids, dtype=np.uint32)

            # Run each prompt token through the model to populate the KV cache.
            last_logits = None
            for i, tok in enumerate(tok_ids):
                # Physical KV cache slot for this token (paged addressing).
                blk_idx = i // block_size
                if blk_idx >= len(blk_ids):
                    raise RuntimeError(
                        f"block table too short for req {rid}: token {i} needs block "
                        f"{blk_idx} but only {len(blk_ids)} blocks allocated"
                    )
                slot = blk_ids[blk_idx] * block_size + (i % block_size)

                class _PM:
                    _slot = slot
                    _bt = bt
                    _ctx = i + 1
                    slot_mapping = [_slot]
                    block_tables = [_bt]
                    max_decode_seq_len = _ctx

                last_logits = self.model.forward(
                    np.array([tok], dtype=np.uint32),
                    np.array([i], dtype=np.uint32),
                    _PM(),
                )

            if last_logits is None:
                continue

            first_decode_tok = int(last_logits.argmax(axis=-1)[0])
            all_req_ids.append(rid)
            all_sampled.append(first_decode_tok)
            # Store last sampled token; decode path needs it (new_token_ids is empty without PP).
            self._req_state[rid] = {"pos": len(tok_ids), "block_ids": blk_ids, "last_tok": first_decode_tok}

        # ── Decode: cached requests ────────────────────────────────────────────
        # new_token_ids is empty without pipeline parallelism (vLLM design).
        # Use last_tok stored in _req_state from the previous step instead.
        if hasattr(cached, "req_ids") and cached.req_ids:
            decode_req_ids: list[str] = []
            input_ids_list: list[int] = []
            positions_list: list[int] = []
            slot_mappings: list[int] = []
            bt_list: list[np.ndarray] = []

            new_block_ids = getattr(cached, "new_block_ids", [])

            # Collect decode batch inputs; don't mutate _req_state until after forward succeeds.
            _staged: list[tuple[str, int, list[int]]] = []  # (rid, new_pos, new_blk_ids)
            for i, rid in enumerate(cached.req_ids):
                state = self._req_state.get(rid, {"pos": 0, "block_ids": [], "last_tok": 0})
                tok = state["last_tok"]
                pos = state["pos"]
                blk_ids = list(state.get("block_ids", []))
                if new_block_ids and i < len(new_block_ids) and new_block_ids[i]:
                    blk_ids.extend(self._flat_block_ids(new_block_ids[i]))

                if pos // block_size >= len(blk_ids):
                    raise RuntimeError(
                        f"block table too short for req {rid}: pos={pos} needs block "
                        f"{pos // block_size} but only {len(blk_ids)} blocks allocated"
                    )
                slot = blk_ids[pos // block_size] * block_size + (pos % block_size)
                decode_req_ids.append(rid)
                input_ids_list.append(tok)
                positions_list.append(pos)
                slot_mappings.append(slot)
                bt_list.append(np.array(blk_ids, dtype=np.uint32))
                _staged.append((rid, pos + 1, blk_ids))

            if decode_req_ids:
                class _DM:
                    _sm = slot_mappings
                    _bt = bt_list
                    _ctx = max(positions_list) + 1
                    slot_mapping = _sm
                    block_tables = _bt
                    max_decode_seq_len = _ctx

                logits = self.model.forward(
                    np.array(input_ids_list, dtype=np.uint32),
                    np.array(positions_list, dtype=np.uint32),
                    _DM(),
                )
                self._last_logits = logits
                sampled = logits.argmax(axis=-1).tolist()
                if not isinstance(sampled, list):
                    sampled = [sampled]
                # Commit state only after successful forward.
                for (rid, new_pos, new_blk_ids), stok in zip(_staged, sampled):
                    self._req_state[rid] = {"pos": new_pos, "block_ids": new_blk_ids, "last_tok": int(stok)}
                all_req_ids.extend(decode_req_ids)
                all_sampled.extend(sampled)

        # Return empty output rather than None when no requests scheduled.
        # vLLM's batch queue raises "unexpected error" on None from execute_model.
        return self._make_model_output(all_req_ids, all_sampled)

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
        # Grammar/structured output is not supported — return cached output as-is.
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

