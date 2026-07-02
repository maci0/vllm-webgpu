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
}


def _build_model(arch: str, model_config: Any, wgpu_device: Any, pipeline_cache: Any) -> "BaseWebGPUModel":
    family = ARCH_MAP.get(arch)
    if family == "llama":
        from vllm_webgpu.models.llama import LlamaWebGPUModel
        return LlamaWebGPUModel(model_config, wgpu_device, pipeline_cache)
    if family == "gemma4":
        from vllm_webgpu.models.gemma4 import Gemma4WebGPUModel
        return Gemma4WebGPUModel(model_config, wgpu_device, pipeline_cache)
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
        self._last_logits: np.ndarray | None = None  # cached for sample_tokens()

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

        num_kv_heads = hf.num_key_value_heads
        # Use explicit head_dim when present (Gemma4 sets it independently of hidden/heads).
        head_dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
        block_size = self.webgpu_config.block_size

        num_blocks = cc.num_gpu_blocks
        planner = WebGPUCachePlanner.from_runner(self.wgpu_device, self)
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
        if FullAttentionSpec is not None:
            for i in range(mc.num_hidden_layers):
                spec[f"model.layers.{i}.self_attn"] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=mc.num_key_value_heads,
                    head_size=getattr(mc, "head_dim", mc.hidden_size // mc.num_attention_heads),
                    dtype=np.float16,
                    use_mla=False,
                )
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
            seq_groups = scheduler_output.scheduled_seq_groups
            if not seq_groups:
                return None

            input_ids_list: list[int] = []
            positions_list: list[int] = []
            for sg in seq_groups:
                seq = sg.seq_group.seqs[0]
                tokens = seq.get_output_token_ids() or seq.get_prompt_token_ids()
                input_ids_list.extend(tokens[-1:])   # decode: last token only
                positions_list.append(seq.get_len() - 1)

            input_ids = np.array(input_ids_list, dtype=np.uint32)
            positions = np.array(positions_list, dtype=np.uint32)

            logits = self.model.forward(input_ids, positions, scheduler_output)
            self._last_logits = logits

            if SamplerOutput is None or ModelRunnerOutput is None:
                return None

            token_ids = logits.argmax(axis=-1).tolist()
            sampler_out = SamplerOutput(
                outputs=[],
                sampled_token_ids=token_ids,
                logprobs=None,
                prompt_logprobs=None,
            )
            return ModelRunnerOutput(
                req_ids=[sg.seq_group.request_id for sg in seq_groups],
                req_id_to_index={sg.seq_group.request_id: i for i, sg in enumerate(seq_groups)},
                sampler_output=sampler_out,
                sampler_output_ready_event=None,
                pooler_output=[],
                finished_sending=None,
            )
        except Exception as e:
            logger.exception("execute_model failed: %s", e)
            raise

    def sample_tokens(self, grammar_output: "GrammarOutput | None") -> Any:
        # Structured output / grammar sampling not supported.
        return None

    def supported_worker_tasks(self) -> tuple[Any, ...]:
        if SupportedTask is not Any and SupportedTask is not None:
            return (SupportedTask.GENERATE,)
        return ()

    def reset_mm_cache(self) -> None:
        pass

    def reset_encoder_cache(self) -> None:
        pass
