# vllm-webgpu Design Spec

**Date:** 2026-06-29 (updated 2026-11-12)
**Status:** Architecture overview. The source code defines the exact interfaces and runtime behavior.

## Overview

`vllm-webgpu` is an out-of-tree vLLM platform plugin that runs LLM inference on any WebGPU-capable GPU using hand-tuned WGSL compute kernels. vLLM handles scheduling and serving; the plugin owns all compute via `wgpu-py` (Rust wgpu Python bindings).

Target architectures: Llama 3.x, Qwen 2.5/3.x/3.5/3.6/3.8, Gemma 3/4, Mistral, Mixtral, GPT-OSS, NemotronH (Mamba-2 hybrid), DiffusionGemma, Phi-4/Phi-4-mini, SmolLM3 (NoPE layers), OLMo-2/3, FalconH1 (Mamba-2 parallel hybrid).
Quantization: safetensors f16, GPTQ, AWQ, FP8, NVFP4, Int8, NF4, ct_pack_int4 (Gemma4 QAT), MXFP4/MXFP8 (load-time dequant), MLX affine-int4 (load-time dequant). GGUF format is not supported; use the vllm-gguf plugin instead.  
Platform: cross-platform (macOS via Metal, Windows via DX12, Linux via Vulkan) via wgpu's backend abstraction.

---

## Plugin Registration

vLLM discovers this package through the `vllm.platform_plugins` entry point. The registration callable can run more than once, so it registers environment variables without overwriting existing entries. It returns the platform class path when a hardware WebGPU adapter is available and `None` otherwise. It also yields to native CUDA and ROCm platforms unless `VLLM_WEBGPU_FORCE=1` is set. See [vLLM 0.33.0's platform plugin guide](https://docs.vllm.ai/en/v0.33.0/design/plugin_system/) for the entry point and platform contracts.

```toml
# pyproject.toml
[project.entry-points."vllm.platform_plugins"]
webgpu = "vllm_webgpu:register"
```

The installed entry point resolves `register` through the package's `__getattr__`. Registration defers platform imports until called. vLLM loads plugins in each process, so the plugin function must remain safe to call repeatedly. `WebGPUPlatform` uses `PlatformEnum.OOT` and `cpu` for its PyTorch device type; model computation runs in WebGPU buffers. `check_and_update_config` selects `WebGPUWorker`, and `get_attn_backend_cls` returns the CPU attention backend class required by vLLM's interface. The single-device worker uses the base device communicator with the `gloo` backend. This package pins vLLM because its worker and model runner use v1 internals whose interfaces can change between releases.

---

## Package Structure

```
vllm_webgpu/
├── __init__.py
├── platform.py
├── config.py
├── envs.py
├── utils.py
├── webgpu/
│   ├── __init__.py
│   ├── device.py
│   ├── buffer.py
│   └── pipeline.py
├── shaders/
│   ├── generic/          (40+ shaders — see KERNELS.md for full reference)
│   │   ├── matmul_quant.wgsl
│   │   ├── matmul_quant_mr4.wgsl
│   │   ├── flash_attn_decode.wgsl
│   │   ├── flash_attn_prefill.wgsl
│   │   ├── fused_qkv.wgsl
│   │   ├── fused_qk_norm_rope.wgsl
│   │   ├── fused_gate_act.wgsl
│   │   ├── add_rms_norm.wgsl
│   │   ├── kv_cache_store_both.wgsl
│   │   ├── mamba2_ssm_step.wgsl
│   │   ├── mamba2_causal_conv.wgsl
│   │   ├── mamba2_norm_gate.wgsl
│   │   ├── causal_conv_step.wgsl
│   │   ├── gdn_state_update.wgsl
│   │   ├── topk_sort.wgsl
│   │   └── ...
│   └── gemma/
│       ├── gelu_mul.wgsl
│       ├── logit_softcap.wgsl
│       └── per_head_rms_norm_no_weight.wgsl
├── models/
│   ├── __init__.py
│   ├── base.py
│   ├── llama.py
│   ├── mixtral.py
│   ├── gpt_oss.py
│   ├── gemma4.py
│   ├── qwen35.py
│   ├── nemotron_h.py
│   ├── diffusion_gemma.py
│   ├── phi.py
│   ├── smollm3.py
│   ├── olmo2.py
│   └── falcon_h1.py
├── quant/
│   ├── __init__.py
│   └── weight_loader.py
└── v1/
    ├── __init__.py
    ├── worker.py
    ├── model_runner.py
    └── cache_policy.py
```

---

## Platform

`WebGPUPlatform` extends vLLM's `Platform` base with `PlatformEnum.OOT`. It advertises `cpu` as the PyTorch device type because vLLM's bookkeeping tensors stay on CPU; model computation uses wgpu buffers and WGSL shaders. `is_available()` probes for a hardware WebGPU adapter and rejects software fallback adapters.

During `check_and_update_config`, the platform selects `WebGPUWorker`, the single-process `uni` executor, a compatible KV cache layout, and a WebGPU block size unless the user specified one. It disables chunked prefill, asynchronous scheduling, dual batch overlap, and CUDA graph capture. On macOS it uses the `spawn` worker start method because a Metal device cannot be safely reused after `fork`.

The platform implements vLLM's attention backend hook with `CPU_ATTN` for metadata. Its model runner dispatches the actual attention kernels through WebGPU. The default device communicator suffices for this single-device `gloo` worker.

---

## Config and Environment Variables

`WebGPUConfig` is loaded lazily and validates the adapter power preference. It currently contains `power_preference`; cache sizing and other settings use vLLM's config or the variables below.

```python
@dataclass(frozen=True)
class WebGPUConfig:
    power_preference: str     # "high-performance" | "low-power"
```

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `VLLM_WEBGPU_POWER_PREFERENCE` | `"high-performance"` | Adapter power hint |
| `VLLM_WEBGPU_BLOCK_SIZE` | `"16"` | KV cache block size (tokens per block) |
| `GDN_BF16` | `"0"` | Preserve BF16 GDN projection weights in packed buffers when set to `1` |
| `VLLM_WEBGPU_FORCE` | unset | Select WebGPU even when a native CUDA or ROCm platform is available |

The first three variables are registered in `vllm.envs` on plugin load. `VLLM_WEBGPU_FORCE` is read directly by the entry point.

---

## WebGPU Runtime Layer

### `webgpu/device.py` - Singleton device init

Called once from `WebGPUWorker.init_device()`.

- Requests adapter with configured `power_preference`
- Enables `shader-f16` feature if adapter supports it (faster matmul)
- Exposes `device.limits` for memory planning
- Exposes `supports_f16: bool` for kernel selection

### `webgpu/buffer.py` - GPU buffer management

```python
class WebGPUBuffer:
    buf: wgpu.GPUBuffer
    shape: tuple[int, ...]
    dtype: str   # "f16" | "f32" | "u8"

    @staticmethod
    def from_numpy(device, arr: np.ndarray, usage=STORAGE|COPY_DST) -> "WebGPUBuffer":
        ...  # create buffer, write_buffer(), return

    @staticmethod
    def empty(device, nbytes: int, usage=STORAGE|COPY_DST|COPY_SRC) -> "WebGPUBuffer":
        ...

    def to_numpy(self) -> np.ndarray:
        ...  # map buffer for read - debug / logit readback only
```

Weight buffers are write-once at load time. KV cache buffers are read/write, pre-allocated as a pool. Logits are read back to CPU once per step for sampling.

### `webgpu/pipeline.py` - Compute pipeline cache

WGSL compilation takes 50-200ms per shader. Cache keyed on `(shader_name, specialization_constants)`.

```python
@dataclass(frozen=True)
class PipelineKey:
    shader_name: str
    defines: tuple[tuple[str, int], ...]  # e.g. (("BLOCK_SIZE", 32), ("HEAD_DIM", 128))

class PipelineCache:
    def get_or_create(self, key: PipelineKey, wgsl_source: str) -> wgpu.GPUComputePipeline:
        if key not in self._cache:
            module = device.create_shader_module(code=wgsl_source)
            self._cache[key] = device.create_compute_pipeline(
                layout="auto",
                compute={"module": module, "entry_point": "main",
                         "constants": dict(key.defines)},
            )
        return self._cache[key]
```

---

## WGSL Kernels

Kernels are ported from the Xenova/tylerstraub gemma4-webgpu project and generalized for multiple architectures. Parameterization uses WGSL `override` constants so one shader binary covers multiple shapes without source templating.

```wgsl
// Example: attn_score.wgsl
override BLOCK_SIZE: u32 = 16u;
override HEAD_DIM: u32 = 128u;
override NUM_KV_HEADS: u32 = 8u;
override NUM_Q_HEADS: u32 = 32u;

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) { ... }
```

### Generic kernels

| Kernel | Operation |
|---|---|
| `matmul_quant.wgsl` | GEMV for decode (M=1); USE_QUANT selects f16/GPTQ/AWQ/FP8/NVFP4/Int8/NF4 |
| `matmul_quant_mr4.wgsl` | Tiled GEMM for prefill (M>1, MR×4 output tile) |
| `flash_attn_decode.wgsl` | Fused QK+softmax+V decode attention (always used for decode) |
| `flash_attn_prefill.wgsl` | Causal self-attention for all T prompt tokens in one dispatch |
| `fused_qkv.wgsl` | Q+K+V projections in one dispatch (f16 only) |
| `fused_qk_norm_rope.wgsl` | Q+K per-head norm + RoPE in one dispatch |
| `fused_gate_act.wgsl` | Gate+up GEMV with inline SiLU/GELU (f16 only) |
| `add_rms_norm.wgsl` | Fused residual add + RMSNorm (f16) |
| `add_f32_rms_norm.wgsl` | Fused residual add + RMSNorm (f32 residual, Gemma4) |
| `rms_norm_add.wgsl` | Fused RMSNorm + residual add for post-norm (OLMo-2) |
| `rms_norm_add_f32_rms_norm.wgsl` | Double-norm fusion for Gemma4 sublayer pairs |
| `kv_cache_store_both.wgsl` | K+V paged cache write in one dispatch |
| `embedding_lookup.wgsl` | token_ids → f16 embedding rows |
| `embedding_lookup_f32.wgsl` | token_ids → f32 embedding rows with sqrt(hidden_dim) scale (Gemma4) |
| `causal_conv_step.wgsl` | Single-step causal depthwise conv1d (Qwen3.5 GDN) |
| `gdn_state_update.wgsl` | Delta-rule SSM state update (Qwen3.5 GDN) |
| `linear_attn_norm_gate.wgsl` | GDN output norm + sigmoid gate |
| `mamba2_ssm_step.wgsl` | Mamba-2 selective scan step (NemotronH) |
| `mamba2_causal_conv.wgsl` | Mamba-2 causal depthwise conv1d (NemotronH) |
| `mamba2_norm_gate.wgsl` | Mamba-2 output norm + silu gate (NemotronH) |
| `topk_sort.wgsl` | GPU top-K expert selection for MoE routing |
| `argmax_f16.wgsl` | GPU argmax over f16 logits (greedy decode) |
| `softmax.wgsl` | Online 2-pass softmax |
| `sigmoid_gate.wgsl` | Element-wise sigmoid(gate) * value (Qwen3.5 attn gate) |

### Gemma-specific kernels (shaders/gemma/)

| Kernel | Operation |
|---|---|
| `gelu_mul.wgsl` | Tanh-GELU for Gemma3/4 FFN |
| `logit_softcap.wgsl` | Gemma logit soft-cap: tanh(x / cap) * cap |
| `per_head_rms_norm_no_weight.wgsl` | Weightless per-head RMSNorm for Gemma4 V normalization |

See [KERNELS.md](../KERNELS.md) for the full reference including dispatch shapes and override tables.

---

## Architecture-Specific Models

### `models/base.py`

```python
class BaseWebGPUModel:
    weights: dict[str, WebGPUBuffer]           # "layers.0.attn.q_proj" -> buffer
    kv_pool: list[tuple[WebGPUBuffer, WebGPUBuffer]]  # per-layer (K, V) blocks

    def load_weights(self, path: str) -> None:
        # detect safetensors format (HF hub, single-shard, or MLX)
        # map HF tensor names to weight dict
        # upload all tensors to GPU as WebGPUBuffer

    def forward(self, input_ids, positions, kv_cache, attn_metadata) -> np.ndarray:
        raise NotImplementedError

    def _dispatch(self, shader_name, bindings, constants, workgroups) -> None:
        # encode + submit one compute dispatch
        pipeline = self.pipelines.get_or_create(shader_name, constants)
        encoder = self.device.create_command_encoder()
        pass_ = encoder.begin_compute_pass()
        pass_.set_pipeline(pipeline)
        for slot, buf in enumerate(bindings):
            pass_.set_bind_group(slot, ...)
        pass_.dispatch_workgroups(*workgroups)
        pass_.end()
        self.device.queue.submit([encoder.finish()])
```

### `models/llama.py`

Handles Llama 3.x and Qwen 2.5/3.x (architecturally identical: standard RoPE, SwiGLU FFN, GQA).

Layer execution per token:
```
embedding_lookup
→ N × (rms_norm → qkv_proj → fused_per_head_norm_rope →
        kv_cache_store → attn_score → softmax → attn_output →
        o_proj → add → rms_norm → gate_proj + up_proj → gelu_mul →
        down_proj → add)
→ rms_norm → lm_head → [logits to CPU]
```

### `models/gemma4.py`

Extends `BaseWebGPUModel` with PLE layer injection and Gemma-specific attention:

- Replaces `rope` with `fused_per_head_norm_rope` (using weightless variant for V)
- Injects PLE pipeline (`ple_stage1_fuse → ple_gelu_mul → ple_skip_scale_add`) at configured layer indices
- Applies `logit_softcap` before returning logits

### Architecture detection

```python
# v1/model_runner.py — selected HF architecture mappings; see MODELS.md for the full map
ARCH_MAP = {
    "LlamaForCausalLM":                         "llama",
    "MistralForCausalLM":                        "mixtral",
    "MixtralForCausalLM":                        "mixtral",
    "Qwen2ForCausalLM":                          "llama",
    "Qwen3ForCausalLM":                          "llama",
    "Gemma3ForCausalLM":                         "gemma4",
    "Gemma3ForConditionalGeneration":            "gemma4",
    "Gemma4ForCausalLM":                         "gemma4",
    "Gemma4UnifiedForConditionalGeneration":     "gemma4",
    "Qwen3_5ForConditionalGeneration":           "qwen35",
    "Qwen3_5ForCausalLM":                        "qwen35",
    "Qwen3_5MoeForConditionalGeneration":        "qwen35",
    "DiffusionGemmaForBlockDiffusion":           "diffusion_gemma",
    "GptOssForCausalLM":                         "gpt_oss",
    "NemotronHForCausalLM":                      "nemotron_h",
    "Phi3ForCausalLM":                           "phi",
    "FalconH1ForCausalLM":                       "falcon_h1",
    "SmolLM3ForCausalLM":                        "smollm3",
    "Olmo2ForCausalLM":                          "olmo2",
    "Olmo3ForCausalLM":                          "olmo2",
}
```

Qwen3.8-27B and Qwen3.6-35B-A3B use the `qwen35` backend for text input. Qwen3.8-27B also passed an end-to-end smoke test with a GPTQ INT4 checkpoint; its original BF16 checkpoint has not been run. Qwen3.6-35B-A3B has published-config and synthetic-weight tests only. The latter uses 256 experts and selects eight per token. See [MODELS.md](../MODELS.md#qwen35webgpumodel) for details.

---

## Weight Loading

All weight loading is handled by `quant/weight_loader.py`. Output: `dict[str, WebGPUBuffer]`.

**Supported sources:**
- HuggingFace model ID or local directory (detected by `model.safetensors.index.json`)
- Single-shard `.safetensors` file (detected by extension)
- MLX directory (detected by `.biases` keys in safetensors index)

**Quantization handling at load time:**
- f16/bf16 safetensors: cast to f16, upload as-is (USE_QUANT=0 at runtime)
- GPTQ/AWQ/FP8/NVFP4/Int8/NF4: raw quantized tensors uploaded to GPU, dequantized on-device by `matmul_quant.wgsl`
- MXFP4/MXFP8: exponent-scale dequantized to f16 at load time, run as USE_QUANT=0
- MLX affine-int4: dequantized to f16 at load time
- compressed-tensors ct_pack_int4: config_groups JSON parsed, routed to USE_QUANT=3/5/7 by sub-format

GGUF format is not supported. Use the vllm-gguf plugin for GGUF checkpoints.

Weight tensor names are mapped from HF checkpoint conventions to the plugin's internal key convention via model-specific key mapping (and for NemotronH, via vLLM's built-in `hf_to_vllm_mapper`).

---

## Worker

`WebGPUWorker` inherits vLLM's `WorkerBase`. `init_device` creates the wgpu device, initializes CPU `gloo` distributed state, seeds the process, and creates `WebGPUModelRunner`. `load_model`, `get_kv_cache_spec`, `initialize_from_config`, and `execute_model` delegate to the runner. `determine_available_memory` calls the policy in `v1/cache_policy.py`, which accounts for model weights, device memory, and activation overhead rather than treating WebGPU's per-buffer size limit as total VRAM. `check_health` submits an empty command buffer to verify that the device still responds.

---

## KV Cache

Paged attention. All KV blocks pre-allocated as GPU buffers at startup.

**Buffer layout per layer:**
```
K: WebGPUBuffer  shape [num_blocks, block_size, num_kv_heads, head_dim]  dtype f16
V: WebGPUBuffer  shape [num_blocks, block_size, num_kv_heads, head_dim]  dtype f16
```

**Block table:**
```
block_table: WebGPUBuffer  shape [max_seqs, max_blocks_per_seq]  dtype u32
```

Updated from CPU via `device.queue.write_buffer()` each step. Read by `kv_cache_store.wgsl` and `attn_score.wgsl`.

**Attention dispatch per step:**
1. CPU builds block_table for current batch, writes to GPU
2. `kv_cache_store.wgsl`: writes new K/V tokens into assigned blocks
3. `attn_score.wgsl`: reads K via block table, computes Q·Kᵀ (GQA-aware)
4. `softmax.wgsl`
5. `attn_output.wgsl`: reads V via block table, computes scores·V

Block size = 16 (matches Xenova kernel tile size, aligns with WebGPU subgroup width of 16 on most adapters).

**Memory planning:**

```python
class WebGPUCachePlanner:
    def allocate_kv_pool(self, num_blocks, num_layers, block_size, num_kv_heads, head_dim):
        bytes_per_block = block_size * num_kv_heads * head_dim * 2  # f16
        for _ in range(num_layers):
            k = WebGPUBuffer.empty(device, num_blocks * bytes_per_block)
            v = WebGPUBuffer.empty(device, num_blocks * bytes_per_block)
            self.kv_pool.append((k, v))
```

**Sampling:** logits read back to CPU via `to_numpy()` after final layer. vLLM's sampler runs on CPU (same approach as vllm-metal).

---

## Known limitations

- **Single-sequence only**: one request per `forward()` call. Multi-sequence batching is an architectural change, not planned.
- **No speculative decoding**: no draft model support.
- **No grammar/constrained decoding**: requires full logit distribution on-device, not available in the decode path.
- **No LoRA**: no adapter injection in WGSL dispatch path.
- **No multi-GPU**: single adapter only (`world_size=1`).
- **No chunked prefill**: no varlen attention kernel.
- **No bfloat16 at runtime**: WGSL has no bf16 type. Weights load as f16 (5-bit exponent). Use `--gdn_bf16` to partially mitigate for Qwen3.5 GDN layers.
- **No GGUF**: use the vllm-gguf plugin for GGUF checkpoints.
- **Max context**: 65535 KV slots from the 4096-block table with 16-token blocks; DiffusionGemma is additionally constrained to canvas_length=256.

---

## Dependencies

```toml
[project]
name = "vllm-webgpu"
version = "0.1.0"

dependencies = [
    "vllm==0.33.0",
    "wgpu>=0.20,!=0.32.0",
    "numpy>=1.24",
    "safetensors>=0.4",
    "compressed-tensors>=0.9",
]

[project.entry-points."vllm.platform_plugins"]
webgpu = "vllm_webgpu:register"
```

---

## Testing

Tests cover kernel correctness, quantization, model construction and forward passes, vLLM platform integration, and Qwen3.6 MoE and Qwen3.8 dense config and synthetic weight loading. CI installs the pinned vLLM version and runs the full suite, lint, and version checks.

```bash
pytest tests/ -q
```
