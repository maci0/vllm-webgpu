# vllm-webgpu Design Spec

**Date:** 2026-06-29  
**Status:** Approved

## Overview

`vllm-webgpu` is an out-of-tree vLLM platform plugin that runs LLM inference on any WebGPU-capable GPU using hand-tuned WGSL compute kernels. It mirrors the vllm-metal architecture: vLLM handles scheduling and serving; the plugin owns all compute via `wgpu-py` (Rust wgpu Python bindings).

Target architectures: Llama 3.x, Qwen 2.5/3.x, Gemma 4.  
Quantization: GGUF Q4_K_M and safetensors f16.  
Platform: cross-platform (macOS via Metal, Windows via DX12, Linux via Vulkan) via wgpu's backend abstraction.

---

## Plugin Registration

vLLM discovers plugins via Python entry points. Registration is a single callable that returns the platform class path if the platform is available, or `None` if not.

```toml
# pyproject.toml
[project.entry-points."vllm.platform_plugins"]
webgpu = "vllm_webgpu:register"
```

```python
# vllm_webgpu/__init__.py
def _register() -> str | None:
    _configure_logging()
    import vllm.envs
    from vllm_webgpu.envs import environment_variables
    vllm.envs.environment_variables.update(environment_variables)
    from vllm_webgpu.compat import apply_compat_patches
    apply_compat_patches()
    from vllm_webgpu.platform import WebGPUPlatform
    if WebGPUPlatform.is_available():
        return "vllm_webgpu.platform.WebGPUPlatform"
    return None
```

---

## Package Structure

```
vllm_webgpu/
├── __init__.py
├── platform.py
├── config.py
├── envs.py
├── compat.py
├── utils.py
├── webgpu/
│   ├── __init__.py
│   ├── device.py
│   ├── buffer.py
│   └── pipeline.py
├── shaders/
│   ├── generic/
│   │   ├── matmul_quant.wgsl
│   │   ├── matmul_quant_mr4.wgsl
│   │   ├── embedding_lookup.wgsl
│   │   ├── rms_norm.wgsl
│   │   ├── rope.wgsl
│   │   ├── fused_per_head_norm_rope.wgsl
│   │   ├── per_head_rms_norm.wgsl
│   │   ├── attn_score.wgsl
│   │   ├── attn_output.wgsl
│   │   ├── kv_cache_store.wgsl
│   │   ├── gelu_mul.wgsl
│   │   ├── fused_norm_add.wgsl
│   │   ├── add.wgsl
│   │   ├── softmax.wgsl
│   │   ├── argmax.wgsl
│   │   └── topk256.wgsl
│   └── gemma/
│       ├── ple_stage1_fuse.wgsl
│       ├── ple_gelu_mul.wgsl
│       ├── ple_skip_scale_add.wgsl
│       ├── logit_softcap.wgsl
│       └── per_head_rms_norm_no_weight.wgsl
├── models/
│   ├── __init__.py
│   ├── base.py
│   ├── llama.py
│   └── gemma4.py
├── quant/
│   ├── __init__.py
│   └── gguf_loader.py
└── v1/
    ├── __init__.py
    ├── worker.py
    ├── model_runner.py
    └── cache_policy.py
```

---

## Platform

`WebGPUPlatform` extends vLLM's `Platform` base. PyTorch device stays CPU throughout; all compute goes through wgpu-py buffers.

```python
class WebGPUPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name = "cpu"
    device_type = "cpu"
    dispatch_key = "CPU"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import wgpu
            adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
            return adapter is not None
        except Exception:
            return False

    @classmethod
    def get_device_capability(cls, device_id=0) -> DeviceCapability:
        return DeviceCapability(major=8, minor=0)  # compatibility shim

    @classmethod
    def get_device_count(cls) -> int:
        return 1

    @classmethod
    def check_and_update_config(cls, vllm_config) -> None:
        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm_webgpu.v1.worker.WebGPUWorker"
        parallel_config.distributed_executor_backend = "uni"
        parallel_config.disable_custom_all_reduce = True
        # no varlen kernel yet; disable chunked prefill
        vllm_config.scheduler_config.enable_chunked_prefill = False

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, attn_selector_config, num_heads=None) -> str:
        # attention is handled by WGSL kernels in the model runner
        return AttentionBackendEnum.CPU_ATTN.get_path()

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        return False
```

---

## Config and Environment Variables

Single global singleton, loaded lazily from env vars at first access.

```python
@dataclass
class WebGPUConfig:
    memory_fraction: float    # -1.0 = auto
    power_preference: str     # "high-performance" | "low-power"
    quantization: str         # "q4_k_m" | "f16" | "auto"
    block_size: int           # KV cache block size, default 16
    debug: bool

    @property
    def is_auto_memory(self) -> bool:
        return self.memory_fraction == -1.0
```

Environment variables (merged into `vllm.envs` at registration):

| Variable | Default | Description |
|---|---|---|
| `VLLM_WEBGPU_MEMORY_FRACTION` | `"auto"` | Fraction of GPU memory for KV cache, or "auto" |
| `VLLM_WEBGPU_POWER_PREFERENCE` | `"high-performance"` | Adapter power hint |
| `VLLM_WEBGPU_QUANTIZATION` | `"auto"` | Force quantization format, or detect from checkpoint |
| `VLLM_WEBGPU_BLOCK_SIZE` | `"16"` | KV cache block size (tokens per block) |
| `VLLM_WEBGPU_DEBUG` | `"0"` | Enable verbose kernel dispatch logging |

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

### Generic kernels (both Llama and Gemma)

| Kernel | Operation |
|---|---|
| `matmul_quant.wgsl` | GEMV for decode (batch=1); Q4_K_M dequant path or raw f16 path selected via `USE_QUANT` specialization constant |
| `matmul_quant_mr4.wgsl` | Matmul with mr=4 tile for prefill; same Q4_K_M/f16 dual-path as above |
| `embedding_lookup.wgsl` | token_ids → f16 embedding rows |
| `rms_norm.wgsl` | standard RMSNorm with weight |
| `rope.wgsl` | standard RoPE (Llama-style frequencies) |
| `fused_per_head_norm_rope.wgsl` | per-head RMSNorm + RoPE in one pass |
| `per_head_rms_norm.wgsl` | per-head RMSNorm with weight |
| `attn_score.wgsl` | Q·Kᵀ scaled dot product, GQA-aware, paged KV |
| `attn_output.wgsl` | softmax(scores)·V, paged KV |
| `kv_cache_store.wgsl` | write K/V tokens into paged block table |
| `gelu_mul.wgsl` | SwiGLU: gate·SiLU(up) for Llama/Qwen FFN |
| `fused_norm_add.wgsl` | residual add + RMSNorm fused |
| `add.wgsl` | residual add |
| `softmax.wgsl` | numerically stable softmax |
| `argmax.wgsl` | greedy decode |
| `topk256.wgsl` | top-k sampling (k <= 256) |

### Gemma-specific kernels

| Kernel | Operation |
|---|---|
| `ple_stage1_fuse.wgsl` | fused per-layer embedding stage 1 |
| `ple_gelu_mul.wgsl` | PLE-variant gated GELU |
| `ple_skip_scale_add.wgsl` | PLE skip connection + scale + add |
| `logit_softcap.wgsl` | Gemma logit soft-cap (tanh(x / cap) * cap) |
| `per_head_rms_norm_no_weight.wgsl` | weightless per-head RMSNorm (weight=1, dropped from ckpt) |

For f16 safetensors checkpoints, `matmul_quant.wgsl` skips the dequant path and operates on raw f16 blocks.

---

## Architecture-Specific Models

### `models/base.py`

```python
class BaseWebGPUModel:
    weights: dict[str, WebGPUBuffer]           # "layers.0.attn.q_proj" -> buffer
    kv_pool: list[tuple[WebGPUBuffer, WebGPUBuffer]]  # per-layer (K, V) blocks

    def load_weights(self, path: str) -> None:
        # detect GGUF vs safetensors
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
# v1/model_runner.py
ARCH_MAP = {
    "LlamaForCausalLM": LlamaWebGPUModel,
    "Qwen2ForCausalLM": LlamaWebGPUModel,
    "Qwen3ForCausalLM": LlamaWebGPUModel,
    "Gemma3ForCausalLM": Gemma4WebGPUModel,
}
arch = vllm_config.model_config.architectures[0]
model_cls = ARCH_MAP.get(arch)
if model_cls is None:
    raise NotImplementedError(f"Unsupported architecture: {arch}")
```

---

## Weight Loading

Two paths, same output: `dict[str, WebGPUBuffer]`.

**GGUF (Q4_K_M):**
- `quant/gguf_loader.py` reads GGUF metadata and tensor blocks
- Q4_K_M blocks uploaded as `u8` buffers (32 weights + f16 scale per block)
- `matmul_quant.wgsl` dequantizes on the GPU during dispatch

**Safetensors (f16/bf16):**
- Load with `safetensors` library, cast bf16 to f16 via numpy
- Upload as f16 WebGPUBuffer
- `matmul_quant.wgsl` and `matmul_quant_mr4.wgsl` use `USE_QUANT=0` specialization constant, skipping dequant

Both paths map HF checkpoint tensor names to the plugin's weight key convention (`"layers.{i}.attn.q_proj"` etc.).

---

## Worker

```python
class WebGPUWorker(WorkerBase):

    def init_device(self) -> None:
        self.wgpu_device = WebGPUDevice.initialize(
            power_preference=self.webgpu_config.power_preference
        )
        self.device = torch.device("cpu")   # PyTorch stays CPU
        init_worker_distributed_environment(...)  # gloo backend
        set_random_seed(self.model_config.seed)
        self.model_runner = WebGPUModelRunner(self.vllm_config, self.wgpu_device)

    def load_model(self) -> None:
        self.model_runner.load_model()

    def determine_available_memory(self) -> int:
        limits = self.wgpu_device.device.limits
        total = limits["max_buffer_size"]
        model_mem = sum(buf.buf.size for buf in self.model_runner.model.weights.values())
        config = get_config()
        if config.is_auto_memory:
            return total - model_mem - _OVERHEAD_BYTES
        return int(total * config.memory_fraction) - model_mem

    def initialize_from_config(self, kv_cache_config) -> None:
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def execute_model(self, scheduler_output) -> ModelRunnerOutput | None:
        return self.model_runner.execute_model(scheduler_output)

    def check_health(self) -> None:
        # submit a trivial 1-element dispatch to verify device is alive
        ...
```

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

## Unsupported (initial release)

- LoRA (no adapter injection in WGSL dispatch path)
- Multi-GPU (single adapter only, `world_size=1`)
- Chunked prefill (no varlen attention kernel)
- Sleep/wake mode
- BF16 at runtime (weights cast to f16 on upload; WGSL has no bf16)
- MoE architectures (no expert routing kernels)

---

## Dependencies

```toml
[project]
name = "vllm-webgpu"
version = "0.1.0"

[project.dependencies]
vllm = ">=0.20"
wgpu = ">=0.20"
numpy = ">=1.24"
safetensors = ">=0.4"
gguf = ">=0.10"
psutil = ">=5.9"

[project.entry-points."vllm.platform_plugins"]
webgpu = "vllm_webgpu:register"
```

---

## Testing

- `tests/test_platform.py` - `is_available()`, `check_and_update_config()`
- `tests/test_buffer.py` - round-trip upload/download, dtype handling
- `tests/test_pipeline.py` - cache hit/miss, specialization constants
- `tests/test_kernels.py` - each WGSL kernel vs numpy reference (rms_norm, rope, softmax, attn_score)
- `tests/test_models.py` - single forward pass for Llama and Gemma4, compare logits to reference
- `tests/test_gguf_loader.py` - Q4_K_M block loading and dequant correctness
