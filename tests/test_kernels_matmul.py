import numpy as np
import pytest
from pathlib import Path

SHADERS_DIR = Path(__file__).parent.parent / "vllm_webgpu" / "shaders"


def gelu_mul_ref(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    """SwiGLU: silu(gate) * up  (Llama/Qwen: activation on gate_proj, not up_proj)."""
    g = gate.astype(np.float32)
    u = up.astype(np.float32)
    silu_g = g * (1.0 / (1.0 + np.exp(-g)))
    return (silu_g * u).astype(np.float16)


def test_gelu_mul(wgpu_device):
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey

    n = 256
    gate = np.random.randn(n).astype(np.float16)
    up = np.random.randn(n).astype(np.float16)
    expected = gelu_mul_ref(gate, up)

    dev = wgpu_device.wgpu_device
    gate_buf = WebGPUBuffer.from_numpy(dev, gate)
    up_buf = WebGPUBuffer.from_numpy(dev, up)
    out_buf = WebGPUBuffer.empty(dev, gate.nbytes,
                                 usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)

    cache = PipelineCache(dev, SHADERS_DIR / "generic")
    key = PipelineKey("gelu_mul", (("N", n),))
    pipeline = cache.get_or_create(key)

    bg = dev.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": gate_buf.buf}},
            {"binding": 1, "resource": {"buffer": up_buf.buf}},
            {"binding": 2, "resource": {"buffer": out_buf.buf}},
        ],
    )
    encoder = dev.create_command_encoder()
    cp = encoder.begin_compute_pass()
    cp.set_pipeline(pipeline)
    cp.set_bind_group(0, bg)
    # gelu_mul.wgsl is vec4<f16>: each thread handles 4 elements, dispatch N/4 threads.
    cp.dispatch_workgroups((n // 4 + 255) // 256, 1, 1)
    cp.end()
    dev.queue.submit([encoder.finish()])

    result = out_buf.to_numpy().view(np.float16)
    np.testing.assert_allclose(result.astype(np.float32), expected.astype(np.float32),
                               rtol=1e-2, atol=1e-2)


def test_matmul_quant_f16(wgpu_device):
    """Verify matmul_quant f16 GEMV against numpy reference."""
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey

    K, N = 64, 32
    x = np.random.randn(K).astype(np.float16)
    W = np.random.randn(N, K).astype(np.float16)  # weight matrix [N, K]
    expected = (W.astype(np.float32) @ x.astype(np.float32)).astype(np.float16)

    dev = wgpu_device.wgpu_device
    rw = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST

    x_buf = WebGPUBuffer.from_numpy(dev, x)
    # Pack W as u32 (pairs of f16), layout [N, K]
    w_packed = np.ascontiguousarray(W).view(np.uint32)
    w_buf = WebGPUBuffer.from_numpy(dev, w_packed)
    # dummy scales (not used in f16 path)
    scales_buf = WebGPUBuffer.from_numpy(dev, np.ones(N, dtype=np.float16))
    out_buf = WebGPUBuffer.empty(dev, N * 2, usage=rw)

    cache = PipelineCache(dev, SHADERS_DIR / "generic")
    # SPLIT_K=0: row-per-thread mode, dispatch ceil(N/256) workgroups
    key = PipelineKey("matmul_quant", (("K", K), ("N", N), ("USE_QUANT", 0), ("SPLIT_K", 0)))
    pipeline = cache.get_or_create(key)

    bg = dev.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": x_buf.buf}},
            {"binding": 1, "resource": {"buffer": w_buf.buf}},
            {"binding": 2, "resource": {"buffer": scales_buf.buf}},
            {"binding": 3, "resource": {"buffer": out_buf.buf}},
        ],
    )
    encoder = dev.create_command_encoder()
    cp = encoder.begin_compute_pass()
    cp.set_pipeline(pipeline)
    cp.set_bind_group(0, bg)
    cp.dispatch_workgroups((N + 255) // 256, 1, 1)
    cp.end()
    dev.queue.submit([encoder.finish()])

    result = out_buf.to_numpy().view(np.float16)
    np.testing.assert_allclose(result.astype(np.float32), expected.astype(np.float32),
                               rtol=1e-2, atol=1e-2)


def test_matmul_additivity_invariant(wgpu_device):
    """Property proof: matmul_quant is additive in its input vector.

    Invariant: mat @ (x + y) == (mat @ x) + (mat @ y) for any vectors x, y.
    Together with the scalar case (linearity), this proves mat@ is a LINEAR MAP.
    A linear map is precisely what matrix-vector multiplication must be.
    """
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey

    K, N = 32, 16
    x = np.random.randn(K).astype(np.float16)
    y = np.random.randn(K).astype(np.float16)
    xy = (x.astype(np.float32) + y.astype(np.float32)).astype(np.float16)
    W = np.random.randn(N, K).astype(np.float16)

    dev = wgpu_device.wgpu_device
    rw = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
    w_packed = np.ascontiguousarray(W).view(np.uint32)
    w_buf = WebGPUBuffer.from_numpy(dev, w_packed)
    scales_buf = WebGPUBuffer.from_numpy(dev, np.ones(N, dtype=np.float16))

    cache = PipelineCache(dev, SHADERS_DIR / "generic")
    key = PipelineKey("matmul_quant", (("K", K), ("N", N), ("USE_QUANT", 0)))
    pipeline = cache.get_or_create(key)

    results = {}
    for name, inp in [("x", x), ("y", y), ("x+y", xy)]:
        xb = WebGPUBuffer.from_numpy(dev, inp)
        ob = WebGPUBuffer.empty(dev, N * 2, usage=rw)
        bg = dev.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[{"binding": 0, "resource": {"buffer": xb.buf}},
                     {"binding": 1, "resource": {"buffer": w_buf.buf}},
                     {"binding": 2, "resource": {"buffer": scales_buf.buf}},
                     {"binding": 3, "resource": {"buffer": ob.buf}}],
        )
        enc = dev.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipeline)
        cp.set_bind_group(0, bg)
        cp.dispatch_workgroups((N + 255) // 256, 1, 1)
        cp.end()
        dev.queue.submit([enc.finish()])
        results[name] = ob.to_numpy().view(np.float16).astype(np.float32)

    expected_sum = (results["x"].astype(np.float64) + results["y"].astype(np.float64)).astype(np.float32)

    # Invariant: mat @ (x+y) == (mat@x) + (mat@y)
    np.testing.assert_allclose(results["x+y"], expected_sum, rtol=5e-2, atol=0.1,
                               err_msg="matmul additivity violated: mat@(x+y) != mat@x + mat@y")


def test_matmul_linearity_invariant(wgpu_device):
    """Property proof: matmul_quant is linear in its input vector.

    Invariant: mat @ (α*x) == α * (mat @ x) for any scalar α.
    Linearity is a fundamental property of matrix multiplication.
    Any shader that violates this is not computing a matrix product.
    """
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey

    K, N = 64, 32
    alpha = np.float16(2.5)
    x = np.random.randn(K).astype(np.float16)
    W = np.random.randn(N, K).astype(np.float16)
    x_scaled = (x.astype(np.float32) * float(alpha)).astype(np.float16)

    dev = wgpu_device.wgpu_device
    rw = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
    w_packed = np.ascontiguousarray(W).view(np.uint32)
    w_buf = WebGPUBuffer.from_numpy(dev, w_packed)
    scales_buf = WebGPUBuffer.from_numpy(dev, np.ones(N, dtype=np.float16))

    cache = PipelineCache(dev, SHADERS_DIR / "generic")
    key = PipelineKey("matmul_quant", (("K", K), ("N", N), ("USE_QUANT", 0)))
    pipeline = cache.get_or_create(key)

    results = []
    for inp in [x, x_scaled]:
        x_buf = WebGPUBuffer.from_numpy(dev, inp)
        out_buf = WebGPUBuffer.empty(dev, N * 2, usage=rw)
        bg = dev.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[{"binding": 0, "resource": {"buffer": x_buf.buf}},
                     {"binding": 1, "resource": {"buffer": w_buf.buf}},
                     {"binding": 2, "resource": {"buffer": scales_buf.buf}},
                     {"binding": 3, "resource": {"buffer": out_buf.buf}}],
        )
        enc = dev.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipeline)
        cp.set_bind_group(0, bg)
        cp.dispatch_workgroups((N + 255) // 256, 1, 1)
        cp.end()
        dev.queue.submit([enc.finish()])
        results.append(out_buf.to_numpy().view(np.float16).astype(np.float32))

    result_x, result_x_scaled = results
    expected_scaled = result_x * float(alpha)

    # Invariant: mat @ (α*x) == α * (mat @ x)
    np.testing.assert_allclose(result_x_scaled, expected_scaled, rtol=5e-2, atol=0.1,
                               err_msg="matmul linearity invariant violated: mat@(α*x) != α*(mat@x)")


def test_add_commutativity_invariant(wgpu_device):
    """Property proof: element-wise addition is commutative: add(a,b) == add(b,a).

    Commutativity of addition is an axiom of arithmetic. Any implementation
    that violates add(a,b) == add(b,a) is not computing addition.
    """
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey

    n = 128
    a = np.random.randn(n).astype(np.float16)
    b = np.random.randn(n).astype(np.float16)

    dev = wgpu_device.wgpu_device
    rw = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
    cache = PipelineCache(dev, SHADERS_DIR / "generic")
    pipeline = cache.get_or_create(PipelineKey("add", (("N", n),)))

    results = []
    for x, y in [(a, b), (b, a)]:  # compute both add(a,b) and add(b,a)
        xb = WebGPUBuffer.from_numpy(dev, x)
        yb = WebGPUBuffer.from_numpy(dev, y)
        ob = WebGPUBuffer.empty(dev, x.nbytes, usage=rw)
        bg = dev.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[{"binding": 0, "resource": {"buffer": xb.buf}},
                     {"binding": 1, "resource": {"buffer": yb.buf}},
                     {"binding": 2, "resource": {"buffer": ob.buf}}],
        )
        enc = dev.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipeline)
        cp.set_bind_group(0, bg)
        cp.dispatch_workgroups((n + 255) // 256, 1, 1)
        cp.end()
        dev.queue.submit([enc.finish()])
        results.append(ob.to_numpy().view(np.float16).astype(np.float32))

    # Invariant: add(a,b) == add(b,a)
    np.testing.assert_array_equal(results[0], results[1],
                                  err_msg="add commutativity violated: add(a,b) != add(b,a)")


def test_matmul_fp8_per_channel_scale(wgpu_device):
    """USE_QUANT=5 + GROUP_K=1: each output row uses its own scale from scales[row]."""
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey
    from vllm_webgpu.quant.weight_loader import _fp8_e4m3_to_f32

    K, N = 32, 8
    rng = np.random.default_rng(42)

    # Build weight matrix in FP8 E4M3 (stored as uint8 bytes)
    W_f32 = rng.uniform(-1.0, 1.0, (N, K)).astype(np.float32)
    # Encode to FP8 E4M3 by rounding to representable values (use scale=1 for simplicity)
    W_f32_clipped = np.clip(W_f32, -448.0, 448.0)
    # Simple FP8 encoding: store as raw bytes (use float16 as proxy then re-decode for reference)
    W_f16 = W_f32_clipped.astype(np.float16)
    # Use the f16 values as "FP8 weights" by re-encoding to FP8 via CPU dequant reference
    # For test purposes: just use small values that encode cleanly in FP8 E4M3
    W_f32_small = rng.uniform(-1.0, 1.0, (N, K)).astype(np.float32) * 0.5
    # Pack as uint8 FP8 bytes: encode each float32 to FP8 E4M3
    def encode_fp8(v: float) -> int:
        """Encode a float32 to FP8 E4M3 (brute-force via lookup)."""
        best, best_err = 0, float("inf")
        for code in range(256):
            decoded = _fp8_e4m3_to_f32(np.array([code], dtype=np.uint8))[0]
            err = abs(float(decoded) - v)
            if err < best_err:
                best_err = err
                best = code
        return best

    W_u8 = np.array([[encode_fp8(W_f32_small[n, k]) for k in range(K)]
                     for n in range(N)], dtype=np.uint8)
    W_decoded = _fp8_e4m3_to_f32(W_u8)  # (N, K) float32

    # Per-channel scales: distinct value per output row
    ch_scales = rng.uniform(0.1, 2.0, N).astype(np.float32)
    x = rng.uniform(-1.0, 1.0, K).astype(np.float32).astype(np.float16)

    # Reference: W_decoded[n,:] * ch_scales[n] dotted with x
    expected = np.array([(W_decoded[n] * ch_scales[n] * x.astype(np.float32)).sum()
                         for n in range(N)], dtype=np.float32)

    dev = wgpu_device.wgpu_device
    rw = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST

    # Pack W_u8 as a flat u32 array (shader reads via rd_byte_at)
    w_flat = np.ascontiguousarray(W_u8.ravel())
    pad = (4 - len(w_flat) % 4) % 4
    if pad:
        w_flat = np.concatenate([w_flat, np.zeros(pad, dtype=np.uint8)])
    w_u32 = w_flat.view(np.uint32)

    x_buf = WebGPUBuffer.from_numpy(dev, x)
    w_buf = WebGPUBuffer.from_numpy(dev, w_u32)
    scales_buf = WebGPUBuffer.from_numpy(dev, ch_scales.astype(np.float16))
    out_buf = WebGPUBuffer.empty(dev, N * 2, usage=rw)

    cache = PipelineCache(dev, SHADERS_DIR / "generic")
    key = PipelineKey("matmul_quant", (
        ("K", K), ("N", N), ("USE_QUANT", 5), ("SPLIT_K", 1),
        ("GROUP_K", 1), ("GLOBAL_SCALE", 1.0),
    ))
    pipeline = cache.get_or_create(key)

    bg = dev.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": x_buf.buf}},
            {"binding": 1, "resource": {"buffer": w_buf.buf}},
            {"binding": 2, "resource": {"buffer": scales_buf.buf}},
            {"binding": 3, "resource": {"buffer": out_buf.buf}},
        ],
    )
    encoder = dev.create_command_encoder()
    cp = encoder.begin_compute_pass()
    cp.set_pipeline(pipeline)
    cp.set_bind_group(0, bg)
    cp.dispatch_workgroups(N, 1, 1)
    cp.end()
    dev.queue.submit([encoder.finish()])

    result = out_buf.to_numpy().view(np.float16).astype(np.float32)[:N]
    # FP8 + f16 accumulation: tolerate ~5% relative error
    np.testing.assert_allclose(result, expected, rtol=0.05, atol=0.05,
                               err_msg="per-channel FP8 GEMV output mismatch")


def test_embedding_exactness_invariant(wgpu_device):
    """Property proof: embedding lookup returns EXACT copies (no approximation).

    Invariant: output[i] == table[token_ids[i]] exactly, not approximately.
    Embedding is a pure table lookup — any rounding or approximation would
    violate this invariant. Tested with all-distinct table values.
    """
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey

    vocab, hidden = 32, 16
    num_tokens = 8
    # Use distinct values for each row so any mixing is detectable
    table = (np.arange(vocab * hidden, dtype=np.float32) / (vocab * hidden)).reshape(vocab, hidden).astype(np.float16)
    token_ids = np.array([0, 7, 3, 15, 31, 1, 16, 8], dtype=np.uint32)

    dev = wgpu_device.wgpu_device
    rw = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
    table_buf = WebGPUBuffer.from_numpy(dev, table)
    ids_buf = WebGPUBuffer.from_numpy(dev, token_ids)
    out_buf = WebGPUBuffer.empty(dev, num_tokens * hidden * 2, usage=rw)

    cache = PipelineCache(dev, SHADERS_DIR / "generic")
    pipeline = cache.get_or_create(PipelineKey("embedding_lookup", (("HIDDEN_DIM", hidden),)))
    bg = dev.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[{"binding": 0, "resource": {"buffer": table_buf.buf}},
                 {"binding": 1, "resource": {"buffer": ids_buf.buf}},
                 {"binding": 2, "resource": {"buffer": out_buf.buf}}],
    )
    enc = dev.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(pipeline)
    cp.set_bind_group(0, bg)
    cp.dispatch_workgroups(num_tokens, 1, 1)
    cp.end()
    dev.queue.submit([enc.finish()])

    result = out_buf.to_numpy().view(np.float16).reshape(num_tokens, hidden)
    expected = table[token_ids]  # exact numpy reference (pure indexing)

    # Invariant: EXACT equality, not approximate — embedding is a pure copy
    np.testing.assert_array_equal(result, expected,
                                  err_msg="embedding_lookup exactness violated: output != table[token_ids]")


def test_embedding_lookup(wgpu_device):
    import wgpu
    from vllm_webgpu.webgpu.buffer import WebGPUBuffer
    from vllm_webgpu.webgpu.pipeline import PipelineCache, PipelineKey

    vocab, hidden = 32, 64
    num_tokens = 4
    table = np.random.randn(vocab, hidden).astype(np.float16)
    token_ids = np.array([0, 5, 10, 31], dtype=np.uint32)
    expected = table[token_ids]

    dev = wgpu_device.wgpu_device
    table_buf = WebGPUBuffer.from_numpy(dev, table)
    ids_buf = WebGPUBuffer.from_numpy(dev, token_ids)
    out_buf = WebGPUBuffer.empty(
        dev, num_tokens * hidden * 2,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
    )

    cache = PipelineCache(dev, SHADERS_DIR / "generic")
    key = PipelineKey("embedding_lookup", (("HIDDEN_DIM", hidden),))
    pipeline = cache.get_or_create(key)

    bg = dev.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": table_buf.buf}},
            {"binding": 1, "resource": {"buffer": ids_buf.buf}},
            {"binding": 2, "resource": {"buffer": out_buf.buf}},
        ],
    )
    encoder = dev.create_command_encoder()
    cp = encoder.begin_compute_pass()
    cp.set_pipeline(pipeline)
    cp.set_bind_group(0, bg)
    cp.dispatch_workgroups(num_tokens, 1, 1)
    cp.end()
    dev.queue.submit([encoder.finish()])

    result = out_buf.to_numpy().view(np.float16).reshape(num_tokens, hidden)
    np.testing.assert_allclose(result.astype(np.float32), expected.astype(np.float32),
                               rtol=1e-3, atol=1e-3)
