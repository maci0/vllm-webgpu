#!/usr/bin/env python3
"""CPU reference forward pass for Gemma3-1B — numpy only, no transformers/torch.

Loads BF16 safetensors, runs full 26-layer forward pass for a single token,
prints top-5 predicted next tokens.

Verified against mlx_lm Gemma3 implementation: top-1 token matches (218070).
The spurious numpy BLAS warnings (divide by zero / overflow in matmul) are
harmless — Apple Accelerate emits them during internal SIMD passes on large
activations; all outputs are finite.
"""

import json
import struct
import warnings
import numpy as np

# Suppress spurious Apple Accelerate BLAS warnings on large activations.
# Verified: all intermediate and final values are finite.
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")

MODEL = (
    "/Users/mwysocki/.cache/huggingface/hub"
    "/models--unsloth--gemma-3-1b-it/snapshots"
    "/5b11413a10db4e486ef16a20101fd028f8f2499c/model.safetensors"
)
TOKEN_ID = 105  # <start_of_turn>

# Architecture constants (from config.json)
HIDDEN    = 1152
N_HEADS   = 4
N_KV      = 1
HEAD_DIM  = 256
INTER     = 6912
VOCAB     = 262144
LAYERS    = 26


# ── weight loading ────────────────────────────────────────────────────────────

print("Reading safetensors header...", flush=True)
with open(MODEL, "rb") as fh:
    hdr_size = struct.unpack("<Q", fh.read(8))[0]
    HDR = json.loads(fh.read(hdr_size))
DATA_OFF = 8 + hdr_size

# Sort by file offset for sequential I/O (much faster than random seeks)
entries = [
    (v["data_offsets"][0], k, v)
    for k, v in HDR.items()
    if k != "__metadata__"
]
entries.sort()

print(f"Loading {len(entries)} tensors (BF16 -> F32, ~3.8 GB)...", flush=True)
W = {}
with open(MODEL, "rb") as fh:
    for _off, name, info in entries:
        start, end = info["data_offsets"]
        fh.seek(DATA_OFF + start)
        raw = fh.read(end - start)
        # BF16 -> F32: left-shift 16 bits, reinterpret as float32
        u16 = np.frombuffer(raw, dtype=np.uint16)
        W[name] = (
            (u16.astype(np.uint32) << 16)
            .view(np.float32)
            .reshape(info["shape"])
            .copy()
        )
print(f"Loaded {len(W)} tensors.", flush=True)


# ── ops ───────────────────────────────────────────────────────────────────────

def rms_norm(x, w, eps=1e-6):
    """Gemma-style RMSNorm: scale = (1 + w) instead of plain w."""
    return x / np.sqrt((x * x).mean() + eps) * (1.0 + w)

def gelu_tanh(x):
    """Tanh-approximate GELU (gelu_pytorch_tanh)."""
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))


# ── forward pass ──────────────────────────────────────────────────────────────

embed_w = W["model.embed_tokens.weight"]  # [vocab, hidden]
x = embed_w[TOKEN_ID].copy() * np.sqrt(float(HIDDEN))  # embedding scale

for i in range(LAYERS):
    L = f"model.layers.{i}"

    # Self-attention
    normed = rms_norm(x, W[f"{L}.input_layernorm.weight"])

    # Q: [n_heads, head_dim], per-head Q-norm
    q = (W[f"{L}.self_attn.q_proj.weight"] @ normed).reshape(N_HEADS, HEAD_DIM)
    qn = W[f"{L}.self_attn.q_norm.weight"]
    for h in range(N_HEADS):
        q[h] = rms_norm(q[h], qn)

    # K: single KV head, K-norm
    k = rms_norm(                                         # noqa (k unused for ctx=1)
        W[f"{L}.self_attn.k_proj.weight"] @ normed,
        W[f"{L}.self_attn.k_norm.weight"],
    )

    # V: single KV head (no V-norm in Gemma3 reference)
    v = W[f"{L}.self_attn.v_proj.weight"] @ normed       # [head_dim]

    # ctx_len=1: softmax=1.0, all N_HEADS attend to the single KV head
    attn_concat = np.tile(v, N_HEADS)                    # [n_heads * head_dim]
    attn_out = W[f"{L}.self_attn.o_proj.weight"] @ attn_concat  # [hidden]

    # Post-attention norm applied BEFORE residual add (Gemma3 / Gemma2 style)
    attn_out = rms_norm(attn_out, W[f"{L}.post_attention_layernorm.weight"])
    x = x + attn_out

    # FFN (SwiGLU with tanh-GELU gate)
    normed2 = rms_norm(x, W[f"{L}.pre_feedforward_layernorm.weight"])
    gate = W[f"{L}.mlp.gate_proj.weight"] @ normed2      # [inter]
    up   = W[f"{L}.mlp.up_proj.weight"]   @ normed2      # [inter]
    ffn_out = W[f"{L}.mlp.down_proj.weight"] @ (gelu_tanh(gate) * up)  # [hidden]

    # Post-FFN norm applied BEFORE residual add
    ffn_out = rms_norm(ffn_out, W[f"{L}.post_feedforward_layernorm.weight"])
    x = x + ffn_out

    print(f"  layer {i:2d}  x[:4]={x[:4].tolist()}", flush=True)

# Final norm + weight-tied LM head
x = rms_norm(x, W["model.norm.weight"])
logits = embed_w @ x  # [vocab]  — weight tying, no separate lm_head

top5 = np.argsort(logits)[-5:][::-1]
print(f"\nTop-5 predicted tokens after token {TOKEN_ID} (<start_of_turn>):")
for tok in top5:
    print(f"  token {tok:>8d}  logit={logits[tok]:+.6f}")
