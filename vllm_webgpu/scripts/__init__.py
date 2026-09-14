"""Shared utilities for vllm-webgpu standalone scripts."""
from __future__ import annotations


def resolve_model_path(model_arg: str) -> str:
    """Return a local path for model_arg, downloading from HuggingFace if needed.

    Existing local directories and files (e.g. a standalone ``.safetensors``
    checkpoint) are returned unchanged. Only non-existent paths are treated as
    HuggingFace repo IDs and passed to ``snapshot_download``.
    """
    from pathlib import Path
    from huggingface_hub import snapshot_download
    p = Path(model_arg)
    if p.is_dir() or p.is_file():
        return str(p)
    return snapshot_download(model_arg)


def config_dir_for_model_path(model_path: str) -> str:
    """Directory that holds config.json / tokenizer files for model_path.

    ``model_path`` may be a model directory or a single weight file; HF loaders
    need the containing directory in the file case.
    """
    from pathlib import Path
    p = Path(model_path)
    return str(p.parent if p.is_file() else p)


def apply_chat_template_or_encode(tok, prompt: str) -> list[int]:
    """Apply the tokenizer's chat template to prompt, falling back to tok.encode.

    Returns the token IDs as a plain list of ints. The fallback fires when
    apply_chat_template raises any exception (e.g. the model has no chat
    template defined), in which case the raw prompt is encoded directly.

    Transformers 5.x returns a BatchEncoding from `apply_chat_template(...,
    tokenize=True)` rather than the flat list 4.x returned, so unwrap
    `input_ids`. Iterating the BatchEncoding instead yields its *keys*, which
    surfaced far downstream as `int('input_ids')` during prefill.
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        encoded = tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
    except Exception as _e:
        print(f"  Warning: apply_chat_template failed ({_e}), falling back to tok.encode")
        encoded = tok.encode(prompt)
    return _as_token_ids(encoded)


def _as_token_ids(encoded) -> list[int]:
    """Flat list of token ids from a tokenizer return value.

    Accepts a plain sequence, a BatchEncoding/dict carrying ``input_ids``, and
    either of those batched one level deep (``[[id, ...]]``).
    """
    ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    ids = list(ids)
    if ids and isinstance(ids[0], (list, tuple)):
        if len(ids) != 1:
            raise ValueError(f"expected a single sequence, got a batch of {len(ids)}")
        ids = list(ids[0])
    return [int(i) for i in ids]
