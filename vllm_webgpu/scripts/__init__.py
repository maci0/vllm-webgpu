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
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    except Exception as _e:
        print(f"  Warning: apply_chat_template failed ({_e}), falling back to tok.encode")
        return tok.encode(prompt)
