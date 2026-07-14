"""Shared utilities for vllm-webgpu standalone scripts."""
from __future__ import annotations


def resolve_model_path(model_arg: str) -> str:
    """Return a local path for model_arg, downloading from HuggingFace if needed.

    When model_arg is an existing local directory it is returned unchanged.
    Otherwise snapshot_download is called to fetch the repo and the resulting
    cache path is returned.
    """
    from pathlib import Path
    from huggingface_hub import snapshot_download
    return model_arg if Path(model_arg).is_dir() else snapshot_download(model_arg)


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
