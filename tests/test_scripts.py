"""Tests for standalone script path helpers."""
from pathlib import Path


def test_resolve_model_path_local_dir(tmp_path):
    from vllm_webgpu.scripts import resolve_model_path

    d = tmp_path / "model"
    d.mkdir()
    assert resolve_model_path(str(d)) == str(d)


def test_resolve_model_path_local_safetensors_file(tmp_path):
    """A local .safetensors file must not be treated as a HuggingFace repo id."""
    from vllm_webgpu.scripts import resolve_model_path, config_dir_for_model_path

    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"fake")
    assert resolve_model_path(str(weight)) == str(weight)
    assert config_dir_for_model_path(str(weight)) == str(tmp_path)


def test_config_dir_for_model_path_directory(tmp_path):
    from vllm_webgpu.scripts import config_dir_for_model_path

    assert config_dir_for_model_path(str(tmp_path)) == str(tmp_path)
