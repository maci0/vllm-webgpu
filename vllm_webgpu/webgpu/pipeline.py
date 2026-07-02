from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineKey:
    shader_name: str
    defines: tuple[tuple[str, int | float], ...]


class PipelineCache:
    def __init__(self, wgpu_device, shaders_dir: Path) -> None:
        self._device = wgpu_device
        self._shaders_dir = Path(shaders_dir)
        self._cache: dict[PipelineKey, object] = {}

    def get_or_create(self, key: PipelineKey) -> object:
        if key in self._cache:
            return self._cache[key]

        shader_path = self._shaders_dir / f"{key.shader_name}.wgsl"
        if not shader_path.exists():
            raise FileNotFoundError(
                f"Shader not found: {shader_path}. "
                f"Check shader_subdir and shader_name in the _dispatch call."
            )
        wgsl_source = shader_path.read_text()

        module = self._device.create_shader_module(code=wgsl_source)
        constants = {name: value for name, value in key.defines}
        pipeline = self._device.create_compute_pipeline(
            layout="auto",
            compute={
                "module": module,
                "entry_point": "main",
                "constants": constants,
            },
        )
        self._cache[key] = pipeline
        logger.debug("Compiled pipeline: %s defines=%s", key.shader_name, key.defines)
        return pipeline
