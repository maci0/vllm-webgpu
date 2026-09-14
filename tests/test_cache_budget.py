"""KV cache budget sizing: which memory pool it draws from."""

from types import SimpleNamespace

import pytest

from vllm_webgpu.v1 import cache_policy
from vllm_webgpu.v1.cache_policy import (
    _UNMEASURABLE_VRAM_KV_BUDGET,
    determine_available_memory,
)

GIB = 1024 * 1024 * 1024


def _worker(*, discrete: bool, model_bytes: int = GIB, utilization: float = 1.0):
    """Worker stub with one loaded weight buffer of `model_bytes`."""
    device = SimpleNamespace(is_discrete_gpu=discrete)
    model = SimpleNamespace(weights={"w": SimpleNamespace(nbytes=model_bytes)})
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            kv_cache_memory_bytes=0, gpu_memory_utilization=utilization
        ),
        model_runner=SimpleNamespace(model=model, wgpu_device=device),
    )


def test_discrete_gpu_budget_comes_from_vram_not_system_ram(monkeypatch):
    """The bug this guards: a discrete GPU sized its KV cache from host RAM.

    On a 24GB card in a 128GB host that produced a 52GB budget, and
    over-committing VRAM does not raise -- it loses the device context and takes
    the process down during warmup. The budget must not exceed free VRAM.
    """
    vram_total, vram_used = 24 * GIB, 4 * GIB
    monkeypatch.setattr(
        cache_policy, "_discrete_vram_bytes", lambda: (vram_total, vram_used)
    )
    # Host RAM an order of magnitude larger, as on the machine that hit this.
    monkeypatch.setattr(cache_policy, "get_visible_memory_node", lambda: [0])
    monkeypatch.setattr(
        cache_policy,
        "get_memory_node_info",
        lambda _n: SimpleNamespace(
            total_memory=128 * GIB, available_memory=100 * GIB
        ),
    )

    available = determine_available_memory(_worker(discrete=True))

    assert 0 < available <= vram_total - vram_used
    assert available < 100 * GIB, "budget was drawn from system RAM"


def test_unified_memory_budget_still_comes_from_system_ram(monkeypatch):
    """Apple Silicon UMA, integrated GPUs and software adapters share host
    memory, so the original behaviour has to survive for them."""
    monkeypatch.setattr(
        cache_policy,
        "_discrete_vram_bytes",
        lambda: pytest.fail("VRAM must not be probed for a non-discrete adapter"),
    )
    monkeypatch.setattr(cache_policy, "get_visible_memory_node", lambda: [0])
    monkeypatch.setattr(
        cache_policy,
        "get_memory_node_info",
        lambda _n: SimpleNamespace(total_memory=64 * GIB, available_memory=32 * GIB),
    )

    available = determine_available_memory(_worker(discrete=False))

    assert available > 16 * GIB


def test_unmeasurable_vram_falls_back_to_a_small_fixed_budget(monkeypatch):
    """No VRAM reading (NVIDIA's proprietary driver, two cards, non-Linux) must
    not silently fall back to the host-RAM figure."""
    monkeypatch.setattr(cache_policy, "_discrete_vram_bytes", lambda: None)
    monkeypatch.setattr(cache_policy, "get_visible_memory_node", lambda: [0])
    monkeypatch.setattr(
        cache_policy,
        "get_memory_node_info",
        lambda _n: SimpleNamespace(
            total_memory=128 * GIB, available_memory=100 * GIB
        ),
    )

    available = determine_available_memory(_worker(discrete=True))

    assert available == _UNMEASURABLE_VRAM_KV_BUDGET


def test_explicit_kv_cache_bytes_wins_over_every_probe(monkeypatch):
    monkeypatch.setattr(
        cache_policy,
        "_discrete_vram_bytes",
        lambda: pytest.fail("must not probe when the budget is set explicitly"),
    )
    worker = _worker(discrete=True)
    worker.cache_config.kv_cache_memory_bytes = 7 * GIB

    assert determine_available_memory(worker) == 7 * GIB


def test_vram_probe_declines_when_the_card_is_ambiguous(monkeypatch, tmp_path):
    """Two VRAM-reporting nodes: nothing says which one wgpu opened, so the
    probe must return None rather than guess."""
    for card in ("card0", "card1"):
        d = tmp_path / card / "device"
        d.mkdir(parents=True)
        (d / "mem_info_vram_total").write_text(str(8 * GIB))
        (d / "mem_info_vram_used").write_text(str(GIB))
    monkeypatch.setattr(
        cache_policy, "_VRAM_TOTAL_GLOB", str(tmp_path / "card*/device/mem_info_vram_total")
    )

    assert cache_policy._discrete_vram_bytes() is None


def test_vram_probe_reads_total_and_used(monkeypatch, tmp_path):
    d = tmp_path / "card0" / "device"
    d.mkdir(parents=True)
    (d / "mem_info_vram_total").write_text(str(24 * GIB))
    (d / "mem_info_vram_used").write_text(str(3 * GIB))
    monkeypatch.setattr(
        cache_policy, "_VRAM_TOTAL_GLOB", str(tmp_path / "card*/device/mem_info_vram_total")
    )

    assert cache_policy._discrete_vram_bytes() == (24 * GIB, 3 * GIB)
