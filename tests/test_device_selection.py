"""Tests for conservative, non-hardcoded training-device selection."""

from __future__ import annotations

import pytest

from dma_wdf.utils.device import (
    GPUStatus,
    gpu_rejection_reasons,
    select_gpu_statuses,
)


def _gpu(
    index: int,
    *,
    free: float,
    used: float,
    utilization: float | None,
) -> GPUStatus:
    return GPUStatus(
        logical_index=index,
        name=f"GPU-{index}",
        total_memory_mib=free + used,
        free_memory_mib=free,
        used_memory_mib=used,
        utilization_percent=utilization,
    )


def test_selector_rejects_busy_cards_and_chooses_most_free() -> None:
    statuses = [
        _gpu(0, free=4000, used=20000, utilization=0),
        _gpu(1, free=22000, used=1000, utilization=80),
        _gpu(2, free=21000, used=1000, utilization=0),
        _gpu(3, free=23000, used=500, utilization=0),
    ]
    selected = select_gpu_statuses(
        statuses,
        count=2,
        minimum_free_memory_mib=8192,
        maximum_used_memory_mib=2048,
        maximum_gpu_utilization_percent=20,
    )
    assert [gpu.logical_index for gpu in selected] == [3, 2]


def test_selector_starts_none_when_not_enough_cards() -> None:
    statuses = [
        _gpu(0, free=4000, used=20000, utilization=0),
        _gpu(1, free=22000, used=1000, utilization=80),
        _gpu(2, free=21000, used=1000, utilization=0),
    ]
    with pytest.raises(RuntimeError, match="Need 2 eligible"):
        select_gpu_statuses(
            statuses,
            count=2,
            minimum_free_memory_mib=8192,
            maximum_used_memory_mib=2048,
            maximum_gpu_utilization_percent=20,
        )


def test_unknown_utilization_can_still_be_memory_eligible() -> None:
    status = _gpu(
        0,
        free=22000,
        used=1000,
        utilization=None,
    )
    assert gpu_rejection_reasons(
        status,
        minimum_free_memory_mib=8192,
        maximum_used_memory_mib=2048,
        maximum_gpu_utilization_percent=20,
    ) == []
