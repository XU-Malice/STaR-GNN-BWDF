"""Tests for physical-GPU visibility-independent recommendations."""

from __future__ import annotations

from dma_wdf.utils.gpu_inspector import (
    parse_nvidia_smi_csv,
    recommend_physical_gpus,
    visible_logical_mapping,
)


SAMPLE = """\
4, NVIDIA GeForce RTX 4090, 24564, 19208, 5356, 0
5, NVIDIA GeForce RTX 4090, 24564, 22773, 1791, 0
6, NVIDIA GeForce RTX 4090, 24564, 0, 24564, 0
7, NVIDIA GeForce RTX 4090, 24564, 15254, 9310, 0
"""


def test_recommends_only_physical_gpu_6() -> None:
    statuses = parse_nvidia_smi_csv(SAMPLE)
    recommended = recommend_physical_gpus(statuses, count=2)
    assert [value.physical_index for value in recommended] == [6]


def test_visible_mapping_explains_physical_4_5_as_logical_0_1() -> None:
    statuses = parse_nvidia_smi_csv(SAMPLE)
    mapping = visible_logical_mapping(statuses, "4,5")
    assert mapping[0]["logical_index"] == 0
    assert mapping[0]["physical_index"] == 4
    assert mapping[1]["logical_index"] == 1
    assert mapping[1]["physical_index"] == 5
