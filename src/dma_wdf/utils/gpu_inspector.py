"""Read-only physical NVIDIA GPU inventory and recommendation logic."""

from __future__ import annotations

import csv
import os
import subprocess
from dataclasses import asdict, dataclass
from io import StringIO
from typing import Any, Sequence


@dataclass(frozen=True)
class PhysicalGPUStatus:
    physical_index: int
    name: str
    total_memory_mib: float
    used_memory_mib: float
    free_memory_mib: float
    utilization_percent: float

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_nvidia_smi_csv(text: str) -> list[PhysicalGPUStatus]:
    """Parse the stable no-header, no-unit nvidia-smi CSV format."""
    statuses: list[PhysicalGPUStatus] = []
    for row in csv.reader(StringIO(text.strip())):
        if not row:
            continue
        if len(row) != 6:
            raise ValueError(f"Unexpected nvidia-smi row: {row!r}")
        statuses.append(
            PhysicalGPUStatus(
                physical_index=int(row[0].strip()),
                name=row[1].strip(),
                total_memory_mib=float(row[2].strip()),
                used_memory_mib=float(row[3].strip()),
                free_memory_mib=float(row[4].strip()),
                utilization_percent=float(row[5].strip()),
            )
        )
    return statuses


def query_physical_gpus() -> list[PhysicalGPUStatus]:
    """Query every physical GPU without changing CUDA visibility."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("nvidia-smi was not found.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"nvidia-smi failed: {exc.stderr.strip()}"
        ) from exc
    return parse_nvidia_smi_csv(completed.stdout)


def physical_gpu_rejection_reasons(
    status: PhysicalGPUStatus,
    *,
    minimum_free_memory_mib: float,
    maximum_used_memory_mib: float,
    maximum_gpu_utilization_percent: float,
) -> list[str]:
    reasons: list[str] = []
    if status.free_memory_mib < minimum_free_memory_mib:
        reasons.append(
            f"free={status.free_memory_mib:.0f}"
            f"<{minimum_free_memory_mib:.0f}MiB"
        )
    if status.used_memory_mib > maximum_used_memory_mib:
        reasons.append(
            f"used={status.used_memory_mib:.0f}"
            f">{maximum_used_memory_mib:.0f}MiB"
        )
    if status.utilization_percent > maximum_gpu_utilization_percent:
        reasons.append(
            f"util={status.utilization_percent:.0f}"
            f">{maximum_gpu_utilization_percent:.0f}%"
        )
    return reasons


def recommend_physical_gpus(
    statuses: Sequence[PhysicalGPUStatus],
    *,
    count: int,
    minimum_free_memory_mib: float = 8192,
    maximum_used_memory_mib: float = 2048,
    maximum_gpu_utilization_percent: float = 20,
) -> list[PhysicalGPUStatus]:
    """Return lightly occupied physical GPUs, most-free first."""
    if int(count) <= 0:
        raise ValueError("count must be positive.")
    eligible = [
        status
        for status in statuses
        if not physical_gpu_rejection_reasons(
            status,
            minimum_free_memory_mib=minimum_free_memory_mib,
            maximum_used_memory_mib=maximum_used_memory_mib,
            maximum_gpu_utilization_percent=maximum_gpu_utilization_percent,
        )
    ]
    eligible.sort(
        key=lambda value: (
            -value.free_memory_mib,
            value.utilization_percent,
            value.physical_index,
        )
    )
    return eligible[: int(count)]


def visible_logical_mapping(
    statuses: Sequence[PhysicalGPUStatus],
    cuda_visible_devices: str | None = None,
) -> list[dict[str, Any]]:
    """Map numeric CUDA_VISIBLE_DEVICES entries to PyTorch logical IDs."""
    value = (
        os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is None
        else cuda_visible_devices
    )
    physical = {status.physical_index: status for status in statuses}
    if value is None or not value.strip():
        identifiers = [str(status.physical_index) for status in statuses]
    else:
        identifiers = [item.strip() for item in value.split(",") if item.strip()]
    mapping: list[dict[str, Any]] = []
    for logical_index, identifier in enumerate(identifiers):
        try:
            physical_index = int(identifier)
        except ValueError:
            physical_index = None
        status = physical.get(physical_index) if physical_index is not None else None
        mapping.append(
            {
                "logical_index": logical_index,
                "cuda_visible_identifier": identifier,
                "physical_index": physical_index,
                "status": None if status is None else status.state_dict(),
            }
        )
    return mapping
