"""CUDA device discovery and conservative training-device selection.

The selector only sees GPUs exposed to the current process, so it naturally
respects ``CUDA_VISIBLE_DEVICES``.  Returned indices are PyTorch *logical*
indices.  No CPU fallback is performed for ``requested="auto"``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch


MIB = 1024**2


@dataclass(frozen=True)
class GPUStatus:
    """One visible CUDA device at selection time."""

    logical_index: int
    name: str
    total_memory_mib: float
    free_memory_mib: float
    used_memory_mib: float
    utilization_percent: float | None

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeviceSelection:
    """Resolved torch device plus the discovery snapshot."""

    requested: str
    device: torch.device
    selected_gpu: GPUStatus | None
    visible_gpus: tuple[GPUStatus, ...]
    explicit_cpu: bool

    def state_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "device": str(self.device),
            "selected_gpu": (
                None
                if self.selected_gpu is None
                else self.selected_gpu.state_dict()
            ),
            "visible_gpus": [
                status.state_dict()
                for status in self.visible_gpus
            ],
            "explicit_cpu": self.explicit_cpu,
        }


def query_visible_gpus() -> list[GPUStatus]:
    """Query memory and utilization for every PyTorch-visible GPU."""
    if not torch.cuda.is_available():
        return []

    statuses: list[GPUStatus] = []
    for logical_index in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(logical_index)
        free_mib = float(free_bytes / MIB)
        total_mib = float(total_bytes / MIB)
        utilization: float | None
        try:
            utilization = float(torch.cuda.utilization(logical_index))
        except (
            AttributeError,
            ImportError,
            RuntimeError,
            OSError,
            ValueError,
        ):
            utilization = None
        statuses.append(
            GPUStatus(
                logical_index=logical_index,
                name=torch.cuda.get_device_name(logical_index),
                total_memory_mib=total_mib,
                free_memory_mib=free_mib,
                used_memory_mib=total_mib - free_mib,
                utilization_percent=utilization,
            )
        )
    return statuses


def gpu_rejection_reasons(
    status: GPUStatus,
    *,
    minimum_free_memory_mib: float,
    maximum_used_memory_mib: float,
    maximum_gpu_utilization_percent: float,
) -> list[str]:
    """Return every reason a GPU is unsuitable for a new training run."""
    reasons: list[str] = []
    if status.free_memory_mib < float(minimum_free_memory_mib):
        reasons.append(
            "free_memory_mib="
            f"{status.free_memory_mib:.0f}"
            f"<{float(minimum_free_memory_mib):.0f}"
        )
    if status.used_memory_mib > float(maximum_used_memory_mib):
        reasons.append(
            "used_memory_mib="
            f"{status.used_memory_mib:.0f}"
            f">{float(maximum_used_memory_mib):.0f}"
        )
    if (
        status.utilization_percent is not None
        and status.utilization_percent
        > float(maximum_gpu_utilization_percent)
    ):
        reasons.append(
            "utilization_percent="
            f"{status.utilization_percent:.0f}"
            f">{float(maximum_gpu_utilization_percent):.0f}"
        )
    return reasons


def select_gpu_statuses(
    statuses: Sequence[GPUStatus],
    *,
    count: int,
    minimum_free_memory_mib: float,
    maximum_used_memory_mib: float,
    maximum_gpu_utilization_percent: float,
) -> list[GPUStatus]:
    """Select distinct eligible GPUs, preferring more free memory."""
    count = int(count)
    if count <= 0:
        raise ValueError("count must be positive.")

    eligible = [
        status
        for status in statuses
        if not gpu_rejection_reasons(
            status,
            minimum_free_memory_mib=minimum_free_memory_mib,
            maximum_used_memory_mib=maximum_used_memory_mib,
            maximum_gpu_utilization_percent=(
                maximum_gpu_utilization_percent
            ),
        )
    ]
    eligible.sort(
        key=lambda status: (
            -status.free_memory_mib,
            (
                float("inf")
                if status.utilization_percent is None
                else status.utilization_percent
            ),
            status.logical_index,
        )
    )
    if len(eligible) < count:
        details = []
        for status in statuses:
            reasons = gpu_rejection_reasons(
                status,
                minimum_free_memory_mib=minimum_free_memory_mib,
                maximum_used_memory_mib=maximum_used_memory_mib,
                maximum_gpu_utilization_percent=(
                    maximum_gpu_utilization_percent
                ),
            )
            details.append(
                f"cuda:{status.logical_index} {status.name}: "
                + (", ".join(reasons) if reasons else "eligible")
            )
        raise RuntimeError(
            f"Need {count} eligible GPU(s), found {len(eligible)}. "
            "No training process was started.\n"
            + "\n".join(details)
        )
    return eligible[:count]


def _runtime_thresholds(
    runtime_config: dict[str, Any],
) -> dict[str, float]:
    return {
        "minimum_free_memory_mib": float(
            runtime_config["minimum_free_memory_mib"]
        ),
        "maximum_used_memory_mib": float(
            runtime_config["maximum_used_memory_mib"]
        ),
        "maximum_gpu_utilization_percent": float(
            runtime_config["maximum_gpu_utilization_percent"]
        ),
    }


def resolve_training_device(
    requested: str,
    *,
    runtime_config: dict[str, Any],
) -> DeviceSelection:
    """Resolve ``auto``, explicit ``cuda:N``, or explicit debug ``cpu``.

    ``auto`` never falls back to CPU.  ``cpu`` is accepted only when the
    caller explicitly requests it, which is useful for smoke tests.
    """
    requested = str(requested).strip().lower()
    if requested == "cpu":
        return DeviceSelection(
            requested=requested,
            device=torch.device("cpu"),
            selected_gpu=None,
            visible_gpus=tuple(query_visible_gpus()),
            explicit_cpu=True,
        )
    if requested == "cuda":
        raise ValueError(
            "Use --device auto or an explicit logical index such as "
            "--device cuda:0; bare 'cuda' is intentionally rejected."
        )
    if requested != "auto" and not requested.startswith("cuda:"):
        raise ValueError(
            "device must be 'auto', 'cpu', or 'cuda:N'."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable and automatic CPU fallback is disabled."
        )

    statuses = query_visible_gpus()
    thresholds = _runtime_thresholds(runtime_config)
    if requested == "auto":
        selected = select_gpu_statuses(
            statuses,
            count=1,
            **thresholds,
        )[0]
    else:
        try:
            logical_index = int(requested.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(
                f"Invalid CUDA device string: {requested!r}."
            ) from exc
        matches = [
            status
            for status in statuses
            if status.logical_index == logical_index
        ]
        if not matches:
            raise RuntimeError(
                f"{requested} is not visible to PyTorch; "
                f"visible logical indices are "
                f"{[s.logical_index for s in statuses]}."
            )
        selected = matches[0]
        reasons = gpu_rejection_reasons(
            selected,
            **thresholds,
        )
        if reasons:
            raise RuntimeError(
                f"{requested} does not satisfy training thresholds: "
                + ", ".join(reasons)
            )

    torch.cuda.set_device(selected.logical_index)
    return DeviceSelection(
        requested=requested,
        device=torch.device(f"cuda:{selected.logical_index}"),
        selected_gpu=selected,
        visible_gpus=tuple(statuses),
        explicit_cpu=False,
    )
