"""STGCN baseline for multi-DMA water-demand forecasting.

This module implements the convolutional architecture introduced by
Yu, Yin, and Zhu (IJCAI 2018), adapted to the fixed BWDF experiment
contract:

* two spatio-temporal convolution blocks;
* gated temporal convolutions;
* Chebyshev spectral graph convolution;
* one training-period positive-Pearson graph shared by both tasks;
* direct 24 h or 168 h decoding with known future calendar features.

The graph relationship is identical to the DCRNN baseline, but its model
support is not.  DCRNN consumes the random-walk matrix.  STGCN derives a
scaled symmetric-normalized Laplacian from ``static_adj`` and constructs
Chebyshev polynomial supports.  The identity Chebyshev term supplies self
information, so the stored adjacency remains loop-free.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from dma_wdf.data.graph import load_graph


def _positive_int(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def validate_static_adjacency(
    adjacency: torch.Tensor,
    *,
    num_nodes: int,
    atol: float = 1.0e-6,
) -> torch.Tensor:
    """Validate the fixed undirected non-negative loop-free adjacency."""
    matrix = torch.as_tensor(
        adjacency,
        dtype=torch.float64,
    ).detach().clone()
    expected = (int(num_nodes), int(num_nodes))
    if matrix.ndim != 2 or tuple(matrix.shape) != expected:
        raise ValueError(
            f"static_adj must have shape {expected}, got "
            f"{tuple(matrix.shape)}."
        )
    if not torch.isfinite(matrix).all():
        raise ValueError("static_adj contains NaN/Inf.")
    if torch.any(matrix < -atol):
        raise ValueError("static_adj must be non-negative.")
    if not torch.allclose(
        matrix,
        matrix.T,
        atol=atol,
        rtol=0.0,
    ):
        raise ValueError("static_adj must be symmetric.")
    if not torch.allclose(
        torch.diag(matrix),
        torch.zeros(num_nodes, dtype=matrix.dtype),
        atol=atol,
        rtol=0.0,
    ):
        raise ValueError("static_adj diagonal must be zero.")
    degree = matrix.sum(dim=1)
    if torch.any(degree <= atol):
        raise ValueError("static_adj contains a zero-degree node.")
    return matrix


def build_chebyshev_supports(
    adjacency: torch.Tensor,
    *,
    chebyshev_order: int,
) -> tuple[torch.Tensor, float]:
    """Return ``[T_0(L~), ..., T_(K-1)(L~)]`` and ``lambda_max``.

    ``L = I - D^-1/2 A D^-1/2`` and
    ``L~ = 2 L / lambda_max - I``.  ``chebyshev_order`` is the number of
    retained polynomial bases, matching the ``K_s`` convention in STGCN.
    """
    order = _positive_int("chebyshev_order", chebyshev_order)
    matrix = torch.as_tensor(adjacency, dtype=torch.float64)
    num_nodes = int(matrix.shape[0])
    degree = matrix.sum(dim=1)
    inv_sqrt = torch.rsqrt(degree)
    normalized_adj = (
        inv_sqrt[:, None] * matrix * inv_sqrt[None, :]
    )
    identity = torch.eye(num_nodes, dtype=torch.float64)
    laplacian = identity - normalized_adj
    eigenvalues = torch.linalg.eigvalsh(laplacian)
    lambda_max = float(eigenvalues.max().item())
    if not np.isfinite(lambda_max) or lambda_max <= 1.0e-12:
        raise ValueError(
            f"Invalid normalized-Laplacian lambda_max={lambda_max}."
        )
    scaled = (2.0 / lambda_max) * laplacian - identity

    supports = [identity]
    if order >= 2:
        supports.append(scaled)
    for _ in range(2, order):
        supports.append(
            2.0 * scaled @ supports[-1] - supports[-2]
        )
    stacked = torch.stack(supports, dim=0).to(torch.float32)
    if not torch.isfinite(stacked).all():
        raise ValueError("Chebyshev supports contain NaN/Inf.")
    return stacked, lambda_max


class TemporalGatedConv(nn.Module):
    """Valid temporal convolution with a GLU and aligned residual path."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        self.input_channels = _positive_int(
            "input_channels",
            input_channels,
        )
        self.output_channels = _positive_int(
            "output_channels",
            output_channels,
        )
        self.kernel_size = _positive_int("kernel_size", kernel_size)
        self.convolution = nn.Conv2d(
            self.input_channels,
            2 * self.output_channels,
            kernel_size=(self.kernel_size, 1),
        )
        self.residual_projection = (
            nn.Conv2d(
                self.input_channels,
                self.output_channels,
                kernel_size=(1, 1),
            )
            if self.input_channels != self.output_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "TemporalGatedConv expects (B,C,T,N), got "
                f"{tuple(x.shape)}."
            )
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} channels, got "
                f"{x.shape[1]}."
            )
        if x.shape[2] < self.kernel_size:
            raise ValueError(
                f"Temporal length {x.shape[2]} is shorter than "
                f"kernel_size={self.kernel_size}."
            )
        convolution = self.convolution(x)
        candidate, gate = convolution.chunk(2, dim=1)
        residual = self.residual_projection(
            x[:, :, self.kernel_size - 1 :, :]
        )
        return (candidate + residual) * torch.sigmoid(gate)


class ChebyshevGraphConv(nn.Module):
    """Chebyshev spectral graph convolution at every temporal position."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        chebyshev_order: int,
    ) -> None:
        super().__init__()
        self.input_channels = _positive_int(
            "input_channels",
            input_channels,
        )
        self.output_channels = _positive_int(
            "output_channels",
            output_channels,
        )
        self.chebyshev_order = _positive_int(
            "chebyshev_order",
            chebyshev_order,
        )
        self.weight = nn.Parameter(
            torch.empty(
                self.chebyshev_order,
                self.input_channels,
                self.output_channels,
            )
        )
        self.bias = nn.Parameter(torch.zeros(self.output_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        x: torch.Tensor,
        supports: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "ChebyshevGraphConv expects (B,C,T,N), got "
                f"{tuple(x.shape)}."
            )
        expected_support = (
            self.chebyshev_order,
            x.shape[3],
            x.shape[3],
        )
        if tuple(supports.shape) != expected_support:
            raise ValueError(
                f"supports must have shape {expected_support}, got "
                f"{tuple(supports.shape)}."
            )
        propagated = torch.einsum(
            "knm,bctm->bkctn",
            supports,
            x,
        )
        output = torch.einsum(
            "bkctn,kco->botn",
            propagated,
            self.weight,
        )
        return output + self.bias.view(1, -1, 1, 1)


class STConvBlock(nn.Module):
    """Temporal gated convolution -> graph convolution -> temporal gate."""

    def __init__(
        self,
        *,
        input_channels: int,
        temporal_channels: int,
        spatial_channels: int,
        output_channels: int,
        temporal_kernel_size: int,
        chebyshev_order: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0,1).")
        self.temporal_in = TemporalGatedConv(
            input_channels,
            temporal_channels,
            temporal_kernel_size,
        )
        self.spatial = ChebyshevGraphConv(
            temporal_channels,
            spatial_channels,
            chebyshev_order,
        )
        self.temporal_out = TemporalGatedConv(
            spatial_channels,
            output_channels,
            temporal_kernel_size,
        )
        self.activation = nn.ReLU()
        self.normalization = nn.LayerNorm(output_channels)
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        x: torch.Tensor,
        supports: torch.Tensor,
    ) -> torch.Tensor:
        output = self.temporal_in(x)
        output = self.activation(self.spatial(output, supports))
        output = self.temporal_out(output)
        output = output.permute(0, 2, 3, 1)
        output = self.normalization(output)
        output = output.permute(0, 3, 1, 2)
        return self.dropout(output)


class STGCN(nn.Module):
    """Task-configurable direct-decoding STGCN.

    Historical features pass through two ST-Conv blocks.  A learned temporal
    collapse uses every surviving historical position.  The resulting node
    context is fused with known future calendar variables and a learned lead
    embedding, then projected directly to all forecast steps.  No target is
    accepted by the decoder, preventing teacher-forcing leakage by design.
    """

    model_name = "stgcn"

    def __init__(
        self,
        *,
        chebyshev_supports: torch.Tensor,
        input_dim: int,
        future_exog_dim: int,
        history_hours: int,
        horizon: int,
        num_nodes: int,
        temporal_kernel_size: int,
        block_channels: Sequence[Sequence[int]],
        head_channels: int,
        dropout: float = 0.0,
        graph_metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = _positive_int("input_dim", input_dim)
        self.future_exog_dim = int(future_exog_dim)
        if self.future_exog_dim < 0:
            raise ValueError("future_exog_dim must be non-negative.")
        self.history_hours = _positive_int(
            "history_hours",
            history_hours,
        )
        self.horizon = _positive_int("horizon", horizon)
        self.num_nodes = _positive_int("num_nodes", num_nodes)
        self.temporal_kernel_size = _positive_int(
            "temporal_kernel_size",
            temporal_kernel_size,
        )
        self.head_channels = _positive_int(
            "head_channels",
            head_channels,
        )

        supports = torch.as_tensor(
            chebyshev_supports,
            dtype=torch.float32,
        ).detach().clone()
        if supports.ndim != 3:
            raise ValueError(
                "chebyshev_supports must have shape (K,N,N)."
            )
        if tuple(supports.shape[1:]) != (
            self.num_nodes,
            self.num_nodes,
        ):
            raise ValueError(
                "Chebyshev support node shape does not match num_nodes."
            )
        if not torch.isfinite(supports).all():
            raise ValueError("chebyshev_supports contains NaN/Inf.")
        self.chebyshev_order = int(supports.shape[0])
        self.register_buffer(
            "chebyshev_supports",
            supports,
            persistent=True,
        )
        self.graph_metadata = dict(graph_metadata or {})

        parsed_channels: list[tuple[int, int, int]] = []
        for index, channels in enumerate(block_channels):
            values = tuple(int(value) for value in channels)
            if len(values) != 3 or any(value <= 0 for value in values):
                raise ValueError(
                    f"block_channels[{index}] must contain three "
                    f"positive integers, got {channels!r}."
                )
            parsed_channels.append(values)
        if len(parsed_channels) != 2:
            raise ValueError(
                "The fixed STGCN baseline requires exactly two "
                "ST-Conv blocks."
            )
        self.block_channels = tuple(parsed_channels)

        blocks: list[STConvBlock] = []
        current_channels = self.input_dim
        remaining_steps = self.history_hours
        for temporal, spatial, output in parsed_channels:
            blocks.append(
                STConvBlock(
                    input_channels=current_channels,
                    temporal_channels=temporal,
                    spatial_channels=spatial,
                    output_channels=output,
                    temporal_kernel_size=self.temporal_kernel_size,
                    chebyshev_order=self.chebyshev_order,
                    dropout=float(dropout),
                )
            )
            current_channels = output
            remaining_steps -= 2 * (
                self.temporal_kernel_size - 1
            )
        if remaining_steps <= 0:
            raise ValueError(
                "history_hours is too short for the configured temporal "
                "kernels and two ST-Conv blocks."
            )
        self.remaining_temporal_steps = int(remaining_steps)
        self.blocks = nn.ModuleList(blocks)

        # This is the long-history counterpart of STGCN's output temporal
        # convolution: it collapses every encoded historical position.
        self.temporal_collapse = nn.Conv2d(
            current_channels,
            self.head_channels,
            kernel_size=(self.remaining_temporal_steps, 1),
        )
        self.future_projection = (
            nn.Linear(self.future_exog_dim, self.head_channels)
            if self.future_exog_dim > 0
            else None
        )
        self.horizon_embedding = nn.Embedding(
            self.horizon,
            self.head_channels,
        )
        self.fusion_norm = nn.LayerNorm(self.head_channels)
        self.output_projection = nn.Linear(self.head_channels, 1)

    def forward(
        self,
        x_past: torch.Tensor,
        *,
        y_target: torch.Tensor | None = None,
        x_future_exog: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        if y_target is not None:
            raise ValueError(
                "STGCN is a direct decoder and never accepts y_target."
            )
        if float(teacher_forcing_ratio) != 0.0:
            raise ValueError(
                "STGCN does not use teacher forcing; ratio must be 0."
            )
        expected_prefix = (
            self.history_hours,
            self.num_nodes,
            self.input_dim,
        )
        if x_past.ndim != 4 or tuple(x_past.shape[1:]) != expected_prefix:
            raise ValueError(
                "x_past must have shape "
                f"(B,{self.history_hours},{self.num_nodes},"
                f"{self.input_dim}), got {tuple(x_past.shape)}."
            )
        if not torch.is_floating_point(x_past):
            raise TypeError("x_past must be floating point.")

        batch_size = int(x_past.shape[0])
        if self.future_exog_dim > 0:
            expected_future = (
                batch_size,
                self.horizon,
                self.num_nodes,
                self.future_exog_dim,
            )
            if (
                x_future_exog is None
                or tuple(x_future_exog.shape) != expected_future
            ):
                observed = (
                    None
                    if x_future_exog is None
                    else tuple(x_future_exog.shape)
                )
                raise ValueError(
                    f"x_future_exog must have shape {expected_future}, "
                    f"got {observed}."
                )
        elif x_future_exog is not None and x_future_exog.shape[-1] != 0:
            raise ValueError(
                "x_future_exog supplied although future_exog_dim=0."
            )

        encoded = x_past.permute(0, 3, 1, 2)
        for block in self.blocks:
            encoded = block(encoded, self.chebyshev_supports)
        context = self.temporal_collapse(encoded)
        if tuple(context.shape[2:]) != (1, self.num_nodes):
            raise RuntimeError(
                "Temporal collapse did not produce one context step."
            )
        context = context.squeeze(2).permute(0, 2, 1)
        fused = context[:, None, :, :].expand(
            -1,
            self.horizon,
            -1,
            -1,
        )
        if self.future_projection is not None:
            assert x_future_exog is not None
            fused = fused + self.future_projection(x_future_exog)
        lead = torch.arange(
            self.horizon,
            device=x_past.device,
        )
        fused = fused + self.horizon_embedding(lead)[None, :, None, :]
        fused = torch.tanh(self.fusion_norm(fused))
        return self.output_projection(fused).squeeze(-1)

    def model_metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "input_dim": self.input_dim,
            "future_exog_dim": self.future_exog_dim,
            "history_hours": self.history_hours,
            "horizon": self.horizon,
            "num_nodes": self.num_nodes,
            "temporal_kernel_size": self.temporal_kernel_size,
            "chebyshev_order": self.chebyshev_order,
            "block_channels": [
                list(values)
                for values in self.block_channels
            ],
            "head_channels": self.head_channels,
            "remaining_temporal_steps": (
                self.remaining_temporal_steps
            ),
            "decoder": "direct_future_conditioned",
            "graph": dict(self.graph_metadata),
        }


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_graph_contract(
    graph: dict[str, Any],
    graph_config: dict[str, Any],
) -> torch.Tensor:
    expectations = {
        "graph_method": graph_config["expected_method"],
        "corr_threshold": graph_config["expected_corr_threshold"],
        "negative_policy": graph_config["expected_negative_policy"],
        "self_loop_in_adjacency": graph_config[
            "expected_self_loop_in_adjacency"
        ],
        "static": True,
    }
    for key, expected in expectations.items():
        if graph.get(key) != expected:
            raise ValueError(
                f"Graph contract mismatch for {key}: "
                f"{graph.get(key)!r} != {expected!r}."
            )
    names = [str(value) for value in graph_config["node_names"]]
    columns = [str(value) for value in graph_config["dma_columns"]]
    if graph["node_names"] != names:
        raise ValueError("Graph node order mismatch.")
    if graph["dma_columns"] != columns:
        raise ValueError("Graph DMA column order mismatch.")
    return validate_static_adjacency(
        torch.from_numpy(
            np.asarray(graph["static_adj"], dtype=np.float64)
        ),
        num_nodes=int(graph_config["expected_nodes"]),
    )


def build_stgcn_model(
    config: dict[str, Any],
    *,
    project_root: Path,
    input_dim: int,
    future_exog_dim: int,
    horizon: int,
    history_hours: int | None = None,
    device: torch.device | str | None = None,
) -> STGCN:
    """Load the frozen Pearson adjacency and construct STGCN."""
    if "model" not in config or "graph" not in config:
        raise ValueError("config must contain model and graph sections.")
    model_config = config["model"]
    graph_config = config["graph"]
    if str(model_config["name"]).lower() != "stgcn":
        raise ValueError(
            "build_stgcn_model requires model.name='stgcn'."
        )
    if graph_config["matrix_key"] != "static_adj":
        raise ValueError(
            "STGCN requires graph.matrix_key='static_adj'."
        )
    if int(horizon) not in {24, 168}:
        raise ValueError(
            f"BWDF horizon must be 24 or 168, got {horizon}."
        )
    resolved_history = (
        int(history_hours)
        if history_hours is not None
        else int(config["task"]["history_hours"])
    )

    project_root = project_root.resolve()
    graph_path = _resolve(
        project_root,
        graph_config["artifact_path"],
    )
    if not graph_path.is_file():
        raise FileNotFoundError(
            f"Graph artifact does not exist: {graph_path}"
        )
    graph = load_graph(graph_path)
    adjacency = _validate_graph_contract(graph, graph_config)
    chebyshev_order = int(model_config["chebyshev_order"])
    supports, lambda_max = build_chebyshev_supports(
        adjacency,
        chebyshev_order=chebyshev_order,
    )

    graph_metadata = {
        "artifact_path": str(graph_path),
        "artifact_sha256": _sha256(graph_path),
        "demand_sha256": graph["demand_sha256"],
        "graph_method": graph["graph_method"],
        "corr_threshold": graph["corr_threshold"],
        "normalization": "scaled_symmetric_normalized_laplacian",
        "source_artifact_normalization": graph["normalization"],
        "matrix_key": graph_config["matrix_key"],
        "chebyshev_order": chebyshev_order,
        "lambda_max": lambda_max,
        "fit_start": graph["fit_start"],
        "fit_end": graph["fit_end"],
        "fit_rows": graph["fit_rows"],
        "node_names": graph["node_names"],
        "dma_columns": graph["dma_columns"],
    }
    model = STGCN(
        chebyshev_supports=supports,
        input_dim=int(input_dim),
        future_exog_dim=int(future_exog_dim),
        history_hours=resolved_history,
        horizon=int(horizon),
        num_nodes=int(model_config["num_nodes"]),
        temporal_kernel_size=int(
            model_config["temporal_kernel_size"]
        ),
        block_channels=model_config["block_channels"],
        head_channels=int(model_config["head_channels"]),
        dropout=float(model_config.get("dropout", 0.0)),
        graph_metadata=graph_metadata,
    )
    if device is not None:
        model = model.to(device)
    return model
