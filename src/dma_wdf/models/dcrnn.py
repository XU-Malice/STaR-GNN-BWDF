"""DCRNN model components for multi-DMA water-demand forecasting.

The core model classes contain model mathematics only. The convenience
function :func:`build_dcrnn_model` loads and validates the fixed graph
artifact, then constructs the model. This module does not build datasets,
create optimizers, implement early stopping, or compute metrics.

The BWDF graph is an undirected positive-Pearson graph exported as one
row-stochastic random-walk matrix ``P``. Because forward and reverse random
walks are identical for this graph construction, the model uses one support
instead of duplicating two identical supports.

For maximum diffusion step ``K=2``, every diffusion convolution learns from
the three bases ``[X, P X, P² X]``. The identity term supplies self
information; therefore the saved adjacency does not need self-loops.

Reference
---------
Li et al., "Diffusion Convolutional Recurrent Neural Network:
Data-Driven Traffic Forecasting", ICLR 2018.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from dma_wdf.data.graph import load_graph


DEFAULT_MAX_DIFFUSION_STEP = 2
GATE_BIAS_START = 1.0


def _validate_positive_int(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _validate_probability(value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            "teacher_forcing_ratio must be in [0, 1], "
            f"got {value}."
        )
    return value


def validate_random_walk(
    random_walk: torch.Tensor,
    *,
    num_nodes: int,
    atol: float = 1.0e-6,
) -> torch.Tensor:
    """Validate and return a detached float tensor random-walk matrix."""
    support = torch.as_tensor(random_walk, dtype=torch.float32).detach().clone()
    if support.ndim != 2 or support.shape != (num_nodes, num_nodes):
        raise ValueError(
            "random_walk must have shape "
            f"({num_nodes}, {num_nodes}), got {tuple(support.shape)}."
        )
    if not torch.isfinite(support).all():
        raise ValueError("random_walk contains NaN/Inf.")
    if torch.any(support < -atol):
        raise ValueError("random_walk must be non-negative.")

    row_sums = support.sum(dim=1)
    if not torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=atol,
        rtol=0.0,
    ):
        raise ValueError(
            "Every random_walk row must sum to one; observed "
            f"{row_sums.tolist()}."
        )
    return support


class DiffusionConv(nn.Module):
    """Learnable diffusion convolution over one fixed random walk.

    For input ``X`` and maximum diffusion step ``K``, the layer concatenates
    ``[X, PX, ..., P^K X]`` along the feature dimension and applies one
    learned affine projection.

    Args:
        input_dim: Input features per node.
        output_dim: Output features per node.
        max_diffusion_step: Highest included diffusion order.
        bias_start: Constant used to initialise the affine bias.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        max_diffusion_step: int = DEFAULT_MAX_DIFFUSION_STEP,
        bias_start: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = _validate_positive_int("input_dim", input_dim)
        self.output_dim = _validate_positive_int("output_dim", output_dim)
        self.max_diffusion_step = int(max_diffusion_step)
        if self.max_diffusion_step < 0:
            raise ValueError(
                "max_diffusion_step must be non-negative, got "
                f"{self.max_diffusion_step}."
            )

        self.num_diffusion_terms = self.max_diffusion_step + 1
        self.weight = nn.Parameter(
            torch.empty(
                self.input_dim * self.num_diffusion_terms,
                self.output_dim,
            )
        )
        self.bias = nn.Parameter(torch.empty(self.output_dim))
        self._bias_start = float(bias_start)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, self._bias_start)

    def diffusion_terms(
        self,
        x: torch.Tensor,
        random_walk: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Return ``[X, PX, ..., P^K X]`` without applying weights."""
        if x.ndim != 3:
            raise ValueError(
                "DiffusionConv input must have shape (B,N,C), got "
                f"{tuple(x.shape)}."
            )
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {x.shape[-1]}."
            )
        if random_walk.shape != (x.shape[1], x.shape[1]):
            raise ValueError(
                "random_walk shape does not match input node count: "
                f"{tuple(random_walk.shape)} vs N={x.shape[1]}."
            )

        terms = [x]
        propagated = x
        for _ in range(self.max_diffusion_step):
            propagated = torch.einsum(
                "nm,bmc->bnc",
                random_walk,
                propagated,
            )
            terms.append(propagated)
        return terms

    def forward(
        self,
        x: torch.Tensor,
        random_walk: torch.Tensor,
    ) -> torch.Tensor:
        terms = self.diffusion_terms(x, random_walk)
        features = torch.cat(terms, dim=-1)
        return torch.matmul(features, self.weight) + self.bias

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"max_diffusion_step={self.max_diffusion_step}, "
            f"num_diffusion_terms={self.num_diffusion_terms}"
        )


class DCGRUCell(nn.Module):
    """Diffusion-convolutional GRU cell.

    The reset and update gates are obtained jointly from
    ``[X_t, H_{t-1}]``. The candidate state is computed from
    ``[X_t, r_t ⊙ H_{t-1}]``, matching the DCRNN formulation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        max_diffusion_step: int = DEFAULT_MAX_DIFFUSION_STEP,
    ) -> None:
        super().__init__()
        self.input_dim = _validate_positive_int("input_dim", input_dim)
        self.hidden_dim = _validate_positive_int("hidden_dim", hidden_dim)
        joint_dim = self.input_dim + self.hidden_dim

        # The official DCRNN implementation starts reset/update biases at 1.
        self.gate_conv = DiffusionConv(
            joint_dim,
            2 * self.hidden_dim,
            max_diffusion_step=max_diffusion_step,
            bias_start=GATE_BIAS_START,
        )
        self.candidate_conv = DiffusionConv(
            joint_dim,
            self.hidden_dim,
            max_diffusion_step=max_diffusion_step,
            bias_start=0.0,
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor,
        random_walk: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.shape[:-1] != x.shape[:-1]:
            raise ValueError(
                "x and hidden must share batch/node dimensions, got "
                f"{tuple(x.shape)} and {tuple(hidden.shape)}."
            )
        if hidden.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected hidden_dim={self.hidden_dim}, "
                f"got {hidden.shape[-1]}."
            )

        gates = torch.sigmoid(
            self.gate_conv(
                torch.cat([x, hidden], dim=-1),
                random_walk,
            )
        )
        reset, update = gates.chunk(2, dim=-1)
        candidate = torch.tanh(
            self.candidate_conv(
                torch.cat([x, reset * hidden], dim=-1),
                random_walk,
            )
        )
        return update * hidden + (1.0 - update) * candidate


class DCRNNEncoder(nn.Module):
    """Stacked DCGRU encoder over historical node features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        max_diffusion_step: int = DEFAULT_MAX_DIFFUSION_STEP,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = _validate_positive_int("input_dim", input_dim)
        self.hidden_dim = _validate_positive_int("hidden_dim", hidden_dim)
        self.num_layers = _validate_positive_int("num_layers", num_layers)
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.layers = nn.ModuleList(
            [
                DCGRUCell(
                    self.input_dim if layer == 0 else self.hidden_dim,
                    self.hidden_dim,
                    max_diffusion_step=max_diffusion_step,
                )
                for layer in range(self.num_layers)
            ]
        )
        self.dropout = (
            nn.Dropout(float(dropout))
            if float(dropout) > 0.0
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        random_walk: torch.Tensor,
        initial_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "Encoder input must have shape (B,L,N,C), got "
                f"{tuple(x.shape)}."
            )
        batch_size, sequence_length, num_nodes, _ = x.shape

        if initial_hidden is None:
            states = [
                x.new_zeros(batch_size, num_nodes, self.hidden_dim)
                for _ in range(self.num_layers)
            ]
        else:
            expected = (
                self.num_layers,
                batch_size,
                num_nodes,
                self.hidden_dim,
            )
            if tuple(initial_hidden.shape) != expected:
                raise ValueError(
                    f"initial_hidden must have shape {expected}, got "
                    f"{tuple(initial_hidden.shape)}."
                )
            states = [
                initial_hidden[layer]
                for layer in range(self.num_layers)
            ]

        for step in range(sequence_length):
            layer_input = x[:, step]
            next_states: list[torch.Tensor] = []
            for layer_index, cell in enumerate(self.layers):
                if layer_index > 0:
                    layer_input = self.dropout(layer_input)
                state = cell(
                    layer_input,
                    states[layer_index],
                    random_walk,
                )
                next_states.append(state)
                layer_input = state
            states = next_states
        return torch.stack(states, dim=0)


class DCRNNDecoder(nn.Module):
    """Autoregressive stacked-DCGRU decoder.

    Scheduled sampling is controlled by ``teacher_forcing_ratio`` but the
    schedule itself belongs to the later training stage.
    """

    def __init__(
        self,
        hidden_dim: int,
        horizon: int,
        future_exog_dim: int = 0,
        num_layers: int = 1,
        max_diffusion_step: int = DEFAULT_MAX_DIFFUSION_STEP,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = _validate_positive_int("hidden_dim", hidden_dim)
        self.horizon = _validate_positive_int("horizon", horizon)
        self.future_exog_dim = int(future_exog_dim)
        if self.future_exog_dim < 0:
            raise ValueError("future_exog_dim must be non-negative.")
        self.num_layers = _validate_positive_int("num_layers", num_layers)
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        decoder_input_dim = 1 + self.future_exog_dim
        self.layers = nn.ModuleList(
            [
                DCGRUCell(
                    decoder_input_dim if layer == 0 else self.hidden_dim,
                    self.hidden_dim,
                    max_diffusion_step=max_diffusion_step,
                )
                for layer in range(self.num_layers)
            ]
        )
        self.dropout = (
            nn.Dropout(float(dropout))
            if float(dropout) > 0.0
            else nn.Identity()
        )
        self.output_projection = nn.Linear(self.hidden_dim, 1)

    def forward(
        self,
        encoder_hidden: torch.Tensor,
        random_walk: torch.Tensor,
        *,
        target: torch.Tensor | None = None,
        future_exog: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        ratio = _validate_probability(teacher_forcing_ratio)
        if encoder_hidden.ndim != 4:
            raise ValueError(
                "encoder_hidden must have shape (layers,B,N,H)."
            )
        if encoder_hidden.shape[0] != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} decoder layers, got "
                f"{encoder_hidden.shape[0]}."
            )

        batch_size = encoder_hidden.shape[1]
        num_nodes = encoder_hidden.shape[2]
        if target is not None:
            expected_target = (batch_size, self.horizon, num_nodes)
            if tuple(target.shape) != expected_target:
                raise ValueError(
                    f"target must have shape {expected_target}, got "
                    f"{tuple(target.shape)}."
                )
        if self.training and ratio > 0.0 and target is None:
            raise ValueError(
                "target is required during training when "
                "teacher_forcing_ratio > 0."
            )

        if self.future_exog_dim > 0:
            expected_exog = (
                batch_size,
                self.horizon,
                num_nodes,
                self.future_exog_dim,
            )
            if future_exog is None:
                raise ValueError(
                    "future_exog is required because future_exog_dim="
                    f"{self.future_exog_dim}."
                )
            if tuple(future_exog.shape) != expected_exog:
                raise ValueError(
                    f"future_exog must have shape {expected_exog}, got "
                    f"{tuple(future_exog.shape)}."
                )
        elif future_exog is not None and future_exog.shape[-1] != 0:
            raise ValueError(
                "future_exog was supplied although future_exog_dim=0."
            )

        states = [
            encoder_hidden[layer]
            for layer in range(self.num_layers)
        ]
        previous_demand = encoder_hidden.new_zeros(
            batch_size,
            num_nodes,
        )
        predictions: list[torch.Tensor] = []

        for step in range(self.horizon):
            layer_input = previous_demand.unsqueeze(-1)
            if future_exog is not None:
                layer_input = torch.cat(
                    [layer_input, future_exog[:, step]],
                    dim=-1,
                )

            next_states: list[torch.Tensor] = []
            for layer_index, cell in enumerate(self.layers):
                if layer_index > 0:
                    layer_input = self.dropout(layer_input)
                state = cell(
                    layer_input,
                    states[layer_index],
                    random_walk,
                )
                next_states.append(state)
                layer_input = state
            states = next_states

            prediction = self.output_projection(layer_input).squeeze(-1)
            predictions.append(prediction)

            if self.training and target is not None and ratio > 0.0:
                if ratio >= 1.0:
                    previous_demand = target[:, step]
                else:
                    # The official DCRNN decoder draws one Bernoulli choice
                    # per decoding step and broadcasts it across the batch.
                    teacher_mask = (
                        torch.rand(
                            (),
                            device=prediction.device,
                        )
                        < ratio
                    )
                    previous_demand = torch.where(
                        teacher_mask,
                        target[:, step],
                        prediction,
                    )
            else:
                previous_demand = prediction

        return torch.stack(predictions, dim=1)


class DCRNN(nn.Module):
    """Task-configurable DCRNN with one fixed graph support.

    The same class is used for both 24 h and 168 h forecasting. Only the
    constructor's ``horizon`` changes; the graph artifact remains identical.
    """

    model_name = "dcrnn"

    def __init__(
        self,
        *,
        random_walk: torch.Tensor,
        input_dim: int,
        hidden_dim: int = 32,
        horizon: int = 24,
        num_nodes: int = 10,
        num_rnn_layers: int = 1,
        max_diffusion_step: int = DEFAULT_MAX_DIFFUSION_STEP,
        future_exog_dim: int = 0,
        dropout: float = 0.0,
        graph_metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = _validate_positive_int("input_dim", input_dim)
        self.hidden_dim = _validate_positive_int("hidden_dim", hidden_dim)
        self.horizon = _validate_positive_int("horizon", horizon)
        self.num_nodes = _validate_positive_int("num_nodes", num_nodes)
        self.num_rnn_layers = _validate_positive_int(
            "num_rnn_layers",
            num_rnn_layers,
        )
        self.max_diffusion_step = int(max_diffusion_step)
        if self.max_diffusion_step < 0:
            raise ValueError("max_diffusion_step must be non-negative.")
        self.future_exog_dim = int(future_exog_dim)
        if self.future_exog_dim < 0:
            raise ValueError("future_exog_dim must be non-negative.")

        support = validate_random_walk(
            random_walk,
            num_nodes=self.num_nodes,
        )
        self.register_buffer(
            "random_walk",
            support,
            persistent=True,
        )
        self.graph_metadata = dict(graph_metadata or {})

        self.encoder = DCRNNEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_rnn_layers,
            max_diffusion_step=self.max_diffusion_step,
            dropout=dropout,
        )
        self.decoder = DCRNNDecoder(
            hidden_dim=self.hidden_dim,
            horizon=self.horizon,
            future_exog_dim=self.future_exog_dim,
            num_layers=self.num_rnn_layers,
            max_diffusion_step=self.max_diffusion_step,
            dropout=dropout,
        )

    def forward(
        self,
        x_past: torch.Tensor,
        *,
        y_target: torch.Tensor | None = None,
        x_future_exog: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        if x_past.ndim != 4:
            raise ValueError(
                "x_past must have shape (B,L,N,C), got "
                f"{tuple(x_past.shape)}."
            )
        if x_past.shape[2] != self.num_nodes:
            raise ValueError(
                f"Expected {self.num_nodes} nodes, got {x_past.shape[2]}."
            )
        if x_past.shape[3] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, "
                f"got {x_past.shape[3]}."
            )
        if not torch.is_floating_point(x_past):
            raise TypeError("x_past must be a floating-point tensor.")
        ratio = _validate_probability(teacher_forcing_ratio)

        # Evaluation/inference is structurally prevented from reading targets.
        if not self.training:
            y_target = None
            ratio = 0.0

        encoded = self.encoder(
            x_past,
            self.random_walk,
        )
        return self.decoder(
            encoded,
            self.random_walk,
            target=y_target,
            future_exog=x_future_exog,
            teacher_forcing_ratio=ratio,
        )

    def model_metadata(self) -> dict[str, Any]:
        """Return serialisable architecture and graph-contract metadata."""
        return {
            "model_name": self.model_name,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "horizon": self.horizon,
            "num_nodes": self.num_nodes,
            "num_rnn_layers": self.num_rnn_layers,
            "max_diffusion_step": self.max_diffusion_step,
            "num_graph_supports": 1,
            "diffusion_terms": [
                f"P^{order}"
                for order in range(self.max_diffusion_step + 1)
            ],
            "future_exog_dim": self.future_exog_dim,
            "graph": dict(self.graph_metadata),
        }

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"horizon={self.horizon}, num_nodes={self.num_nodes}, "
            f"num_rnn_layers={self.num_rnn_layers}, "
            f"max_diffusion_step={self.max_diffusion_step}, "
            f"future_exog_dim={self.future_exog_dim}"
        )


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_graph_contract(
    graph: dict[str, Any],
    graph_config: dict[str, Any],
) -> None:
    """Validate that an artifact is the frozen BWDF Pearson graph."""
    expectations = {
        "graph_method": graph_config["expected_method"],
        "corr_threshold": graph_config["expected_corr_threshold"],
        "negative_policy": graph_config["expected_negative_policy"],
        "self_loop_in_adjacency": graph_config[
            "expected_self_loop_in_adjacency"
        ],
        "static": True,
        "normalization": "random_walk",
    }
    for key, expected in expectations.items():
        observed = graph.get(key)
        if observed != expected:
            raise ValueError(
                f"Graph contract mismatch for {key}: "
                f"observed={observed!r}, expected={expected!r}."
            )

    expected_names = [
        str(value)
        for value in graph_config["node_names"]
    ]
    expected_columns = [
        str(value)
        for value in graph_config["dma_columns"]
    ]
    if graph["node_names"] != expected_names:
        raise ValueError(
            "Graph node order mismatch: "
            f"{graph['node_names']} != {expected_names}."
        )
    if graph["dma_columns"] != expected_columns:
        raise ValueError(
            "Graph DMA column order mismatch: "
            f"{graph['dma_columns']} != {expected_columns}."
        )

    random_walk = np.asarray(graph["random_walk"], dtype=np.float64)
    expected_nodes = int(graph_config["expected_nodes"])
    if random_walk.shape != (expected_nodes, expected_nodes):
        raise ValueError(
            "Graph random_walk shape mismatch: "
            f"{random_walk.shape} != "
            f"({expected_nodes}, {expected_nodes})."
        )
    if not np.isfinite(random_walk).all():
        raise ValueError("Graph random_walk contains NaN/Inf.")
    if np.any(random_walk < -1.0e-10):
        raise ValueError("Graph random_walk contains negative entries.")
    if not np.allclose(
        random_walk.sum(axis=1),
        1.0,
        atol=1.0e-10,
        rtol=0.0,
    ):
        raise ValueError("Graph random_walk rows do not sum to one.")


def build_dcrnn_model(
    config: dict[str, Any],
    *,
    project_root: Path,
    input_dim: int,
    future_exog_dim: int,
    horizon: int,
    device: torch.device | str | None = None,
) -> DCRNN:
    """Load the fixed graph and construct DCRNN for 24 h or 168 h.

    This is the only model-construction entry point required in stage 3.
    The later unified train/test scripts will call this function whenever
    ``model.name`` is ``dcrnn``.
    """
    if "model" not in config or "graph" not in config:
        raise ValueError("config must contain model and graph sections.")
    model_config = config["model"]
    graph_config = config["graph"]
    if str(model_config["name"]).lower() != "dcrnn":
        raise ValueError(
            "build_dcrnn_model requires model.name='dcrnn'."
        )
    if graph_config["matrix_key"] != "random_walk":
        raise ValueError(
            "DCRNN requires graph.matrix_key='random_walk'."
        )
    if int(model_config["max_diffusion_step"]) != 2:
        raise ValueError(
            "The fixed BWDF DCRNN protocol requires "
            "max_diffusion_step=2."
        )
    if int(model_config["num_graph_supports"]) != 1:
        raise ValueError(
            "The undirected BWDF graph uses one random-walk support."
        )
    if int(horizon) not in {24, 168}:
        raise ValueError(
            f"BWDF horizon must be 24 or 168, got {horizon}."
        )

    project_root = project_root.resolve()
    graph_path = _resolve_path(
        project_root,
        graph_config["artifact_path"],
    )
    if not graph_path.is_file():
        raise FileNotFoundError(
            f"Graph artifact does not exist: {graph_path}"
        )
    graph = load_graph(graph_path)
    _validate_graph_contract(graph, graph_config)

    expected_nodes = int(graph_config["expected_nodes"])
    if int(model_config["num_nodes"]) != expected_nodes:
        raise ValueError(
            "model.num_nodes and graph.expected_nodes must match."
        )

    graph_metadata = {
        "artifact_path": str(graph_path),
        "artifact_sha256": _file_sha256(graph_path),
        "demand_sha256": graph["demand_sha256"],
        "graph_method": graph["graph_method"],
        "corr_threshold": graph["corr_threshold"],
        "normalization": graph["normalization"],
        "matrix_key": graph_config["matrix_key"],
        "fit_start": graph["fit_start"],
        "fit_end": graph["fit_end"],
        "fit_rows": graph["fit_rows"],
        "node_names": graph["node_names"],
        "dma_columns": graph["dma_columns"],
    }
    model = DCRNN(
        random_walk=torch.from_numpy(
            np.asarray(graph["random_walk"], dtype=np.float32)
        ),
        input_dim=int(input_dim),
        hidden_dim=int(model_config["hidden_dim"]),
        horizon=int(horizon),
        num_nodes=int(model_config["num_nodes"]),
        num_rnn_layers=int(model_config["num_rnn_layers"]),
        max_diffusion_step=int(
            model_config["max_diffusion_step"]
        ),
        future_exog_dim=int(future_exog_dim),
        dropout=float(model_config.get("dropout", 0.0)),
        graph_metadata=graph_metadata,
    )
    if device is not None:
        model = model.to(device)
    return model
