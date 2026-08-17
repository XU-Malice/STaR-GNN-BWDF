"""Model-only tests for the DCRNN implementation."""

from __future__ import annotations

import pytest
import torch

from dma_wdf.models.dcrnn import (
    DCGRUCell,
    DCRNN,
    DiffusionConv,
)


@pytest.fixture
def random_walk_10() -> torch.Tensor:
    """Positive row-stochastic 10-node support without self-loops."""
    matrix = torch.ones(10, 10, dtype=torch.float32)
    matrix.fill_diagonal_(0.0)
    return matrix / matrix.sum(dim=1, keepdim=True)


def _model(
    random_walk: torch.Tensor,
    *,
    horizon: int,
    input_dim: int = 12,
    hidden_dim: int = 8,
    future_exog_dim: int = 7,
    dropout: float = 0.0,
) -> DCRNN:
    return DCRNN(
        random_walk=random_walk,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        horizon=horizon,
        num_nodes=10,
        num_rnn_layers=1,
        max_diffusion_step=2,
        future_exog_dim=future_exog_dim,
        dropout=dropout,
    )


def test_k2_has_identity_first_and_second_order() -> None:
    conv = DiffusionConv(
        input_dim=1,
        output_dim=1,
        max_diffusion_step=2,
    )
    assert conv.num_diffusion_terms == 3
    assert conv.weight.shape == (3, 1)


def test_diffusion_terms_equal_x_px_p2x() -> None:
    support = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    x = torch.tensor(
        [[[1.0], [2.0], [4.0]]],
        dtype=torch.float32,
    )
    conv = DiffusionConv(1, 1, max_diffusion_step=2)
    terms = conv.diffusion_terms(x, support)
    assert len(terms) == 3
    assert torch.allclose(terms[0], x)
    assert torch.allclose(terms[1], support @ x)
    assert torch.allclose(terms[2], support @ support @ x)


def test_k2_projection_can_select_only_p2x() -> None:
    support = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    x = torch.tensor(
        [[[1.0], [2.0], [4.0]]],
        dtype=torch.float32,
    )
    conv = DiffusionConv(1, 1, max_diffusion_step=2)
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[2, 0] = 1.0
        conv.bias.zero_()
    observed = conv(x, support)
    expected = support @ support @ x
    assert torch.allclose(observed, expected)


def test_dcgru_gate_bias_matches_official_initialisation() -> None:
    cell = DCGRUCell(
        input_dim=3,
        hidden_dim=5,
        max_diffusion_step=2,
    )
    assert torch.allclose(
        cell.gate_conv.bias,
        torch.ones_like(cell.gate_conv.bias),
    )
    assert torch.allclose(
        cell.candidate_conv.bias,
        torch.zeros_like(cell.candidate_conv.bias),
    )


@pytest.mark.parametrize(
    ("horizon", "batch_size"),
    [(24, 1), (24, 4), (168, 2)],
)
def test_output_shapes(
    random_walk_10: torch.Tensor,
    horizon: int,
    batch_size: int,
) -> None:
    model = _model(random_walk_10, horizon=horizon)
    x = torch.randn(batch_size, 12, 10, 12)
    future = torch.randn(batch_size, horizon, 10, 7)
    model.eval()
    with torch.no_grad():
        output = model(
            x,
            x_future_exog=future,
        )
    assert output.shape == (batch_size, horizon, 10)
    assert torch.isfinite(output).all()


def test_full_history_contract_672h(
    random_walk_10: torch.Tensor,
) -> None:
    model = _model(
        random_walk_10,
        horizon=24,
        hidden_dim=4,
    )
    x = torch.randn(1, 672, 10, 12)
    future = torch.randn(1, 24, 10, 7)
    model.eval()
    with torch.no_grad():
        output = model(
            x,
            x_future_exog=future,
        )
    assert output.shape == (1, 24, 10)


def test_training_forward_with_teacher_forcing(
    random_walk_10: torch.Tensor,
) -> None:
    model = _model(
        random_walk_10,
        horizon=24,
    )
    model.train()
    x = torch.randn(3, 8, 10, 12)
    target = torch.randn(3, 24, 10)
    future = torch.randn(3, 24, 10, 7)
    output = model(
        x,
        y_target=target,
        x_future_exog=future,
        teacher_forcing_ratio=0.5,
    )
    assert output.shape == target.shape
    assert torch.isfinite(output).all()


def test_support_is_fixed_persistent_buffer(
    random_walk_10: torch.Tensor,
) -> None:
    model = _model(random_walk_10, horizon=24)
    buffers = dict(model.named_buffers())
    parameters = dict(model.named_parameters())
    state = model.state_dict()
    assert "random_walk" in buffers
    assert "random_walk" in state
    assert "random_walk" not in parameters
    assert buffers["random_walk"].requires_grad is False
    assert torch.equal(buffers["random_walk"], random_walk_10)


def test_state_dict_roundtrip_preserves_support(
    random_walk_10: torch.Tensor,
) -> None:
    first = _model(random_walk_10, horizon=24)
    second = _model(
        torch.eye(10),
        horizon=24,
    )
    second.load_state_dict(first.state_dict())
    assert torch.equal(first.random_walk, second.random_walk)


def test_metadata_declares_single_k2_support(
    random_walk_10: torch.Tensor,
) -> None:
    model = _model(random_walk_10, horizon=168)
    metadata = model.model_metadata()
    assert metadata["horizon"] == 168
    assert metadata["max_diffusion_step"] == 2
    assert metadata["num_graph_supports"] == 1
    assert metadata["diffusion_terms"] == ["P^0", "P^1", "P^2"]


def test_missing_future_exog_rejected(
    random_walk_10: torch.Tensor,
) -> None:
    model = _model(random_walk_10, horizon=24)
    x = torch.randn(2, 8, 10, 12)
    with pytest.raises(ValueError, match="future_exog is required"):
        model(x)


def test_wrong_node_count_rejected(
    random_walk_10: torch.Tensor,
) -> None:
    model = _model(random_walk_10, horizon=24)
    x = torch.randn(2, 8, 9, 12)
    future = torch.randn(2, 24, 9, 7)
    with pytest.raises(ValueError, match="Expected 10 nodes"):
        model(x, x_future_exog=future)


def test_invalid_random_walk_rejected() -> None:
    bad = torch.ones(10, 10)
    with pytest.raises(ValueError, match="row must sum to one"):
        _model(bad, horizon=24)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
def test_cuda_forward(random_walk_10: torch.Tensor) -> None:
    model = _model(
        random_walk_10,
        horizon=24,
        hidden_dim=4,
    ).to("cuda")
    x = torch.randn(2, 8, 10, 12, device="cuda")
    future = torch.randn(2, 24, 10, 7, device="cuda")
    model.eval()
    with torch.no_grad():
        output = model(x, x_future_exog=future)
    assert output.device.type == "cuda"
    assert output.shape == (2, 24, 10)
