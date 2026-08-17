"""Model-level tests that inference cannot read future true demand.

Scheduled-sampling schedules are deliberately not tested here. Their decay,
early stopping, and optimiser interactions belong to stage 4 training tests.
"""

from __future__ import annotations

import pytest
import torch

from dma_wdf.models.dcrnn import DCRNN


@pytest.fixture
def model() -> DCRNN:
    support = torch.ones(10, 10, dtype=torch.float32)
    support.fill_diagonal_(0.0)
    support = support / support.sum(dim=1, keepdim=True)
    return DCRNN(
        random_walk=support,
        input_dim=12,
        hidden_dim=8,
        horizon=24,
        num_nodes=10,
        num_rnn_layers=1,
        max_diffusion_step=2,
        future_exog_dim=7,
        dropout=0.0,
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    x = torch.randn(3, 12, 10, 12)
    future = torch.randn(3, 24, 10, 7)
    return x, future


def test_eval_target_is_structurally_ignored(model: DCRNN) -> None:
    x, future = _inputs()
    target_a = torch.zeros(3, 24, 10)
    target_b = torch.full((3, 24, 10), 1000.0)

    model.eval()
    with torch.no_grad():
        prediction_none = model(
            x,
            y_target=None,
            x_future_exog=future,
            teacher_forcing_ratio=0.0,
        )
        prediction_a = model(
            x,
            y_target=target_a,
            x_future_exog=future,
            teacher_forcing_ratio=1.0,
        )
        prediction_b = model(
            x,
            y_target=target_b,
            x_future_exog=future,
            teacher_forcing_ratio=1.0,
        )

    assert torch.equal(prediction_none, prediction_a)
    assert torch.equal(prediction_none, prediction_b)


def test_training_ratio_zero_ignores_target(model: DCRNN) -> None:
    x, future = _inputs()
    target_a = torch.zeros(3, 24, 10)
    target_b = torch.full((3, 24, 10), 1000.0)

    model.train()
    prediction_a = model(
        x,
        y_target=target_a,
        x_future_exog=future,
        teacher_forcing_ratio=0.0,
    )
    prediction_b = model(
        x,
        y_target=target_b,
        x_future_exog=future,
        teacher_forcing_ratio=0.0,
    )
    assert torch.equal(prediction_a, prediction_b)


def test_training_full_teacher_forcing_uses_target(
    model: DCRNN,
) -> None:
    x, future = _inputs()
    target_a = torch.zeros(3, 24, 10)
    target_b = torch.full((3, 24, 10), 5.0)

    model.train()
    prediction_a = model(
        x,
        y_target=target_a,
        x_future_exog=future,
        teacher_forcing_ratio=1.0,
    )
    prediction_b = model(
        x,
        y_target=target_b,
        x_future_exog=future,
        teacher_forcing_ratio=1.0,
    )
    assert not torch.allclose(prediction_a, prediction_b)


def test_training_teacher_forcing_requires_target(
    model: DCRNN,
) -> None:
    x, future = _inputs()
    model.train()
    with pytest.raises(ValueError, match="target is required"):
        model(
            x,
            y_target=None,
            x_future_exog=future,
            teacher_forcing_ratio=0.5,
        )


@pytest.mark.parametrize("ratio", [-0.01, 1.01])
def test_teacher_forcing_ratio_bounds(
    model: DCRNN,
    ratio: float,
) -> None:
    x, future = _inputs()
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        model(
            x,
            x_future_exog=future,
            teacher_forcing_ratio=ratio,
        )


def test_known_future_exog_affects_predictions(
    model: DCRNN,
) -> None:
    x, future = _inputs()
    model.eval()
    with torch.no_grad():
        first = model(
            x,
            x_future_exog=future,
        )
        second = model(
            x,
            x_future_exog=future + 2.0,
        )
    assert not torch.allclose(first, second)
