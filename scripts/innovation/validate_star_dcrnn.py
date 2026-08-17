#!/usr/bin/env python
"""Mechanism, identity, leakage, and gradient checks for STaR-DCRNN."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dma_wdf.models.dcrnn import build_dcrnn_model  # noqa: E402
from dma_wdf.models.star_dcrnn import (  # noqa: E402
    FORMAL_VARIANTS,
    STaRDCRNN,
    STaRForwardDetails,
    build_star_dcrnn_model,
)
from dma_wdf.training.engine import set_reproducible_seed  # noqa: E402
from dma_wdf.utils.config import (  # noqa: E402
    load_config_with_inheritance,
)


def _config(task: str) -> dict:
    return load_config_with_inheritance(
        PROJECT_ROOT,
        PROJECT_ROOT
        / "configs"
        / "train"
        / f"star_dcrnn_{task}.yaml",
    )


def _tiny_gradient_check() -> dict[str, float]:
    model = STaRDCRNN(
        random_walk=torch.eye(2),
        input_dim=3,
        hidden_dim=4,
        horizon=24,
        history_hours=672,
        num_nodes=2,
        future_exog_dim=1,
        use_dssn_sasr=True,
        use_fa_dpr=True,
        attention_dim=3,
        condition_on_future_calendar=True,
    )
    model.train()
    x = torch.randn(1, 672, 2, 3)
    y = torch.randn(1, 24, 2)
    future = torch.randn(1, 24, 2, 1)
    details = model(
        x,
        y_target=y,
        x_future_exog=future,
        teacher_forcing_ratio=0.0,
        return_details=True,
    )
    assert isinstance(details, STaRForwardDetails)
    assert details.dssn_sasr is not None
    state_loss, _ = model.state_supervision_loss(
        target=y,
        state=details.dssn_sasr,
    )
    (F.l1_loss(details.prediction, y) + 0.1 * state_loss).backward()
    assert model.fa_dpr is not None
    assert model.dssn_sasr is not None
    parameters = {
        "encoder": model.encoder.layers[0].gate_conv.weight,
        "decoder": model.decoder.output_projection.weight,
        "attention": model.fa_dpr.query_projection.weight,
        "future_calendar": model.fa_dpr.future_projection.weight,
        "fusion_gate": model.fa_dpr.fusion_gate.weight,
        "alpha_mean": model.dssn_sasr.restorer.alpha_mean_logits,
        "alpha_std": model.dssn_sasr.restorer.alpha_std_logits,
    }
    norms: dict[str, float] = {}
    for name, parameter in parameters.items():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"Missing/non-finite gradient: {name}")
        norm = float(parameter.grad.abs().sum().item())
        if norm <= 0.0:
            raise AssertionError(f"Zero gradient: {name}")
        norms[name] = norm
    return norms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    checks: list[dict] = []

    for task, horizon in (("24h", 24), ("168h", 168)):
        config = _config(task)
        set_reproducible_seed(123)
        model = build_star_dcrnn_model(
            config,
            project_root=PROJECT_ROOT,
            input_dim=12,
            future_exog_dim=7,
            horizon=horizon,
            history_hours=672,
            variant="full",
            device=device,
        )
        model.eval()
        x = torch.randn(1, 672, 10, 12, device=device)
        future = torch.randn(1, horizon, 10, 7, device=device)
        target_a = torch.randn(1, horizon, 10, device=device)
        target_b = target_a + 1000.0
        with torch.inference_mode():
            details = model(
                x,
                y_target=target_a,
                x_future_exog=future,
                return_details=True,
            )
            prediction_b = model(
                x,
                y_target=target_b,
                x_future_exog=future,
            )
        assert isinstance(details, STaRForwardDetails)
        assert details.dssn_sasr is not None
        assert details.fa_dpr is not None
        torch.testing.assert_close(
            details.prediction,
            prediction_b,
            atol=0.0,
            rtol=0.0,
        )
        state = details.dssn_sasr
        readout = details.fa_dpr
        assert details.prediction.shape == (1, horizon, 10)
        assert state.normalized_history.shape == (1, 672, 10)
        assert state.future_mean.shape == (1, horizon, 10)
        assert torch.all(state.future_std > 0.0)
        assert torch.all((state.alpha_mean >= 0) & (state.alpha_mean <= 1))
        assert torch.all((state.alpha_std >= 0) & (state.alpha_std <= 1))
        assert readout.daily_tokens.shape == (1, 28, 10, 32)
        assert model.fa_dpr is not None
        assert model.fa_dpr.token_projection.in_features == 32
        assert model.fa_dpr.token_projection.out_features == 32
        assert model.fa_dpr.future_projection is not None
        assert model.fa_dpr.future_projection.in_features == 7
        assert readout.attention_weights.shape == (1, horizon, 10, 28)
        torch.testing.assert_close(
            readout.attention_weights.sum(dim=-1),
            torch.ones(1, horizon, 10, device=device),
            atol=1.0e-6,
            rtol=0.0,
        )
        checks.append(
            {
                "task": task,
                "output_shape": list(details.prediction.shape),
                "target_isolated_in_evaluation": True,
                "parameters": sum(p.numel() for p in model.parameters()),
            }
        )

    config = _config("24h")
    dcrnn_config = deepcopy(config)
    dcrnn_config["model"]["name"] = "dcrnn"
    set_reproducible_seed(456)
    baseline = build_dcrnn_model(
        dcrnn_config,
        project_root=PROJECT_ROOT,
        input_dim=12,
        future_exog_dim=7,
        horizon=24,
        device=device,
    )
    set_reproducible_seed(456)
    disabled = build_star_dcrnn_model(
        config,
        project_root=PROJECT_ROOT,
        input_dim=12,
        future_exog_dim=7,
        horizon=24,
        history_hours=672,
        variant="backbone",
        device=device,
    )
    baseline.eval()
    disabled.eval()
    x = torch.randn(1, 672, 10, 12, device=device)
    future = torch.randn(1, 24, 10, 7, device=device)
    with torch.inference_mode():
        expected = baseline(x, x_future_exog=future)
        observed = disabled(x, x_future_exog=future)
    max_difference = float((expected - observed).abs().max().item())
    if max_difference != 0.0:
        raise AssertionError(
            f"Disabled STaR is not exact DCRNN: max diff={max_difference}."
        )

    payload = {
        "all_passed": True,
        "model": "star_dcrnn",
        "formal_variants": list(FORMAL_VARIANTS),
        "device": str(device),
        "checks": checks,
        "exact_dcrnn_max_abs_difference": max_difference,
        "joint_gradient_l1_norms": _tiny_gradient_check(),
        "test_target_passed_to_inference": False,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
