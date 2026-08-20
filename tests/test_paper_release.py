from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _protocol():
    return yaml.safe_load(
        (ROOT / "configs/paper/protocol.yaml").read_text(encoding="utf-8")
    )


def _count_relations(expected, relations):
    metrics = ("MAE", "MAPE", "RMSE", "NSE")
    passed = 0
    for task in ("24h", "168h"):
        for left, right in relations:
            for metric in metrics:
                a = expected[task][left][metric]
                b = expected[task][right][metric]
                passed += int(a > b if metric == "NSE" else a < b)
    return passed


def test_paper_setting_is_frozen_to_validation_selected_candidate():
    setting = _protocol()["paper_setting"]
    assert setting == {
        "learning_rate": 0.0003,
        "weight_decay": 0.0,
        "cl_decay_steps": 500,
        "state_loss_weight": 0.03,
        "max_epochs": 100,
        "seed": 0,
    }


def test_formal_star_configs_match_paper_setting():
    for task in ("24h", "168h"):
        config = yaml.safe_load(
            (ROOT / f"configs/train/star_dcrnn_{task}.yaml").read_text(
                encoding="utf-8"
            )
        )
        assert config["training"]["learning_rate"] == 0.0003
        assert config["training"]["weight_decay"] == 0.0
        assert (
            config["training"]["scheduled_sampling"]["cl_decay_steps"]
            == 500
        )
        assert config["innovation"]["dssn_sasr"]["state_loss_weight"] == 0.03


def test_legacy_aggregate_demand_view_remains_31_of_32():
    protocol = _protocol()
    expected = protocol["expected_common46_test"]
    assert protocol["metric_views"]["expected_common46_test"]["role"] == (
        "internal_aggregate_demand_diagnostic"
    )
    relations = (
        ("State", "Base"),
        ("FA-DPR", "Base"),
        ("Full", "State"),
        ("Full", "FA-DPR"),
    )
    assert _count_relations(expected, relations) == 31

    # Under the internal aggregate-demand MAE view, Full is best on all four
    # metrics for both horizons.  This is retained only for backward-compatible
    # release diagnostics, not as the manuscript Table 2 ranking.
    metrics = ("MAE", "MAPE", "RMSE", "NSE")
    for task in ("24h", "168h"):
        for metric in metrics:
            values = {name: expected[task][name][metric] for name in expected[task]}
            winner = max(values, key=values.get) if metric == "NSE" else min(
                values, key=values.get
            )
            assert winner == "Full"


def test_manuscript_factorial_ablation_is_four_models_and_30_of_32():
    protocol = _protocol()
    expected = protocol["expected_manuscript_common46_test"]
    assert protocol["metric_views"]["expected_manuscript_common46_test"]["role"] == (
        "manuscript_publisher_compatible_factorial_ablation"
    )
    model_order = (
        "DCRNN",
        "DCRNN + SAS-Norm",
        "DCRNN + FA-DPR",
        "STaR-GNN",
    )
    for task in ("24h", "168h"):
        assert tuple(expected[task]) == model_order
        assert "STGCN" not in expected[task]

    relations = (
        ("DCRNN + SAS-Norm", "DCRNN"),
        ("DCRNN + FA-DPR", "DCRNN"),
        ("STaR-GNN", "DCRNN + SAS-Norm"),
        ("STaR-GNN", "DCRNN + FA-DPR"),
    )
    assert _count_relations(expected, relations) == 30

    # 24 h: Full is best on all metrics.
    for metric in ("MAE", "MAPE", "RMSE", "NSE"):
        values = {name: expected["24h"][name][metric] for name in model_order}
        winner = max(values, key=values.get) if metric == "NSE" else min(
            values, key=values.get
        )
        assert winner == "STaR-GNN"

    # 168 h: SAS-Norm-only has the marginally lower publisher-compatible MAE;
    # Full is best on MAPE, RMSE, and NSE.
    assert expected["168h"]["DCRNN + SAS-Norm"]["MAE"] < expected["168h"]["STaR-GNN"]["MAE"]
    for metric in ("MAPE", "RMSE", "NSE"):
        values = {name: expected["168h"][name][metric] for name in model_order}
        winner = max(values, key=values.get) if metric == "NSE" else min(
            values, key=values.get
        )
        assert winner == "STaR-GNN"


def test_protocol_selection_evidence_keeps_test_out_of_selection():
    protocol = _protocol()
    evidence = protocol["selection_evidence"]
    assert protocol["selection_policy"] == "validation_first_test_once"
    assert evidence["test_used_for_training_or_selection"] is False
    assert evidence["aggregate_demand_test_relations_passed"] == 31
    assert evidence["aggregate_demand_test_relations_total"] == 32
    assert evidence["manuscript_factorial_cells_passed"] == 30
    assert evidence["manuscript_factorial_cells_total"] == 32
    assert protocol["dataset"]["expected_common_test_origins"] == 46
