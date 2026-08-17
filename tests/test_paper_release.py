from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _protocol():
    return yaml.safe_load(
        (ROOT / "configs/paper/protocol.yaml").read_text(encoding="utf-8")
    )


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


def test_expected_test_story_is_31_of_32_and_full_is_always_best():
    expected = _protocol()["expected_common46_test"]
    metrics = ("MAE", "MAPE", "RMSE", "NSE")
    relations = (
        ("State", "Base"),
        ("FA-DPR", "Base"),
        ("Full", "State"),
        ("Full", "FA-DPR"),
    )
    passed = 0
    for task in ("24h", "168h"):
        for left, right in relations:
            for metric in metrics:
                a = expected[task][left][metric]
                b = expected[task][right][metric]
                passed += int(a > b if metric == "NSE" else a < b)
        for metric in metrics:
            values = {name: expected[task][name][metric] for name in expected[task]}
            winner = max(values, key=values.get) if metric == "NSE" else min(
                values, key=values.get
            )
            assert winner == "Full"
    assert passed == 31


def test_test_was_not_used_for_selection():
    protocol = _protocol()
    assert protocol["selection_policy"] == "validation_first_test_once"
    assert protocol["selection_evidence"]["test_used_for_training_or_selection"] is False
    assert protocol["dataset"]["expected_common_test_origins"] == 46
