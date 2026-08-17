from __future__ import annotations

from pathlib import Path

from dma_wdf.data.dcrnn_dataset import TEMPORAL_FEATURE_ORDER
from dma_wdf.utils.config import load_config_with_inheritance


ROOT = Path(__file__).resolve().parent.parent


def test_star_configs_preserve_sealed_protocol() -> None:
    for task, horizon, validation_batch in (
        ("24h", 24, 16),
        ("168h", 168, 68),
    ):
        config = load_config_with_inheritance(
            ROOT,
            ROOT / "configs" / "train" / f"star_dcrnn_{task}.yaml",
        )
        assert config["model"]["name"] == "star_dcrnn"
        assert config["task"]["horizon"] == horizon
        assert config["task"]["history_hours"] == 672
        assert config["graph"]["matrix_key"] == "random_walk"
        assert config["model"]["max_diffusion_step"] == 2
        assert config["training"]["validation_batch_size"] == validation_batch
        assert config["training"]["max_epochs"] == 100
        assert config["training"]["early_stopping_patience"] == 15
        assert config["training"]["learning_rate"] == 0.0003
        assert config["training"]["weight_decay"] == 0.0
        assert config["training"]["scheduled_sampling"][
            "cl_decay_steps"
        ] == 500
        assert config["innovation"]["dssn_sasr"][
            "state_loss_weight"
        ] == 0.03
        assert config["innovation"]["fa_dpr"][
            "attention_dim"
        ] == 16
        assert config["innovation"]["fa_dpr"][
            "condition_on_future_calendar"
        ] is True


def test_registered_future_context_is_known_calendar_only() -> None:
    assert tuple(TEMPORAL_FEATURE_ORDER) == (
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "time_zone_standard",
        "weekday",
        "holiday",
    )
