"""Strict, non-training metric audits for the Que et al. reconstruction.

``pooled`` reproduces the existing flattened-origin convention. ``origin_mean``
is a *diagnostic hypothesis*, not a verified publisher convention: compute each
origin's complete metric, then take its arithmetic mean. Never mix conventions
across models, horizons, DMAs, or metrics to obtain a closer paper table.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Mapping, Sequence

import numpy as np

METRIC_MODES = ("pooled", "origin_mean")
METRICS = ("MAE", "MAPE", "RMSE", "NSE")


def array_sha256(values: Any) -> str:
    """Hash array content together with its shape and dtype (no pickle)."""
    array = np.asarray(values)
    if array.dtype.hasobject:
        raise ValueError("Object arrays cannot be used as prediction evidence.")
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def validate_prediction_bundle(
    arrays: Mapping[str, Any],
    *,
    expected_sequences: int = 46,
    expected_dma_letters: Sequence[str] = tuple("ABCDEFGHIJ"),
    prediction_atol: float = 1.0e-5,
    prediction_rtol: float = 1.0e-5,
    require_first_day_consistency: bool = True,
) -> dict[str, Any]:
    """Require finite complete arrays, common origins, and frozen first-day output.

    Matching first-day arrays is necessary but does not prove that checkpoints
    were never changed. Checkpoint provenance remains a separate recorded claim.
    """
    required = {
        "y_true_24h", "y_pred_24h", "y_true_168h", "y_pred_168h",
        "forecast_starts", "dma_letters",
    }
    missing = required.difference(arrays.keys())
    if missing:
        raise ValueError(f"Missing prediction fields: {sorted(missing)}")
    letters = np.asarray(arrays["dma_letters"])
    if letters.ndim != 1 or letters.tolist() != list(expected_dma_letters):
        raise ValueError("DMA order must match the declared A-J order exactly.")
    starts = np.asarray(arrays["forecast_starts"])
    if starts.shape != (expected_sequences,) or len(set(starts.tolist())) != expected_sequences:
        raise ValueError("Forecast origins must be one unique origin per sequence.")
    try:
        parsed_starts = [datetime.fromisoformat(str(value).replace("Z", "+00:00")) for value in starts]
        if any(left >= right for left, right in zip(parsed_starts, parsed_starts[1:])):
            raise ValueError("Forecast origins must be in strictly increasing time order.")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Forecast origins must be ordered ISO timestamps: {exc}") from exc
    for task, horizon in (("24h", 24), ("168h", 168)):
        expected = (expected_sequences, horizon, len(expected_dma_letters))
        for kind in ("true", "pred"):
            values = np.asarray(arrays[f"y_{kind}_{task}"])
            if values.shape != expected or values.dtype.kind not in "fiu":
                raise ValueError(f"y_{kind}_{task} must be numeric with shape {expected}.")
            if not np.isfinite(values).all():
                raise ValueError(f"y_{kind}_{task} contains non-finite values; masking is forbidden.")
    if not np.array_equal(arrays["y_true_24h"], np.asarray(arrays["y_true_168h"])[:, :24]):
        raise ValueError("24h truth differs from the first 24 hours of 168h truth.")
    p24 = np.asarray(arrays["y_pred_24h"], dtype=np.float64)
    p168 = np.asarray(arrays["y_pred_168h"], dtype=np.float64)[:, :24]
    first_day_consistent = bool(np.allclose(p24, p168, rtol=prediction_rtol, atol=prediction_atol))
    if require_first_day_consistency and not first_day_consistent:
        raise ValueError("24h prediction differs from the first day of the frozen 168h rollout.")
    return {
        "test_sequences": expected_sequences,
        "dma_letters": letters.tolist(),
        "forecast_starts": starts.tolist(),
        "origins_sha256": array_sha256(starts),
        "array_hashes": {key: array_sha256(arrays[key]) for key in sorted(required)},
        "first_day_max_absolute_difference": float(np.max(np.abs(p24 - p168))),
        "first_day_consistent": first_day_consistent,
        "first_day_absolute_tolerance": prediction_atol,
        "first_day_relative_tolerance": prediction_rtol,
        "finite_complete_arrays": True,
    }


def _metric_values(truth: np.ndarray, prediction: np.ndarray, mode: str) -> dict[str, float]:
    axes: int | None = None if mode == "pooled" else 1
    error = truth - prediction
    absolute = np.mean(np.abs(error), axis=axes)
    mse = np.mean(error**2, axis=axes)
    variance = np.var(truth, axis=axes)
    # Undefined values are reported, never silently dropped from origin means.
    with np.errstate(divide="ignore", invalid="ignore"):
        percentage = np.mean(np.where(np.abs(truth) >= 1e-12, np.abs(error / truth), np.nan), axis=axes)
        efficiency = np.where(variance > 1e-12, 1.0 - mse / variance, np.nan)
    return {
        "MAE": float(np.mean(absolute)),
        "MAPE": float(np.mean(percentage)),
        "RMSE": float(np.mean(np.sqrt(mse))),
        "NSE": float(np.mean(efficiency)),
    }


def compute_reproduction_metrics(
    truth: Any,
    prediction: Any,
    dma_letters: Sequence[str] = tuple("ABCDEFGHIJ"),
    *,
    mode: str = "pooled",
) -> list[dict[str, Any]]:
    """Return complete per-DMA and total rows without finite-pair masking.

    The paper-style ``total/MAE`` is the sum of DMA MAEs. The distinct
    ``physical_total/MAE`` is MAE of summed demand, not the same quantity.
    All aggregation uses float64; old float32 sums can differ at roundoff scale.
    """
    if mode not in METRIC_MODES:
        raise ValueError(f"Unknown metric mode: {mode}")
    true, pred = np.asarray(truth, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if true.ndim != 3 or true.shape != pred.shape or true.shape[2] != len(dma_letters):
        raise ValueError("Expected matching [origin, horizon, DMA] arrays.")
    if min(true.shape) <= 0 or len(set(dma_letters)) != len(dma_letters):
        raise ValueError("Empty arrays or duplicate DMA labels are not valid.")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("Non-finite observations/predictions cannot be masked.")
    rows: list[dict[str, Any]] = []
    dma_mae: list[float] = []
    for index, letter in enumerate(dma_letters):
        values = _metric_values(true[:, :, index], pred[:, :, index], mode)
        dma_mae.append(values["MAE"])
        rows.extend({"mode": mode, "series": letter, "metric": metric, "value": value}
                    for metric, value in values.items())
    total = _metric_values(true.sum(axis=2), pred.sum(axis=2), mode)
    rows.append({"mode": mode, "series": "physical_total", "metric": "MAE", "value": total["MAE"]})
    total["MAE"] = float(sum(dma_mae))
    rows.extend({"mode": mode, "series": "total", "metric": metric, "value": value}
                for metric, value in total.items())
    return rows


def rmse_nse_feasibility(
    truth: Any,
    target_rmse: float,
    target_nse: float,
    *,
    error_relative_tolerance: float = 0.05,
    nse_absolute_tolerance: float = 0.01,
    rounding_half_unit: float = 0.0,
) -> dict[str, Any]:
    """Necessary pooled RMSE/NSE feasibility under one *fixed* truth array.

    RMSE**2 / (1-NSE) equals population truth variance. This bound cannot be
    applied to origin-mean metrics. A positive result does not guarantee that
    MAE/MAPE or any actual model can satisfy the complete paper table.
    ``rounding_half_unit=.0005`` gives a conservative three-decimal sensitivity.
    """
    true = np.asarray(truth, dtype=np.float64)
    numbers = [target_rmse, target_nse, error_relative_tolerance, nse_absolute_tolerance, rounding_half_unit]
    if true.size == 0 or not np.isfinite(true).all() or not np.isfinite(numbers).all():
        raise ValueError("Feasibility requires finite targets and nonempty finite truth.")
    if target_rmse < 0 or target_nse > 1 or not 0 <= error_relative_tolerance < 1:
        raise ValueError("Invalid RMSE, NSE, or relative tolerance.")
    if nse_absolute_tolerance < 0 or rounding_half_unit < 0:
        raise ValueError("Tolerances and rounding uncertainty must be nonnegative.")
    variance = float(np.var(true))
    low_rmse = max(0.0, target_rmse - rounding_half_unit) * (1 - error_relative_tolerance)
    high_rmse = (target_rmse + rounding_half_unit) * (1 + error_relative_tolerance)
    low_nse = target_nse - rounding_half_unit - nse_absolute_tolerance
    high_nse = min(1.0, target_nse + rounding_half_unit + nse_absolute_tolerance)
    result: dict[str, Any] = {
        "mode": "pooled", "truth_variance": variance,
        "target_rmse": target_rmse, "target_nse": target_nse,
        "error_relative_tolerance": error_relative_tolerance,
        "nse_absolute_tolerance": nse_absolute_tolerance,
        "rounding_half_unit": rounding_half_unit,
        "allowed_rmse_low": low_rmse, "allowed_rmse_high": high_rmse,
        "allowed_nse_low": low_nse, "allowed_nse_high": high_nse,
        "paper_pair_implied_variance": target_rmse**2 / (1-target_nse) if target_nse < 1 else None,
    }
    if variance <= 1e-12:
        return {**result, "pair_feasible": False, "reason": "NSE_UNDEFINED_CONSTANT_TRUTH"}
    implied_low, implied_high = 1-high_rmse**2/variance, 1-low_rmse**2/variance
    min_rmse_nse = float(np.sqrt(max(0.0, variance*(1-high_nse))))
    feasible = max(implied_low, low_nse) <= min(implied_high, high_nse) + 1e-14
    return {
        **result, "implied_nse_low": implied_low, "implied_nse_high": implied_high,
        "minimum_rmse_for_nse_tolerance": min_rmse_nse,
        "minimum_relative_rmse_increase_for_nse": max(0.0, min_rmse_nse/target_rmse-1) if target_rmse else None,
        "pair_feasible": bool(feasible),
        "reason": "PAIR_NOT_RULED_OUT" if feasible else "DISJOINT_POOLED_RMSE_NSE_INTERVALS",
    }
