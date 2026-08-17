"""Forecast evaluation metrics.

Implements MAE, MAPE, RMSE, and NSE as used in the MSCMNet paper
(Water Research X, 2024).  MAPE is returned as a **fraction**
(e.g. 0.026 = 2.6%), matching the supplementary S1 tables.
"""

from __future__ import annotations

from typing import Any

import numpy as np

EPSILON: float = 1.0e-12


def _as_float_array(values: Any) -> np.ndarray:
    """Convert input to a 1-D float64 array."""
    arr = np.asarray(values, dtype=np.float64)
    return arr.ravel()


def _finite_pair(
    y_true: Any,
    y_pred: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate shapes and filter to finite values only.

    Returns:
        Tuple of (y_true_finite, y_pred_finite) as 1-D float arrays.
    """
    true = _as_float_array(y_true)
    pred = _as_float_array(y_pred)
    if true.shape != pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {true.shape} vs y_pred {pred.shape}"
        )
    finite = np.isfinite(true) & np.isfinite(pred)
    return true[finite], pred[finite]


def mae(y_true: Any, y_pred: Any) -> float:
    """Mean Absolute Error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        MAE as a float.  Non-finite pairs are excluded.
    """
    true, pred = _finite_pair(y_true, y_pred)
    if len(true) == 0:
        return float("nan")
    return float(np.mean(np.abs(true - pred)))


def mape(y_true: Any, y_pred: Any, epsilon: float = EPSILON) -> float:
    """Mean Absolute Percentage Error (fraction, not percent).

    Values where ``|y_true| < epsilon`` are masked to avoid
    division by zero.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        epsilon: Threshold below which true values are considered zero.

    Returns:
        MAPE as a **fraction** (e.g. 0.026 means 2.6%).
    """
    true, pred = _finite_pair(y_true, y_pred)
    mask = np.abs(true) >= epsilon
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((true[mask] - pred[mask]) / true[mask])))


def rmse(y_true: Any, y_pred: Any) -> float:
    """Root Mean Squared Error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        RMSE as a float.
    """
    true, pred = _finite_pair(y_true, y_pred)
    if len(true) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def nse(y_true: Any, y_pred: Any) -> float:
    """Nash-Sutcliffe Efficiency.

    .. math::

        NSE = 1 - \\frac{\\sum (y - \\hat{y})^2}
                       {\\sum (y - \\bar{y})^2}

    Returns ``nan`` if the denominator is near zero.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        NSE as a float (higher is better, max = 1.0).
    """
    true, pred = _finite_pair(y_true, y_pred)
    if len(true) == 0:
        return float("nan")
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    if ss_tot < EPSILON:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute all four standard forecast metrics.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Dict with keys ``"MAE"``, ``"MAPE"``, ``"RMSE"``, ``"NSE"``.
        MAPE is returned as a fraction.
    """
    return {
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "NSE": nse(y_true, y_pred),
    }
