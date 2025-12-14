# src/evaluation/metrics.py
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute common regression metrics on the original scale.

    Returns:
        dict with rmse, mse, mae, r2
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))  # can be negative in finance

    return {"rmse": rmse, "mse": mse, "mae": mae, "r2": r2}


def directional_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Directional metrics for return prediction.

    Directional accuracy:
        mean(sign(y_true) == sign(y_pred))

    F1 score:
        treat positive return as class 1, otherwise 0
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    dir_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred)))

    y_true_bin = (y_true > 0).astype(int)
    y_pred_bin = (y_pred > 0).astype(int)
    f1 = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))

    return {"directional_accuracy": dir_acc, "f1": f1}

def buy_and_hold_summary(
    y_true_returns: np.ndarray,
    risk_free: float = 0.0,
) -> dict:
    """
    Buy-and-hold baseline: always long the market.
    """
    r = np.asarray(y_true_returns).reshape(-1)

    return {
        "sharpe": sharpe_ratio(r, risk_free=risk_free),
        "sortino": sortino_ratio(r, risk_free=risk_free),
        "max_drawdown": max_drawdown(r, use_log=False),
        "mean_return": float(np.mean(r)) if len(r) else 0.0,
        "std_return": float(np.std(r, ddof=1)) if len(r) > 1 else 0.0,
        "n_days": int(len(r)),
    }

