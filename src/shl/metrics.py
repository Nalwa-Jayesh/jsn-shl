"""Regression metrics. No official competition metric is known."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import pearsonr, spearmanr


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else math.nan
    pearson = float(pearsonr(y_true, y_pred).statistic) if np.unique(y_pred).size > 1 else math.nan
    spearman = float(spearmanr(y_true, y_pred).statistic) if np.unique(y_pred).size > 1 else math.nan
    return {
        "mae": mae,
        "rmse": rmse,
        "pearson": pearson,
        "spearman": spearman,
        "r2": r2,
        "official_metric": math.nan,
    }
