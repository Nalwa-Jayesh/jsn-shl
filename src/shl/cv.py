"""Leakage-safe stratified folds. The bin label is a split key, never a feature."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold

SEED = 42
N_FOLDS = 5


def target_bin(label: float) -> str:
    if label == 0.0:
        return "0"
    if label <= 2.0:
        return "1-2"
    if label <= 3.0:
        return "2.5-3"
    if label <= 4.0:
        return "3.5-4"
    return "4.5-5"


def make_folds(labels: np.ndarray, n_folds: int = N_FOLDS, seed: int = SEED) -> np.ndarray:
    bins = np.asarray([target_bin(float(value)) for value in labels])
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = np.empty(len(labels), dtype=np.int8)
    for fold, (_, valid_index) in enumerate(splitter.split(np.zeros(len(labels)), bins)):
        folds[valid_index] = fold
    return folds
