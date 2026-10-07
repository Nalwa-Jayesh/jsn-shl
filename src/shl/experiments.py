"""Append-only experiment log. Never rewrites an existing experiment_id."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

COLUMNS = [
    "experiment_id",
    "timestamp",
    "feature_set",
    "model",
    "hyperparameters",
    "cv_strategy",
    "cv_mean",
    "cv_std",
    "mae",
    "rmse",
    "pearson",
    "spearman",
    "official_metric",
    "notes",
]


def append_result(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    if path.exists():
        with path.open(newline="") as handle:
            for old in csv.DictReader(handle):
                existing.add(old["experiment_id"])
    experiment_id = str(row["experiment_id"])
    if experiment_id in existing:
        raise ValueError(f"{experiment_id} already exists in {path}; refusing to overwrite")
    payload = {column: row.get(column, "") for column in COLUMNS}
    payload["timestamp"] = payload["timestamp"] or datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(payload)
