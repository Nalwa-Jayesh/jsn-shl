#!/usr/bin/env python3
"""Concatenate cached train features. No new encoders. No test. No stacking."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shl import paths as _paths
from shl.cv import N_FOLDS, SEED, make_folds
from shl.experiments import append_result
from shl.metrics import regression_metrics

HIGH_ID = 5000
CV_STRATEGY = f"StratifiedKFold bins n={N_FOLDS} seed={SEED}"
BLOCKS = {
    "acoustic": _paths.acoustic_cache(),
    "linguistic": _paths.asr_dir() / "train_ling_v1.npz",
    "speech": _paths.speech_cache(),
}
# Best single-modality models from experiments/results.csv, by pooled OOF RMSE.
# TF-IDF and MiniLM lost. They are not fused.
SINGLES = ("E004", "E005", "E010")
RUNS = (
    ("E011", "acoustic+linguistic", ("acoustic", "linguistic"), "Ridge"),
    ("E012", "acoustic+linguistic", ("acoustic", "linguistic"), "HistGradientBoostingRegressor"),
    ("E013", "acoustic+speech", ("acoustic", "speech"), "Ridge"),
    ("E014", "acoustic+speech", ("acoustic", "speech"), "HistGradientBoostingRegressor"),
    ("E015", "linguistic+speech", ("linguistic", "speech"), "Ridge"),
    ("E016", "linguistic+speech", ("linguistic", "speech"), "HistGradientBoostingRegressor"),
    ("E017", "acoustic+linguistic+speech", ("acoustic", "linguistic", "speech"), "Ridge"),
    ("E018", "acoustic+linguistic+speech", ("acoustic", "linguistic", "speech"), "HistGradientBoostingRegressor"),
)
LABELS = {
    "E004": "acoustic HistGradientBoosting",
    "E005": "linguistic Ridge",
    "E010": "wav2vec2 HistGradientBoosting",
    "E011": "acoustic+linguistic Ridge",
    "E012": "acoustic+linguistic HistGradientBoosting",
    "E013": "acoustic+speech Ridge",
    "E014": "acoustic+speech HistGradientBoosting",
    "E015": "linguistic+speech Ridge",
    "E016": "linguistic+speech HistGradientBoosting",
    "E017": "all three Ridge",
    "E018": "all three HistGradientBoosting",
}


def load_train(data_dir: Path) -> list[dict[str, str]]:
    with (data_dir / "train.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_block(path: Path, ids: list[str]) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    if list(payload["sample_id"].astype(str)) != ids:
        raise SystemExit(f"{path} sample_id order does not match train.csv")
    matrix = payload["features"] if "features" in payload.files else payload["embeddings"]
    return np.asarray(matrix, dtype=np.float64)


def hstack(names: tuple[str, ...], blocks: dict[str, np.ndarray]) -> np.ndarray:
    return np.hstack([blocks[name] for name in names])


def ridge(width: int) -> Pipeline:
    # 768-d speech block needed a stronger penalty than the 166-d acoustic block.
    alpha = 10.0 if width >= 700 else 1.0
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def hist_gbm() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.05,
        max_iter=200,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=SEED,
    )


def oof_predict(kind: str, x: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    pred = np.empty(len(y), dtype=np.float64)
    for fold in range(N_FOLDS):
        train_mask = folds != fold
        valid_mask = folds == fold
        model = ridge(x.shape[1]) if kind == "Ridge" else hist_gbm()
        model.fit(x[train_mask], y[train_mask])
        pred[valid_mask] = model.predict(x[valid_mask])
    return pred


def write_oof(path: Path, rows: list[dict[str, str]], y: np.ndarray, folds: np.ndarray, pred: np.ndarray) -> None:
    if path.exists():
        raise SystemExit(f"{path} already exists; refusing to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "filename", "fold", "target", "prediction", "zero_cohort"],
        )
        writer.writeheader()
        for row, target, fold, score in zip(rows, y, folds, pred, strict=True):
            filename_int = int("".join(ch for ch in row["filename"] if ch.isdigit()))
            writer.writerow(
                {
                    "sample_id": f"train/{row['filename']}",
                    "filename": row["filename"],
                    "fold": int(fold),
                    "target": f"{target:.1f}",
                    "prediction": f"{float(score):.6f}",
                    "zero_cohort": int(filename_int >= HIGH_ID),
                }
            )


def load_oof(path: Path, ids: list[str]) -> np.ndarray:
    loaded = list(csv.DictReader(path.open()))
    if [row["sample_id"] for row in loaded] != ids:
        raise SystemExit(f"{path} order does not match train.csv")
    return np.asarray([float(row["prediction"]) for row in loaded])


def fold_rmse(y: np.ndarray, pred: np.ndarray, folds: np.ndarray) -> list[float]:
    return [
        float(np.sqrt(np.mean((pred[folds == fold] - y[folds == fold]) ** 2)))
        for fold in range(N_FOLDS)
    ]


def log_experiment(
    results_path: Path,
    experiment_id: str,
    feature_set: str,
    model: str,
    hyperparameters: dict[str, object],
    y: np.ndarray,
    pred: np.ndarray,
    folds: np.ndarray,
) -> None:
    metrics = regression_metrics(y, pred)
    per_fold = fold_rmse(y, pred, folds)
    append_result(
        results_path,
        {
            "experiment_id": experiment_id,
            "feature_set": feature_set,
            "model": model,
            "hyperparameters": json.dumps(hyperparameters, sort_keys=True),
            "cv_strategy": CV_STRATEGY,
            "cv_mean": f"{float(np.mean(per_fold)):.6f}",
            "cv_std": f"{float(np.std(per_fold, ddof=1)):.6f}",
            "mae": f"{metrics['mae']:.6f}",
            "rmse": f"{metrics['rmse']:.6f}",
            "pearson": "" if np.isnan(metrics["pearson"]) else f"{metrics['pearson']:.6f}",
            "spearman": "" if np.isnan(metrics["spearman"]) else f"{metrics['spearman']:.6f}",
            "official_metric": "",
            "notes": "cached feature concat; imputer+scaler fit in-fold; no test; no stacking; no fine-tune",
        },
    )
    print(experiment_id, f"rmse={metrics['rmse']:.4f}", f"mae={metrics['mae']:.4f}", flush=True)


def write_report(
    path: Path,
    y: np.ndarray,
    folds: np.ndarray,
    cohort: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> None:
    lines = [
        "# Multimodal report",
        "",
        "Generated by `scripts/run_multimodal.py`.",
        "Concatenation of cached train features only. No new encoder. No test audio. No stacking.",
        "Official metric: unknown. Comparison number is pooled OOF RMSE.",
        f"Validation: {CV_STRATEGY}. Same `make_folds` as E001–E010.",
        "Imputer and scaler are fit on the training rows of each fold. HistGradientBoosting does not use the scaler.",
        "TF-IDF (E006) and MiniLM (E008) are omitted. Both lost to linguistic features and to speech embeddings.",
        "",
        "Blocks:",
        "",
        "- acoustic: `acoustic_v1`, 166-d, best single E004",
        "- linguistic: `asr_ling_v1`, 19-d, best single E005",
        "- speech: `wav2vec2_base_mean_v1`, 768-d mean pool, best single E010",
        "",
        "| experiment | model | RMSE | MAE | Pearson | Spearman | R² | fold RMSE std | zero RMSE | rest RMSE |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    order = (*SINGLES, *[item[0] for item in RUNS])
    for experiment_id in order:
        pred = predictions[experiment_id]
        metrics = regression_metrics(y, pred)
        per_fold = fold_rmse(y, pred, folds)
        zero = float(np.sqrt(np.mean((pred[cohort] - y[cohort]) ** 2)))
        rest = float(np.sqrt(np.mean((pred[~cohort] - y[~cohort]) ** 2)))
        pearson = "n/a" if np.isnan(metrics["pearson"]) else f"{metrics['pearson']:.4f}"
        spearman = "n/a" if np.isnan(metrics["spearman"]) else f"{metrics['spearman']:.4f}"
        lines.append(
            f"| {experiment_id} | {LABELS[experiment_id]} | {metrics['rmse']:.4f} | {metrics['mae']:.4f} | {pearson} | {spearman} | {metrics['r2']:.4f} | "
            f"{float(np.std(per_fold, ddof=1)):.4f} | {zero:.4f} | {rest:.4f} |"
        )

    fused_ids = [item[0] for item in RUNS]
    best_fused = min(fused_ids, key=lambda key: regression_metrics(y, predictions[key])["rmse"])
    best_single = min(SINGLES, key=lambda key: regression_metrics(y, predictions[key])["rmse"])
    fused = regression_metrics(y, predictions[best_fused])["rmse"]
    single = regression_metrics(y, predictions[best_single])["rmse"]
    rest = ~cohort
    rest_fused = float(np.sqrt(np.mean((predictions[best_fused][rest] - y[rest]) ** 2)))
    rest_single = float(np.sqrt(np.mean((predictions[best_single][rest] - y[rest]) ** 2)))
    lines.extend(
        [
            "",
            f"Best fusion: `{best_fused}`. Best single: `{best_single}`.",
            f"Pooled RMSE delta vs best single: {single - fused:+.4f} (positive means fusion is better).",
            f"Non-zero-row RMSE: fusion {rest_fused:.4f}, best single {rest_single:.4f}.",
            "",
            "## Decision",
            "",
        ]
    )
    if single - fused > 0.01 and rest_single - rest_fused > 0.01:
        lines.append("Concatenation beats the best single modality on pooled RMSE and on the 732 non-zero rows. Keep that feature set. Do not stack yet.")
    elif single - fused > 0.01:
        lines.append("Pooled gain is not matched on the 732 non-zero rows. Do not replace the best single modality.")
    else:
        lines.append("Concatenation does not beat the best single modality by a meaningful RMSE margin. Do not add a fusion model.")
    lines.extend(["", "No weighted average. No stacking. No test predictions. No submission.", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=_paths.data_dir())
    parser.add_argument("--oof-dir", type=Path, default=_paths.oof_dir())
    parser.add_argument("--results", type=Path, default=_paths.results_csv())
    parser.add_argument("--report", type=Path, default=_paths.reports_dir() / "multimodal_report.md")
    args = parser.parse_args()

    rows = load_train(args.data_dir)
    y = np.asarray([float(row["label"]) for row in rows], dtype=np.float64)
    ids = [f"train/{row['filename']}" for row in rows]
    folds = make_folds(y)
    cohort = np.asarray([int("".join(ch for ch in row["filename"] if ch.isdigit())) >= HIGH_ID for row in rows])
    blocks = {name: load_block(path, ids) for name, path in BLOCKS.items()}

    predictions = {experiment_id: load_oof(args.oof_dir / f"{experiment_id}_oof.csv", ids) for experiment_id in SINGLES}
    for experiment_id, feature_set, block_names, model_name in RUNS:
        matrix = hstack(block_names, blocks)
        pred = oof_predict(model_name, matrix, y, folds)
        predictions[experiment_id] = pred
        alpha = 10.0 if matrix.shape[1] >= 700 else 1.0
        hyperparameters: dict[str, object] = {"blocks": list(block_names), "width": int(matrix.shape[1])}
        if model_name == "Ridge":
            hyperparameters.update({"alpha": alpha, "scaler": "StandardScaler", "imputer": "median"})
        else:
            hyperparameters.update(
                {
                    "max_depth": 3,
                    "learning_rate": 0.05,
                    "max_iter": 200,
                    "min_samples_leaf": 20,
                    "l2_regularization": 1.0,
                    "random_state": SEED,
                }
            )
        log_experiment(args.results, experiment_id, feature_set, model_name, hyperparameters, y, pred, folds)
        write_oof(args.oof_dir / f"{experiment_id}_oof.csv", rows, y, folds, pred)

    write_report(args.report, y, folds, cohort, predictions)
    fused_ids = [item[0] for item in RUNS]
    best = min(fused_ids, key=lambda key: regression_metrics(y, predictions[key])["rmse"])
    single = min(SINGLES, key=lambda key: regression_metrics(y, predictions[key])["rmse"])
    delta = regression_metrics(y, predictions[single])["rmse"] - regression_metrics(y, predictions[best])["rmse"]
    print(f"wrote {args.report}", flush=True)
    print("best", best, "delta_vs_best_single", f"{delta:+.4f}", "single", single, flush=True)


if __name__ == "__main__":
    main()
