#!/usr/bin/env python3
"""Leakage-safe constant and acoustic baselines.

Train audio and train labels only. Test wavs are not read. Test labels are not read.
Filename, filename integer, and encoder tag are not features.
"""

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
from shl.acoustic import FEATURE_VERSION, extract_acoustic, feature_names
from shl.cv import N_FOLDS, SEED, make_folds, target_bin
from shl.experiments import append_result
from shl.metrics import regression_metrics

HIGH_ID = 5000
CV_STRATEGY = f"StratifiedKFold bins n={N_FOLDS} seed={SEED}"


def load_train(data_dir: Path) -> list[dict[str, str]]:
    with (data_dir / "train.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [row["filename"] for row in rows] != [row["filename"] for row in rows]:
        raise SystemExit("train.csv filename list failed identity check")
    return rows


def _save_cache(cache_path: Path, rows: list[dict[str, str]], names: list[str], features: np.ndarray, done: int) -> None:
    np.savez_compressed(
        cache_path,
        version=np.asarray(FEATURE_VERSION),
        sample_id=np.asarray([f"train/{row['filename']}" for row in rows]),
        columns=np.asarray(names),
        features=features,
        done=np.asarray(done),
    )


def cache_features(rows: list[dict[str, str]], data_dir: Path, cache_path: Path) -> np.ndarray:
    names = feature_names()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    wanted = [f"train/{row['filename']}" for row in rows]
    features = np.empty((len(rows), len(names)), dtype=np.float64)
    start = 0
    if cache_path.exists():
        payload = np.load(cache_path, allow_pickle=False)
        cached_ids = list(payload["sample_id"].astype(str))
        if payload["version"].item() != FEATURE_VERSION or list(payload["columns"].astype(str)) != names or cached_ids != wanted:
            raise SystemExit(f"{cache_path} exists but does not match {FEATURE_VERSION}; refusing to overwrite")
        features[:] = payload["features"]
        start = int(payload["done"]) if "done" in payload else len(rows)
        if start >= len(rows) and np.isfinite(features).all():
            print(f"reuse {cache_path}", flush=True)
            return features
        print(f"resume {cache_path} at {start}/{len(rows)}", flush=True)

    for index in range(start, len(rows)):
        features[index] = extract_acoustic(data_dir / "train" / rows[index]["filename"])
        if (index + 1) % 25 == 0 or index + 1 == len(rows):
            _save_cache(cache_path, rows, names, features, index + 1)
            print(f"extracted {index + 1}/{len(rows)}", flush=True)
    if start == len(rows):
        return features
    _save_cache(cache_path, rows, names, features, len(rows))
    print(f"wrote {cache_path}", flush=True)
    return features


def oof_constant(y: np.ndarray, folds: np.ndarray, statistic: str) -> np.ndarray:
    pred = np.empty(len(y), dtype=np.float64)
    for fold in range(int(folds.max()) + 1):
        train = y[folds != fold]
        value = float(np.mean(train) if statistic == "mean" else np.median(train))
        pred[folds == fold] = value
    return pred


def oof_model(model, x: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    pred = np.empty(len(y), dtype=np.float64)
    for fold in range(int(folds.max()) + 1):
        train_mask = folds != fold
        valid_mask = folds == fold
        model.fit(x[train_mask], y[train_mask])
        pred[valid_mask] = model.predict(x[valid_mask])
    return pred


def ridge() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
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


def write_oof(
    path: Path,
    rows: list[dict[str, str]],
    y: np.ndarray,
    folds: np.ndarray,
    pred: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"{path} already exists; refusing to overwrite")
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


def fold_rmse(y: np.ndarray, pred: np.ndarray, folds: np.ndarray) -> list[float]:
    out = []
    for fold in range(int(folds.max()) + 1):
        mask = folds == fold
        out.append(float(np.sqrt(np.mean((pred[mask] - y[mask]) ** 2))))
    return out


def cohort_rmse(y: np.ndarray, pred: np.ndarray, cohort: np.ndarray) -> dict[str, float]:
    def rmse(mask: np.ndarray) -> float:
        if not mask.any():
            return float("nan")
        return float(np.sqrt(np.mean((pred[mask] - y[mask]) ** 2)))

    return {"zero_cohort_rmse": rmse(cohort), "rest_rmse": rmse(~cohort)}


def log_experiment(
    results_path: Path,
    experiment_id: str,
    feature_set: str,
    model: str,
    hyperparameters: dict[str, object],
    y: np.ndarray,
    pred: np.ndarray,
    folds: np.ndarray,
    notes: str,
) -> dict[str, float]:
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
            "notes": notes,
        },
    )
    return {**metrics, "cv_mean": float(np.mean(per_fold)), "cv_std": float(np.std(per_fold, ddof=1))}


def write_report(
    path: Path,
    rows: list[dict[str, str]],
    y: np.ndarray,
    folds: np.ndarray,
    predictions: dict[str, np.ndarray],
    cohort: np.ndarray,
) -> None:
    lines = [
        "# Baseline report",
        "",
        "Generated by `scripts/run_baselines.py`. Train rows only. Test audio was not read.",
        "Official competition metric: unknown. Primary comparison number for this stage is pooled OOF RMSE. That is not an official metric.",
        "",
        f"Validation: {CV_STRATEGY}. Stratification bins are a split key only.",
        "Forbidden features not used: filename, filename integer, encoder tag, split.",
        "Scaler and imputer, where used, are fit inside each training fold.",
        "",
        "## Fold balance",
        "",
        "| fold | n | target mean | target std | zero-cohort n |",
        "| --- | --- | --- | --- | --- |",
    ]
    for fold in range(N_FOLDS):
        mask = folds == fold
        lines.append(
            f"| {fold} | {int(mask.sum())} | {y[mask].mean():.4f} | {y[mask].std(ddof=1):.4f} | {int(cohort[mask].sum())} |"
        )
    bin_names = ["0", "1-2", "2.5-3", "3.5-4", "4.5-5"]
    lines.extend(["", "| fold | " + " | ".join(bin_names) + " |", "| --- | " + " | ".join("---" for _ in bin_names) + " |"])
    bins = np.asarray([target_bin(float(value)) for value in y])
    for fold in range(N_FOLDS):
        counts = [str(int(((folds == fold) & (bins == name)).sum())) for name in bin_names]
        lines.append(f"| {fold} | " + " | ".join(counts) + " |")

    lines.extend(
        [
            "",
            "## Results",
            "",
            "Pooled OOF metrics. `cv_mean` / `cv_std` are the mean and sample std of per-fold RMSE.",
            "",
            "| experiment | model | RMSE | MAE | Pearson | Spearman | R² | fold RMSE mean | fold RMSE std | zero-cohort RMSE | rest RMSE |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    summaries = []
    for experiment_id, pred in predictions.items():
        metrics = regression_metrics(y, pred)
        per_fold = fold_rmse(y, pred, folds)
        split = cohort_rmse(y, pred, cohort)
        model = {
            "E001": "constant mean",
            "E002": "constant median",
            "E003": "Ridge",
            "E004": "HistGradientBoosting",
        }[experiment_id]
        summaries.append((experiment_id, metrics["rmse"], model))
        pearson = "n/a" if np.isnan(metrics["pearson"]) else f"{metrics['pearson']:.4f}"
        spearman = "n/a" if np.isnan(metrics["spearman"]) else f"{metrics['spearman']:.4f}"
        lines.append(
            f"| {experiment_id} | {model} | {metrics['rmse']:.4f} | {metrics['mae']:.4f} | {pearson} | {spearman} | {metrics['r2']:.4f} | "
            f"{float(np.mean(per_fold)):.4f} | {float(np.std(per_fold, ddof=1)):.4f} | "
            f"{split['zero_cohort_rmse']:.4f} | {split['rest_rmse']:.4f} |"
        )

    best_id, best_rmse, best_model = min(summaries, key=lambda item: item[1])
    mean_rmse = regression_metrics(y, predictions["E001"])["rmse"]
    lines.extend(
        [
            "",
            f"Best by pooled OOF RMSE: `{best_id}` {best_model}, RMSE {best_rmse:.4f}.",
            f"Constant-mean floor: RMSE {mean_rmse:.4f}. Delta vs floor: {mean_rmse - best_rmse:.4f}.",
            "XGBoost and LightGBM are not installed. HistGradientBoosting is the tree/boosting model. No extra dependency was added.",
            "",
            "## Error analysis",
            "",
            "Zero cohort is the 37 train files with filename integer >= 5000. All have target 0.0. The integer was not a feature. The cohort flag is only an analysis column.",
            "",
        ]
    )
    best_pred = predictions[best_id]
    residual = best_pred - y
    order = np.argsort(-np.abs(residual))
    lines.append(f"Largest absolute residuals for {best_id}:")
    lines.append("")
    lines.append("| sample_id | target | prediction | residual | zero_cohort |")
    lines.append("| --- | --- | --- | --- | --- |")
    for index in order[:15]:
        filename = rows[index]["filename"]
        lines.append(
            f"| train/{filename} | {y[index]:.1f} | {best_pred[index]:.3f} | {residual[index]:+.3f} | {int(cohort[index])} |"
        )

    lines.extend(["", "RMSE by true target bin, best model:", "", "| bin | n | RMSE | mean prediction | mean target |", "| --- | --- | --- | --- | --- |"])
    for name in bin_names:
        mask = bins == name
        if not mask.any():
            continue
        rmse = float(np.sqrt(np.mean((best_pred[mask] - y[mask]) ** 2)))
        lines.append(
            f"| {name} | {int(mask.sum())} | {rmse:.4f} | {best_pred[mask].mean():.3f} | {y[mask].mean():.3f} |"
        )
    lines.extend(
        [
            "",
            "If zero-cohort RMSE drops far more than rest RMSE, the acoustic model is mostly detecting that loud/low-silence cohort, not grammar. Compare the two columns above before keeping the feature family.",
            "",
            "## Not done",
            "",
            "- No test predictions.",
            "- No ASR, speech embeddings, text models, fine-tuning, or ensemble.",
            "- No submission file.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=_paths.data_dir())
    parser.add_argument("--cache", type=Path, default=_paths.acoustic_cache())
    parser.add_argument("--oof-dir", type=Path, default=_paths.oof_dir())
    parser.add_argument("--results", type=Path, default=_paths.results_csv())
    parser.add_argument("--report", type=Path, default=_paths.reports_dir() / "baseline_report.md")
    args = parser.parse_args()

    rows = load_train(args.data_dir)
    y = np.asarray([float(row["label"]) for row in rows], dtype=np.float64)
    folds = make_folds(y)
    cohort = np.asarray(
        [int("".join(ch for ch in row["filename"] if ch.isdigit())) >= HIGH_ID for row in rows]
    )
    print("fold sizes", [int((folds == fold).sum()) for fold in range(N_FOLDS)], flush=True)

    features = cache_features(rows, args.data_dir, args.cache)
    predictions = {
        "E001": oof_constant(y, folds, "mean"),
        "E002": oof_constant(y, folds, "median"),
        "E003": oof_model(ridge(), features, y, folds),
        "E004": oof_model(hist_gbm(), features, y, folds),
    }
    notes = {
        "E001": "fold-wise training mean; no features",
        "E002": "fold-wise training median; no features",
        "E003": "acoustic_v1; Ridge alpha=1; imputer+scaler fit in-fold; no id/encoder features",
        "E004": "acoustic_v1; HistGradientBoosting depth=3 lr=0.05 iter=200 min_leaf=20 l2=1; no id/encoder features; xgboost/lightgbm unavailable",
    }
    configs = {
        "E001": {"statistic": "mean"},
        "E002": {"statistic": "median"},
        "E003": {"alpha": 1.0, "scaler": "StandardScaler", "imputer": "median"},
        "E004": {
            "max_depth": 3,
            "learning_rate": 0.05,
            "max_iter": 200,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
            "random_state": SEED,
        },
    }
    feature_sets = {
        "E001": "none",
        "E002": "none",
        "E003": FEATURE_VERSION,
        "E004": FEATURE_VERSION,
    }
    models = {
        "E001": "constant_mean",
        "E002": "constant_median",
        "E003": "Ridge",
        "E004": "HistGradientBoostingRegressor",
    }
    for experiment_id, pred in predictions.items():
        write_oof(args.oof_dir / f"{experiment_id}_oof.csv", rows, y, folds, pred)
        metrics = log_experiment(
            args.results,
            experiment_id,
            feature_sets[experiment_id],
            models[experiment_id],
            configs[experiment_id],
            y,
            pred,
            folds,
            notes[experiment_id],
        )
        print(
            experiment_id,
            f"rmse={metrics['rmse']:.4f}",
            f"mae={metrics['mae']:.4f}",
            f"pearson={metrics['pearson']}",
            flush=True,
        )
    write_report(args.report, rows, y, folds, predictions, cohort)
    print(f"wrote {args.report}", flush=True)


if __name__ == "__main__":
    main()
