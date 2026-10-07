#!/usr/bin/env python3
"""Frozen wav2vec2-base mean-pool embeddings. Train audio only. No fine-tune."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
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

MODEL_NAME = "facebook/wav2vec2-base"
VERSION = "wav2vec2_base_mean_v1"
HIGH_ID = 5000
CV_STRATEGY = f"StratifiedKFold bins n={N_FOLDS} seed={SEED}"
LABELS = {
    "E004": "acoustic HistGradientBoosting",
    "E005": "linguistic Ridge",
    "E006": "TF-IDF Ridge",
    "E008": "MiniLM HistGradientBoosting",
    "E009": "wav2vec2 Ridge",
    "E010": "wav2vec2 HistGradientBoosting",
}
COMPARE = ("E004", "E005", "E006", "E008", "E009", "E010")


def load_train(data_dir: Path) -> list[dict[str, str]]:
    with (data_dir / "train.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def cache_embeddings(rows: list[dict[str, str]], data_dir: Path, cache_path: Path, device: str) -> np.ndarray:
    ids = [f"train/{row['filename']}" for row in rows]
    names = feature_columns()
    embeddings = np.empty((len(rows), 768), dtype=np.float32)
    start = 0
    if cache_path.exists():
        payload = np.load(cache_path, allow_pickle=False)
        if (
            payload["version"].item() != VERSION
            or payload["model"].item() != MODEL_NAME
            or list(payload["sample_id"].astype(str)) != ids
            or payload["embeddings"].shape[1] != 768
        ):
            raise SystemExit(f"{cache_path} exists but does not match {VERSION}; refusing to overwrite")
        embeddings[:] = payload["embeddings"]
        start = int(payload["done"]) if "done" in payload else len(rows)
        if start >= len(rows) and np.isfinite(embeddings).all():
            print(f"reuse {cache_path} {embeddings.shape}", flush=True)
            return embeddings
        print(f"resume {cache_path} at {start}/{len(rows)}", flush=True)

    if start < len(rows):
        from transformers import Wav2Vec2Model, Wav2Vec2Processor

        processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
        model.eval()
        torch_device = torch.device(device)
        model.to(torch_device)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        for index in range(start, len(rows)):
            audio, sample_rate = sf.read(data_dir / "train" / rows[index]["filename"], dtype="float32", always_2d=False)
            if audio.ndim != 1 or sample_rate != 16000:
                raise SystemExit(f"{rows[index]['filename']} is not 16 kHz mono")
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            inputs = {key: value.to(torch_device) for key, value in inputs.items()}
            with torch.no_grad():
                hidden = model(**inputs).last_hidden_state
            embeddings[index] = hidden.mean(dim=1).squeeze(0).float().cpu().numpy()
            if (index + 1) % 25 == 0 or index + 1 == len(rows):
                _save(cache_path, ids, names, embeddings, index + 1)
                print(f"encoded {index + 1}/{len(rows)}", flush=True)
        _save(cache_path, ids, names, embeddings, len(rows))
        print(f"wrote {cache_path} {embeddings.shape}", flush=True)
    return embeddings


def feature_columns() -> list[str]:
    return [f"w2v_{index}" for index in range(768)]


def _save(path: Path, ids: list[str], columns: list[str], embeddings: np.ndarray, done: int) -> None:
    np.savez_compressed(
        path,
        version=np.asarray(VERSION),
        model=np.asarray(MODEL_NAME),
        pooling=np.asarray("mean"),
        sample_id=np.asarray(ids),
        columns=np.asarray(columns),
        embeddings=embeddings,
        done=np.asarray(done),
    )


def oof_predict(factory, x: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    pred = np.empty(len(y), dtype=np.float64)
    for fold in range(int(folds.max()) + 1):
        train_mask = folds != fold
        valid_mask = folds == fold
        model = factory()
        model.fit(x[train_mask], y[train_mask])
        pred[valid_mask] = model.predict(x[valid_mask])
    return pred


def ridge() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
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
    rows = list(csv.DictReader(path.open()))
    if [row["sample_id"] for row in rows] != ids:
        raise SystemExit(f"{path} order does not match train.csv")
    return np.asarray([float(row["prediction"]) for row in rows])


def fold_rmse(y: np.ndarray, pred: np.ndarray, folds: np.ndarray) -> list[float]:
    return [
        float(np.sqrt(np.mean((pred[folds == fold] - y[folds == fold]) ** 2)))
        for fold in range(N_FOLDS)
    ]


def log_experiment(
    results_path: Path,
    experiment_id: str,
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
            "feature_set": VERSION,
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
            "notes": "frozen facebook/wav2vec2-base last_hidden_state mean-pool; no fine-tune; no id/encoder features; test audio not read",
        },
    )
    print(experiment_id, f"rmse={metrics['rmse']:.4f}", f"mae={metrics['mae']:.4f}", flush=True)


def write_report(
    path: Path,
    y: np.ndarray,
    folds: np.ndarray,
    cohort: np.ndarray,
    predictions: dict[str, np.ndarray],
    device: str,
) -> None:
    lines = [
        "# Speech embedding report",
        "",
        "Generated by `scripts/run_speech_embeddings.py`.",
        f"Frozen `{MODEL_NAME}`, last hidden state, mean pool, one 768-d vector per clip.",
        f"Device used for extraction: `{device}`. No fine-tune. Test audio was not read.",
        "Official metric: unknown. Comparison number is pooled OOF RMSE.",
        f"Validation: {CV_STRATEGY}. Same `make_folds` as E001–E008.",
        "Ridge uses alpha=10 because the embedding is 768-d on 769 rows. Scaler fit inside each fold.",
        "",
        "| experiment | model | RMSE | MAE | Pearson | Spearman | R² | fold RMSE std | zero RMSE | rest RMSE |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for experiment_id in COMPARE:
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

    best_new = min(("E009", "E010"), key=lambda key: regression_metrics(y, predictions[key])["rmse"])
    current_best = min(("E004", "E005"), key=lambda key: regression_metrics(y, predictions[key])["rmse"])
    new = regression_metrics(y, predictions[best_new])["rmse"]
    current = regression_metrics(y, predictions[current_best])["rmse"]
    rest = ~cohort
    rest_new = float(np.sqrt(np.mean((predictions[best_new][rest] - y[rest]) ** 2)))
    rest_current = float(np.sqrt(np.mean((predictions[current_best][rest] - y[rest]) ** 2)))
    lines.extend(
        [
            "",
            f"Best speech model: `{best_new}` `{MODEL_NAME}` mean-pool.",
            f"Pooled RMSE delta vs current best `{current_best}`: {current - new:+.4f} (positive means speech is better).",
            f"Non-zero-row RMSE: {best_new} {rest_new:.4f}, {current_best} {rest_current:.4f}.",
            "",
            "## Decision",
            "",
        ]
    )
    if current - new > 0.01 and rest_current - rest_new > 0.01:
        lines.append("Speech embeddings beat the current best baseline on pooled RMSE and on the 732 non-zero rows. Keep the family. Do not fine-tune yet.")
    elif current - new > 0.01:
        lines.append("Pooled gain is not matched on the 732 non-zero rows. Do not replace the current best baseline.")
    else:
        lines.append("Speech embeddings do not beat the current best baseline by a meaningful RMSE margin. Do not carry this encoder forward. Do not fine-tune it on 769 labels.")
    lines.extend(["", "No ensemble. No test predictions. No submission.", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=_paths.data_dir())
    parser.add_argument("--cache", type=Path, default=_paths.speech_cache())
    parser.add_argument("--oof-dir", type=Path, default=_paths.oof_dir())
    parser.add_argument("--results", type=Path, default=_paths.results_csv())
    parser.add_argument("--report", type=Path, default=_paths.reports_dir() / "speech_embedding_report.md")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rows = load_train(args.data_dir)
    y = np.asarray([float(row["label"]) for row in rows], dtype=np.float64)
    ids = [f"train/{row['filename']}" for row in rows]
    folds = make_folds(y)
    cohort = np.asarray([int("".join(ch for ch in row["filename"] if ch.isdigit())) >= HIGH_ID for row in rows])
    embeddings = cache_embeddings(rows, args.data_dir, args.cache, args.device)

    predictions = {
        "E004": load_oof(args.oof_dir / "E004_oof.csv", ids),
        "E005": load_oof(args.oof_dir / "E005_oof.csv", ids),
        "E006": load_oof(args.oof_dir / "E006_oof.csv", ids),
        "E008": load_oof(args.oof_dir / "E008_oof.csv", ids),
        "E009": oof_predict(ridge, embeddings, y, folds),
        "E010": oof_predict(hist_gbm, embeddings, y, folds),
    }
    log_experiment(
        args.results,
        "E009",
        "Ridge",
        {"alpha": 10.0, "scaler": "StandardScaler", "model_name": MODEL_NAME, "pooling": "mean"},
        y,
        predictions["E009"],
        folds,
    )
    log_experiment(
        args.results,
        "E010",
        "HistGradientBoostingRegressor",
        {
            "max_depth": 3,
            "learning_rate": 0.05,
            "max_iter": 200,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
            "random_state": SEED,
            "model_name": MODEL_NAME,
            "pooling": "mean",
        },
        y,
        predictions["E010"],
        folds,
    )
    write_oof(args.oof_dir / "E009_oof.csv", rows, y, folds, predictions["E009"])
    write_oof(args.oof_dir / "E010_oof.csv", rows, y, folds, predictions["E010"])
    write_report(args.report, y, folds, cohort, predictions, args.device)
    best = min(("E009", "E010"), key=lambda key: regression_metrics(y, predictions[key])["rmse"])
    current = min(("E004", "E005"), key=lambda key: regression_metrics(y, predictions[key])["rmse"])
    delta = regression_metrics(y, predictions[current])["rmse"] - regression_metrics(y, predictions[best])["rmse"]
    print(f"wrote {args.report}", flush=True)
    print("best", best, "delta_vs_current_best", f"{delta:+.4f}", "current", current, flush=True)


if __name__ == "__main__":
    main()
