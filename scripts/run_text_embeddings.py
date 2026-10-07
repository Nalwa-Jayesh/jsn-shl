#!/usr/bin/env python3
"""Frozen MiniLM embeddings on cached train transcripts. No fine-tune. No test text."""

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
from shl.asr import ASR_VERSION, load_transcript, transcript_path
from shl.cv import N_FOLDS, SEED, make_folds
from shl.experiments import append_result
from shl.metrics import regression_metrics

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VERSION = "minilm_l6_v1"
HIGH_ID = 5000
CV_STRATEGY = f"StratifiedKFold bins n={N_FOLDS} seed={SEED}"
LABELS = {
    "E004": "acoustic HistGradientBoosting",
    "E005": "linguistic Ridge",
    "E006": "TF-IDF Ridge",
    "E007": "MiniLM Ridge",
    "E008": "MiniLM HistGradientBoosting",
}


def load_train(data_dir: Path) -> list[dict[str, str]]:
    with (data_dir / "train.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_texts(rows: list[dict[str, str]], asr_dir: Path) -> list[str]:
    texts = []
    for row in rows:
        payload = load_transcript(transcript_path(asr_dir, "train", row["filename"]))
        if payload is None:
            raise SystemExit(f"missing transcript for train/{row['filename']}; run scripts/run_asr_baselines.py first")
        if payload.get("version") != ASR_VERSION:
            raise SystemExit(f"unexpected transcript version for {row['filename']}")
        texts.append(str(payload.get("text") or ""))
    return texts


def cache_embeddings(rows: list[dict[str, str]], texts: list[str], cache_path: Path) -> np.ndarray:
    ids = [f"train/{row['filename']}" for row in rows]
    if cache_path.exists():
        payload = np.load(cache_path, allow_pickle=False)
        if payload["version"].item() != VERSION or list(payload["sample_id"].astype(str)) != ids:
            raise SystemExit(f"{cache_path} exists but does not match {VERSION}; refusing to overwrite")
        print(f"reuse {cache_path} {payload['embeddings'].shape}", flush=True)
        return payload["embeddings"]

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(MODEL_NAME, device="cpu")
    embeddings = encoder.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        version=np.asarray(VERSION),
        model=np.asarray(MODEL_NAME),
        sample_id=np.asarray(ids),
        embeddings=embeddings,
    )
    print(f"wrote {cache_path} {embeddings.shape}", flush=True)
    return embeddings


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
            "notes": "frozen all-MiniLM-L6-v2 on cached faster-whisper train transcripts; no fine-tune; no id/encoder features; test not used",
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
        "# Text embedding report",
        "",
        "Generated by `scripts/run_text_embeddings.py`.",
        f"Frozen `{MODEL_NAME}` on cached train transcripts only. No fine-tune. Test transcripts not used.",
        "Official metric: unknown. Comparison number is pooled OOF RMSE.",
        f"Validation: {CV_STRATEGY}. Same `make_folds` as E001–E006.",
        "Scaler fit inside each fold for Ridge. Embedding model was not trained on labels.",
        "",
        "| experiment | model | RMSE | MAE | Pearson | Spearman | R² | fold RMSE std | zero RMSE | rest RMSE |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for experiment_id, pred in predictions.items():
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

    best_new = min(("E007", "E008"), key=lambda key: regression_metrics(y, predictions[key])["rmse"])
    new = regression_metrics(y, predictions[best_new])["rmse"]
    tfidf = regression_metrics(y, predictions["E006"])["rmse"]
    e005 = regression_metrics(y, predictions["E005"])["rmse"]
    e004 = regression_metrics(y, predictions["E004"])["rmse"]
    rest = ~cohort
    rest_new = float(np.sqrt(np.mean((predictions[best_new][rest] - y[rest]) ** 2)))
    rest_e005 = float(np.sqrt(np.mean((predictions["E005"][rest] - y[rest]) ** 2)))
    rest_e004 = float(np.sqrt(np.mean((predictions["E004"][rest] - y[rest]) ** 2)))
    delta = tfidf - new
    lines.extend(
        [
            "",
            f"Best embedding model: `{best_new}`. Pooled RMSE delta vs E006 TF-IDF: {delta:+.4f} (positive means embeddings are better).",
            f"Same delta vs E005: {e005 - new:+.4f}. Same delta vs E004: {e004 - new:+.4f}.",
            f"Non-zero-row RMSE: {best_new} {rest_new:.4f}, E005 {rest_e005:.4f}, E004 {rest_e004:.4f}.",
            "",
            "## Decision",
            "",
        ]
    )
    if delta > 0.01 and rest_new + 0.01 < min(rest_e004, rest_e005):
        lines.append("Embeddings beat TF-IDF and also beat E004 and E005 on the 732 non-zero rows. Keep the family.")
    elif delta > 0.01:
        lines.append(
            "Embeddings beat TF-IDF by more than 0.01 RMSE. Keep the family only as a complement. "
            "Do not replace E004 or E005 unless the non-zero-row RMSE above is also better."
        )
    else:
        lines.append("Embeddings do not beat TF-IDF by a meaningful RMSE margin. Do not carry this encoder forward.")
    lines.extend(["", "No ensemble. No test predictions. No submission.", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=_paths.data_dir())
    parser.add_argument("--asr-dir", type=Path, default=_paths.asr_dir())
    parser.add_argument("--cache", type=Path, default=_paths.text_cache())
    parser.add_argument("--oof-dir", type=Path, default=_paths.oof_dir())
    parser.add_argument("--results", type=Path, default=_paths.results_csv())
    parser.add_argument("--report", type=Path, default=_paths.reports_dir() / "text_embedding_report.md")
    args = parser.parse_args()

    rows = load_train(args.data_dir)
    y = np.asarray([float(row["label"]) for row in rows], dtype=np.float64)
    ids = [f"train/{row['filename']}" for row in rows]
    folds = make_folds(y)
    cohort = np.asarray([int("".join(ch for ch in row["filename"] if ch.isdigit())) >= HIGH_ID for row in rows])
    texts = load_texts(rows, args.asr_dir)
    embeddings = cache_embeddings(rows, texts, args.cache)

    predictions = {
        "E004": load_oof(args.oof_dir / "E004_oof.csv", ids),
        "E005": load_oof(args.oof_dir / "E005_oof.csv", ids),
        "E006": load_oof(args.oof_dir / "E006_oof.csv", ids),
        "E007": oof_predict(ridge, embeddings, y, folds),
        "E008": oof_predict(hist_gbm, embeddings, y, folds),
    }
    log_experiment(
        args.results,
        "E007",
        "Ridge",
        {"alpha": 1.0, "scaler": "StandardScaler", "model_name": MODEL_NAME},
        y,
        predictions["E007"],
        folds,
    )
    log_experiment(
        args.results,
        "E008",
        "HistGradientBoostingRegressor",
        {
            "max_depth": 3,
            "learning_rate": 0.05,
            "max_iter": 200,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
            "random_state": SEED,
            "model_name": MODEL_NAME,
        },
        y,
        predictions["E008"],
        folds,
    )
    write_oof(args.oof_dir / "E007_oof.csv", rows, y, folds, predictions["E007"])
    write_oof(args.oof_dir / "E008_oof.csv", rows, y, folds, predictions["E008"])
    write_report(args.report, y, folds, cohort, predictions)
    best = min(("E007", "E008"), key=lambda key: regression_metrics(y, predictions[key])["rmse"])
    delta = regression_metrics(y, predictions["E006"])["rmse"] - regression_metrics(y, predictions[best])["rmse"]
    print(f"wrote {args.report}", flush=True)
    print("best", best, "delta_vs_tfidf", f"{delta:+.4f}", flush=True)


if __name__ == "__main__":
    main()
