#!/usr/bin/env python3
"""ASR linguistic features and TF-IDF Ridge. Train only.

Test audio is not transcribed. Filename and encoder tag are not features.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shl import paths as _paths
from shl.asr import ASR_VERSION, transcript_path, transcribe_file, load_transcript, save_transcript
from shl.cv import N_FOLDS, SEED, make_folds, target_bin
from shl.experiments import append_result
from shl.metrics import regression_metrics
from shl.text import FEATURE_NAMES, linguistic_features

HIGH_ID = 5000
CV_STRATEGY = f"StratifiedKFold bins n={N_FOLDS} seed={SEED}"
LING_VERSION = "asr_ling_v1"


def load_train(data_dir: Path) -> list[dict[str, str]]:
    with (data_dir / "train.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def ensure_transcripts(rows: list[dict[str, str]], data_dir: Path, cache_dir: Path) -> list[dict[str, object]]:
    pending = []
    cached: dict[str, dict[str, object]] = {}
    for row in rows:
        path = transcript_path(cache_dir, "train", row["filename"])
        payload = load_transcript(path)
        if payload is None:
            pending.append(row)
        else:
            cached[row["filename"]] = payload
    print(f"cached {len(cached)} pending {len(pending)}", flush=True)
    if pending:
        from faster_whisper import WhisperModel

        model = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=8)
        for index, row in enumerate(pending, start=1):
            payload = transcribe_file(model, data_dir / "train" / row["filename"], "train", row["filename"])
            save_transcript(transcript_path(cache_dir, "train", row["filename"]), payload)
            cached[row["filename"]] = payload
            if index % 10 == 0 or index == len(pending):
                print(f"transcribed {index}/{len(pending)}", flush=True)
    return [cached[row["filename"]] for row in rows]


def ling_matrix(payloads: list[dict[str, object]]) -> np.ndarray:
    rows = [
        linguistic_features(payload, float(payload.get("duration") or 0.0))
        for payload in payloads
    ]
    return np.vstack(rows)


def oof_ridge_table(x: np.ndarray, y: np.ndarray, folds: np.ndarray, alpha: float) -> np.ndarray:
    pred = np.empty(len(y), dtype=np.float64)
    for fold in range(int(folds.max()) + 1):
        train_mask = folds != fold
        valid_mask = folds == fold
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=alpha)),
            ]
        )
        pipe.fit(x[train_mask], y[train_mask])
        pred[valid_mask] = pipe.predict(x[valid_mask])
    return pred


def oof_tfidf(texts: list[str], y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    pred = np.empty(len(y), dtype=np.float64)
    for fold in range(int(folds.max()) + 1):
        train_mask = folds != fold
        valid_mask = folds == fold
        train_text = [texts[index] for index in np.flatnonzero(train_mask)]
        valid_text = [texts[index] for index in np.flatnonzero(valid_mask)]
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=5000,
            sublinear_tf=True,
        )
        x_train = vectorizer.fit_transform(train_text)
        x_valid = vectorizer.transform(valid_text)
        model = Ridge(alpha=1.0)
        model.fit(x_train, y[train_mask])
        pred[valid_mask] = model.predict(x_valid)
    return pred


def write_oof(path: Path, rows: list[dict[str, str]], y: np.ndarray, folds: np.ndarray, pred: np.ndarray) -> None:
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
    return [
        float(np.sqrt(np.mean((pred[folds == fold] - y[folds == fold]) ** 2)))
        for fold in range(int(folds.max()) + 1)
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
    notes: str,
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
            "notes": notes,
        },
    )
    print(experiment_id, f"rmse={metrics['rmse']:.4f}", f"mae={metrics['mae']:.4f}", flush=True)


def load_oof(path: Path) -> np.ndarray:
    rows = list(csv.DictReader(path.open()))
    return np.asarray([float(row["prediction"]) for row in rows])


def write_report(
    path: Path,
    rows: list[dict[str, str]],
    y: np.ndarray,
    folds: np.ndarray,
    predictions: dict[str, np.ndarray],
    payloads: list[dict[str, object]],
    cohort: np.ndarray,
) -> None:
    word_counts = np.asarray([len(str(payload.get("text") or "").split()) for payload in payloads])
    empty = int((word_counts == 0).sum())
    lines = [
        "# ASR baseline report",
        "",
        "Generated by `scripts/run_asr_baselines.py`. Train audio only. Test audio was not transcribed.",
        "Official competition metric: unknown. Comparison number is pooled OOF RMSE. That is not an official metric.",
        "",
        f"ASR: faster-whisper `{ASR_VERSION}`, CPU int8, language=en, beam_size=1, vad_filter=true, word timestamps on.",
        "GPU path failed: `libcublas.so.12` is not installed. CPU int8 was used instead.",
        f"Validation: {CV_STRATEGY}. Same folds as E001–E004 (`make_folds`, seed 42).",
        "TF-IDF vectorizer is fit on the training fold only.",
        "",
        "## Transcripts",
        "",
        f"- cached files: {len(payloads)}",
        f"- empty transcripts: {empty}",
        f"- words per clip: min {int(word_counts.min())}, median {float(np.median(word_counts)):.0f}, mean {float(word_counts.mean()):.1f}, max {int(word_counts.max())}",
        "",
        "## Results",
        "",
        "| experiment | model | RMSE | MAE | Pearson | Spearman | R² | fold RMSE mean | fold RMSE std | zero RMSE | rest RMSE |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    labels = {
        "E004": "acoustic HistGradientBoosting",
        "E005": "linguistic Ridge",
        "E006": "TF-IDF Ridge",
    }
    for experiment_id, pred in predictions.items():
        metrics = regression_metrics(y, pred)
        per_fold = fold_rmse(y, pred, folds)
        zero = float(np.sqrt(np.mean((pred[cohort] - y[cohort]) ** 2)))
        rest = float(np.sqrt(np.mean((pred[~cohort] - y[~cohort]) ** 2)))
        pearson = "n/a" if np.isnan(metrics["pearson"]) else f"{metrics['pearson']:.4f}"
        spearman = "n/a" if np.isnan(metrics["spearman"]) else f"{metrics['spearman']:.4f}"
        lines.append(
            f"| {experiment_id} | {labels[experiment_id]} | {metrics['rmse']:.4f} | {metrics['mae']:.4f} | {pearson} | {spearman} | {metrics['r2']:.4f} | "
            f"{float(np.mean(per_fold)):.4f} | {float(np.std(per_fold, ddof=1)):.4f} | {zero:.4f} | {rest:.4f} |"
        )

    acoustic = predictions["E004"]
    best_text_id = min(("E005", "E006"), key=lambda key: regression_metrics(y, predictions[key])["rmse"])
    best_text = predictions[best_text_id]
    delta = regression_metrics(y, acoustic)["rmse"] - regression_metrics(y, best_text)["rmse"]
    rest = ~cohort
    rest_delta = (
        float(np.sqrt(np.mean((acoustic[rest] - y[rest]) ** 2)))
        - float(np.sqrt(np.mean((best_text[rest] - y[rest]) ** 2)))
    )
    lines.extend(
        [
            "",
            f"Best text model: `{best_text_id}`. Pooled RMSE delta vs E004: {delta:+.4f} (positive means text is better).",
            f"Same delta on the 732 non-zero rows: {rest_delta:+.4f}.",
            "",
            "## Decision",
            "",
        ]
    )
    if delta > 0.01 and rest_delta > 0.01:
        lines.append("Text beats the acoustic baseline on pooled RMSE and on the non-zero rows. Keep the family.")
    elif delta > 0.01 and rest_delta <= 0.01:
        lines.append("Pooled gain is concentrated on the zero cohort. Do not treat text as a grammar improvement yet.")
    else:
        lines.append("Text does not beat E004 by a meaningful RMSE margin. Do not replace the acoustic baseline with text alone.")

    residual = best_text - y
    order = np.argsort(-np.abs(residual))
    lines.extend(["", f"Largest absolute residuals for {best_text_id}:", "", "| sample_id | target | prediction | residual | words | zero_cohort |", "| --- | --- | --- | --- | --- | --- |"])
    for index in order[:12]:
        lines.append(
            f"| train/{rows[index]['filename']} | {y[index]:.1f} | {best_text[index]:.3f} | {residual[index]:+.3f} | {int(word_counts[index])} | {int(cohort[index])} |"
        )
    bins = np.asarray([target_bin(float(value)) for value in y])
    lines.extend(["", "| bin | n | E004 RMSE | best text RMSE |", "| --- | --- | --- | --- |"])
    for name in ["0", "1-2", "2.5-3", "3.5-4", "4.5-5"]:
        mask = bins == name
        e004 = float(np.sqrt(np.mean((acoustic[mask] - y[mask]) ** 2)))
        text = float(np.sqrt(np.mean((best_text[mask] - y[mask]) ** 2)))
        lines.append(f"| {name} | {int(mask.sum())} | {e004:.4f} | {text:.4f} |")
    lines.extend(
        [
            "",
            "## Not done",
            "",
            "- No test transcription.",
            "- No text embeddings, speech embeddings, fine-tuning, or ensemble.",
            "- No submission file.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=_paths.data_dir())
    parser.add_argument("--asr-dir", type=Path, default=_paths.asr_dir())
    parser.add_argument("--oof-dir", type=Path, default=_paths.oof_dir())
    parser.add_argument("--results", type=Path, default=_paths.results_csv())
    parser.add_argument("--report", type=Path, default=_paths.reports_dir() / "asr_baseline_report.md")
    parser.add_argument("--transcribe-only", action="store_true")
    args = parser.parse_args()

    rows = load_train(args.data_dir)
    y = np.asarray([float(row["label"]) for row in rows], dtype=np.float64)
    folds = make_folds(y)
    cohort = np.asarray([int("".join(ch for ch in row["filename"] if ch.isdigit())) >= HIGH_ID for row in rows])
    payloads = ensure_transcripts(rows, args.data_dir, args.asr_dir)
    if args.transcribe_only:
        print("transcribe-only done", flush=True)
        return

    texts = [str(payload.get("text") or "") for payload in payloads]
    features = ling_matrix(payloads)
    np.savez_compressed(
        args.asr_dir / "train_ling_v1.npz",
        version=np.asarray(LING_VERSION),
        sample_id=np.asarray([f"train/{row['filename']}" for row in rows]),
        columns=np.asarray(FEATURE_NAMES),
        features=features,
    )
    predictions = {
        "E004": load_oof(args.oof_dir / "E004_oof.csv"),
        "E005": oof_ridge_table(features, y, folds, alpha=1.0),
        "E006": oof_tfidf(texts, y, folds),
    }
    if list(csv.DictReader((args.oof_dir / "E004_oof.csv").open()))[0]["sample_id"] != f"train/{rows[0]['filename']}":
        raise SystemExit("E004 OOF order does not match train.csv")
    log_experiment(
        args.results,
        "E005",
        LING_VERSION,
        "Ridge",
        {"alpha": 1.0, "features": FEATURE_NAMES},
        y,
        predictions["E005"],
        folds,
        "faster-whisper small int8 CPU; handcrafted counts/rate/ttr/fillers/pauses; no id/encoder features",
    )
    log_experiment(
        args.results,
        "E006",
        "tfidf_word_1-2",
        "Ridge",
        {"alpha": 1.0, "ngram_range": [1, 2], "min_df": 2, "max_features": 5000, "sublinear_tf": True},
        y,
        predictions["E006"],
        folds,
        "TF-IDF word 1-2 grams fit in-fold on faster-whisper transcripts; no id/encoder features",
    )
    write_oof(args.oof_dir / "E005_oof.csv", rows, y, folds, predictions["E005"])
    write_oof(args.oof_dir / "E006_oof.csv", rows, y, folds, predictions["E006"])
    write_report(args.report, rows, y, folds, predictions, payloads, cohort)
    print(f"wrote {args.report}", flush=True)


if __name__ == "__main__":
    main()
