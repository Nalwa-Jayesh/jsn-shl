#!/usr/bin/env python3
"""OOF ensembles. Weights and the meta-model are fit on the other folds only."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shl import paths as _paths
from shl.cv import N_FOLDS
from shl.experiments import append_result
from shl.metrics import regression_metrics

HIGH_ID = 5000
CV_STRATEGY = "StratifiedKFold bins n=5 seed=42; ensemble weights fit on the other folds only"
WEIGHTS = np.round(np.linspace(0.0, 1.0, 11), 1)
MEMBERS = {
    "E004": "acoustic HistGradientBoosting",
    "E005": "linguistic Ridge",
    "E010": "wav2vec2 HistGradientBoosting",
    "E012": "acoustic+linguistic HistGradientBoosting",
    "E016": "linguistic+speech HistGradientBoosting",
    "E018": "acoustic+linguistic+speech HistGradientBoosting",
}


def load_train_ids(data_dir: Path) -> list[str]:
    with (data_dir / "train.csv").open(newline="") as handle:
        return [f"train/{row['filename']}" for row in csv.DictReader(handle)]


def load_oof(path: Path, ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(path.open()))
    if [row["sample_id"] for row in rows] != ids:
        raise SystemExit(f"{path} order does not match train.csv")
    folds = np.asarray([int(row["fold"]) for row in rows])
    y = np.asarray([float(row["target"]) for row in rows])
    pred = np.asarray([float(row["prediction"]) for row in rows])
    cohort = np.asarray([int(row["zero_cohort"]) for row in rows])
    if set(folds.tolist()) != set(range(N_FOLDS)):
        raise SystemExit(f"{path} does not contain folds 0-{N_FOLDS - 1}")
    return folds, y, pred, cohort


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def nested_weight(primary: np.ndarray, other: np.ndarray, y: np.ndarray, folds: np.ndarray) -> tuple[np.ndarray, list[float]]:
    pred = np.empty(len(y), dtype=np.float64)
    chosen = []
    for fold in range(N_FOLDS):
        train = folds != fold
        valid = folds == fold
        best_weight = 1.0
        best_score = float("inf")
        for weight in WEIGHTS:
            blended = weight * primary[train] + (1.0 - weight) * other[train]
            score = rmse(y[train], blended)
            if score < best_score:
                best_score = score
                best_weight = float(weight)
        chosen.append(best_weight)
        pred[valid] = best_weight * primary[valid] + (1.0 - best_weight) * other[valid]
    return pred, chosen


def nested_ridge(columns: list[np.ndarray], y: np.ndarray, folds: np.ndarray) -> tuple[np.ndarray, list[list[float]]]:
    matrix = np.column_stack(columns)
    pred = np.empty(len(y), dtype=np.float64)
    coefs = []
    for fold in range(N_FOLDS):
        train = folds != fold
        valid = folds == fold
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        )
        model.fit(matrix[train], y[train])
        pred[valid] = model.predict(matrix[valid])
        coefs.append([float(value) for value in model.named_steps["model"].coef_])
    return pred, coefs


def write_oof(
    path: Path,
    ids: list[str],
    folds: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray,
    cohort: np.ndarray,
) -> None:
    if path.exists():
        raise SystemExit(f"{path} already exists; refusing to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "filename", "fold", "target", "prediction", "zero_cohort"],
        )
        writer.writeheader()
        for sample_id, fold, target, score, flag in zip(ids, folds, y, pred, cohort, strict=True):
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "filename": sample_id.split("/", 1)[1],
                    "fold": int(fold),
                    "target": f"{target:.1f}",
                    "prediction": f"{float(score):.6f}",
                    "zero_cohort": int(flag),
                }
            )


def fold_rmse(y: np.ndarray, pred: np.ndarray, folds: np.ndarray) -> list[float]:
    return [rmse(y[folds == fold], pred[folds == fold]) for fold in range(N_FOLDS)]


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


def write_report(
    path: Path,
    y: np.ndarray,
    folds: np.ndarray,
    cohort: np.ndarray,
    members: dict[str, np.ndarray],
    ensembles: dict[str, np.ndarray],
    notes: dict[str, str],
) -> None:
    lines = [
        "# Ensemble report",
        "",
        "Generated by `scripts/run_ensemble.py`.",
        "Members are saved OOF predictions. No base model was refit. Test predictions were not used.",
        "Official metric: unknown. Comparison number is pooled OOF RMSE.",
        "Weights and the Ridge meta-model are chosen inside each outer fold, using only the other four folds.",
        "A weight picked on the full OOF set would be optimistic. That selection was not used for the logged score.",
        "",
        "Member check: same 769 `sample_id`s, same fold ids, same targets, folds 0-4 present.",
        "",
        "## Why these members",
        "",
        "E018 is the best completed model (pooled RMSE 0.6068). E012 and E016 are the next two, but they are not diverse:",
        "",
    ]
    for experiment_id in ("E012", "E016", "E010", "E005", "E004"):
        pred_corr = float(np.corrcoef(members["E018"], members[experiment_id])[0, 1])
        resid_corr = float(np.corrcoef(members["E018"] - y, members[experiment_id] - y)[0, 1])
        lines.append(
            f"- `{experiment_id}` prediction correlation with E018 {pred_corr:.3f}, residual correlation {resid_corr:.3f}"
        )
    lines.extend(
        [
            "",
            "E012 and E016 share the learner and most of the features with E018. Averaging them is not a new signal.",
            "E004 and E005 disagree more with E018. They are the complementary members.",
            "",
            "## Results",
            "",
            "| experiment | model | RMSE | MAE | Pearson | Spearman | R² | fold RMSE std | zero RMSE | rest RMSE |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    labels = {
        "E018": "best individual, all-three HistGradientBoosting",
        "E019": "equal average E004+E005+E010",
        "E020": "nested weight E018+E005",
        "E021": "nested weight E018+E004",
        "E022": "nested Ridge stack E004+E005+E010",
        "E023": "nested Ridge stack E012+E016+E018",
    }
    order = ("E018", "E019", "E020", "E021", "E022", "E023")
    combined = {"E018": members["E018"], **ensembles}
    for experiment_id in order:
        pred = combined[experiment_id]
        metrics = regression_metrics(y, pred)
        per_fold = fold_rmse(y, pred, folds)
        zero = rmse(y[cohort == 1], pred[cohort == 1])
        rest = rmse(y[cohort == 0], pred[cohort == 0])
        lines.append(
            f"| {experiment_id} | {labels[experiment_id]} | {metrics['rmse']:.4f} | {metrics['mae']:.4f} | "
            f"{metrics['pearson']:.4f} | {metrics['spearman']:.4f} | {metrics['r2']:.4f} | "
            f"{float(np.std(per_fold, ddof=1)):.4f} | {zero:.4f} | {rest:.4f} |"
        )

    best_id = min(order, key=lambda key: regression_metrics(y, combined[key])["rmse"])
    best = regression_metrics(y, combined[best_id])["rmse"]
    base = regression_metrics(y, members["E018"])["rmse"]
    rest = cohort == 0
    rest_best = rmse(y[rest], combined[best_id][rest])
    rest_base = rmse(y[rest], members["E018"][rest])
    lines.extend(
        [
            "",
            "## Selection detail",
            "",
        ]
    )
    for experiment_id, note in notes.items():
        lines.append(f"- `{experiment_id}`: {note}")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Best individual: `E018`, pooled RMSE {base:.4f}.",
            f"Best ensemble: `{best_id}`, pooled RMSE {best:.4f}.",
            f"Pooled delta vs E018: {base - best:+.4f} (positive means the ensemble is better).",
            f"Non-zero-row RMSE: ensemble {rest_best:.4f}, E018 {rest_base:.4f}.",
            "",
        ]
    )
    if base - best > 0.01 and rest_base - rest_best > 0.01:
        lines.append("The ensemble beats E018 on pooled RMSE and on the 732 non-zero rows. Keep it. The gain is still small relative to the E010 to E018 fusion jump.")
    elif base - best > 0.005:
        lines.append("The ensemble is only marginally better than E018. Treat it as tied. Prefer E018 unless a later error pass shows the extra member fixes a real failure mode.")
    else:
        lines.append("No ensemble beats E018 by a meaningful RMSE margin. Keep E018. E004 and E005 are complementary in residual correlation, but that disagreement is not large enough to improve the fused model once the weight is chosen out of fold.")
    lines.extend(
        [
            "",
            "E012 and E016 do not add a useful second signal on top of E018. Their residuals already match it.",
            "No test predictions. No refit of base models. No submission.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=_paths.data_dir())
    parser.add_argument("--oof-dir", type=Path, default=_paths.oof_dir())
    parser.add_argument("--results", type=Path, default=_paths.results_csv())
    parser.add_argument("--report", type=Path, default=_paths.reports_dir() / "ensemble_report.md")
    args = parser.parse_args()

    ids = load_train_ids(args.data_dir)
    loaded = {name: load_oof(args.oof_dir / f"{name}_oof.csv", ids) for name in MEMBERS}
    folds, y, _, cohort = loaded["E018"]
    for name, (other_folds, other_y, _, other_cohort) in loaded.items():
        if not np.array_equal(other_folds, folds) or not np.allclose(other_y, y) or not np.array_equal(other_cohort, cohort):
            raise SystemExit(f"{name} OOF fold, target, or cohort columns do not match E018")
    members = {name: item[2] for name, item in loaded.items()}

    equal = (members["E004"] + members["E005"] + members["E010"]) / 3.0
    weight_text, text_weights = nested_weight(members["E018"], members["E005"], y, folds)
    weight_acoustic, acoustic_weights = nested_weight(members["E018"], members["E004"], y, folds)
    stack_diverse, diverse_coefs = nested_ridge(
        [members["E004"], members["E005"], members["E010"]],
        y,
        folds,
    )
    stack_strong, strong_coefs = nested_ridge(
        [members["E012"], members["E016"], members["E018"]],
        y,
        folds,
    )
    ensembles = {
        "E019": equal,
        "E020": weight_text,
        "E021": weight_acoustic,
        "E022": stack_diverse,
        "E023": stack_strong,
    }
    specs = {
        "E019": (
            "oof:E004+E005+E010",
            "equal_average",
            {"members": ["E004", "E005", "E010"], "weights": [1 / 3, 1 / 3, 1 / 3]},
            "equal average of diverse single-modality OOFs",
        ),
        "E020": (
            "oof:E018+E005",
            "nested_weighted_average",
            {"members": ["E018", "E005"], "grid": WEIGHTS.tolist(), "chosen_e018_weight_by_fold": text_weights},
            "E018 weight chosen per fold on the other folds; grid 0.0-1.0 step 0.1",
        ),
        "E021": (
            "oof:E018+E004",
            "nested_weighted_average",
            {"members": ["E018", "E004"], "grid": WEIGHTS.tolist(), "chosen_e018_weight_by_fold": acoustic_weights},
            "E018 weight chosen per fold on the other folds; grid 0.0-1.0 step 0.1",
        ),
        "E022": (
            "oof:E004+E005+E010",
            "nested_ridge_stack",
            {"members": ["E004", "E005", "E010"], "alpha": 1.0, "coef_by_fold": diverse_coefs},
            "Ridge meta-model fit per fold on OOF predictions from the other folds",
        ),
        "E023": (
            "oof:E012+E016+E018",
            "nested_ridge_stack",
            {"members": ["E012", "E016", "E018"], "alpha": 1.0, "coef_by_fold": strong_coefs},
            "Ridge meta-model on the three strongest fused OOFs; expected to be redundant",
        ),
    }
    detail = {
        "E020": "chosen E018 weights by fold " + ", ".join(f"{value:.1f}" for value in text_weights),
        "E021": "chosen E018 weights by fold " + ", ".join(f"{value:.1f}" for value in acoustic_weights),
        "E022": "Ridge coefs by fold (E004, E005, E010) " + "; ".join(
            "[" + ", ".join(f"{value:.3f}" for value in coef) + "]" for coef in diverse_coefs
        ),
        "E023": "Ridge coefs by fold (E012, E016, E018) " + "; ".join(
            "[" + ", ".join(f"{value:.3f}" for value in coef) + "]" for coef in strong_coefs
        ),
    }
    for experiment_id, pred in ensembles.items():
        feature_set, model, hyperparameters, notes = specs[experiment_id]
        log_experiment(args.results, experiment_id, feature_set, model, hyperparameters, y, pred, folds, notes)
        write_oof(args.oof_dir / f"{experiment_id}_oof.csv", ids, folds, y, pred, cohort)

    write_report(args.report, y, folds, cohort, members, ensembles, detail)
    best = min(("E018", *ensembles), key=lambda key: rmse(y, members[key] if key == "E018" else ensembles[key]))
    base = rmse(y, members["E018"])
    chosen = rmse(y, members[best] if best == "E018" else ensembles[best])
    print(f"wrote {args.report}", flush=True)
    print("best", best, "delta_vs_E018", f"{base - chosen:+.4f}", flush=True)


if __name__ == "__main__":
    main()
