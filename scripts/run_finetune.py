#!/usr/bin/env python3
"""Selective wav2vec2 fine-tune. Train audio only. Same 5 folds. No test.

E018 reference (do not refit): frozen facebook/wav2vec2-base last_hidden_state
mean-pool (768-d) + acoustic_v1 (166-d) + asr_ling_v1 (19-d), HistGradientBoosting
depth=3 lr=0.05 iter=200 min_leaf=20 l2=1. Pooled OOF RMSE 0.6068.

E024: encoder fully frozen, torch head only, on cached full-clip mean embeddings.
E025: unfreeze encoder.layers.11 + encoder.layer_norm only, same head,
      very small encoder LR, early stopping on an inner split.
Both are speech-only. Comparison target is E018 (fusion) and E010 (frozen speech HGB).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shl import paths as _paths
from shl.cv import N_FOLDS, SEED, make_folds, target_bin
from shl.experiments import append_result
from shl.metrics import regression_metrics

MODEL_NAME = "facebook/wav2vec2-base"
HIGH_ID = 5000
CV_STRATEGY = f"StratifiedKFold bins n={N_FOLDS} seed={SEED}; inner 80/20 split per outer fold for early stopping"
CROP_SEC = 20
BATCH = 2
MAX_EPOCHS = 8
PATIENCE = 2
HEAD_LR = 1e-3
ENC_LR = 1e-5
WEIGHT_DECAY = 0.01
UNFROZEN_PREFIXES = ("encoder.layers.11", "encoder.layer_norm")
LABELS = {
    "E018": "fusion HistGradientBoosting (frozen, best)",
    "E010": "wav2vec2 frozen HGB",
    "E024": "wav2vec2 frozen, torch head",
    "E025": "wav2vec2 last-layer fine-tune + head",
}


class Head(nn.Module):
    def __init__(self, dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(dim, 1)
        nn.init.normal_(self.out.weight, std=0.02)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.drop(self.norm(x))).squeeze(-1)


def load_train(data_dir: Path) -> list[dict[str, str]]:
    with (data_dir / "train.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_cached_speech(path: Path, ids: list[str]) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    if list(payload["sample_id"].astype(str)) != ids:
        raise SystemExit(f"{path} order does not match train.csv")
    key = "embeddings" if "embeddings" in payload.files else "features"
    return np.asarray(payload[key], dtype=np.float32)


def load_oof(path: Path, ids: list[str]) -> np.ndarray:
    rows = list(csv.DictReader(path.open()))
    if [r["sample_id"] for r in rows] != ids:
        raise SystemExit(f"{path} order does not match train.csv")
    return np.asarray([float(r["prediction"]) for r in rows])


def inner_split(outer_train: np.ndarray, y: np.ndarray, fold: int) -> tuple[np.ndarray, np.ndarray]:
    bins = np.asarray([target_bin(float(v)) for v in y[outer_train]])
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED + fold)
    inner_train_rel, inner_val_rel = next(splitter.split(np.zeros(len(outer_train)), bins))
    return outer_train[inner_train_rel], outer_train[inner_val_rel]


def train_head_cached(
    x: np.ndarray,
    y: np.ndarray,
    inner_train: np.ndarray,
    inner_val: np.ndarray,
    device: torch.device,
    seed: int,
) -> Head:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    head = Head().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(x[inner_train]), torch.from_numpy(y[inner_train].astype(np.float32))
    )
    loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True, generator=gen)
    val_x = torch.from_numpy(x[inner_val]).to(device)
    val_y = torch.from_numpy(y[inner_val].astype(np.float32)).to(device)
    best = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    bad = 0
    for _ in range(MAX_EPOCHS):
        head.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            opt.zero_grad()
            loss = loss_fn(head(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
        head.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(head(val_x), val_y))
        if val_loss < best - 1e-6:
            best = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        head.load_state_dict(best_state)
    return head


@torch.no_grad()
def clip_vector(
    model, processor, audio: np.ndarray, device: torch.device, crop_samples: int
) -> np.ndarray:
    """Full-clip mean embedding via non-overlapping crops. No grad."""
    model.eval()
    chunks = []
    for start in range(0, len(audio), crop_samples):
        piece = audio[start : start + crop_samples]
        if len(piece) < crop_samples // 4:
            break
        inputs = processor(piece, sampling_rate=16000, return_tensors="pt")
        hidden = model(inputs["input_values"].to(device)).last_hidden_state
        chunks.append(hidden.mean(dim=1).squeeze(0).cpu())
    if not chunks:
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        hidden = model(inputs["input_values"].to(device)).last_hidden_state
        return hidden.mean(dim=1).squeeze(0).cpu().numpy()
    return torch.stack(chunks).mean(dim=0).numpy()


def run_e025_fold(
    train_files: list[str],
    y: np.ndarray,
    inner_train: np.ndarray,
    inner_val: np.ndarray,
    outer_val: np.ndarray,
    data_dir: Path,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    from transformers import Wav2Vec2Model, Wav2Vec2Processor

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if name.startswith(UNFROZEN_PREFIXES):
            param.requires_grad = True
    head = Head().to(device)
    trainable = [p for p in list(model.parameters()) + list(head.parameters()) if p.requires_grad]
    head_params = list(head.parameters())
    enc_params = [p for p in trainable if not any(p is h for h in head_params)]
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": ENC_LR}, {"params": head_params, "lr": HEAD_LR}],
        weight_decay=WEIGHT_DECAY,
    )
    loss_fn = nn.MSELoss()
    crop_samples = CROP_SEC * 16000
    audios: dict[int, np.ndarray] = {}
    for index in list(inner_train) + list(inner_val) + list(outer_val):
        audio, rate = sf.read(data_dir / "train" / train_files[index], dtype="float32", always_2d=False)
        assert rate == 16000 and audio.ndim == 1
        audios[int(index)] = np.ascontiguousarray(audio)

    def val_rmse(indices: np.ndarray) -> float:
        model.eval()
        head.eval()
        preds = []
        with torch.no_grad():
            for index in indices:
                vec = clip_vector(model, processor, audios[int(index)], device, crop_samples)
                preds.append(head(torch.from_numpy(vec).to(device)).item())
        preds_arr = np.asarray(preds)
        return float(np.sqrt(np.mean((preds_arr - y[indices]) ** 2)))

    best = float("inf")
    best_model: dict[str, torch.Tensor] | None = None
    best_head: dict[str, torch.Tensor] | None = None
    bad = 0
    order = np.asarray(inner_train)
    for _ in range(MAX_EPOCHS):
        model.train()
        head.train()
        rng.shuffle(order)
        for start in range(0, len(order), BATCH):
            batch = order[start : start + BATCH]
            inputs_list = []
            for index in batch:
                audio = audios[int(index)]
                if len(audio) > crop_samples:
                    offset = int(rng.integers(0, len(audio) - crop_samples + 1))
                    piece = audio[offset : offset + crop_samples]
                else:
                    piece = audio
                encoded = processor(piece, sampling_rate=16000, return_tensors="pt")
                inputs_list.append(encoded["input_values"].squeeze(0))
            longest = max(t.shape[0] for t in inputs_list)
            padded = torch.stack(
                [torch.nn.functional.pad(t, (0, longest - t.shape[0])) for t in inputs_list]
            ).to(device)
            attention = torch.stack(
                [
                    torch.cat([torch.ones(t.shape[0]), torch.zeros(longest - t.shape[0])])
                    for t in inputs_list
                ]
            ).long().to(device)
            hidden = model(input_values=padded, attention_mask=attention).last_hidden_state
            pooled = hidden.mean(dim=1)
            targets = torch.from_numpy(y[batch].astype(np.float32)).to(device)
            opt.zero_grad()
            loss = loss_fn(head(pooled), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
        score = val_rmse(inner_val)
        if score < best - 1e-6:
            best = score
            best_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_head = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_model is not None:
        model.load_state_dict(best_model)
        head.load_state_dict(best_head)  # type: ignore[union-attr]
    model.eval()
    head.eval()
    out = np.empty(len(outer_val))
    with torch.no_grad():
        for pos, index in enumerate(outer_val):
            vec = clip_vector(model, processor, audios[int(index)], device, crop_samples)
            out[pos] = head(torch.from_numpy(vec).to(device)).item()
    del model, head, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def write_oof(path: Path, ids: list[str], folds: np.ndarray, y: np.ndarray, pred: np.ndarray, cohort: np.ndarray) -> None:
    if path.exists():
        raise SystemExit(f"{path} already exists; refusing to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["sample_id", "filename", "fold", "target", "prediction", "zero_cohort"]
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
    return [float(np.sqrt(np.mean((pred[folds == f] - y[folds == f]) ** 2))) for f in range(N_FOLDS)]


def log(results: Path, eid: str, feature_set: str, model: str, hyper: dict[str, object], y, pred, folds, notes: str) -> None:
    metrics = regression_metrics(y, pred)
    per_fold = fold_rmse(y, pred, folds)
    append_result(
        results,
        {
            "experiment_id": eid,
            "feature_set": feature_set,
            "model": model,
            "hyperparameters": json.dumps(hyper, sort_keys=True),
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
    print(eid, f"rmse={metrics['rmse']:.4f}", f"mae={metrics['mae']:.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=_paths.data_dir())
    parser.add_argument("--speech-cache", type=Path, default=_paths.speech_cache())
    parser.add_argument("--oof-dir", type=Path, default=_paths.oof_dir())
    parser.add_argument("--results", type=Path, default=_paths.results_csv())
    parser.add_argument("--report", type=Path, default=_paths.reports_dir() / "finetuning_report.md")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true", help="fold 0 only, quick check")
    args = parser.parse_args()

    rows = load_train(args.data_dir)
    y = np.asarray([float(r["label"]) for r in rows], dtype=np.float64)
    ids = [f"train/{r['filename']}" for r in rows]
    files = [r["filename"] for r in rows]
    folds = make_folds(y)
    cohort = np.asarray([int("".join(c for c in f if c.isdigit())) >= HIGH_ID for f in files])
    device = torch.device(args.device)
    outer_folds = [0] if args.smoke else list(range(N_FOLDS))

    x_speech = load_cached_speech(args.speech_cache, ids)
    e018 = load_oof(args.oof_dir / "E018_oof.csv", ids)
    e010 = load_oof(args.oof_dir / "E010_oof.csv", ids)

    pred24 = np.empty(len(y))
    pred25 = np.empty(len(y))
    pred24[:] = np.nan
    pred25[:] = np.nan
    for fold in outer_folds:
        outer_val = np.flatnonzero(folds == fold)
        outer_train = np.flatnonzero(folds != fold)
        inner_train, inner_val = inner_split(outer_train, y, fold)
        head = train_head_cached(x_speech, y, inner_train, inner_val, device, SEED + fold)
        head.eval()
        with torch.no_grad():
            pred24[outer_val] = head(torch.from_numpy(x_speech[outer_val]).to(device)).cpu().numpy()
        print(f"E024 fold {fold} done", flush=True)
        pred25[outer_val] = run_e025_fold(files, y, inner_train, inner_val, outer_val, args.data_dir, device, SEED + 100 + fold)
        print(f"E025 fold {fold} done", flush=True)

    if args.smoke:
        print("smoke: fold 0 only, no logging", flush=True)
        for name, pred in (("E024", pred24), ("E025", pred25)):
            mask = folds == 0
            print(name, f"fold0_rmse={float(np.sqrt(np.mean((pred[mask] - y[mask]) ** 2))):.4f}", flush=True)
        return

    preds = {"E018": e018, "E010": e010, "E024": pred24, "E025": pred25}
    log(args.results, "E024", "wav2vec2_base_mean_v1", "torch_linear_head",
        {"head": "LayerNorm-Dropout(0.1)-Linear", "lr": HEAD_LR, "epochs": MAX_EPOCHS, "patience": PATIENCE},
        y, pred24, folds, "frozen wav2vec2-base mean-pool, torch head with inner early stopping; speech-only control")
    log(args.results, "E025", "wav2vec2_base_lastlayer_ft_v1", "wav2vec2_lastlayer+head",
        {"unfrozen": list(UNFROZEN_PREFIXES), "enc_lr": ENC_LR, "head_lr": HEAD_LR, "crop_s": CROP_SEC,
         "batch": BATCH, "epochs": MAX_EPOCHS, "patience": PATIENCE},
        y, pred25, folds, "unfreeze encoder.layers.11 + layer_norm only; random 20s crops train, full-clip chunk mean at predict; inner early stopping")
    write_oof(args.oof_dir / "E024_oof.csv", ids, folds, y, pred24, cohort)
    write_oof(args.oof_dir / "E025_oof.csv", ids, folds, y, pred25, cohort)

    lines = [
        "# Fine-tuning report",
        "",
        "Generated by `scripts/run_finetune.py`.",
        f"E018 setup (reference, not refit): `{MODEL_NAME}` last_hidden_state mean-pool (768-d) + acoustic_v1 (166-d) + asr_ling_v1 (19-d), HistGradientBoosting depth=3 lr=0.05 iter=200 min_leaf=20 l2=1.",
        "E024: same frozen mean-pool embeddings, torch LayerNorm-Dropout-Linear head, inner 80/20 early stopping per outer fold.",
        f"E025: unfreeze `{UNFROZEN_PREFIXES[0]}` + layer_norm only (~7M params), encoder LR {ENC_LR}, head LR {HEAD_LR}, random {CROP_SEC}s crops in training, full-clip chunk mean at prediction, inner early stopping.",
        "Official metric: unknown. Comparison number is pooled OOF RMSE.",
        f"Validation: {CV_STRATEGY}. Outer validation folds were never used for fitting or early stopping.",
        "",
        "| experiment | model | RMSE | MAE | Pearson | Spearman | R² | fold RMSE std | zero RMSE | rest RMSE |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for eid in ("E018", "E010", "E024", "E025"):
        pred = preds[eid]
        m = regression_metrics(y, pred)
        per_fold = fold_rmse(y, pred, folds)
        zero = float(np.sqrt(np.mean((pred[cohort == 1] - y[cohort == 1]) ** 2)))
        rest = float(np.sqrt(np.mean((pred[cohort == 0] - y[cohort == 0]) ** 2)))
        lines.append(
            f"| {eid} | {LABELS[eid]} | {m['rmse']:.4f} | {m['mae']:.4f} | {m['pearson']:.4f} | {m['spearman']:.4f} | {m['r2']:.4f} | "
            f"{float(np.std(per_fold, ddof=1)):.4f} | {zero:.4f} | {rest:.4f} |"
        )
    base = regression_metrics(y, e018)["rmse"]
    best_new = min(("E024", "E025"), key=lambda k: regression_metrics(y, preds[k])["rmse"])
    new = regression_metrics(y, preds[best_new])["rmse"]
    rest = cohort == 0
    rest_new = float(np.sqrt(np.mean((preds[best_new][rest] - y[rest]) ** 2)))
    rest_base = float(np.sqrt(np.mean((e018[rest] - y[rest]) ** 2)))
    lines.extend(
        [
            "",
            f"Best fine-tune run: `{best_new}`. Pooled RMSE delta vs E018: {base - new:+.4f} (positive means fine-tune is better).",
            f"Non-zero-row RMSE: {best_new} {rest_new:.4f}, E018 {rest_base:.4f}.",
            "",
            "## Decision",
            "",
        ]
    )
    if base - new > 0.01 and rest_base - rest_new > 0.01:
        lines.append("Fine-tuning beats frozen E018 on pooled RMSE and on the 732 non-zero rows. Keep it as a new speech block and re-run fusion before replacing E018.")
    else:
        lines.append("Fine-tuning does not beat frozen E018 by a meaningful margin. Keep E018. Do not fine-tune further on 769 labels.")
    lines.extend(["", "No test audio. No submission. No full-encoder fine-tune was attempted.", ""])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines))
    print(f"wrote {args.report}", flush=True)
    print("best_new", best_new, "delta_vs_E018", f"{base - new:+.4f}", flush=True)


if __name__ == "__main__":
    main()
