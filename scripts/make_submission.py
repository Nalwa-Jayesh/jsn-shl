#!/usr/bin/env python3
"""Final prediction. Retrains the E018 feature set on all train rows, predicts test.

E018 feature set: acoustic_v1 (166-d) + asr_ling_v1 (19-d) +
wav2vec2-base mean-pool (768-d), HistGradientBoostingRegressor
depth=3 lr=0.05 iter=200 min_leaf=20 l2=1.0 seed=42.

Reads competition CSVs from --data-dir (never writes there).
Writes submission plus reused feature caches under the working outputs dir.
Run only after the methodology is frozen. Do not tune against the output.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shl import paths as _paths
from shl.acoustic import extract_acoustic, feature_names as acoustic_names
from shl.asr import transcript_path, transcribe_file, load_transcript, save_transcript
from shl.cv import SEED
from shl.text import FEATURE_NAMES as LING_NAMES, linguistic_features

MODEL_NAME = "facebook/wav2vec2-base"
CLIP_MIN, CLIP_MAX = 0.0, 5.0


def read_index(data_dir: Path, name: str) -> list[dict[str, str]]:
    with (data_dir / name).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [str(row["filename"]) for row in rows] != [str(row["filename"]) for row in rows]:
        raise SystemExit(f"{name} filename list failed identity check")
    return rows


def acoustic_block(files: list[str], audio_dir: Path, cache: Path | None) -> np.ndarray:
    names = acoustic_names()
    matrix = np.empty((len(files), len(names)), dtype=np.float64)
    for index, filename in enumerate(files):
        matrix[index] = extract_acoustic(audio_dir / filename)
        if (index + 1) % 50 == 0 or index + 1 == len(files):
            print(f"acoustic {index + 1}/{len(files)}", flush=True)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, columns=np.asarray(names), features=matrix)
    return matrix


def transcripts(
    rows: list[dict[str, str]], split: str, data_dir: Path, asr_dir: Path, device_note: str
) -> list[dict[str, object]]:
    from faster_whisper import WhisperModel

    model = None
    out = []
    for index, row in enumerate(rows, start=1):
        path = transcript_path(asr_dir, split, row["filename"])
        payload = load_transcript(path)
        if payload is None:
            if model is None:
                print(f"loading faster-whisper small ({device_note})", flush=True)
                model = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=8)
            payload = transcribe_file(model, data_dir / split / row["filename"], split, row["filename"])
            save_transcript(path, payload)
        out.append(payload)
        if index % 25 == 0 or index == len(rows):
            print(f"asr {split} {index}/{len(rows)}", flush=True)
    return out


def ling_block(payloads: list[dict[str, object]]) -> np.ndarray:
    return np.vstack(
        [linguistic_features(p, float(p.get("duration") or 0.0)) for p in payloads]
    )


def speech_block(
    files: list[str], audio_dir: Path, cache: Path | None, device: str
) -> np.ndarray:
    import soundfile as sf
    import torch
    from transformers import Wav2Vec2Model, Wav2Vec2Processor

    torch_device = torch.device(device)
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    model.eval()
    model.to(torch_device)
    matrix = np.empty((len(files), 768), dtype=np.float64)
    with torch.no_grad():
        for index, filename in enumerate(files):
            audio, rate = sf.read(audio_dir / filename, dtype="float32", always_2d=False)
            assert rate == 16000 and audio.ndim == 1
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            hidden = model(inputs["input_values"].to(torch_device)).last_hidden_state
            matrix[index] = hidden.mean(dim=1).squeeze(0).cpu().numpy()
            if (index + 1) % 25 == 0 or index + 1 == len(files):
                print(f"speech {index + 1}/{len(files)}", flush=True)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, model=np.asarray(MODEL_NAME), features=matrix)
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=_paths.data_dir())
    parser.add_argument("--work-dir", type=Path, default=_paths.work_root())
    parser.add_argument("--submission", type=Path, default=_paths.submission_path())
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    args = parser.parse_args()

    import torch as _torch

    device = args.device or ("cuda" if _torch.cuda.is_available() else "cpu")
    data_dir = args.data_dir
    work = args.work_dir
    feat = work / "outputs" / "features"

    train_rows = read_index(data_dir, "train.csv")
    test_rows = read_index(data_dir, "test.csv")
    train_ids = [r["filename"] for r in train_rows]
    test_ids = [r["filename"] for r in test_rows]
    if len(set(test_ids)) != len(test_ids):
        raise SystemExit("duplicate filenames in test.csv")
    y = np.asarray([float(r["label"]) for r in train_rows], dtype=np.float64)

    asr_dir = feat / "asr"
    print("acoustic train", flush=True)
    x_train_ac = acoustic_block(train_ids, data_dir / "train", feat / "acoustic" / "final_acoustic_train.npz")
    print("acoustic test", flush=True)
    x_test_ac = acoustic_block(test_ids, data_dir / "test", feat / "acoustic" / "final_acoustic_test.npz")

    print("asr train", flush=True)
    train_payloads = transcripts(train_rows, "train", data_dir, asr_dir, device)
    print("asr test", flush=True)
    test_payloads = transcripts(test_rows, "test", data_dir, asr_dir, device)
    x_train_ling = ling_block(train_payloads)
    x_test_ling = ling_block(test_payloads)

    print(f"speech train ({device})", flush=True)
    x_train_sp = speech_block(train_ids, data_dir / "train", feat / "speech_embeddings" / "final_speech_train.npz", device)
    print(f"speech test ({device})", flush=True)
    x_test_sp = speech_block(test_ids, data_dir / "test", feat / "speech_embeddings" / "final_speech_test.npz", device)

    x_train = np.hstack([x_train_ac, x_train_ling, x_train_sp])
    x_test = np.hstack([x_test_ac, x_test_ling, x_test_sp])
    print("train", x_train.shape, "test", x_test.shape, flush=True)

    model = HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.05,
        max_iter=200,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=SEED,
    )
    model.fit(x_train, y)
    pred = np.clip(model.predict(x_test), CLIP_MIN, CLIP_MAX)
    if not np.isfinite(pred).all():
        raise SystemExit("non-finite predictions")

    args.submission.parent.mkdir(parents=True, exist_ok=True)
    with args.submission.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "label"])
        writer.writeheader()
        for row, score in zip(test_rows, pred, strict=True):
            writer.writerow({"filename": row["filename"], "label": f"{float(score):.4f}"})
    print(f"wrote {args.submission} rows={len(test_rows)}", flush=True)
    print(f"range {float(pred.min()):.3f}..{float(pred.max()):.3f} mean {float(pred.mean()):.3f}", flush=True)


if __name__ == "__main__":
    main()
