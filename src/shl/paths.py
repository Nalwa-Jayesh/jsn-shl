"""Environment-independent paths. Local repo layout and Kaggle notebooks.

Local: competition files live in `<repo>/data`, generated files in
`<repo>/outputs` and `<repo>/experiments`.

Kaggle: competition files are read-only under `/kaggle/input/...`, generated
files go under `/kaggle/working`. Detection is `Path("/kaggle/input").is_dir()`
or the `KAGGLE_KERNEL_RUN` env var. Raw competition files are never written.
"""

from __future__ import annotations

import os
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")


def is_kaggle() -> bool:
    return KAGGLE_INPUT.is_dir() or bool(os.environ.get("KAGGLE_KERNEL_RUN"))


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _env(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def find_dataset_root() -> Path:
    """Directory holding train.csv, test.csv, train/, test/."""
    override = _env("SHL_DATA_DIR")
    if override is not None:
        return override
    if is_kaggle():
        hint = _env("SHL_KAGGLE_DATASET")
        if hint is not None:
            return hint
        candidates = sorted(
            p.parent for p in KAGGLE_INPUT.rglob("train.csv") if p.is_file()
        )
        if candidates:
            return candidates[0]
        return KAGGLE_INPUT
    return repo_root() / "data"


def work_root() -> Path:
    """Writable root for generated files."""
    override = _env("SHL_WORK_DIR")
    if override is not None:
        return override
    if is_kaggle():
        return KAGGLE_WORKING
    return repo_root()


def data_dir() -> Path:
    return find_dataset_root()


def outputs_dir() -> Path:
    return work_root() / "outputs"


def experiments_dir() -> Path:
    return work_root() / "experiments"


def acoustic_cache() -> Path:
    return outputs_dir() / "features" / "acoustic" / "train_acoustic_v1.npz"


def asr_dir() -> Path:
    return outputs_dir() / "features" / "asr"


def speech_cache() -> Path:
    return outputs_dir() / "features" / "speech_embeddings" / "train_wav2vec2_base_mean_v1.npz"


def text_cache() -> Path:
    return outputs_dir() / "features" / "text_embeddings" / "train_minilm_l6_v1.npz"


def oof_dir() -> Path:
    return outputs_dir() / "predictions" / "oof"


def results_csv() -> Path:
    return experiments_dir() / "results.csv"


def reports_dir() -> Path:
    return outputs_dir() / "reports"


def submission_path() -> Path:
    return outputs_dir() / "submission.csv"


def describe() -> dict[str, str]:
    return {
        "kaggle": str(is_kaggle()),
        "data_dir": str(data_dir()),
        "work_root": str(work_root()),
        "outputs": str(outputs_dir()),
        "experiments": str(experiments_dir()),
    }
