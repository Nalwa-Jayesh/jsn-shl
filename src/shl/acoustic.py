"""Deterministic acoustic features. No filename, encoder, or label inputs."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

FEATURE_VERSION = "acoustic_v1"
N_MFCC = 13
HOP = 512
FRAME = 2048
SILENCE_ABS = 200 / 32768
PAUSE_HOP = 1600  # 100 ms at 16 kHz
PAUSE_MIN_FRAMES = 3  # 300 ms
SUMMARY_STATS = ("mean", "std", "min", "max", "p10", "p90")


def feature_names() -> list[str]:
    names = [
        "duration_s",
        "rms_global",
        "peak",
        "silence_ratio",
        "leading_silence_s",
        "trailing_silence_s",
        "speech_ratio",
        "pause_count",
        "pause_duration_s",
        "mean_pause_s",
        "voiced_ratio",
        "f0_mean",
        "f0_std",
        "f0_min",
        "f0_max",
        "f0_range",
    ]
    frame_groups = [
        "rms",
        "zcr",
        "centroid",
        "bandwidth",
        "rolloff",
        *(f"contrast_{index}" for index in range(7)),
        *(f"mfcc_{index}" for index in range(1, N_MFCC + 1)),
    ]
    for group in frame_groups:
        for stat in SUMMARY_STATS:
            names.append(f"{group}_{stat}")
    return names


def _frame_rms(y: np.ndarray) -> np.ndarray:
    if y.size < FRAME:
        return np.asarray([[float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))]])
    frames = librosa.util.frame(y, frame_length=FRAME, hop_length=HOP)
    return np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=0, keepdims=True))


def _summarize(values: np.ndarray) -> list[float]:
    flat = np.asarray(values, dtype=np.float64).ravel()
    if flat.size == 0 or not np.isfinite(flat).any():
        return [0.0] * len(SUMMARY_STATS)
    flat = flat[np.isfinite(flat)]
    return [
        float(flat.mean()),
        float(flat.std()),
        float(flat.min()),
        float(flat.max()),
        float(np.percentile(flat, 10)),
        float(np.percentile(flat, 90)),
    ]


def _silence_edges(abs_y: np.ndarray, sample_rate: int) -> tuple[float, float, float]:
    silent = abs_y <= SILENCE_ABS
    if silent.size == 0:
        return 0.0, 0.0, 1.0
    lead = int(np.argmax(~silent)) if not silent.all() else silent.size
    trail = int(np.argmax(~silent[::-1])) if not silent.all() else silent.size
    return lead / sample_rate, trail / sample_rate, float(silent.mean())


def _pauses(abs_y: np.ndarray, sample_rate: int) -> tuple[int, float, float]:
    n_frames = abs_y.size // PAUSE_HOP
    if n_frames == 0:
        return 0, 0.0, 0.0
    frames = abs_y[: n_frames * PAUSE_HOP].reshape(n_frames, PAUSE_HOP)
    quiet = frames.max(axis=1) <= SILENCE_ABS
    count = 0
    samples = 0
    run = 0
    for is_quiet in quiet:
        if is_quiet:
            run += 1
            continue
        if run >= PAUSE_MIN_FRAMES:
            count += 1
            samples += run * PAUSE_HOP
        run = 0
    if run >= PAUSE_MIN_FRAMES:
        count += 1
        samples += run * PAUSE_HOP
    duration = samples / sample_rate
    mean = duration / count if count else 0.0
    return count, duration, mean


def _pitch(y: np.ndarray, sample_rate: int) -> tuple[float, float, float, float, float, float]:
    # 8 kHz is enough for an F0 summary and about 10x cheaper than full-rate yin.
    step = max(1, sample_rate // 8000)
    y_ds = np.ascontiguousarray(y[::step])
    sr_ds = sample_rate // step
    if y_ds.size < 1024:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    f0 = librosa.yin(
        y_ds,
        fmin=50,
        fmax=400,
        sr=sr_ds,
        frame_length=1024,
        hop_length=1024,
    )
    finite = f0[np.isfinite(f0)]
    if finite.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # yin fills unvoiced frames with the fmin bound. Treat that bound as unvoiced.
    voiced = finite[finite > 55]
    voiced_ratio = float(voiced.size / finite.size)
    if voiced.size == 0:
        return voiced_ratio, 0.0, 0.0, 0.0, 0.0, 0.0
    return (
        voiced_ratio,
        float(voiced.mean()),
        float(voiced.std()),
        float(voiced.min()),
        float(voiced.max()),
        float(voiced.max() - voiced.min()),
    )


def extract_acoustic(path: Path) -> np.ndarray:
    y, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim != 1:
        raise ValueError(f"{path} is not mono")
    if sample_rate != 16000:
        raise ValueError(f"{path} sample rate {sample_rate}, expected 16000")
    y = np.ascontiguousarray(y, dtype=np.float32)
    abs_y = np.abs(y)
    duration = float(y.size / sample_rate)
    lead, trail, silence_ratio = _silence_edges(abs_y, sample_rate)
    pause_count, pause_duration, mean_pause = _pauses(abs_y, sample_rate)
    voiced_ratio, f0_mean, f0_std, f0_min, f0_max, f0_range = _pitch(y, sample_rate)

    rms = _frame_rms(y)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=FRAME, hop_length=HOP)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sample_rate, n_fft=FRAME, hop_length=HOP)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sample_rate, n_fft=FRAME, hop_length=HOP)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sample_rate, n_fft=FRAME, hop_length=HOP)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sample_rate, n_fft=FRAME, hop_length=HOP)
    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=N_MFCC, n_fft=FRAME, hop_length=HOP)

    values = [
        duration,
        float(np.sqrt(np.mean(np.square(y, dtype=np.float64)))),
        float(abs_y.max() if abs_y.size else 0.0),
        silence_ratio,
        lead,
        trail,
        1.0 - silence_ratio,
        float(pause_count),
        pause_duration,
        mean_pause,
        voiced_ratio,
        f0_mean,
        f0_std,
        f0_min,
        f0_max,
        f0_range,
    ]
    for series in (rms, zcr, centroid, bandwidth, rolloff):
        values.extend(_summarize(series))
    for band in contrast:
        values.extend(_summarize(band))
    for coeff in mfcc:
        values.extend(_summarize(coeff))
    out = np.asarray(values, dtype=np.float64)
    if out.shape != (len(feature_names()),):
        raise RuntimeError(f"feature length {out.shape[0]} != {len(feature_names())}")
    if not np.isfinite(out).all():
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out
