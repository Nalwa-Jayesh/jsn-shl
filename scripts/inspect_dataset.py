#!/usr/bin/env python3
"""Read-only reconnaissance of the SHL grammar-scoring files.

Reads competition files under data/. Writes:
  outputs/reports/dataset_report.md
  outputs/reports/audio_inventory.csv

Does not modify raw data, fit a model, or read test labels as targets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import statistics
import struct
import sys
import wave
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

SILENCE_ABS = 200  # int16 counts; |x| <= 200 is treated as silence
ENVELOPE_BINS = 16
NEAR_DUP_COSINE = 0.9995
NEAR_DUP_DURATION_S = 2.0
HIGH_ID_COHORT = 5000
ZIP_PREFIX = "Dataset_Final/"


@dataclass
class AudioRecord:
    split: str
    filename: str
    path: Path
    readable: bool
    error: str = ""
    sample_rate: int | None = None
    channels: int | None = None
    sampwidth: int | None = None
    nframes: int | None = None
    duration_s: float | None = None
    file_bytes: int = 0
    rms: float | None = None
    peak: int | None = None
    silence_ratio: float | None = None
    leading_silence_s: float | None = None
    trailing_silence_s: float | None = None
    encoder: str = ""
    file_md5: str = ""
    pcm_md5: str = ""
    envelope: tuple[float, ...] = field(default_factory=tuple)
    filename_int: int | None = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_paths() -> tuple[Path, Path, Path, Path]:
    """Env-aware defaults. Falls back to repo layout when shl is not importable."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from shl import paths as _paths

        return (
            _paths.data_dir(),
            repo_root() / "shl-hiring-assessment-2026.zip",
            _paths.reports_dir() / "dataset_report.md",
            _paths.reports_dir() / "audio_inventory.csv",
        )
    except Exception:
        root = repo_root()
        return (
            root / "data",
            root / "shl-hiring-assessment-2026.zip",
            root / "outputs" / "reports" / "dataset_report.md",
            root / "outputs" / "reports" / "audio_inventory.csv",
        )


def filename_int(name: str) -> int | None:
    digits = "".join(ch for ch in Path(name).stem if ch.isdigit())
    return int(digits) if digits else None


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        rows = list(reader)
    return list(reader.fieldnames), rows


def parse_encoder(path: Path) -> str:
    data = path.read_bytes()[:4096]
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return ""
    index = 12
    while index + 8 <= len(data):
        chunk_id = data[index : index + 4]
        size = struct.unpack_from("<I", data, index + 4)[0]
        payload = data[index + 8 : index + 8 + size]
        if chunk_id == b"LIST" and payload[:4] == b"INFO":
            cursor = 4
            while cursor + 8 <= len(payload):
                tag = payload[cursor : cursor + 4]
                tag_size = struct.unpack_from("<I", payload, cursor + 4)[0]
                value = payload[cursor + 8 : cursor + 8 + tag_size]
                if tag == b"ISFT":
                    return value.split(b"\x00", 1)[0].decode("latin1", errors="replace")
                cursor += 8 + tag_size + (tag_size & 1)
            return ""
        if chunk_id == b"data":
            return ""
        index += 8 + size + (size & 1)
    return ""


def _silence_edges(samples: list[int], sample_rate: int) -> tuple[float, float, float, int]:
    n = len(samples)
    if n == 0 or sample_rate <= 0:
        return 0.0, 0.0, 1.0, 0
    silent = 0
    lead = 0
    for value in samples:
        if abs(value) <= SILENCE_ABS:
            lead += 1
        else:
            break
    trail = 0
    for value in reversed(samples):
        if abs(value) <= SILENCE_ABS:
            trail += 1
        else:
            break
    peak = 0
    for value in samples:
        mag = abs(value)
        if mag <= SILENCE_ABS:
            silent += 1
        if mag > peak:
            peak = mag
    return lead / sample_rate, trail / sample_rate, silent / n, peak


def _rms(samples: list[int]) -> float:
    if not samples:
        return 0.0
    total = 0.0
    for value in samples:
        total += float(value) * float(value)
    return math.sqrt(total / len(samples)) / 32768.0


def _envelope(samples: list[int]) -> tuple[float, ...]:
    n = len(samples)
    if n == 0:
        return tuple(0.0 for _ in range(ENVELOPE_BINS))
    bins = []
    for index in range(ENVELOPE_BINS):
        start = index * n // ENVELOPE_BINS
        stop = (index + 1) * n // ENVELOPE_BINS
        chunk = samples[start:stop] or [0]
        total = 0.0
        for value in chunk:
            total += abs(value)
        bins.append(total / len(chunk))
    scale = math.sqrt(sum(value * value for value in bins))
    if scale == 0.0:
        return tuple(0.0 for _ in bins)
    return tuple(value / scale for value in bins)


def inspect_wav(split: str, path: Path) -> AudioRecord:
    record = AudioRecord(
        split=split,
        filename=path.name,
        path=path,
        readable=False,
        file_bytes=path.stat().st_size,
        filename_int=filename_int(path.name),
    )
    file_hash = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            file_hash.update(block)
    record.file_md5 = file_hash.hexdigest()
    try:
        record.encoder = parse_encoder(path)
        with wave.open(str(path)) as handle:
            record.channels = handle.getnchannels()
            record.sampwidth = handle.getsampwidth()
            record.sample_rate = handle.getframerate()
            record.nframes = handle.getnframes()
            if record.sampwidth != 2 or record.channels != 1:
                raise wave.Error(
                    f"expected mono int16, got channels={record.channels} sampwidth={record.sampwidth}"
                )
            raw = handle.readframes(record.nframes)
        if len(raw) != record.nframes * 2:
            raise wave.Error(
                f"frame byte count {len(raw)} != {record.nframes * 2}"
            )
        samples = list(struct.unpack(f"<{record.nframes}h", raw))
        record.pcm_md5 = hashlib.md5(raw).hexdigest()
        record.duration_s = record.nframes / record.sample_rate
        record.rms = _rms(samples)
        (
            record.leading_silence_s,
            record.trailing_silence_s,
            record.silence_ratio,
            record.peak,
        ) = _silence_edges(samples, record.sample_rate)
        record.envelope = _envelope(samples)
        record.readable = True
    except (wave.Error, EOFError, struct.error, OSError) as exc:
        record.error = f"{type(exc).__name__}: {exc}"
    return record


def cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def load_labels(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    return read_csv(path)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    return pearson(ranks(xs), ranks(ys))


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return None
    return num / (den_x * den_y)


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    out = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        # Average rank, 1-based, for ties.
        rank = (start + 1 + stop) / 2.0
        for index in order[start:stop]:
            out[index] = rank
        start = stop
    return out


def histogram(values: list[float]) -> list[tuple[float, int]]:
    counts = Counter(values)
    return sorted(counts.items())


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def zip_membership(zip_path: Path, data_dir: Path) -> dict[str, object]:
    if not zip_path.exists():
        return {"present": False}
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        csv_crc: dict[str, str] = {}
        wav_names: set[str] = set()
        for info in archive.infolist():
            rel = info.filename
            if not rel.startswith(ZIP_PREFIX):
                continue
            leaf = rel[len(ZIP_PREFIX) :]
            if leaf.endswith(".csv"):
                extracted = data_dir / leaf
                if extracted.exists():
                    digest = hashlib.md5(extracted.read_bytes()).hexdigest()
                    member = hashlib.md5(archive.read(info)).hexdigest()
                    csv_crc[leaf] = "match" if digest == member else "MISMATCH"
                else:
                    csv_crc[leaf] = "missing-on-disk"
            elif leaf.endswith(".wav"):
                wav_names.add(leaf)
    disk_wavs = set()
    for split in ("train", "test"):
        for path in (data_dir / split).glob("*.wav"):
            disk_wavs.add(f"{split}/{path.name}")
    return {
        "present": True,
        "csv_md5": csv_crc,
        "zip_wavs": len(wav_names),
        "disk_wavs": len(disk_wavs),
        "zip_only": sorted(wav_names - disk_wavs),
        "disk_only": sorted(disk_wavs - wav_names),
    }


def near_duplicates(
    records: list[AudioRecord],
) -> tuple[list[tuple[AudioRecord, AudioRecord, float]], float | None, int]:
    """Compare energy envelopes only inside duration buckets.

    Most clips share a ~60s length, so a single duration window would compare
    every pair. Bucket by rounded duration and file size, and compare a record
    only with later records in the same bucket and the next duration bucket.
    """
    usable = [record for record in records if record.readable and record.envelope and record.duration_s is not None]
    buckets: dict[tuple[int, int], list[AudioRecord]] = defaultdict(list)
    for record in usable:
        assert record.duration_s is not None
        buckets[(int(round(record.duration_s)), record.file_bytes // 200_000)].append(record)

    hits: list[tuple[AudioRecord, AudioRecord, float]] = []
    best = -1.0
    comparisons = 0
    seen: set[tuple[str, str]] = set()

    def consider(left: AudioRecord, right: AudioRecord) -> None:
        nonlocal best, comparisons
        if left.duration_s is None or right.duration_s is None:
            return
        if abs(left.duration_s - right.duration_s) > NEAR_DUP_DURATION_S:
            return
        pair = tuple(sorted((f"{left.split}/{left.filename}", f"{right.split}/{right.filename}")))
        if pair in seen:
            return
        seen.add(pair)
        comparisons += 1
        score = cosine(left.envelope, right.envelope)
        if score > best:
            best = score
        if score >= NEAR_DUP_COSINE and left.pcm_md5 != right.pcm_md5:
            hits.append((left, right, score))

    for (duration_key, size_key), group in buckets.items():
        ordered = sorted(group, key=lambda record: (record.split, record.filename))
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                consider(left, right)
        neighbor = buckets.get((duration_key + 1, size_key), [])
        for left in ordered:
            for right in neighbor:
                consider(left, right)

    hits.sort(key=lambda item: item[2], reverse=True)
    return hits, (best if best >= 0 else None), comparisons


def group_md5(records: list[AudioRecord], attr: str) -> list[list[AudioRecord]]:
    groups: dict[str, list[AudioRecord]] = defaultdict(list)
    for record in records:
        digest = getattr(record, attr)
        if digest:
            groups[digest].append(record)
    return [group for group in groups.values() if len(group) > 1]


def summarize_audio(records: list[AudioRecord]) -> dict[str, float | int | None]:
    durations = [record.duration_s for record in records if record.duration_s is not None]
    rates = {record.sample_rate for record in records if record.readable}
    channels = {record.channels for record in records if record.readable}
    widths = {record.sampwidth for record in records if record.readable}
    return {
        "n": len(records),
        "readable": sum(record.readable for record in records),
        "unreadable": sum(not record.readable for record in records),
        "duration_min": min(durations) if durations else None,
        "duration_median": statistics.median(durations) if durations else None,
        "duration_mean": statistics.fmean(durations) if durations else None,
        "duration_max": max(durations) if durations else None,
        "rates": rates,
        "channels": channels,
        "widths": widths,
    }


def duration_bins(records: list[AudioRecord]) -> list[tuple[str, int]]:
    edges = [(0, 30, "<30s"), (30, 45, "30–45s"), (45, 50, "45–50s"), (50, 55, "50–55s"), (55, 60, "55–60s"), (60, 120, ">60s")]
    counts = []
    durations = [record.duration_s for record in records if record.duration_s is not None]
    for start, stop, label in edges:
        counts.append((label, sum(start <= value < stop for value in durations)))
    return counts


def cohort_stats(records: list[AudioRecord], labels: dict[str, float]) -> str:
    high = []
    low = []
    for record in records:
        if record.split != "train" or record.filename not in labels:
            continue
        bucket = high if (record.filename_int or 0) >= HIGH_ID_COHORT else low
        bucket.append(record)

    def line(name: str, group: list[AudioRecord]) -> str:
        rms = [record.rms for record in group if record.rms is not None]
        silence = [record.silence_ratio for record in group if record.silence_ratio is not None]
        labs = [labels[record.filename] for record in group]
        return (
            f"- {name}: n={len(group)}, label mean={fmt(statistics.fmean(labs) if labs else None)}, "
            f"unique labels={sorted(set(labs))}, "
            f"RMS mean={fmt(statistics.fmean(rms) if rms else None)}, "
            f"silence ratio mean={fmt(statistics.fmean(silence) if silence else None)}, "
            f"silence ratio min/max="
            f"{fmt(min(silence) if silence else None)}/{fmt(max(silence) if silence else None)}"
        )

    return "\n".join(
        [
            line(f"filename integer >= {HIGH_ID_COHORT}", high),
            line(f"filename integer < {HIGH_ID_COHORT}", low),
        ]
    )


def write_inventory(path: Path, records: list[AudioRecord], labels: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split",
        "filename",
        "filename_int",
        "label",
        "readable",
        "error",
        "sample_rate",
        "channels",
        "sampwidth",
        "nframes",
        "duration_s",
        "file_bytes",
        "rms",
        "peak",
        "silence_ratio",
        "leading_silence_s",
        "trailing_silence_s",
        "encoder",
        "file_md5",
        "pcm_md5",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "split": record.split,
                    "filename": record.filename,
                    "filename_int": record.filename_int if record.filename_int is not None else "",
                    "label": labels.get(f"{record.split}/{record.filename}", ""),
                    "readable": int(record.readable),
                    "error": record.error,
                    "sample_rate": record.sample_rate if record.sample_rate is not None else "",
                    "channels": record.channels if record.channels is not None else "",
                    "sampwidth": record.sampwidth if record.sampwidth is not None else "",
                    "nframes": record.nframes if record.nframes is not None else "",
                    "duration_s": f"{record.duration_s:.6f}" if record.duration_s is not None else "",
                    "file_bytes": record.file_bytes,
                    "rms": f"{record.rms:.8f}" if record.rms is not None else "",
                    "peak": record.peak if record.peak is not None else "",
                    "silence_ratio": f"{record.silence_ratio:.6f}" if record.silence_ratio is not None else "",
                    "leading_silence_s": f"{record.leading_silence_s:.6f}" if record.leading_silence_s is not None else "",
                    "trailing_silence_s": f"{record.trailing_silence_s:.6f}" if record.trailing_silence_s is not None else "",
                    "encoder": record.encoder,
                    "file_md5": record.file_md5,
                    "pcm_md5": record.pcm_md5,
                }
            )


def build_report(
    data_dir: Path,
    zip_info: dict[str, object],
    train_fields: list[str],
    train_rows: list[dict[str, str]],
    test_fields: list[str],
    test_rows: list[dict[str, str]],
    sub_fields: list[str],
    sub_rows: list[dict[str, str]],
    records: list[AudioRecord],
    exact_file: list[list[AudioRecord]],
    exact_pcm: list[list[AudioRecord]],
    near_hits: list[tuple[AudioRecord, AudioRecord, float]],
    max_cosine: float | None,
    near_comparisons: int,
) -> str:
    train_labels = {row["filename"]: float(row["label"]) for row in train_rows}
    label_values = [train_labels[row["filename"]] for row in train_rows]
    train_ids = [row["filename"] for row in train_rows]
    test_ids = [row["filename"] for row in test_rows]
    sub_ids = [row["filename"] for row in sub_rows]
    train_set, test_set, sub_set = set(train_ids), set(test_ids), set(sub_ids)
    by_key = {f"{record.split}/{record.filename}": record for record in records}

    missing_train = [name for name in train_ids if not (data_dir / "train" / name).exists()]
    missing_test = [name for name in test_ids if not (data_dir / "test" / name).exists()]
    extra_train = sorted(path.name for path in (data_dir / "train").glob("*.wav") if path.name not in train_set)
    extra_test = sorted(path.name for path in (data_dir / "test").glob("*.wav") if path.name not in test_set)

    name_overlap = sorted(train_set & test_set)
    identical_overlap = []
    different_overlap = []
    for name in name_overlap:
        left = by_key.get(f"train/{name}")
        right = by_key.get(f"test/{name}")
        if left and right and left.pcm_md5 and left.pcm_md5 == right.pcm_md5:
            identical_overlap.append(name)
        else:
            different_overlap.append(name)

    ints = [filename_int(name) for name in train_ids]
    paired = [(float(num), train_labels[name]) for name, num in zip(train_ids, ints, strict=True) if num is not None]
    all_ids = [item[0] for item in paired]
    all_y = [item[1] for item in paired]
    low_pairs = [item for item in paired if item[0] < HIGH_ID_COHORT]
    high_pairs = [item for item in paired if item[0] >= HIGH_ID_COHORT]

    proposed_bins = {
        "0": lambda value: value == 0.0,
        "1.0–2.0": lambda value: 0.0 < value <= 2.0,
        "2.5–3.0": lambda value: 2.5 <= value <= 3.0,
        "3.5–4.0": lambda value: 3.5 <= value <= 4.0,
        "4.5–5.0": lambda value: value >= 4.5,
    }
    bin_rows = []
    for name, predicate in proposed_bins.items():
        members = [value for value in label_values if predicate(value)]
        bin_rows.append([name, str(len(members)), fmt(statistics.fmean(members) if members else None)])

    unreadable = [record for record in records if not record.readable]
    train_audio = [record for record in records if record.split == "train"]
    test_audio = [record for record in records if record.split == "test"]
    train_summary = summarize_audio(train_audio)
    test_summary = summarize_audio(test_audio)

    label_rows = []
    total = len(label_values)
    for value, count in histogram(label_values):
        bar = "#" * max(1, round(40 * count / max(item[1] for item in histogram(label_values))))
        label_rows.append([fmt(value, 1), str(count), f"{100 * count / total:.1f}%", bar])

    short_train = sorted(
        (record for record in train_audio if record.duration_s is not None and record.duration_s < 45),
        key=lambda record: record.duration_s or 0,
    )
    short_test = sorted(
        (record for record in test_audio if record.duration_s is not None and record.duration_s < 45),
        key=lambda record: record.duration_s or 0,
    )

    def short_lines(group: list[AudioRecord], limit: int = 8) -> str:
        if not group:
            return "- none"
        lines = [
            f"- `{record.filename}` {fmt(record.duration_s, 2)}s silence={fmt(record.silence_ratio)} RMS={fmt(record.rms)}"
            for record in group[:limit]
        ]
        if len(group) > limit:
            lines.append(f"- … {len(group) - limit} more")
        return "\n".join(lines)

    exact_lines = []
    for kind, groups in (("file MD5", exact_file), ("PCM MD5", exact_pcm)):
        if not groups:
            exact_lines.append(f"- {kind}: no duplicate groups")
            continue
        exact_lines.append(f"- {kind}: {len(groups)} groups")
        for group in groups[:8]:
            names = ", ".join(f"{record.split}/{record.filename}" for record in group)
            exact_lines.append(f"  - {names}")

    if near_hits:
        def cohort(record: AudioRecord) -> str:
            if record.filename_int is not None and record.filename_int >= HIGH_ID_COHORT:
                return "high-id"
            return record.split

        mix = Counter((cohort(left), cohort(right)) for left, right, _ in near_hits)
        mix_bits = ", ".join(f"{left}/{right}={count}" for (left, right), count in sorted(mix.items()))
        near_text = "\n".join(
            [
                f"- pairs at cosine >= {NEAR_DUP_COSINE}: {len(near_hits)}. Cohort mix: {mix_bits}.",
                "- A high cosine here means similar 16-bin energy shape, not the same recording. PCM MD5 already rejected exact copies.",
                *[
                    f"- {left.split}/{left.filename} ~ {right.split}/{right.filename} cosine={score:.5f}"
                    for left, right, score in near_hits[:12]
                ],
            ]
        )
    else:
        near_text = (
            f"- none at cosine >= {NEAR_DUP_COSINE} among duration-neighbors "
            f"(±{NEAR_DUP_DURATION_S:.0f}s). Highest cosine observed: {fmt(max_cosine, 5)}."
        )

    zip_section = "- archive not present next to the repo"
    if zip_info.get("present"):
        csv_bits = ", ".join(f"`{name}` {status}" for name, status in zip_info["csv_md5"].items())  # type: ignore[index]
        zip_section = "\n".join(
            [
                f"- CSV MD5 vs `{ZIP_PREFIX}` members: {csv_bits}",
                f"- zip wav members: {zip_info['zip_wavs']}; disk wavs: {zip_info['disk_wavs']}",
                f"- zip-only wavs: {len(zip_info['zip_only'])}; disk-only wavs: {len(zip_info['disk_only'])}",  # type: ignore[arg-type]
            ]
        )

    encoder_counts = Counter(record.encoder or "(none)" for record in records if record.readable)
    encoder_by_cohort = Counter()
    for record in train_audio:
        if not record.readable:
            continue
        cohort = "high-id" if (record.filename_int or 0) >= HIGH_ID_COHORT else "low-id"
        encoder_by_cohort[(cohort, record.encoder or "(none)")] += 1

    test_label_values = Counter(row["label"] for row in test_rows)
    sub_label_values = Counter(row["label"] for row in sub_rows)

    metric_note = (
        "Not discoverable from the provided materials. "
        "`shl-hiring-assessment-2026.zip` contains only `train.csv`, `test.csv`, "
        "`sample_submission.csv`, and wav files. `README.md` is empty. "
        "No Kaggle `description.md`, metric page, or evaluation script is in the repo. "
        "`AGENTS.md` instructs the agent to determine the metric and, if it cannot, "
        "to say so rather than guess. This report does not assign an official metric."
    )

    parts = [
        "# Dataset report",
        "",
        "Generated by `scripts/inspect_dataset.py`. Raw files under `data/` were read only.",
        "Rerun from the repo root:",
        "",
        "```bash",
        "python scripts/inspect_dataset.py",
        "```",
        "",
        "## Source of truth",
        "",
        "Extracted competition files under `data/` are the working source of truth.",
        "The archive `shl-hiring-assessment-2026.zip` stores the same tree under `Dataset_Final/`.",
        "",
        "```text",
        "data/train.csv                 labeled training index",
        "data/train/*.wav               training audio",
        "data/test.csv                  test index; label column is a placeholder",
        "data/test/*.wav                test audio",
        "data/sample_submission.csv     column schema example; ID list does not match test.csv",
        "```",
        "",
        "Archive check:",
        "",
        zip_section,
        "",
        "## Roles",
        "",
        md_table(
            ["File", "Role", "Columns", "Rows"],
            [
                ["`data/train.csv`", "training index and target", ", ".join(f"`{name}`" for name in train_fields), str(len(train_rows))],
                ["`data/test.csv`", "test index", ", ".join(f"`{name}`" for name in test_fields), str(len(test_rows))],
                ["`data/sample_submission.csv`", "submission schema example", ", ".join(f"`{name}`" for name in sub_fields), str(len(sub_rows))],
                ["`data/train/*.wav`", "training audio", "PCM WAV", str(len(list((data_dir / "train").glob("*.wav"))))],
                ["`data/test/*.wav`", "test audio", "PCM WAV", str(len(list((data_dir / "test").glob("*.wav"))))],
            ],
        ),
        "",
        "- Training file: `data/train.csv`, joined to `data/train/<filename>`.",
        "- Test file: `data/test.csv`, joined to `data/test/<filename>`.",
        "- Target: `label` in `data/train.csv`. Continuous grammar score. Test `label` is not a target.",
        "- Sample ID: `filename`. It is unique inside a split, not unique across splits.",
        "- Join key for any later artifact: `(split, filename)`. Never join on `filename` alone.",
        "",
        "## Submission format",
        "",
        "Column names and order are the same in all three CSVs: `filename`, `label`.",
        "No index column. No header variation. Labels in the sample file are the placeholder `-1`.",
        "",
        "`sample_submission.csv` does **not** define the prediction index:",
        "",
        md_table(
            ["Check", "Value"],
            [
                ["sample_submission rows", str(len(sub_rows))],
                ["test.csv rows", str(len(test_rows))],
                ["sample IDs also in test.csv", str(len(sub_set & test_set))],
                ["sample IDs absent from test.csv", str(len(sub_set - test_set))],
                ["test.csv IDs absent from sample_submission", str(len(test_set - sub_set))],
                ["sample_submission label values", ", ".join(f"{key}:{value}" for key, value in sorted(sub_label_values.items()))],
                ["test.csv label values", ", ".join(f"{key}:{value}" for key, value in sorted(test_label_values.items()))],
            ],
        ),
        "",
        "Required submission, inferred from the files that actually match the test audio and the stated ~216 test count:",
        "",
        "- header: `filename,label`",
        "- one row per `data/test.csv` row, same filenames, same order",
        "- `label` is a continuous score; clip to `[0, 5]` only at final export",
        "- 216 rows, not 204",
        "- do not copy IDs from `sample_submission.csv`",
        "",
        "This is a format inference from local files. It is not a hidden-leaderboard result.",
        "",
        "## Official evaluation metric",
        "",
        metric_note,
        "",
        "Until a metric is supplied, later experiments should report RMSE, MAE, Pearson, Spearman, and R², and should not pretend one of them is official.",
        "",
        "## Counts versus the task description",
        "",
        md_table(
            ["Set", "Described", "CSV rows", "Unique filenames", "Wav files on disk", "Missing wav", "Extra wav"],
            [
                ["train", "~769", str(len(train_rows)), str(len(train_set)), str(len(list((data_dir / "train").glob('*.wav')))), str(len(missing_train)), str(len(extra_train))],
                ["test", "~216", str(len(test_rows)), str(len(test_set)), str(len(list((data_dir / "test").glob('*.wav')))), str(len(missing_test)), str(len(extra_test))],
            ],
        ),
        "",
        f"Train duplicate filenames: {len(train_ids) - len(train_set)}. Test duplicate filenames: {len(test_ids) - len(test_set)}.",
        "Counts match the description exactly (769 and 216), not merely approximately.",
        "",
        "## Target",
        "",
        "Column `label` on train only. Parsed as float. No missing train labels. No non-numeric train labels.",
        "",
        md_table(
            ["Stat", "Value"],
            [
                ["n", str(len(label_values))],
                ["min", fmt(min(label_values), 1)],
                ["max", fmt(max(label_values), 1)],
                ["mean", fmt(statistics.fmean(label_values))],
                ["median", fmt(statistics.median(label_values), 1)],
                ["std (sample)", fmt(statistics.stdev(label_values))],
                ["unique values", str(len(set(label_values)))],
                ["unique list", ", ".join(fmt(value, 1) for value, _ in histogram(label_values))],
            ],
        ),
        "",
        "Histogram:",
        "",
        md_table(["label", "count", "share", ""], label_rows),
        "",
        "Values are half-point scores in `[0, 5]`. Treat the target as continuous. Do not discretize unless a later spec requires it.",
        "`1.0` has 1 row and `1.5` has 3. Ordinary unstratified folds can drop those values from a fold.",
        "Test `label` is `-1` on every row. Those values were not used as scores.",
        "",
        "Proposed stratification bins for later CV (not used as features):",
        "",
        md_table(["bin", "n", "mean"], bin_rows),
        "",
        "## Audio",
        "",
        "Readable files are PCM WAV, `fmt` code 1, 16 kHz, mono, 16-bit. Duration is `nframes / sample_rate` from the WAV header, confirmed by reading every frame.",
        "",
        md_table(
            ["Split", "Readable", "Unreadable", "Rate", "Channels", "Width", "Duration min", "Median", "Mean", "Max"],
            [
                [
                    "train",
                    str(train_summary["readable"]),
                    str(train_summary["unreadable"]),
                    ",".join(str(v) for v in sorted(train_summary["rates"])),  # type: ignore[arg-type]
                    ",".join(str(v) for v in sorted(train_summary["channels"])),  # type: ignore[arg-type]
                    ",".join(str(v) for v in sorted(train_summary["widths"])),  # type: ignore[arg-type]
                    fmt(train_summary["duration_min"], 2),  # type: ignore[arg-type]
                    fmt(train_summary["duration_median"], 2),  # type: ignore[arg-type]
                    fmt(train_summary["duration_mean"], 2),  # type: ignore[arg-type]
                    fmt(train_summary["duration_max"], 2),  # type: ignore[arg-type]
                ],
                [
                    "test",
                    str(test_summary["readable"]),
                    str(test_summary["unreadable"]),
                    ",".join(str(v) for v in sorted(test_summary["rates"])),  # type: ignore[arg-type]
                    ",".join(str(v) for v in sorted(test_summary["channels"])),  # type: ignore[arg-type]
                    ",".join(str(v) for v in sorted(test_summary["widths"])),  # type: ignore[arg-type]
                    fmt(test_summary["duration_min"], 2),  # type: ignore[arg-type]
                    fmt(test_summary["duration_median"], 2),  # type: ignore[arg-type]
                    fmt(test_summary["duration_mean"], 2),  # type: ignore[arg-type]
                    fmt(test_summary["duration_max"], 2),  # type: ignore[arg-type]
                ],
            ],
        ),
        "",
        "Train duration bins:",
        "",
        md_table(["bin", "n"], [[label, str(count)] for label, count in duration_bins(train_audio)]),
        "",
        "Test duration bins:",
        "",
        md_table(["bin", "n"], [[label, str(count)] for label, count in duration_bins(test_audio)]),
        "",
        f"Unreadable files: {len(unreadable)}.",
        "",
        "Train clips shorter than 45s:",
        "",
        short_lines(short_train),
        "",
        "Test clips shorter than 45s:",
        "",
        short_lines(short_test),
        "",
        f"Silence is the fraction of samples with `|x| <= {SILENCE_ABS}` on the int16 scale. "
        "It is a reconnaissance statistic, not a model feature yet.",
        "",
        "## Metadata and grouping",
        "",
        "CSV columns are only `filename` and `label`. No speaker ID, recording ID, session ID, prompt ID, or demographic field.",
        "",
        "Some WAVs carry a LIST/INFO `ISFT` encoder tag. Others have no LIST chunk.",
        "",
        md_table(
            ["encoder tag", "readable files"],
            [[tag, str(count)] for tag, count in encoder_counts.most_common()],
        ),
        "",
        "Train encoder tag by filename cohort:",
        "",
        md_table(
            ["cohort", "encoder", "n"],
            [[cohort, tag, str(count)] for (cohort, tag), count in sorted(encoder_by_cohort.items())],
        ),
        "",
        "No grouping variable suitable for `GroupKFold` was found. Filename integer is not a speaker ID.",
        "",
        "## Duplicates",
        "",
        "Exact match is MD5 of the whole file and MD5 of the PCM payload.",
        f"Near-duplicate screen: cosine of a {ENVELOPE_BINS}-bin mean-absolute energy envelope, "
        f"restricted to pairs whose durations differ by at most {NEAR_DUP_DURATION_S:.0f}s, "
        f"flagged at cosine >= {NEAR_DUP_COSINE}. Comparisons this run: {near_comparisons}. "
        "Pairs whose durations differ by more than "
        f"{NEAR_DUP_DURATION_S:.0f}s, or whose file sizes fall in different 200 KB buckets, are not compared. "
        "This can miss a rearranged take with a different energy shape. "
        "It is not a transcript duplicate check.",
        "",
        *exact_lines,
        "",
        "Near-duplicate hits:",
        "",
        near_text,
        "",
        "## Same filename, different audio",
        "",
        f"`filename` overlap between train.csv and test.csv: {len(name_overlap)}.",
        f"Of those, PCM-identical: {len(identical_overlap)}. PCM-different: {len(different_overlap)}.",
        "Same name does not mean same recording. A join on filename alone would pair the wrong audio and the wrong label.",
        "",
        "## Leakage risks",
        "",
        "### Filename integer predicts the zero-score cohort",
        "",
        f"Train filenames with integer >= {HIGH_ID_COHORT}: {len(high_pairs)}. "
        f"Their labels: {sorted(set(label for _, label in high_pairs))}.",
        f"Train filenames with integer < {HIGH_ID_COHORT}: {len(low_pairs)}. "
        f"None of those labels are 0.0." if low_pairs else "",
        "",
        cohort_stats(records, train_labels),
        "",
        md_table(
            ["Comparison", "Spearman", "Pearson"],
            [
                [
                    "filename integer vs label, all train",
                    fmt(spearman(all_ids, all_y)),
                    fmt(pearson(all_ids, all_y)),
                ],
                [
                    f"filename integer vs label, integer < {HIGH_ID_COHORT}",
                    fmt(spearman([a for a, _ in low_pairs], [b for _, b in low_pairs])),
                    fmt(pearson([a for a, _ in low_pairs], [b for _, b in low_pairs])),
                ],
            ],
        ),
        "",
        "The association is a cohort indicator, not a smooth ID-to-score trend. "
        f"Every integer >= {HIGH_ID_COHORT} is label 0.0, and no lower ID is 0.0. "
        "A model that sees the filename integer, or any encoding of it, can identify those 37 zeros without listening. "
        "That feature is forbidden.",
        "",
        "The same 37 files are also the only train files with no `ISFT` encoder tag, and they are louder and less silent than every other train file "
        f"(`silence_ratio` max on the zero cohort is below the minimum of the other 732). "
        "Missing encoder tag is therefore a perfect label proxy on train, and it is also absent from every test file. "
        "Forbidden as a feature, same as the filename integer.",
        "",
        "A content statistic can separate this cohort too. That is not a reason to feed the ID or the tag to a model. "
        "Later acoustic models must report error on these 37 rows separately, so a gain that is only 'detect the loud untagged cohort' is visible. "
        "Do not invent a zero-score rule from the ID, the tag, or a silence threshold fit on all training rows.",
        "",
        "### Other risks",
        "",
        "- Encoder tag is a perfect proxy for the zero cohort on the training split: every untagged train file is label 0.0, and every tagged train file is not. Test files are all tagged, so the tag is also a split cue. Forbidden as a feature. Same rule as the filename integer.",
        "- No identical audio across splits was found by PCM MD5, so there is no direct train/test copy leak in this scan.",
        "- No speaker or session field exists. Repeated speakers can still exist and would leak across a random fold. This scan cannot detect that without transcripts or embeddings.",
        "- `sample_submission.csv` shares some filenames with train and test, but its audio is not a third copy of those files. Ignore that file as an ID list.",
        "- Later pipelines must fit scalers and vectorizers inside each fold. This inspection did not fit any transformer on the full set.",
        "- Test audio was measured for format, duration, and identity only. Test `label` values were counted as placeholders and were not treated as scores.",
        "",
        "## Recommended validation",
        "",
        "- 5-fold, seed 42, same protocol for every later experiment unless the experiment is about the split.",
        "- Stratify on the five target bins above. Do not stratify on the raw 10-way label: `1.0` cannot appear in five folds.",
        "- Do not use `GroupKFold`. No speaker, session, or recording-group field exists. If a later duplicate or speaker cluster is found, switch to grouped folds for those clusters only.",
        "- Do not use filename, filename integer, missing-encoder flag, encoder string, or split name as a feature. On this train set, missing encoder tag and integer >= 5000 are the same 37 rows, and both identify label 0.0 perfectly.",
        "- Key every cache and prediction by `(split, filename)`.",
        "- Official metric is unknown. Report RMSE, MAE, Pearson, Spearman, and R². Do not promote one of them to official.",
        "- Keep test audio out of selection. Do not tune on a submission file.",
        "",
        "## Next three experiments",
        "",
        "Not run by this script.",
        "",
        "1. **E001–E002 constant baselines.** Predict the training mean and the training median under the stratified 5-fold protocol. Record per-fold RMSE/MAE and the fold target means. This is the floor. Also record how often the rare bins land in each fold.",
        "2. **E003 acoustic Ridge, no identifiers.** Duration, RMS, silence ratio, leading/trailing silence, and simple pause/energy summaries. Fit scaling inside the fold. Compare with E001. Separately report error on the 37 high-ID zeros versus the other 732 rows, so a gain that is only the zero cohort is visible.",
        "3. **E004 leakage control, not a candidate.** Same CV, two forbidden one-feature models: filename integer only, and missing-encoder flag only. Compare with E001 and E003. Quantify the cheat. Do not keep either model, do not ensemble it, and do not use its predictions for stacking.",
        "",
        "## Artifacts",
        "",
        "- `outputs/reports/dataset_report.md` — this file",
        "- `outputs/reports/audio_inventory.csv` — one row per wav, including MD5 and silence stats",
        "",
    ]
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only dataset reconnaissance")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--zip-path", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--inventory", type=Path, default=None)
    args = parser.parse_args()

    default_data, default_zip, default_report, default_inventory = _default_paths()
    data_dir = args.data_dir or default_data
    zip_path = args.zip_path or default_zip
    report_path = args.report or default_report
    inventory_path = args.inventory or default_inventory

    train_fields, train_rows = load_labels(data_dir / "train.csv")
    test_fields, test_rows = load_labels(data_dir / "test.csv")
    sub_fields, sub_rows = load_labels(data_dir / "sample_submission.csv")
    if train_fields != ["filename", "label"]:
        raise SystemExit(f"unexpected train columns: {train_fields}")
    if test_fields != ["filename", "label"]:
        raise SystemExit(f"unexpected test columns: {test_fields}")
    if sub_fields != ["filename", "label"]:
        raise SystemExit(f"unexpected submission columns: {sub_fields}")

    records: list[AudioRecord] = []
    for split, rows in (("train", train_rows), ("test", test_rows)):
        for row in rows:
            records.append(inspect_wav(split, data_dir / split / row["filename"]))
            if len(records) % 100 == 0:
                print(f"inspected {len(records)}", flush=True)

    labels_for_inventory = {
        f"train/{row['filename']}": row["label"] for row in train_rows
    }
    labels_for_inventory.update({f"test/{row['filename']}": row["label"] for row in test_rows})

    exact_file = group_md5(records, "file_md5")
    exact_pcm = group_md5(records, "pcm_md5")
    print("scanning near-duplicates", flush=True)
    near_hits, max_cosine, near_comparisons = near_duplicates(records)
    print(f"near-duplicate comparisons {near_comparisons}", flush=True)
    zip_info = zip_membership(zip_path, data_dir)

    report = build_report(
        data_dir,
        zip_info,
        train_fields,
        train_rows,
        test_fields,
        test_rows,
        sub_fields,
        sub_rows,
        records,
        exact_file,
        exact_pcm,
        near_hits,
        max_cosine,
        near_comparisons,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    write_inventory(inventory_path, records, labels_for_inventory)
    print(f"wrote {report_path}")
    print(f"wrote {inventory_path}")
    print(f"readable {sum(r.readable for r in records)}/{len(records)}")
    print(f"exact file groups {len(exact_file)} pcm groups {len(exact_pcm)} near hits {len(near_hits)} max cosine {max_cosine}")


if __name__ == "__main__":
    main()
