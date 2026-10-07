"""Cached faster-whisper transcripts. Train or test is chosen by the caller.

The transcript text is a feature source. Filename and encoder tag are not.
"""

from __future__ import annotations

import json
from pathlib import Path

ASR_VERSION = "fw_small_int8_v1"
MODEL_SIZE = "small"
COMPUTE_TYPE = "int8"
LANGUAGE = "en"
BEAM_SIZE = 1


def transcript_path(cache_dir: Path, split: str, filename: str) -> Path:
    return cache_dir / split / f"{Path(filename).stem}.json"


def load_transcript(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if payload.get("version") != ASR_VERSION:
        return None
    return payload


def transcribe_file(model, audio_path: Path, split: str, filename: str) -> dict[str, object]:
    segments, info = model.transcribe(
        str(audio_path),
        language=LANGUAGE,
        beam_size=BEAM_SIZE,
        vad_filter=True,
        word_timestamps=True,
        condition_on_previous_text=False,
    )
    words: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []
    texts: list[str] = []
    for segment in segments:
        texts.append(segment.text.strip())
        segment_rows.append(
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "avg_logprob": segment.avg_logprob,
                "no_speech_prob": segment.no_speech_prob,
            }
        )
        for word in segment.words or []:
            words.append(
                {
                    "start": word.start,
                    "end": word.end,
                    "word": word.word,
                    "probability": word.probability,
                }
            )
    return {
        "version": ASR_VERSION,
        "model": MODEL_SIZE,
        "compute_type": COMPUTE_TYPE,
        "language": LANGUAGE,
        "beam_size": BEAM_SIZE,
        "vad_filter": True,
        "sample_id": f"{split}/{filename}",
        "duration": info.duration,
        "language_probability": info.language_probability,
        "text": " ".join(part for part in texts if part),
        "segments": segment_rows,
        "words": words,
    }


def save_transcript(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload))
    temporary.replace(path)
