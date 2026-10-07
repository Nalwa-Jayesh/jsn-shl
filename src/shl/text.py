"""Handcrafted linguistic features from a cached transcript. No labels."""

from __future__ import annotations

import math
import re

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9']+")
SENTENCE_RE = re.compile(r"[.!?]+")
FILLERS = frozenset({"um", "uh", "er", "ah", "hmm", "mm", "mmm", "like", "uhh", "umm"})

FEATURE_NAMES = [
    "word_count",
    "char_count",
    "sentence_count",
    "words_per_sentence",
    "speech_rate_wps",
    "type_token_ratio",
    "root_ttr",
    "repeat_word_ratio",
    "repeat_bigram_ratio",
    "filler_ratio",
    "filler_count",
    "mean_word_prob",
    "low_conf_word_ratio",
    "segment_count",
    "mean_segment_logprob",
    "mean_no_speech_prob",
    "word_pause_count",
    "word_pause_mean_s",
    "empty_transcript",
]


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def linguistic_features(payload: dict[str, object], duration_s: float) -> np.ndarray:
    text = str(payload.get("text") or "")
    tokens = tokenize(text)
    n_words = len(tokens)
    sentences = [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]
    n_sentences = max(len(sentences), 1 if n_words else 0)
    duration = duration_s if duration_s > 0 else float(payload.get("duration") or 0.0)
    unique = len(set(tokens))
    repeats = n_words - unique
    bigrams = list(zip(tokens, tokens[1:], strict=False))
    unique_bigrams = len(set(bigrams))
    filler_count = sum(token in FILLERS for token in tokens)
    words = payload.get("words") or []
    probs = [float(word["probability"]) for word in words if word.get("probability") is not None]
    gaps = []
    timed = [word for word in words if word.get("start") is not None and word.get("end") is not None]
    for left, right in zip(timed, timed[1:], strict=False):
        gap = float(right["start"]) - float(left["end"])
        if gap >= 0.3:
            gaps.append(gap)
    segments = payload.get("segments") or []
    logprobs = [float(seg["avg_logprob"]) for seg in segments if seg.get("avg_logprob") is not None]
    no_speech = [float(seg["no_speech_prob"]) for seg in segments if seg.get("no_speech_prob") is not None]
    values = [
        float(n_words),
        float(len(text)),
        float(n_sentences),
        float(n_words / n_sentences) if n_sentences else 0.0,
        float(n_words / duration) if duration else 0.0,
        float(unique / n_words) if n_words else 0.0,
        float(unique / math.sqrt(n_words)) if n_words else 0.0,
        float(repeats / n_words) if n_words else 0.0,
        float(1.0 - unique_bigrams / len(bigrams)) if bigrams else 0.0,
        float(filler_count / n_words) if n_words else 0.0,
        float(filler_count),
        float(np.mean(probs)) if probs else 0.0,
        float(np.mean(np.asarray(probs) < 0.5)) if probs else 0.0,
        float(len(segments)),
        float(np.mean(logprobs)) if logprobs else 0.0,
        float(np.mean(no_speech)) if no_speech else 1.0,
        float(len(gaps)),
        float(np.mean(gaps)) if gaps else 0.0,
        float(n_words == 0),
    ]
    return np.asarray(values, dtype=np.float64)
