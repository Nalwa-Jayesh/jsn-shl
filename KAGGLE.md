# Kaggle runs

Same scripts, same folds, same `experiments/results.csv` schema locally and on Kaggle.
`src/shl/paths.py` picks directories. CLI flags override everything.

## Detection

Kaggle when `/kaggle/input` exists or `KAGGLE_KERNEL_RUN` is set.

| Env | Local | Kaggle |
|---|---|---|
| Competition data | `<repo>/data` | `/kaggle/input/...` (auto-found by `train.csv`) |
| Generated files | `<repo>/outputs`, `<repo>/experiments` | `/kaggle/working/outputs`, `/kaggle/working/experiments` |

Overrides: `SHL_DATA_DIR`, `SHL_WORK_DIR`, `SHL_KAGGLE_DATASET`.

## Required input

Attach the competition dataset so it mounts under `/kaggle/input`. Expected tree:

```text
<dataset>/
  train.csv                filename,label (769 rows)
  test.csv                 filename,label placeholder (216 rows)
  sample_submission.csv    filename,label
  train/*.wav              16 kHz mono
  test/*.wav               16 kHz mono
```

Read-only. Scripts never write under `/kaggle/input`. No `cp` of wavs.

## Dependencies

Notebook needs internet once, or pre-install via Settings > Internet on.

```bash
pip install -U pip
pip install -e . --no-deps
pip install faster-whisper librosa soundfile scikit-learn scipy numpy pandas \
  sentence-transformers torch torchaudio transformers
```

GPU note: `faster-whisper` needs `libcublas.so.12` for CUDA. Without it the
scripts fall back to CPU int8 (`run_asr_baselines.py`, `make_submission.py`).

## Commands

Repo root is `/kaggle/working/shl` if cloned there, else `cd` to it first.
Every command resumes its own cache. Reruns skip finished files.

```bash
python scripts/inspect_dataset.py
python scripts/run_baselines.py
python scripts/run_asr_baselines.py --transcribe-only
python scripts/run_asr_baselines.py
python scripts/run_text_embeddings.py
python scripts/run_speech_embeddings.py
python scripts/run_multimodal.py
python scripts/run_ensemble.py
python scripts/run_finetune.py --smoke
python scripts/run_finetune.py
python scripts/make_submission.py
```

Single-stage rerun example (cache + OOF + report only):

```bash
python scripts/run_multimodal.py \
  --data-dir /kaggle/input/<dataset> \
  --oof-dir /kaggle/working/outputs/predictions/oof
```

## Outputs

- Features: `/kaggle/working/outputs/features/{acoustic,asr,speech_embeddings,text_embeddings}/`
- OOF: `/kaggle/working/outputs/predictions/oof/E*.csv`
- Log: `/kaggle/working/experiments/results.csv` (append-only, same 5-fold seed 42)
- Reports: `/kaggle/working/outputs/reports/*.md`
- Submission: `/kaggle/working/outputs/submission.csv` (`filename,label`, test.csv order, clipped to [0, 5])

Download `submission.csv` from the notebook output panel.
