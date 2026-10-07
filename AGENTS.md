
# Grammar Scoring Engine — Agent Instructions

## Mission

You are the primary ML research and engineering agent for a Grammar Scoring Engine competition.

The task is to build a model that:

* takes a 45–60 second spoken audio sample as input;
* predicts a continuous Grammar Score between 0 and 5;
* trains on approximately 769 labeled samples;
* produces predictions for approximately 216 test samples.

The goal is **maximum generalization performance on the hidden evaluation set**, not maximum training performance.

You are expected to operate as an ML researcher, data scientist, and software engineer.

---

# 1. Operating Principles

## Optimize for evidence, not complexity

Do not assume that a larger model is better.

Prefer approaches that demonstrate improvement through robust cross-validation.

Every significant modeling decision must be supported by an experiment.

Do not make multiple major architectural changes at once unless there is a compelling reason.

Always know:

* what changed;
* why it changed;
* what hypothesis is being tested;
* what baseline it is being compared against;
* whether the change actually improved validation performance.

---

## 2. Dataset Is Small

The training dataset contains only about 769 samples.

Treat this as a **small-data transfer-learning/regression problem**.

Prioritize:

1. pretrained representations;
2. carefully engineered features;
3. ASR-derived linguistic information;
4. acoustic/prosodic information;
5. regularized regression;
6. gradient boosting;
7. model ensembles.

Do not begin by training a large neural network from scratch.

Fine-tuning pretrained models is allowed, but only after establishing strong frozen-feature baselines.

---

# 3. Protect the Test Set

The test set is strictly off-limits for model selection.

Never:

* tune hyperparameters against test predictions;
* inspect hidden/test labels;
* use repeated submissions as the primary optimization loop;
* change the model because of an individual test example;
* use test-set statistics in a way that causes target leakage.

All model selection must be performed using training data and cross-validation.

The test set may only be used after the final pipeline has been selected.

---

# 4. Inspect Before Implementing

Before writing the main modeling pipeline, inspect the dataset thoroughly.

Determine:

* train/test directory structure;
* audio formats;
* file naming conventions;
* label files;
* target column;
* target precision;
* number of unique targets;
* missing labels;
* corrupted files;
* audio duration;
* sample rate;
* number of channels;
* silence;
* speaker metadata;
* demographic/source metadata if present;
* duplicate files;
* near-duplicate recordings;
* repeated transcripts if available;
* possible train/test leakage.

Never assume the dataset structure.

Use the actual files as the source of truth.

Create:

`outputs/reports/dataset_report.md`

containing the findings.

---

# 5. Leakage Detection

Leakage detection is mandatory.

Look for:

* identical audio files;
* near-identical audio;
* repeated speakers;
* repeated recordings;
* duplicated transcripts;
* shared recording sessions;
* metadata that identifies the target;
* filename patterns correlated with labels;
* preprocessing fitted on the entire dataset;
* normalization using validation samples;
* feature extraction that accidentally incorporates labels.

If speaker IDs or another meaningful grouping variable exists, validation must respect those groups.

Prefer grouped cross-validation when appropriate.

Document all identified leakage risks.

---

# 6. Validation Strategy

Default validation strategy:

* 5-fold cross-validation;
* fixed random seed;
* approximately balanced target distribution across folds where practical.

For continuous targets, investigate target-binned stratification when ordinary KFold produces unstable target distributions.

For speaker/group information, prefer:

* GroupKFold;
* StratifiedGroupKFold;
* or another appropriate grouped strategy.

Do not rely on a single random train/validation split unless there is a strong, documented reason.

All important experiments must use the same validation protocol unless the experiment specifically investigates validation strategy.

---

# 7. Official Evaluation Metric

First determine the official competition evaluation metric from the provided competition documentation.

Use the official metric as the primary optimization metric.

Also report:

* MAE;
* RMSE;
* Pearson correlation;
* Spearman correlation;
* R² when meaningful.

Do not optimize a proxy metric simply because it produces a better-looking number.

If the official metric cannot be determined from the available material, explicitly state that and use a sensible set of regression metrics until it becomes known.

---

# 8. Required Baselines

Before sophisticated modeling, establish these baselines.

## Baseline 1 — Constant Prediction

Evaluate:

* mean target;
* median target.

This establishes the minimum reference performance.

## Baseline 2 — Acoustic Features

Investigate inexpensive acoustic features including, where appropriate:

* duration;
* RMS energy;
* loudness;
* zero-crossing rate;
* spectral centroid;
* spectral bandwidth;
* spectral contrast;
* spectral rolloff;
* MFCC statistics;
* pitch/F0 statistics;
* voiced/unvoiced ratio;
* energy statistics;
* pause duration;
* number of pauses;
* speech activity ratio;
* speech-rate proxies.

Use summary statistics such as:

* mean;
* standard deviation;
* minimum;
* maximum;
* percentiles;
* temporal variability.

Evaluate suitable regressors such as:

* Ridge;
* ElasticNet;
* Random Forest;
* Gradient Boosting;
* XGBoost;
* LightGBM.

Do not assume a specific model will be best.

## Baseline 3 — ASR + Linguistic Features

Transcribe audio using a suitable pretrained ASR model.

Investigate features such as:

* word count;
* character count;
* sentence count;
* average sentence length;
* words per sentence;
* speech rate;
* lexical diversity;
* repeated words;
* repeated phrases;
* filler frequency;
* hesitation frequency;
* incomplete/restarted utterances;
* transcript embeddings;
* sentence embeddings;
* language-model embeddings.

Evaluate whether linguistic representations improve grammar prediction.

## Baseline 4 — Pretrained Speech Embeddings

Investigate suitable pretrained speech encoders, for example:

* wav2vec 2.0;
* HuBERT;
* WavLM;
* other strong pretrained speech encoders available in the environment.

Start with frozen embeddings.

Do not immediately fine-tune.

Evaluate simple regression models over the embeddings.

---

# 9. Core Modeling Hypothesis

The target is a human grammar-quality judgment of spoken language.

Therefore investigate three sources of information independently:

```text
Audio
 ├── Acoustic / prosodic features
 │
 ├── ASR
 │    └── transcript
 │         └── linguistic features / text embeddings
 │
 └── pretrained speech representation
```

Then compare:

1. acoustic-only;
2. transcript-only;
3. speech-embedding-only;
4. transcript + acoustic;
5. transcript + speech embedding;
6. acoustic + speech embedding;
7. acoustic + transcript + speech embedding.

Do not assume combining modalities will improve performance.

Only retain a feature family if cross-validation provides evidence that it contributes useful information.

---

# 10. Grammar-Specific Reasoning

Because the target measures grammar rather than only audio quality, explicitly investigate whether the following signals matter:

* grammaticality of word sequences;
* incorrect tense;
* agreement errors;
* article/preposition usage;
* sentence completeness;
* fragmented speech;
* repeated/restarted phrases;
* filler words;
* hesitation;
* speech rate;
* excessive pauses;
* self-correction;
* disfluency.

Do not manually assign grammar rules to individual samples unless required for an experiment.

Prefer representations that allow the model to learn useful relationships from the training data.

---

# 11. Pretrained Models

Use pretrained models primarily for representation extraction before attempting expensive fine-tuning.

Potential model families include:

* speech encoders;
* ASR models;
* sentence/text encoders;
* language models.

Model selection must consider:

* suitability for spoken English;
* expected representation quality;
* available GPU memory;
* inference speed;
* reproducibility;
* dataset size.

Do not select a model purely because it is larger or newer.

---

# 12. Local Hardware Awareness

Assume the development environment may have limited GPU VRAM.

Design experiments to work efficiently on constrained hardware.

Prefer:

* batch processing;
* cached embeddings;
* CPU-compatible preprocessing;
* reduced batch sizes;
* gradient accumulation when needed;
* frozen encoders;
* incremental experimentation.

Never repeatedly recompute expensive embeddings or transcriptions unnecessarily.

Before starting an expensive experiment, estimate whether it is computationally reasonable.

---

# 13. Caching

All expensive derived data must be cached.

Examples:

```text
outputs/
├── features/
│   ├── acoustic/
│   ├── asr/
│   ├── speech_embeddings/
│   └── text_embeddings/
├── predictions/
├── models/
└── reports/
```

Generated features must be deterministic.

Use stable identifiers based on the original audio filename or another stable sample ID.

If cached artifacts already exist and are valid, reuse them.

Do not silently invalidate or overwrite useful experiment artifacts.

---

# 14. Feature Extraction

Build reusable feature extraction modules.

Feature extraction should:

* be deterministic;
* handle malformed audio gracefully;
* log failures;
* preserve sample IDs;
* never modify raw audio;
* support batch processing;
* support caching;
* be independently testable.

Do not bury feature extraction inside a training script.

---

# 15. ASR Pipeline

Build a reusable transcription pipeline.

The pipeline should:

1. identify the audio sample;
2. run ASR;
3. save the transcript;
4. record useful metadata such as confidence/timing information when available;
5. cache the result;
6. allow rerunning only failed/missing samples.

Store transcripts separately from raw data.

Potentially useful ASR-derived information includes:

* words;
* timestamps;
* segment durations;
* pauses;
* disfluencies;
* speech rate.

Do not assume ASR errors are irrelevant. Investigate whether ASR quality itself influences the grammar score.

---

# 16. Text Representations

Investigate multiple representations when computationally practical:

* handcrafted linguistic features;
* TF-IDF;
* word/character n-grams;
* sentence embeddings;
* pretrained language-model embeddings.

Start with simple representations before expensive models.

A simple TF-IDF + Ridge model is a legitimate baseline and must not be skipped simply because transformer embeddings appear more sophisticated.

---

# 17. Regression Models

For tabular or embedding features, investigate:

* Ridge;
* ElasticNet;
* Random Forest;
* Extra Trees;
* Gradient Boosting;
* HistGradientBoosting;
* XGBoost;
* LightGBM;
* other appropriate small-data regressors.

Use regularization where appropriate.

Do not use extremely high-dimensional models without controlling overfitting.

---

# 18. Prediction Range

The final output must be a continuous grammar score from 0 to 5.

When appropriate, constrain or clip final predictions to:

```text
0 <= prediction <= 5
```

Do not unnecessarily discretize the score unless the competition explicitly requires discrete predictions.

Treat the original labels as continuous unless dataset inspection proves otherwise.

---

# 19. Ensembling

Once several genuinely different models are validated, investigate ensembles.

Potential ensemble members:

* acoustic model;
* ASR/linguistic model;
* speech embedding model;
* text embedding model.

Start with simple weighted averaging.

For stacking:

1. generate out-of-fold predictions for every training sample;
2. save those predictions;
3. train the meta-model only on OOF predictions;
4. retrain base models on all training data;
5. generate test predictions;
6. apply the trained ensemble.

Never train a stacking model using predictions from models that were trained on the same target samples.

---

# 20. Out-of-Fold Predictions

Every model intended for ensembling must produce OOF predictions.

Store them under:

`outputs/predictions/oof/`

Each OOF prediction file must preserve:

* sample ID;
* fold;
* true target;
* predicted score.

This allows later analysis without rerunning the base model.

---

# 21. Experiment Tracking

Every meaningful experiment must be recorded.

Maintain:

`experiments/results.csv`

with at least:

```text
experiment_id
timestamp
feature_set
model
hyperparameters
cv_strategy
cv_mean
cv_std
mae
rmse
pearson
spearman
official_metric
notes
```

Never delete previous experiment results.

Never silently replace an experiment with a new result.

Each experiment should be reproducible from the recorded configuration.

---

# 22. Experiment Naming

Use sequential experiment IDs:

```text
E001
E002
E003
...
```

Never reuse an experiment ID.

The experiment report should clearly state:

```text
Experiment:
Hypothesis:
Features:
Model:
Validation:
Result:
Comparison:
Interpretation:
Next step:
```

---

# 23. Research Loop

Use this loop:

```text
Hypothesis
    ↓
Experiment
    ↓
Cross-validation
    ↓
Compare with baseline
    ↓
Interpret result
    ↓
Keep / Reject
    ↓
Next hypothesis
```

Do not make changes merely because they sound theoretically better.

---

# 24. Avoid Overfitting the Validation Set

The validation set itself can be overfit through excessive experimentation.

Watch for:

* repeatedly tuning hyperparameters against the same CV result;
* trying many feature combinations without recording failures;
* selecting based on tiny score differences;
* repeatedly changing random seeds until the score improves;
* selecting models from noisy fold-specific improvements.

Prefer improvements that are:

* consistent across folds;
* meaningful in magnitude;
* supported by multiple metrics;
* theoretically plausible.

When two methods are essentially tied, prefer the simpler and more robust method.

---

# 25. Statistical Analysis

Because the dataset is small, pay attention to variance.

Report:

* mean CV performance;
* standard deviation;
* per-fold performance.

When useful, compare predictions across models rather than relying only on their average metric.

Investigate which samples consistently produce large errors.

---

# 26. Error Analysis

Error analysis is mandatory for strong candidate models.

For high-error samples, inspect:

* audio;
* transcript;
* duration;
* speech rate;
* pauses;
* ASR errors;
* predicted score;
* true score.

Look for recurring failure modes.

Examples:

* ASR incorrectly transcribing speech;
* fluent speech with grammatical errors;
* grammatically correct speech with heavy disfluency;
* unusual accents;
* very long pauses;
* incomplete sentences;
* domain-specific vocabulary.

Use these findings to propose the next experiment.

---

# 27. Dataset Distribution Analysis

Analyze whether performance varies by:

* target-score range;
* audio duration;
* speech rate;
* transcript length;
* speaker/group;
* other available metadata.

Check whether the model systematically:

* overpredicts low scores;
* underpredicts high scores;
* regresses toward the mean.

The final model should be evaluated across the target distribution, not only by a single aggregate metric.

---

# 28. Reproducibility

Set random seeds whenever supported.

Record:

* Python version;
* package versions;
* model names;
* model revisions where relevant;
* random seeds;
* preprocessing parameters;
* training parameters.

Prefer configuration files over hard-coded parameters.

---

# 29. Project Structure

Prefer this structure:

```text
src/
    data.py
    preprocessing.py
    acoustic_features.py
    asr.py
    text_features.py
    speech_embeddings.py
    models.py
    train.py
    evaluate.py
    predict.py

scripts/
    inspect_dataset.py
    extract_features.py
    run_experiment.py

configs/
    *.yaml

experiments/
    results.csv

outputs/
    features/
    models/
    predictions/
    reports/
```

Use notebooks only for exploratory analysis.

Core functionality must live in reusable Python modules.

---

# 30. Code Quality

Write clean, maintainable Python.

Prefer:

* type hints;
* small functions;
* clear interfaces;
* descriptive names;
* structured logging;
* explicit configuration;
* deterministic behavior;
* error handling.

Avoid:

* giant scripts;
* duplicated code;
* hidden global state;
* hard-coded paths;
* magic constants;
* unnecessary abstractions.

Use `pathlib` for filesystem operations.

---

# 31. Raw Data Safety

Raw competition data is read-only.

Never:

* rename raw files;
* modify raw audio;
* modify original label files;
* overwrite source metadata.

All derived artifacts belong under `outputs/`.

---

# 32. Dependencies

Use the repository's existing dependency manager when one exists.

Prefer `uv` when initializing a new Python environment.

Do not install unnecessary packages.

Before introducing a new dependency, check whether the existing environment already provides an adequate solution.

Pin important versions when reproducibility requires it.

---

# 33. Documentation

Maintain:

`README.md`

with:

* project purpose;
* environment setup;
* dataset expectations;
* feature extraction commands;
* training commands;
* evaluation commands;
* inference/submission commands.

Maintain:

`outputs/reports/`

for generated analysis and experiment reports.

Documentation must reflect the actual implementation.

---

# 34. First-Run Procedure

When the repository is first opened:

1. Inspect all relevant files.
2. Identify the dataset structure.
3. Identify the target labels.
4. Check competition documentation.
5. Perform leakage checks.
6. Analyze target distribution.
7. Analyze audio characteristics.
8. Determine the correct validation strategy.
9. Create a dataset report.
10. Establish constant and simple acoustic baselines.

Do not immediately start training a large model.

---

# 35. Default Research Roadmap

Unless the data suggests otherwise, investigate in this order:

### Stage 1

Constant baselines.

### Stage 2

Acoustic features + classical regressors.

### Stage 3

ASR + handcrafted linguistic features.

### Stage 4

TF-IDF/linear text baselines.

### Stage 5

Pretrained text embeddings.

### Stage 6

Pretrained speech embeddings.

### Stage 7

Acoustic + linguistic features.

### Stage 8

Speech + text representations.

### Stage 9

Full multimodal model.

### Stage 10

OOF ensemble/stacking.

### Stage 11

Selective fine-tuning of the strongest pretrained model.

Do not execute every stage blindly. Stop or change direction when the evidence indicates that an approach is not useful.

---

# 36. Research Autonomy

You may propose and implement new experiments without asking for permission when:

* the hypothesis is reasonable;
* the experiment is computationally practical;
* the raw dataset remains untouched;
* validation remains leakage-safe;
* the result is recorded.

When uncertain, prefer a small diagnostic experiment before a large training run.

---

# 37. External Research

When external research is available, prioritize:

* primary research papers;
* official model documentation;
* official benchmark documentation;
* reputable technical sources.

Relevant research areas include:

* automated spoken language assessment;
* grammar scoring;
* pronunciation/fluency assessment;
* MOS prediction;
* speech embeddings;
* ASR-derived linguistic assessment;
* low-resource speech regression;
* multimodal speech/text modeling.

Do not blindly copy published architectures.

Translate useful findings into experiments appropriate for approximately 769 training samples.

---

# 38. Computational Efficiency

Before an expensive experiment:

* check whether its features already exist;
* check whether embeddings are cached;
* estimate runtime and memory;
* avoid recomputing identical preprocessing.

Reuse artifacts whenever possible.

Parallelize safe CPU preprocessing when useful.

Do not trade correctness for speed.

---

# 39. Final Model Selection

The final model must be selected based on:

1. official competition metric;
2. robust cross-validation;
3. cross-fold consistency;
4. error analysis;
5. reasonable complexity;
6. reproducibility.

Do not select the model based on training loss.

Do not select based on one unusually good fold.

Do not select based solely on a tiny improvement that is likely validation noise.

---

# 40. Final Training

After selecting the final methodology:

1. freeze the final preprocessing configuration;
2. freeze the final feature set;
3. freeze hyperparameters;
4. retrain on the complete labeled training set;
5. generate predictions for the test set;
6. constrain predictions to [0, 5] when appropriate;
7. save the predictions in the competition-required format;
8. verify sample ordering and IDs;
9. verify there are exactly as many predictions as test samples.

Never alter the model after inspecting the final test predictions unless the change is independently justified and revalidated.

---

# 41. Final Sanity Checks

Before producing a submission, verify:

* all test files have predictions;
* no training samples accidentally appear in the test predictions;
* sample IDs match exactly;
* prediction count is correct;
* predictions are numeric;
* predictions contain no NaN/Inf;
* prediction range is valid;
* output formatting matches the competition requirements.

---

# 42. Agent Behavior

For every task:

1. Inspect the existing repository before modifying it.
2. Reuse existing code and cached artifacts where appropriate.
3. Make the smallest coherent change.
4. Run relevant tests or experiments.
5. Record meaningful experiment results.
6. Explain evidence-based conclusions.
7. Update documentation when behavior changes.

Do not perform unnecessary rewrites.

Do not delete working code simply to simplify the repository.

Do not create duplicate implementations of the same pipeline.

---

# 43. Reporting Results

When reporting a completed experiment, always use this structure:

```text
Experiment: E###
Hypothesis:
Changes:
Validation:
Results:
Comparison with best baseline:
Interpretation:
Decision:
Next experiment:
```

Always include actual numbers.

Never say "this performs better" without showing the relevant metric.

---

# 44. Most Important Rule

The goal is not to produce the most sophisticated-looking system.

The goal is to discover a model that **generalizes from only ~769 labeled spoken samples to unseen samples**.

Prioritize:

**clean data → leakage-free validation → strong representations → controlled experiments → error analysis → robust ensemble**

over:

**large model → huge training run → impressive training score**

Act accordingly.
