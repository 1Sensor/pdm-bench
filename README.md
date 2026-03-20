# pdm-bench Benchmark Core

This repository contains the code, configs, and scripts needed to run the frozen bearing-fault benchmark used in the PHM work on the CWRU and PU datasets.

It is intentionally lean:

- no paper draft
- no checked-in benchmark outputs
- no internal export metadata
- no domain-adaptation or cross-sensor side experiments

The published project name is `pdm-bench`. The internal Python module path remains `pdm_tools`.

## What Is Here

- `src/` benchmark pipelines, loaders, feature extraction, training, evaluation, and tracking
- `config/` Hydra configs for the benchmark scenarios and model presets
- `scripts/` shell and Python entrypoints for running the benchmark and summarizing outputs

## Requirements

- Python 3.10-3.12
- `uv`
- local copies of the CWRU and PU datasets

Install the project with:

```bash
uv sync
```

For local development and tests:

```bash
uv sync --group dev
```

If you want MLflow tracking support:

```bash
uv sync --extra tracking
```

## Environment Setup

Copy `.env.example` to `.env` and set:

```bash
WORKSPACE=/absolute/path/to/this/repo
```

If `WORKSPACE` is unset, the benchmark runner scripts now default it to the repo root automatically.

The checked-in dataset configs expect these paths:

- `${WORKSPACE}/datasets/javadseraj-cwru-bearing-fault-data-set/Datasets/CWRU/`
- `${WORKSPACE}/datasets/paderborn-university-bearing-dataset/`

## Download Datasets

The repo includes a benchmark-only download helper:

```bash
./scripts/download_datasets.sh --all
./scripts/download_datasets.sh --cwru
./scripts/download_datasets.sh --pu
./scripts/download_datasets.sh --cwru --install-kaggle
```

Requirements:

- CWRU download uses the `kaggle` CLI and either `KAGGLE_USERNAME` plus `KAGGLE_KEY`, or your local Kaggle credentials in `~/.kaggle/kaggle.json`
- PU download uses `aria2c` plus either `unrar` or `7z`, and `wget` or `curl`

If the Kaggle CLI is missing, you can either install it yourself with
`uv tool install kaggle`, `python3 -m pip install --user kaggle`, or let the
script bootstrap it with `--install-kaggle`. On systems with externally-managed
Python, `uv tool install kaggle` is usually the cleanest option.

## Validate The Repo

Run the lightweight local validation flow with:

```bash
uv run pytest -q
./scripts/run_phm_benchmark.sh --check-configs --quick
./scripts/run_phm_followup_dl.sh --check-configs --scope winners
```

A matching GitHub Actions workflow runs the test suite on Python 3.10 and 3.12.

## Run The Benchmark

Run one of the frozen benchmark tiers:

```bash
./scripts/run_phm_benchmark.sh --quick
./scripts/run_phm_benchmark.sh --normal
./scripts/run_phm_benchmark.sh --long
```

Check the Hydra-expanded configs without executing runs:

```bash
./scripts/run_phm_benchmark.sh --check-configs --quick
```

Run the repeated-seed DL follow-up:

```bash
./scripts/run_phm_followup_dl.sh --scope all --seeds 41,42,43
```

Outputs are written under `artifacts/benchmarks/`.

## Summarize Results

Build a summary package for a benchmark root:

```bash
uv run python scripts/summarize_benchmark.py \
  artifacts/benchmarks/phm_long_v2 \
  --output-dir artifacts/benchmarks/phm_long_v2/summary
```

Compare benchmark tiers:

```bash
uv run python scripts/compare_benchmark_runs.py \
  artifacts/benchmarks/phm_quick_v1 \
  artifacts/benchmarks/phm_normal \
  artifacts/benchmarks/phm_long_v2 \
  --output-dir artifacts/benchmarks/phm_tier_comparison
```

Summarize the DL follow-up runs:

```bash
uv run python scripts/summarize_followup_dl.py \
  artifacts/benchmarks/phm_followup_dl \
  --output-dir artifacts/benchmarks/phm_followup_dl/summary
```

## Frozen Benchmark Scope

The main benchmark runner covers six scenarios:

- `cwru_cross_load`
- `cwru_cross_fs`
- `cwru_cross_fault_instance`
- `pu_cross_operating_condition`
- `pu_cross_damage_provenance`
- `pu_cross_bearing_instance`

## Extend The Repo

### Add a dataset loader

1. Implement a loader that returns `list[Recording]`.
2. Register it in both pipeline registries:
   `src/pdm_tools/main/pipelines/ml/pipeline.py`
   `src/pdm_tools/main/pipelines/dl/pipeline.py`
3. Add a dataset config under `config/dataset/`.
4. Add or update task configs under `config/task/`.

### Add an ML model

1. Add the estimator to `classifier_factory` in `src/pdm_tools/main/training/ml_classifiers.py`.
2. Add an optional search space in the same file if you want Bayesian tuning support.
3. Reference the classifier name from the Hydra config or benchmark script overrides.

### Add a DL model

1. Register the model in `src/pdm_tools/main/training/dl/models.py`.
2. Add or update task presets under `config/task/<scenario>/`.
3. Include the model in the relevant benchmark script task matrix if it should be part of the frozen benchmark.
