# pdm-bench

`pdm-bench` is a benchmark toolkit for bearing fault diagnosis on the Case Western Reserve University (CWRU) and Paderborn University (PU) datasets. It provides dataset loaders, predefined benchmark scenarios, classical ML and deep learning pipelines, and scripts for running experiments and summarizing results.

This repository accompanies our PHM paper submission, but it is meant to stand on its own as a regular open-source project focused on runnable benchmark code and configuration.

The Python package name is `pdm_bench`.

## Features

- Benchmark coverage for six transfer and generalization scenarios across CWRU and PU
- Classical ML pipeline with feature extraction and Bayesian hyperparameter search
- Deep learning pipeline with MLP, 1D CNN, FFT CNN, and STFT CNN variants
- Hydra-based configuration for datasets, tasks, and run settings
- Download helpers for the public benchmark datasets
- Result summarization scripts for benchmark tiers and DL follow-up runs

## Installation

`pdm-bench` supports Python 3.10 to 3.12 and uses `uv` for dependency management.

Install the project:

```bash
uv sync
```

Install development dependencies:

```bash
uv sync --group dev
```

Install optional MLflow support:

```bash
uv sync --extra tracking
```

## Quick Start

Install dependencies, download data, and validate the setup:

```bash
uv sync --group dev
./scripts/download_datasets.sh --all
uv run pytest -q
./scripts/run_phm_benchmark.sh --check-configs --quick
./scripts/run_phm_followup_dl.sh --check-configs --scope winners
```

If you want a lightweight execution smoke test after the datasets are present:

```bash
./scripts/run_phm_benchmark.sh --smoke --pipelines ml --datasets cwru
```

## Datasets

The benchmark uses local copies of:

- CWRU: Case Western Reserve University Bearing Data Center data
- PU: Paderborn University bearing dataset

The expected dataset locations are:

- `${WORKSPACE}/datasets/javadseraj-cwru-bearing-fault-data-set/Datasets/CWRU/`
- `${WORKSPACE}/datasets/paderborn-university-bearing-dataset/`

You can either set `WORKSPACE` manually in `.env`, or let the runner scripts default it to the repository root.

### Download helper

The repository includes a convenience script for downloading the datasets:

```bash
./scripts/download_datasets.sh --all
./scripts/download_datasets.sh --cwru
./scripts/download_datasets.sh --pu
./scripts/download_datasets.sh --cwru --install-kaggle
```

Notes:

- CWRU download uses the official `kaggle` CLI and requires either `KAGGLE_USERNAME` plus `KAGGLE_KEY`, or `~/.kaggle/kaggle.json`.
- PU download uses `aria2c` plus either `unrar` or `7z`, and `wget` or `curl`.
- On systems with externally managed Python, `uv tool install kaggle` is usually the cleanest way to install the Kaggle CLI.

## Running the Benchmark

The main runner exposes three benchmark profiles:

```bash
./scripts/run_phm_benchmark.sh --quick
./scripts/run_phm_benchmark.sh --normal
./scripts/run_phm_benchmark.sh --long
```

You can inspect the expanded Hydra configs without launching jobs:

```bash
./scripts/run_phm_benchmark.sh --check-configs --quick
```

The repeated-seed DL follow-up runs are handled separately:

```bash
./scripts/run_phm_followup_dl.sh --scope winners --seeds 41,42,43
./scripts/run_phm_followup_dl.sh --scope all --seeds 41,42,43
```

Outputs are written under `artifacts/benchmarks/`.

### Benchmark scenarios

The benchmark includes six scenarios:

- `cwru_cross_load`
- `cwru_cross_fs`
- `cwru_cross_fault_instance`
- `pu_cross_operating_condition`
- `pu_cross_damage_provenance`
- `pu_cross_bearing_instance`

## Summarizing Results

Build a summary package for one benchmark root:

```bash
uv run python scripts/summarize_benchmark.py \
  artifacts/benchmarks/phm_long_v2 \
  --output-dir artifacts/benchmarks/phm_long_v2/summary
```

Compare multiple benchmark tiers:

```bash
uv run python scripts/compare_benchmark_runs.py \
  artifacts/benchmarks/phm_quick_v1 \
  artifacts/benchmarks/phm_normal \
  artifacts/benchmarks/phm_long_v2 \
  --output-dir artifacts/benchmarks/phm_tier_comparison
```

Summarize the repeated-seed DL follow-up runs:

```bash
uv run python scripts/summarize_followup_dl.py \
  artifacts/benchmarks/phm_followup_dl \
  --output-dir artifacts/benchmarks/phm_followup_dl/summary
```

## Project Layout

- `src/pdm_bench/` core loaders, pipelines, training, tracking, and evaluation code
- `config/` Hydra configs for datasets and benchmark scenarios
- `scripts/` runnable entry points for benchmark execution, downloads, and summarization
- `tests/` automated test suite

## Extending the Project

### Add a dataset loader

1. Implement a loader that returns `list[Recording]`.
2. Register it in the ML and DL pipelines.
3. Add a dataset config under `config/dataset/`.
4. Add or update task configs under `config/task/`.

### Add an ML model

1. Add the estimator to `classifier_factory` in `src/pdm_bench/training/ml_classifiers.py`.
2. Add a search space in the same file if you want Bayesian tuning support.
3. Reference the classifier name from the relevant Hydra config or runner overrides.

### Add a DL model

1. Register the model in `src/pdm_bench/training/dl/models.py`.
2. Add or update task presets under `config/task/<scenario>/`.
3. Include the model in the runner matrix if it should become part of the standard benchmark suite.

## Development

Run the test suite:

```bash
uv run pytest -q
```

Run the lightweight config checks:

```bash
./scripts/run_phm_benchmark.sh --check-configs --quick
./scripts/run_phm_followup_dl.sh --check-configs --scope winners
```

GitHub Actions runs the test suite on Python 3.10 and 3.12.

## License

This project is released under the license in `LICENSE`.
