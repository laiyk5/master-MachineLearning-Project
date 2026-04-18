# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CS5487 Machine Learning course project: handwritten digit classification on a 4,000-sample MNIST subset (400 per class, digits 0-9). Features are 784-dimensional (28x28 grayscale, values 0-255). The dataset defines 2 fixed train/test splits (50/50, 2,000 samples each, different writers). Baseline is 1-NN Euclidean at ~91.60%.

## Common Commands

This project uses `uv` for dependency management and script execution. Python >=3.10 is required.

```bash
# Install dependencies
uv sync

# Run the main experiment suite (9 classifiers, prints summary table)
uv run python scripts/run_experiments.py

# Run tests with coverage
uv run pytest

# Run a single test file
uv run pytest tests/test_basic.py

# Lint and format
uv run ruff check src tests scripts
uv run ruff format src tests scripts

# Type check
uv run mypy src

# Launch Jupyter
uv run jupyter notebook notebooks/
```

## High-Level Architecture

### Core Package (`src/`)

The source code is organized around an experiment runner pattern:

- **`src/data_loader.py`** — Loads `digits4000.mat` (MATLAB format via `scipy.io.loadmat`). Provides `load_digits_data()`, `get_train_test_split(data, trial)` where `trial` is 0 or 1, and `normalize_features()` with methods `standard`, `minmax`, `scale255`, or `none`. Note: image vectors are stored column-major (transposed on reshape to display as 28x28).

- **`src/classifiers.py`** — Contains a `OneVsAllClassifier` wrapper and `PCATransform`. `get_classifier(name)` maps strings (`knn`, `svm_linear`, `svm_rbf`, `svm_poly`, `logistic`, `lda`, `qda`, `linear_svc`) to sklearn instances. The one-vs-all wrapper trains 10 binary classifiers and uses `decision_function` or `predict_proba` for scoring.

- **`src/experiments.py`** — `Experiment` class orchestrates trials. `run_experiment()` runs both splits and returns mean/std accuracy. `grid_search()` sweeps a parameter list. Results accumulate in `self.results` and can be saved to JSON.

- **`src/preprocessing.py`** and **`src/features.py`** — Mostly placeholder stubs.
- **`src/utils.py`** — Result directory management (`get_result_dir()`, `get_git_short_hash()`, `setup_run_logging()`). All scripts use this to create timestamped output directories.

### Standalone Scripts (`scripts/`)

All executable scripts live in `scripts/` and write outputs to `results/<git-short-hash>/<timestamp>/`. Each run creates a fresh directory with subfolders `figures/` and `logs/`, plus a `run.log` that captures the full console output.

- **`scripts/run_experiments.py`** — Main entry point. Runs all 9 predefined experiment configurations, prints a formatted results table, and saves `logs/results.json` and `logs/summary.txt`.
- **`scripts/error_analysis.py`** — `ErrorAnalyzer` class: runs a classifier, generates confusion matrices, per-digit accuracy breakdowns, and saves figures to `figures/`.
- **`scripts/sample_error_analysis.py`** — Categorizes individual misclassified samples as easy/medium/hard based on confidence scores.
- **`scripts/class_distribution.py`** — Plots class distribution histograms.
- **`scripts/evaluate_challenge.py`** — Evaluates models trained on MNIST against the challenge dataset (`data/raw/challenge/cdigits.mat`) without retraining.

### Dataset Location

The data file `digits4000.mat` is expected at `data/raw/MINIST/digits4000.mat`. It is not committed to git (download separately). The `.mat` file contains `digits_vec` (784x4000), `digits_labels` (4000,), `trainset` (2x2000), and `testset` (2x2000).

A challenge dataset is also available at `data/raw/challenge/cdigits.mat` (150 samples, professor's handwritten digits).

### Tooling Configuration

All tool config lives in `pyproject.toml`:
- **Ruff**: target `py310`, line length 100, selects `E`, `F`, `I`, `N`, `W`, `UP`, `B`, `C4`, `SIM`.
- **Black**: line length 100.
- **pytest**: verbose, coverage for `src`, term-missing report.
- **mypy**: `py310`, `warn_return_any`, `warn_unused_configs`.

CI runs on Ubuntu with Python 3.10 and 3.11 via `.github/workflows/tests.yml`.
