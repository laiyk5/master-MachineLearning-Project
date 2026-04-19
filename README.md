# Machine Learning Course Project - CS5487

**Team:** Yikai LAI, Zhang Yao  
**Course:** CS5487 Machine Learning, City University of Hong Kong  
**Project:** Handwritten Digit Classification (default project)

## Overview

Classification of a 4,000-sample MNIST subset (400 images per digit 0--9) using methods from the course. The dataset defines two predefined writer-independent train/test splits (50/50). Models are also evaluated on a held-out challenge dataset of 150 professor-written digits without retraining.

## Quick Start

```bash
# Install dependencies
uv sync

# Run the full experiment suite
uv run python scripts/run_experiments.py

# Evaluate on the challenge dataset
uv run python scripts/evaluate_challenge.py
```

## Project Structure

```
├── data/raw/MINIST/digits4000.mat    # Main dataset (download separately)
├── data/raw/challenge/cdigits.mat    # Challenge dataset
├── docs/report/report.tex            # Final report (LaTeX)
├── docs/requirement/                 # Course project requirements
├── notebooks/                        # Data exploration notebooks
├── scripts/                          # Executable analysis scripts
│   ├── run_experiments.py            # Main experiment suite
│   ├── evaluate_challenge.py         # Challenge evaluation
│   ├── error_analysis.py             # Confusion matrices & per-digit accuracy
│   ├── sample_error_analysis.py      # Hard/medium/easy sample analysis
│   ├── class_distribution.py         # Class distribution plots
│   └── evaluate_upper_bound.py       # Optional: CNN / vision LLM upper bound
├── src/                              # Core package
│   ├── data_loader.py                # Load data, splits, normalization
│   ├── classifiers.py                # Classifier factory, OvA wrapper, PCA
│   ├── experiments.py                # Experiment runner
│   ├── augmentation.py               # Geometric data augmentation
│   └── utils.py                      # Result directory management
├── results/                          # Auto-generated: results/<git-hash>/<timestamp>/
└── pyproject.toml                    # Dependencies and tool config
```

## Methods Evaluated

- $k$-Nearest Neighbors ($k$-NN)
- Linear Discriminant Analysis (LDA)
- Quadratic Discriminant Analysis (QDA)
- Support Vector Machines (SVM) — linear and RBF kernels
- Logistic Regression
- Data augmentation (rotation, translation, scaling)
- PCA dimensionality reduction

All binary classifiers use a one-vs-all multi-class strategy. Parameters are selected via cross-validation on the training set only.

## Result Management

All scripts write outputs to `results/<git-short-hash>/<timestamp>/` with subdirectories:

- `figures/` — plots and visualizations
- `logs/` — `results.json`, `summary.txt`, `challenge_results.txt`
- `run.log` — captured console output

Run `scripts/run_all.py` to execute all scripts into a single shared directory.

## Development

```bash
# Lint and format
uv run ruff check src tests scripts
uv run ruff format src tests scripts

# Type check
uv run mypy src

# Launch Jupyter
uv run jupyter notebook notebooks/
```

## Optional: Upper Bound Evaluation

The project includes an optional upper-bound script that evaluates a CNN trained on full MNIST (60K) or a local vision LLM (Qwen2-VL-2B). This is not part of the core coursework.

```bash
# Install optional dependencies
uv sync --extra upperbound

# Run CNN upper bound
uv run python scripts/evaluate_upper_bound.py --backend cnn
```
