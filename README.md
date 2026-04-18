# Machine Learning Course Project - CS5487

**Team:** Yikai LAI (赖奕恺), Zhang Yao
**Course:** CS5487 Machine Learning
**Project:** Default Project - Digit Classification

---

## Project Overview

Handwritten digit classification on a 4,000-sample MNIST subset using multiple machine learning techniques. The project evaluates classifiers across two predefined train/test splits (different writers) and tests generalization on a held-out challenge dataset of 150 professor-written digits.

### Dataset
- **Source:** MNIST digits subset
- **Size:** 4,000 images (400 per class, digits 0-9)
- **Features:** 784 dimensions (28×28 grayscale images, values 0-255)
- **Splits:** 2 predefined train/test splits (50/50, 2,000 samples each, different writers)
- **Location:** `data/raw/MINIST/digits4000.mat`

### Baseline
- 1-NN with Euclidean distance: **91.60% ± 0.35%**

### Best Results
- **MNIST test:** SVM-RBF + PCA50 = **94.98% ± 0.28%**
- **Challenge dataset:** SVM-RBF + PCA50 = **71.67% ± 1.67%** (vs 68.33% baseline)

---

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Run all experiments and analysis (recommended)
uv run python scripts/run_all.py

# 3. Or run individual scripts
uv run python scripts/run_experiments.py
uv run python scripts/evaluate_challenge.py
uv run python scripts/error_analysis.py
uv run python scripts/sample_error_analysis.py
uv run python scripts/class_distribution.py
```

---

## Project Structure

```
MachineLearning/
├── data/
│   └── raw/
│       ├── MINIST/
│       │   └── digits4000.mat          # MNIST subset (download separately)
│       └── challenge/
│           ├── cdigits.mat             # Challenge dataset (150 samples)
│           └── cdigits_digits_labels.txt
├── docs/
│   ├── requirement/
│   │   └── PA-3-courseproject.pdf      # Course project requirements
│   └── report/
│       ├── report.tex                  # Final report (LaTeX)
│       └── report.pdf                  # Compiled report
├── notebooks/
│   └── 01_exploration.ipynb            # Data exploration
├── scripts/                            # Executable analysis scripts
│   ├── run_all.py                      # Run all scripts in one directory
│   ├── run_experiments.py              # Main experiment suite (9 classifiers)
│   ├── evaluate_challenge.py           # Evaluate on challenge dataset
│   ├── error_analysis.py               # Confusion matrices & per-digit accuracy
│   ├── sample_error_analysis.py        # Hard/medium/easy sample analysis
│   └── class_distribution.py           # Class distribution visualization
├── src/
│   ├── __init__.py
│   ├── data_loader.py                  # Load MNIST, train/test splits, normalization
│   ├── classifiers.py                  # Classifier factory, OvA, PCA transform
│   ├── experiments.py                  # Experiment runner
│   ├── utils.py                        # Result directory management, logging
│   ├── preprocessing.py                # Preprocessing utilities
│   └── features.py                     # Feature engineering
├── results/                            # Auto-generated result directories
│   └── <git-hash>/
│       └── <timestamp>/
│           ├── figures/                # All plots and visualizations
│           ├── logs/                   # results.json, summary.txt, challenge_results.txt
│           └── run.log                 # Full captured session output
├── tests/                              # Unit tests
├── pyproject.toml                      # Dependencies and tool config
└── README.md                           # This file
```

---

## Classifiers Implemented

| Classifier | Description | Best Config |
|------------|-------------|-------------|
| `knn` | K-Nearest Neighbors | k=1, no preprocessing |
| `svm_linear` | Linear SVM (OvA) | C=1.0, scale to [0,1] |
| `svm_rbf` | SVM with RBF kernel (OvA) | C=1.0, gamma=scale, scale to [0,1] |
| `svm_poly` | SVM with Polynomial kernel (OvA) | Available, not in main experiments |
| `logistic` | Logistic Regression (OvA) | C=1.0, standardize |
| `lda` | Linear Discriminant Analysis | No preprocessing |
| `qda` | Quadratic Discriminant Analysis | PCA to 50 components |

---

## Result Management

All scripts write outputs to a unified directory structure: `results/<git-short-hash>/<timestamp>/`.

### Run everything at once (recommended)

```bash
uv run python scripts/run_all.py
```

This creates one shared directory and runs all 5 scripts sequentially, capturing output in a single `run.log`.

### Run with a custom name

```bash
uv run python scripts/run_all.py --run-name "final-experiments"
```

### Run individual scripts standalone

```bash
# Auto-generates a new timestamp directory
uv run python scripts/run_experiments.py

# Write to an existing directory
uv run python scripts/evaluate_challenge.py --run-dir results/b875b6b/20260418_205949
```

---

## Usage Examples

### Basic Experiment

```python
from src.experiments import Experiment

exp = Experiment("data/raw/MINIST/digits4000.mat")

# Run SVM with RBF kernel
result = exp.run_experiment(
    classifier_name='svm_rbf',
    classifier_params={'C': 1.0, 'gamma': 'scale'},
    normalize='scale255',
    use_one_vs_all=True
)

print(f"Accuracy: {result['mean_acc']:.4f} ± {result['std_acc']:.4f}")
```

### With PCA

```python
result = exp.run_experiment(
    classifier_name='svm_rbf',
    classifier_params={'C': 1.0, 'gamma': 'scale'},
    normalize='scale255',
    pca_components=50,
    use_one_vs_all=True
)
```

### Challenge Evaluation

```python
from scripts.evaluate_challenge import main
main(output_dir=Path("results/my-run"))
```

---

## Experiment Results Summary

### MNIST Subset (4,000 samples)

| Method | Trial 1 | Trial 2 | Mean ± Std |
|--------|---------|---------|------------|
| 1-NN Baseline | 0.9135 | 0.9185 | 0.9160 ± 0.0035 |
| 3-NN | 0.9180 | 0.9060 | 0.9120 ± 0.0060 |
| 5-NN | 0.9170 | 0.9095 | 0.9133 ± 0.0038 |
| LDA | 0.7845 | 0.7795 | 0.7820 ± 0.0025 |
| QDA + PCA50 | 0.9315 | 0.9440 | 0.9378 ± 0.0063 |
| SVM-Linear | 0.8600 | 0.8580 | 0.8590 ± 0.0010 |
| SVM-RBF | 0.9395 | 0.9335 | 0.9365 ± 0.0030 |
| **SVM-RBF + PCA50** | **0.9525** | **0.9470** | **0.9498 ± 0.0028** |
| Logistic Regression | 0.8680 | 0.8640 | 0.8660 ± 0.0020 |

### Challenge Dataset (150 samples)

| Method | Trial 1 | Trial 2 | Mean ± Std |
|--------|---------|---------|------------|
| 1-NN Baseline | 0.6600 | 0.7067 | 0.6833 ± 0.0233 |
| QDA + PCA50 | 0.7000 | 0.7267 | 0.7133 ± 0.0133 |
| SVM-RBF | 0.6867 | 0.7000 | 0.6933 ± 0.0067 |
| **SVM-RBF + PCA50** | **0.7000** | **0.7333** | **0.7167 ± 0.0167** |

---

## Error Analysis Highlights

- **Hard samples** (all models wrong): 60/4000 (1.5%) — likely ambiguous or mislabeled
- **Medium samples** (some models wrong): 563/4000 (14.1%) — model capability differences matter
- **Easy samples** (all models correct): 3377/4000 (84.4%)
- Most confused pairs: 5→3, 4→9, 9→7, 5→6, 9→4

---

## Development

```bash
# Run tests
uv run pytest

# Lint and format
uv run ruff check src tests scripts
uv run ruff format src tests scripts

# Type check
uv run mypy src

# Launch Jupyter
uv run jupyter notebook notebooks/
```

---

## Notes

- All classifiers use the one-vs-all strategy for multi-class classification
- Parameters are tuned using training set only (no peeking at test set)
- PCA is fit on training data and applied to both train and test
- Challenge evaluation uses models trained on MNIST training splits without retraining
- Group of 2 — individual contributions documented in report
