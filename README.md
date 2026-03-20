# Machine Learning Course Project - CS5487

**Team:** Yikai LAI (赖奕恺), Zhang Yao  
**Course:** CS5487 Machine Learning  
**Project:** Default Project - Digit Classification

---

## Project Overview

Handwritten digit classification on MNIST subset using various machine learning techniques.

### Dataset
- **Source:** MNIST digits subset
- **Size:** 4000 images (400 per class, digits 0-9)
- **Features:** 784 dimensions (28×28 grayscale images, values 0-255)
- **Splits:** 2 predefined train/test splits (50/50, different writers)

### Baseline
- 1-NN with Euclidean distance: **91.60% ± 0.35%**

### Goals
1. ✅ Implement and compare multiple classifiers
2. 🎯 Beat the baseline accuracy
3. 🔬 Experiment with preprocessing (PCA, normalization)
4. 🏆 **Bonus:** Classify professor's handwritten digits

---

## Quick Start

```bash
# 1. Download dataset (digits4000.mat) and place in data/raw/

# 2. Install dependencies
cd master/MachineLearning
uv sync

# 3. Run baseline experiments
uv run python run_experiments.py

# 4. Launch Jupyter for exploration
uv run jupyter notebook notebooks/
```

---

## Project Structure

```
MachineLearning/
├── data/
│   └── raw/
│       └── digits4000.mat          # Dataset (download separately)
├── notebooks/
│   └── 01_exploration.ipynb        # Data exploration
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Load MNIST data
│   ├── classifiers.py              # SVM, LR, KNN, etc.
│   ├── experiments.py              # Experiment runner
│   ├── preprocessing.py            # Data preprocessing
│   └── features.py                 # Feature engineering
├── experiments/
│   └── logs/                       # Experiment results
├── run_experiments.py              # Main script
└── pyproject.toml                  # Dependencies
```

---

## Classifiers Implemented

| Classifier | Description |
|------------|-------------|
| `knn` | K-Nearest Neighbors |
| `svm_linear` | Linear SVM |
| `svm_rbf` | SVM with RBF kernel |
| `svm_poly` | SVM with Polynomial kernel |
| `logistic` | Logistic Regression |
| `lda` | Linear Discriminant Analysis |
| `qda` | Quadratic Discriminant Analysis |

---

## Usage Examples

### Basic Experiment

```python
from src.experiments import Experiment

# Load data
exp = Experiment("data/raw/digits4000.mat")

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
    classifier_name='logistic',
    classifier_params={'C': 1.0},
    normalize='standard',
    pca_components=50,  # Reduce to 50 dimensions
    use_one_vs_all=True
)
```

### Grid Search

```python
from src.experiments import grid_search

param_grid = [
    {'classifier_name': 'svm_rbf', 'classifier_params': {'C': 0.1, 'gamma': 'scale'}},
    {'classifier_name': 'svm_rbf', 'classifier_params': {'C': 1.0, 'gamma': 'scale'}},
    {'classifier_name': 'svm_rbf', 'classifier_params': {'C': 10.0, 'gamma': 'scale'}},
]

results = grid_search(exp, param_grid)
```

---

## Experiment Checklist

### Phase 1: Baseline & Exploration
- [x] Project setup
- [ ] Download digits4000.mat
- [ ] Implement 1-NN baseline (verify ~91.6%)
- [ ] Data visualization (sample digits)
- [ ] Analyze class distribution

### Phase 2: Classifiers
- [ ] Linear SVM (1-vs-all)
- [ ] RBF SVM with different C, gamma
- [ ] Logistic Regression
- [ ] LDA/QDA
- [ ] KNN with different k

### Phase 3: Preprocessing & Features
- [ ] Normalization (standard, minmax, /255)
- [ ] PCA dimensionality reduction
- [ ] Kernel PCA
- [ ] Feature selection

### Phase 4: Analysis
- [ ] Compare all methods
- [ ] Parameter sensitivity analysis
- [ ] Confusion matrices
- [ ] Error analysis (success/failure cases)
- [ ] **Bonus challenge preparation**

### Phase 5: Report & Presentation
- [ ] Project proposal (Due: Week 10)
- [ ] Final report (Due: Week 14)
- [ ] Poster (Due: Week 14)
- [ ] Presentation (Week 14) - **Required for A grade**

---

## Grading Breakdown (30 points total)

| Component | Points | Weight |
|-----------|--------|--------|
| Project proposal | 5 | 16.7% |
| Technical correctness | 5 | 16.7% |
| Experiments (thoroughness) | 5 | 16.7% |
| Analysis (insights) | 5 | 16.7% |
| Report quality | 5 | 16.7% |
| Presentation | 5 | 16.7% |

---

## Timeline

| Week | Task |
|------|------|
| Week 10 | Proposal due (Friday) |
| Week 11-13 | Implementation & experiments |
| Week 14 | Presentation (TBA), Report due (Friday) |

---

## Resources

- Course slides (SVM, Logistic Regression, PCA, etc.)
- [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
- [LIBLINEAR](https://www.csie.ntu.edu.tw/~cjlin/liblinear/)
- sklearn documentation

---

## Notes

- Must use one-vs-all strategy for binary classifiers on multi-class problem
- Can only tune parameters using training set (no peeking at test set!)
- Can use 3rd party code with proper citation
- Group of 2 - document individual contributions
