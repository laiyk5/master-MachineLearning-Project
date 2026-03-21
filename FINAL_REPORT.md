# CS5487 Machine Learning - Course Project Report

**Digit Classification on MNIST Subset**

**Team Members:**  
Yikai LAI (赖奕恺) - 5714XXXX  
Zhang Yao - 5714XXXX

**Course:** CS5487 Machine Learning  
**Date:** March 2026

---

## 1. Introduction

Handwritten digit recognition is a fundamental problem in machine learning with widespread applications in postal services, banking (check processing), document digitization, and optical character recognition (OCR) systems. The task involves classifying grayscale images of handwritten digits (0-9) into their corresponding numerical categories.

This project implements and compares various classification algorithms on a subset of the MNIST dataset containing 4,000 images. The primary objectives are:

1. **Implement multiple classifiers** and establish a performance baseline
2. **Beat the 1-NN baseline** accuracy of 91.6%
3. **Analyze error patterns** to understand model strengths and weaknesses
4. **Optimize performance** through parameter tuning and dimensionality reduction

### 1.1 Problem Statement

Given a 784-dimensional feature vector representing a 28×28 grayscale image, classify the digit into one of 10 classes (0-9). The dataset provides two predefined train/test splits to ensure fair evaluation across different data distributions.

---

## 2. Dataset Description

### 2.1 Data Characteristics

| Property | Value |
|----------|-------|
| **Total Samples** | 4,000 images |
| **Classes** | 10 (digits 0-9) |
| **Features** | 784 (28×28 grayscale pixels, values 0-255) |
| **Class Distribution** | Balanced: 400 samples per class |
| **Train/Test Split** | 50/50 (2,000 training, 2,000 testing) |
| **Splits** | 2 predefined trials with different writers |

### 2.2 Data Visualization

The dataset contains balanced representations of all digit classes. Each image is stored as a column vector in MATLAB format, requiring reshape operations for visualization. Sample images show typical handwritten digit variations including different stroke styles, sizes, and orientations.

![Class Distribution](results/figures/class_distribution.png)
*Figure 1: Dataset class distribution showing perfect balance (400 samples per digit) and train/test split consistency across both trials.*

### 2.3 Preprocessing

Three normalization strategies were evaluated:
- **Standard scaling**: Zero mean, unit variance per feature
- **Min-max scaling**: Scale to [0, 1] range
- **Divide by 255**: Simple pixel value normalization

The optimal preprocessing varied by classifier (see Section 4).

---

## 3. Methodology

### 3.1 Algorithms Implemented

| Algorithm | Type | Multi-class Strategy | Key Parameters |
|-----------|------|---------------------|----------------|
| **K-NN** | Instance-based | Direct | k ∈ {1, 3, 5}, distance metric |
| **SVM (Linear)** | Discriminative | One-vs-All | C (regularization) |
| **SVM (RBF)** | Discriminative | One-vs-All | C, γ (kernel width) |
| **SVM (Poly)** | Discriminative | One-vs-All | C, degree=3 |
| **Logistic Regression** | Probabilistic | One-vs-All | C, solver |
| **LDA** | Generative | Direct | - |
| **QDA** | Generative | Direct | solver='svd' |

### 3.2 One-vs-All Strategy

For binary classifiers (SVM, Logistic Regression), we implemented a One-vs-All (OvA) wrapper:
- Train 10 binary classifiers (one per digit)
- Each classifier distinguishes one digit from all others
- Prediction: select class with highest decision function score

### 3.3 Dimensionality Reduction

Principal Component Analysis (PCA) was applied to reduce the 784-dimensional feature space:
- Explored components: 30, 40, 50, 60, 80, 100
- Optimal value found: **50 components** (explains ~83% variance)
- Benefits: Reduced noise, faster training, often improved generalization

### 3.4 Evaluation Metrics

- **Primary metric**: Classification accuracy
- **Validation**: 2 predefined trials, report mean ± std
- **Statistical significance**: Compare against 1-NN baseline (91.6% ± 0.35%)

---

## 4. Experimental Results

### 4.1 Summary of All Classifiers

| Classifier | Preprocessing | PCA | Trial 1 | Trial 2 | **Mean ± Std** |
|------------|--------------|-----|---------|---------|----------------|
| K-NN (k=1) | None | No | 91.35% | 91.85% | **91.60% ± 0.35%** |
| K-NN (k=3) | None | No | 91.80% | 90.60% | **91.20% ± 0.85%** |
| K-NN (k=5) | None | No | 91.70% | 90.95% | **91.33% ± 0.53%** |
| **SVM RBF** | **Scale/255** | **50** | **95.20%** | **94.75%** | **94.98% ± 0.32%** |
| SVM RBF | Scale/255 | No | 93.95% | 93.35% | 93.65% ± 0.42% |
| SVM Linear | Scale/255 | No | 86.00% | 85.80% | 85.90% ± 0.14% |
| QDA | None | 50 | 93.20% | 94.45% | 93.83% ± 0.88% |
| Logistic | Standard | No | 86.80% | 86.40% | 86.60% ± 0.28% |
| LDA | None | No | 78.45% | 77.95% | 78.20% ± 0.35% |

### 4.2 Best Result

**SVM with RBF kernel + PCA (50 components)** achieved the highest accuracy:
- **Accuracy: 94.98% ± 0.32%**
- Improvement over baseline: **+3.38 percentage points**
- Parameters: C=1.0, γ='scale', normalization='scale255'

### 4.3 Per-Digit Accuracy Analysis

The SVM RBF + PCA model showed relatively balanced performance across digits:

| Digit | Accuracy | Notes |
|-------|----------|-------|
| 0 | 97.75% | Well-separated class |
| 1 | 98.75% | Highest accuracy - distinct features |
| 2 | 93.75% | Occasionally confused with 7 |
| 3 | 93.00% | Sometimes confused with 2, 7, 9 |
| 4 | 95.25% | Occasionally confused with 9 |
| 5 | 92.50% | Most challenging - confused with 3, 6 |
| 6 | 97.25% | Well-separated class |
| 7 | 95.50% | Occasionally confused with 2 |
| 8 | 93.25% | Challenging - confused with multiple digits |
| 9 | 92.50% | Occasionally confused with 4, 7 |

![Per-Digit Accuracy](results/figures/per_digit_accuracy.png)
*Figure 3: Per-digit accuracy comparison across all classifiers. SVM RBF with PCA (red bars) consistently achieves high accuracy across all digits, while Logistic Regression (brown bars) struggles particularly with digits 5 and 8.*

### 4.4 Confusion Matrix Analysis

**Top Confused Pairs** (SVM RBF + PCA):
1. **5 → 3**: 11 errors (2.75%) - Similar curved top
2. **4 → 9**: 9 errors (2.25%) - Similar closed-loop shape
3. **9 → 4**: 9 errors (2.25%) - Reciprocal confusion
4. **9 → 7**: 9 errors (2.25%) - Similar top structure
5. **5 → 6**: 8 errors (2.00%) - Bottom loop similarity

![Confusion Matrices Comparison](results/figures/confusion_comparison.png)
*Figure 2: Confusion matrices for all classifiers. SVM RBF + PCA (bottom-left) shows the cleanest diagonal with minimal off-diagonal errors, while Logistic Regression (bottom-right) shows significant confusion patterns.*

---

## 5. Error Analysis

### 5.1 Sample-Level Error Categorization

To understand whether errors stem from data quality or model limitations, we analyzed all 4,000 test samples across 5 different models:

| Category | Count | Percentage | Interpretation |
|----------|-------|------------|----------------|
| **Easy samples** | 3,379 | 84.5% | All models correct - clear, well-written digits |
| **Medium samples** | 561 | 14.0% | Some models correct, some wrong - model capability issue |
| **Hard samples** | 60 | 1.5% | All models wrong - ambiguous or mislabeled samples |

### 5.2 Hard Samples (Ambiguous Data)

60 samples (1.5%) were misclassified by all 5 models. Visual inspection reveals:
- **Genuinely ambiguous handwriting**: Poorly formed, stylized, or rotated digits
- **Potential label errors**: Some samples appear to be incorrectly labeled in the dataset
- **Borderline cases**: Digits that even humans might struggle to classify

**Examples of hard cases:**
- Sample #2045 (True=0): Predicted as 5 or 8 - written with unusual curl
- Sample #2643 (True=3): Predicted as 2 - flat top resembles digit 2
- Sample #2988 (True=4): Predicted as 9 - closed top resembles 9

![Hard Samples](results/figures/hard_samples.png)
*Figure 4: Examples of "hard" samples that all 5 models misclassified. Left column shows the digit image; right column shows prediction distribution across models. These samples represent genuinely ambiguous or potentially mislabeled data.*

### 5.3 Medium Samples (Model Improvement Opportunity)

561 samples showed disagreement across models, indicating potential for improvement:
- **460 samples**: Models agree on the wrong prediction (systematic confusion)
- **101 samples**: Models disagree on predictions (different model biases)

These represent cases where better models (e.g., deep learning) could potentially improve accuracy.

### 5.4 Model-Specific Error Patterns

**K-NN Errors:**
- Struggles with digits 4, 8, 9
- Confuses 4→9 most frequently (8.5% of digit 4 errors)
- Distance-based approach doesn't capture stroke structure

**Logistic Regression Errors:**
- Linear decision boundaries insufficient for this task
- Poor performance on digits 5 and 8 (74-77% accuracy)
- Confuses 5↔8, 4↔9 pairs

**LDA/QDA Errors:**
- LDA significantly underperforms (78%) - assumes shared covariance
- QDA with PCA achieves 94% - quadratic boundaries better fit the data

---

## 6. Discussion

### 6.1 Key Findings

1. **SVM with RBF kernel significantly outperforms other methods**, achieving 95% accuracy vs. the 91.6% baseline.

2. **PCA improves SVM performance** (93.7% → 95.0%), likely by:
   - Reducing noise in high-dimensional pixel space
   - Focusing on discriminative principal components
   - Mitigating the curse of dimensionality

3. **The dataset has a ceiling around 98.5%**: 1.5% of samples are inherently ambiguous or mislabeled.

4. **Different models have complementary strengths**: The 14% "medium" samples represent opportunities for ensemble methods.

### 6.2 Why SVM RBF Works Best

The RBF kernel implicitly maps data to an infinite-dimensional space, enabling:
- Non-linear decision boundaries
- Local similarity matching (like K-NN but with learned support vectors)
- Robustness to high-dimensional data when combined with PCA

### 6.3 Why Other Methods Fall Short

| Method | Limitation |
|--------|------------|
| K-NN | Treats all pixels equally; no feature learning |
| Linear SVM | Cannot capture non-linear stroke patterns |
| Logistic Regression | Linear boundaries insufficient for digit shapes |
| LDA | Shared covariance assumption violated |
| QDA (no PCA) | Overfits to 784 dimensions |

### 6.4 Recommendations for Further Improvement

1. **Parameter tuning**: Grid search over C and γ for SVM
2. **Feature engineering**: Extract stroke-based features (HOG, SIFT)
3. **Ensemble methods**: Combine SVM, QDA, and K-NN predictions
4. **Data augmentation**: Rotation, scaling, elastic deformation
5. **Neural networks**: CNNs typically achieve >99% on full MNIST

---

## 7. Conclusion

This project successfully implemented and compared multiple classification algorithms for handwritten digit recognition. The key achievements are:

✅ **Beat the baseline**: SVM RBF + PCA achieved 95.0%, exceeding the 91.6% 1-NN baseline by 3.4 percentage points.

✅ **Comprehensive analysis**: Evaluated 7 different classifiers with various preprocessing and dimensionality reduction strategies.

✅ **Error understanding**: Identified 1.5% hard samples (data quality limit) and 14% medium samples (model improvement opportunity).

✅ **Insights gained**: 
- PCA with 50 components is optimal for this dataset
- SVM RBF benefits most from dimensionality reduction
- Digit 5 and 8 are most challenging across all models
- The 4↔9 and 5↔3 confusion pairs are most common

The project demonstrates the importance of algorithm selection, preprocessing, and thorough error analysis in machine learning workflows.

---

## 8. References

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324.

2. Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273-297.

3. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

4. Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.

5. Course lecture notes: CS5487 Machine Learning, City University of Hong Kong.

---

## Appendix A: Division of Labor

| Member | Contribution |
|--------|-------------|
| **Yikai LAI** | SVM implementations, PCA experiments, error analysis, report writing |
| **Zhang Yao** | Logistic Regression, LDA/QDA, data visualization, parameter tuning |
| **Both** | K-NN baseline, integration testing, presentation preparation |

## Appendix B: Code Repository Structure

```
ML-Project-Code/
├── src/
│   ├── classifiers.py      # Classifier implementations
│   ├── data_loader.py      # Data loading and preprocessing
│   ├── experiments.py      # Experiment runner
│   └── features.py         # Feature engineering
├── notebooks/              # Jupyter notebooks for exploration
├── experiments/logs/       # Experiment results (JSON)
├── results/figures/        # Generated visualizations
├── run_experiments.py      # Main experiment script
├── error_analysis.py       # Error analysis tools
├── class_distribution.py   # Data analysis
└── sample_error_analysis.py # Sample-level error analysis
```

## Appendix C: List of Figures

All figures are available in `results/figures/`:

| Figure | Filename | Description |
|--------|----------|-------------|
| **Figure 1** | `class_distribution.png` | Dataset balance and train/test split visualization |
| **Figure 2** | `confusion_comparison.png` | Side-by-side confusion matrices for all classifiers |
| **Figure 3** | `per_digit_accuracy.png` | Accuracy comparison across digits and models |
| **Figure 4** | `hard_samples.png` | Examples of ambiguous samples misclassified by all models |
| Figure 5 | `medium_samples.png` | Samples with model disagreement (improvement opportunity) |
| Figure 6 | `confusion_matrix_knn_pcanone.png` | Individual confusion matrix for K-NN |
| Figure 7 | `confusion_matrix_svm_rbf_pca50.png` | Individual confusion matrix for SVM RBF + PCA |
| Figure 8 | `confusion_matrix_logistic_pcanone.png` | Individual confusion matrix for Logistic Regression |
| Figure 9 | `confused_pairs_knn_pcanone.png` | Top confused pairs for K-NN |
| Figure 10 | `confused_pairs_svm_rbf_pca50.png` | Top confused pairs for SVM RBF + PCA |
