# CS5487 Machine Learning - Project Proposal

**Team Members:**  
Yikai LAI (赖奕恺) - 5714XXXX  
Zhang Yao - 5714XXXX

**Course:** CS5487 Machine Learning  
**Submission Date:** Week 11

---

## 1. Introduction

### 1.1 Problem Background

Handwritten digit recognition is a classic and fundamental problem in machine learning and computer vision, with wide-ranging practical applications including:
- **Postal services**: Automated mail sorting and ZIP code recognition
- **Banking**: Check processing and amount verification
- **Document digitization**: Converting scanned documents to editable text
- **Form processing**: Automated data entry from handwritten forms

Despite being a well-studied problem, digit classification remains challenging due to variations in writing styles, stroke thickness, orientation, and noise in the input data.

### 1.2 Project Objectives

This project aims to implement, evaluate, and compare various machine learning algorithms for handwritten digit classification on a subset of the MNIST dataset. Our specific goals are:

1. **Establish a strong baseline** using 1-Nearest Neighbor (1-NN)
2. **Implement multiple classifiers** including SVM, Logistic Regression, LDA, and QDA
3. **Beat the baseline accuracy** of 91.6% through algorithm selection and optimization
4. **Investigate dimensionality reduction** using PCA to improve performance and efficiency
5. **Conduct thorough error analysis** to understand model limitations
6. **Compete in the bonus challenge** (if time permits)

### 1.3 Dataset Overview

We use a curated subset of the MNIST dataset provided by the course:

| Property | Value |
|----------|-------|
| Total samples | 4,000 images |
| Classes | 10 (digits 0-9) |
| Features | 784 (28×28 grayscale pixels) |
| Pixel values | 0-255 (8-bit grayscale) |
| Class distribution | Balanced (400 samples per class) |
| Train/Test splits | 2 predefined 50/50 splits |

The two predefined splits use different sets of writers, allowing us to evaluate model generalization across different handwriting styles.

---

## 2. Proposed Methodology

### 2.1 Preprocessing Strategies

We will evaluate three normalization approaches:

| Method | Description | Rationale |
|--------|-------------|-----------|
| **Standard Scaling** | Zero mean, unit variance per feature | Gaussian assumption for LDA/QDA |
| **Min-Max Scaling** | Scale to [0, 1] range | Preserves relative distances |
| **Divide by 255** | Simple pixel normalization | Natural for image data [0,1] range |

### 2.2 Dimensionality Reduction

**Principal Component Analysis (PCA)** will be explored to:
- Reduce computational complexity
- Remove noise from high-dimensional pixel space
- Improve generalization by focusing on principal variations

We will experiment with PCA components: 30, 40, 50, 60, 80, and 100.

### 2.3 Algorithms to Implement

| Algorithm | Type | Multi-class Strategy | Key Parameters |
|-----------|------|---------------------|----------------|
| **K-NN** | Instance-based | Direct multi-class | k ∈ {1, 3, 5} |
| **SVM (Linear)** | Discriminative | One-vs-All | C ∈ {0.1, 1.0, 10.0} |
| **SVM (RBF)** | Discriminative | One-vs-All | C, γ (kernel width) |
| **SVM (Poly)** | Discriminative | One-vs-All | C, degree=3 |
| **Logistic Regression** | Probabilistic | One-vs-All | C, solver |
| **LDA** | Generative | Direct | - |
| **QDA** | Generative | Direct | Regularization |

#### 2.3.1 One-vs-All Strategy

For binary classifiers (SVM, Logistic Regression), we will implement a One-vs-All wrapper:
- Train 10 binary classifiers (one per digit)
- For digit i: positive class = i, negative class = all others
- Prediction: argmax of decision function scores across all classifiers

### 2.4 Evaluation Methodology

**Primary Metric**: Classification accuracy

**Validation Protocol**:
- Run experiments on both predefined train/test splits
- Report mean accuracy ± standard deviation across splits
- Statistical comparison against 1-NN baseline (91.6% ± 0.35%)

**Error Analysis**:
- Confusion matrices for all classifiers
- Per-digit accuracy analysis
- Identification of most confused digit pairs
- Sample-level error categorization (easy/medium/hard)

---

## 3. Expected Outcomes

### 3.1 Minimum Goals (Must Achieve)

- [x] Implement 1-NN baseline and verify correctness (~91.6%)
- [x] Implement at least 3 additional classifiers
- [x] Achieve accuracy > 92% on test set
- [x] Complete error analysis and confusion matrices

### 3.2 Stretch Goals (Target)

- [x] Achieve accuracy > 95% on test set
- [x] Extensive parameter tuning for SVM (C, γ)
- [x] PCA analysis to find optimal number of components
- [ ] Kernel PCA or other non-linear dimensionality reduction
- [ ] Compete in bonus challenge (classify professor's handwritten digits)

### 3.3 Anticipated Results

Based on literature and preliminary experiments, we expect:

| Classifier | Expected Accuracy | Notes |
|------------|-------------------|-------|
| 1-NN | 91-92% | Baseline, no training needed |
| K-NN (k=3) | 91-93% | Smoother decision boundaries |
| Linear SVM | 85-88% | May be too restrictive |
| SVM RBF | 93-96% | Non-linear boundaries should help |
| Logistic Regression | 85-88% | Linear model limitation |
| LDA | 75-80% | Shared covariance assumption |
| QDA | 92-95% | Quadratic boundaries, but may overfit |
| SVM RBF + PCA | 94-96% | Noise reduction + non-linear boundaries |

---

## 4. Division of Labor

| Member | Primary Responsibilities |
|--------|-------------------------|
| **Yikai LAI** | • SVM implementations (Linear, RBF, Poly)<br>• PCA experiments and analysis<br>• Error analysis and sample-level investigation<br>• Report writing and documentation |
| **Zhang Yao** | • Logistic Regression implementation<br>• LDA and QDA implementations<br>• Data visualization and figures<br>• Parameter tuning experiments |
| **Both** | • K-NN baseline implementation<br>• Integration testing<br>• Presentation preparation<br>• Results validation |

---

## 5. Project Timeline

| Week | Milestone | Deliverables |
|------|-----------|--------------|
| **Week 10** | Proposal Submission | This proposal document |
| **Week 11** | Baseline + Initial Classifiers | • 1-NN working<br>• 2 additional classifiers<br>• Initial results |
| **Week 12** | Full Implementation | • All classifiers implemented<br>• Parameter tuning begun<br>• PCA experiments |
| **Week 13** | Analysis & Optimization | • Complete error analysis<br>• Final parameter tuning<br>• Best model selected |
| **Week 14** | Final Deliverables | • Final report<br>• Poster<br>• Presentation (Required for A grade) |

---

## 6. Technical Requirements

### 6.1 Software Stack

- **Python 3.12** as primary language
- **scikit-learn** for classifier implementations
- **NumPy/SciPy** for numerical computations
- **Matplotlib/Seaborn** for visualization
- **MATLAB** for dataset loading (provided format)

### 6.2 Development Environment

- Git version control for collaboration
- Virtual environment (uv) for dependency management
- Jupyter notebooks for exploratory analysis
- Automated experiment logging to JSON

---

## 7. Risk Assessment and Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| QDA overfitting on 784D | High | Medium | Use PCA or regularization |
| SVM training time too long | Medium | Low | Use PCA to reduce dimensions |
| Poor multi-class handling | Medium | High | Implement One-vs-All wrapper |
| Dataset loading issues | Low | Medium | Use scipy.io for MATLAB files |
| Convergence warnings | Medium | Low | Increase max_iter, tune parameters |

---

## 8. Preliminary Results (if available)

*To be completed after Week 11 experiments*

---

## 9. References

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324.

2. Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273-297.

3. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.

4. Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.

5. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

6. Course lecture notes: CS5487 Machine Learning, City University of Hong Kong (Weeks 1-10).

7. LIBSVM Documentation: https://www.csie.ntu.edu.tw/~cjlin/libsvm/

8. LIBLINEAR Documentation: https://www.csie.ntu.edu.tw/~cjlin/liblinear/

---

## Appendix A: Algorithm Justification

### Why These Algorithms?

**K-NN**: Simple baseline, no training required, captures local similarity

**SVM**: State-of-the-art for many classification tasks, effective in high dimensions with proper kernel selection

**Logistic Regression**: Fast, interpretable baseline for linear separability assessment

**LDA/QDA**: Generative approach, theoretically motivated, QDA offers non-linear boundaries without kernels

**PCA**: Dimensionality reduction is crucial for 784D data; reduces noise and computational cost

---

## Appendix B: Evaluation Checklist

### Technical Correctness (5 points)
- [ ] Correct implementation of One-vs-All strategy
- [ ] Proper train/test separation (no data leakage)
- [ ] Consistent evaluation across both splits
- [ ] Correct handling of PCA (fit on train, transform both)

### Experiments (5 points)
- [ ] Multiple classifiers compared
- [ ] Parameter tuning performed
- [ ] PCA experiments conducted
- [ ] Error analysis with confusion matrices

### Analysis (5 points)
- [ ] Insights on why certain methods work better
- [ ] Identification of failure cases
- [ ] Statistical significance of results
- [ ] Discussion of limitations

### Report Quality (5 points)
- [ ] Clear structure and formatting
- [ ] Appropriate figures and tables
- [ ] Proper citations
- [ ] Professional presentation

### Presentation (5 points)
- [ ] Clear slides
- [ ] Live demonstration (if possible)
- [ ] Q&A preparation
- [ ] Time management
