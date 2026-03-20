# Project Proposal - CS5487 Course Project

**Team Members:** Yikai LAI (赖奕恺), Zhang Yao  
**Project:** Default Project - Digit Classification  
**Date:** Week 10

---

## 1. Introduction

Handwritten digit recognition is a classic machine learning problem with applications in postal services, banking (check processing), and document digitization. We will implement and compare various classification algorithms on a subset of the MNIST dataset containing 4000 images of digits 0-9.

**Goal:** Implement multiple classifiers, beat the 1-NN baseline (91.6%), and compete in the bonus challenge.

---

## 2. Proposed Methodology

### 2.1 Dataset
- **Source:** MNIST subset (provided)
- **Size:** 4000 images, 400 per class
- **Features:** 784D (28×28 grayscale)
- **Train/Test:** 2 predefined splits (50/50)

### 2.2 Preprocessing
- Normalization: Standard scaling, min-max, divide by 255
- Dimensionality reduction: PCA to reduce from 784D

### 2.3 Algorithms to Implement

| Algorithm | Strategy | Parameters to Tune |
|-----------|----------|-------------------|
| K-NN | Direct multi-class | k=1,3,5, distance metrics |
| SVM (Linear) | One-vs-all | C |
| SVM (RBF) | One-vs-all | C, gamma |
| Logistic Regression | One-vs-all | C, solver |
| LDA | Direct multi-class | - |
| QDA | Direct multi-class | - |

### 2.4 Evaluation
- **Metric:** Classification accuracy
- **Validation:** 2 predefined trials, report mean ± std
- **Baseline to beat:** 1-NN = 91.60% ± 0.35%

---

## 3. Expected Outcomes

### Minimum Goals
- Implement 1-NN baseline (verify correctness)
- Implement at least 3 other classifiers
- Achieve > 92% accuracy

### Stretch Goals
- Achieve > 95% accuracy
- Extensive parameter tuning
- PCA analysis (optimal dimensions)
- Compete in bonus challenge

---

## 4. Division of Labor

| Member | Responsibilities |
|--------|------------------|
| Yikai LAI | SVM implementations, PCA experiments, report writing |
| Zhang Yao | Logistic Regression, LDA/QDA, data visualization |
| Both | K-NN baseline, integration testing, presentation |

---

## 5. Timeline

| Week | Milestone |
|------|-----------|
| Week 10 | Proposal submission |
| Week 11 | Implement baseline + 2 classifiers |
| Week 12 | Add more classifiers, parameter tuning |
| Week 13 | PCA experiments, error analysis |
| Week 14 | Final report, poster, presentation |

---

## 6. References

1. sklearn documentation (scikit-learn.org)
2. LIBSVM: https://www.csie.ntu.edu.tw/~cjlin/libsvm/
3. Course lecture notes (SVM, Logistic Regression, PCA)
