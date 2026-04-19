# CS5487 Course Project — Parsed Requirements

**Source:** `docs/requirement/PA-3-courseproject.pdf`  
**Course:** CS5487 Machine Learning, City University of Hong Kong  
**Instructor:** Antoni Chan

---

## 1. Deliverables & Deadlines

| Deliverable | Due Date | Format | Notes |
|-------------|----------|--------|-------|
| Project Proposal | Friday, Week 10 | At most 1 page | Introduction + precise plan |
| Project Report | Friday, Week 14 | 4–8 pages | Proposal with all details filled in |
| Presentation Poster | Friday, Week 14 | Poster | **Required for grade A** |
| Source Code | Friday, Week 14 | Source files | Submit via Canvas |

**Group requirement:** Exactly 2 students. Report **must state contribution level** of each member.  
**Presentation rule:** Optional, but **mandatory for "A"**. Without it, max grade is **B+**.

---

## 2. Report Required Sections

The report must contain **exactly 4 sections**:

1. **Introduction** — What is the problem? Why is it important?
2. **Methodology** — What algorithms? Technical details? Advantages/disadvantages?
3. **Experimental Setup** — What data? Pre-processing? Algorithms run? Evaluation metric?
4. **Experimental Results** — Results? Insights? Typical success and failure cases?

---

## 3. Grading Breakdown (30 points total)

Each component is worth **5 points (16.7%)**:

| Component | What earns points |
|-----------|-------------------|
| Project Proposal | Clear plan, feasible scope |
| Technical Correctness | Algorithms used correctly |
| Experiments | **Thoroughness**, testing interesting cases, different parameter settings |
| Analysis | **Insightful observations** — more points for deeper insights |
| Report Quality | Organized, complete descriptions |
| Presentation | Clear poster + presentation |

---

## 4. Default Project — Digit Classification

### 4.1 Dataset

| Property | Value |
|----------|-------|
| Total samples | 4,000 images |
| Classes | 10 (digits 0–9) |
| Samples per class | 400 (balanced) |
| Features | 784 (28×28 grayscale pixels) |
| Pixel values | [0, 255] |
| Train/test split | 50/50 (2,000 / 2,000) |
| Number of splits | 2 predefined trials |
| Writer constraint | Same writer **not** in both train and test within a trial |

**MATLAB file contents (`digits4000.mat`):**
- `digits_vec` — 784×4000 matrix, each column is a vectorized image
- `digits_labels` — 1×4000 matrix, labels y ∈ {0, ..., 9}
- `trainset` — 2×2000 matrix, row = training indices for each trial
- `testset` — 2×2000 matrix, row = test indices for each trial

### 4.2 Allowed Methods

Any technique from course material:
- Bayes classifiers, Fisher's Discriminant, SVMs, logistic regression, perceptron, kernel functions
- Can use methods **not** covered in class, but must describe them **in detail** in the report
- Can use 3rd-party libraries (libsvm, liblinear, scikit-learn) with **proper citation**

### 4.3 Preprocessing

- PCA or kernel-PCA for dimensionality reduction
- Normalization
- Image processing techniques

### 4.4 Multi-class Strategy

For binary classifiers (SVM, Logistic Regression), use **1-vs-all**:
- Train 10 binary classifiers, one per digit
- Each classifier distinguishes one digit (+1) vs. all others (-1)
- At test time, select classifier with **highest confidence**:
  - SVM: furthest from the margin
  - Logistic regression: highest calculated probability

### 4.5 Critical Evaluation Rules

- **Train on training set only**
- **May use training set for cross-validation** to select optimal parameters
- **Do NOT tune parameters to optimize test accuracy directly**
- Report accuracy per trial, then mean ± standard deviation

### 4.6 Baseline

| Trial | 1-NN (Euclidean) Accuracy |
|-------|--------------------------|
| 1 | 0.9135 |
| 2 | 0.9185 |
| **Mean ± Std** | **0.9160 ± 0.0035** |

**Research questions to address:**
- Which classifier does better than baseline?
- What preprocessing helps or hurts performance?
- How does performance vary with parameter values?

### 4.7 Bonus Challenge

- New test set: professor's own handwritten digits
- Classify using **trained classifiers without retraining** on challenge data
- Best performance **wins a prize**

---

## 5. Proposal Requirements (Week 10)

At most 1 page containing:
1. Brief introduction stating the problem
2. Precise description of what you plan to do:
   - What features will you use?
   - What algorithms will you use?
   - What dataset will you use?
   - How will you evaluate results?
   - How do you define a good outcome?

The goal is to work out the project in your head before implementation.

---

## 6. Third-Party Code Policy

- 3rd-party source code is allowed (e.g., libsvm)
- Must acknowledge with an **appropriate reference**
- Cannot submit work you have already done (e.g., results from a published paper)
- Can extend prior work with new methods from the course

---

## 7. Compliance Checklist

| Requirement | Required? | Our Status |
|-------------|-----------|------------|
| Group of 2 | Yes | Yikai LAI + Zhang Yao |
| Proposal (Week 10) | Yes | Submitted |
| Report 4–8 pages with 4 sections | Yes | `docs/report/report.tex` (7 pages) |
| Contribution statement | Yes | Included in report |
| Multiple classifiers | Yes | 11 configurations tested |
| Parameter experiments | Yes | k∈{1,3,5}, PCA 35/50, C∈{1,4}, aug variants |
| No test-set tuning | Yes | Parameters selected via CV on training data only |
| Error analysis | Yes | Confusion matrices, per-digit accuracy, hard/medium/easy |
| Challenge evaluation (no retraining) | Yes | `evaluate_challenge.py` |
| Third-party code cited | Yes | scikit-learn referenced |
| Source code submitted | Yes | Full repo on GitHub |
| Presentation poster | Required for A | **Not yet created** |
