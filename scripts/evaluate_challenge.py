#!/usr/bin/env python3
"""Evaluate classifiers trained on MNIST against the challenge dataset.

Trains each model on the MNIST training set (both trials separately) and
evaluates on the challenge digits WITHOUT retraining or tuning on challenge data.
"""

import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import scipy.io as sio

from src.data_loader import (
    load_digits_data,
    get_train_test_split,
    normalize_features,
    evaluate_accuracy,
)
from src.classifiers import OneVsAllClassifier, PCATransform, get_classifier
from src.augmentation import augment_training_data
from src.utils import resolve_run_dir, setup_run_logging


CHALLENGE_MAT = Path(__file__).parent.parent / "data" / "raw" / "challenge" / "cdigits.mat"
MNIST_MAT = Path(__file__).parent.parent / "data" / "raw" / "MINIST" / "digits4000.mat"

# Same experiment configs as run_experiments.py
# (name, classifier, params, normalize, pca, one_vs_all, augment, augment_strength)
EXPERIMENTS = [
    ("1-NN Baseline", "knn", {"n_neighbors": 1}, "none", None, False, 0, "mild"),
    ("3-NN", "knn", {"n_neighbors": 3}, "none", None, False, 0, "mild"),
    ("5-NN", "knn", {"n_neighbors": 5}, "none", None, False, 0, "mild"),
    ("LDA", "lda", {}, "none", None, False, 0, "mild"),
    ("QDA + PCA50", "qda", {"solver": "svd"}, "none", 50, False, 0, "mild"),
    ("SVM-Linear", "svm_linear", {"C": 1.0}, "scale255", None, True, 0, "mild"),
    ("SVM-RBF", "svm_rbf", {"C": 1.0, "gamma": "scale"}, "scale255", None, True, 0, "mild"),
    ("SVM-RBF + PCA50", "svm_rbf", {"C": 1.0, "gamma": "scale"}, "scale255", 50, True, 0, "mild"),
    (
        "Logistic Regression",
        "logistic",
        {"C": 1.0, "max_iter": 1000},
        "standard",
        None,
        True,
        0,
        "mild",
    ),
    # Zhang's improved configurations
    ("SVM-RBF + PCA35 + Aug", "svm_rbf", {"C": 4, "gamma": "scale"}, "scale255", 35, True, 3, "mild"),
    ("SVM-RBF + PCA50 + Aug", "svm_rbf", {"C": 4, "gamma": "scale"}, "scale255", 50, True, 3, "mild"),
]


def load_challenge_data(mat_file: str = None):
    """Load challenge digits from .mat file.

    Returns:
        X: (150, 784) feature matrix
        y: (150,) labels (0-9)
    """
    if mat_file is None:
        mat_file = CHALLENGE_MAT

    data = sio.loadmat(mat_file)
    X = data["cdigits_vec"].T  # (150, 784)
    y = data["cdigits_labels"].flatten()  # (150,)
    return X, y


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_challenge: np.ndarray,
    y_challenge: np.ndarray,
    classifier_name: str,
    classifier_params: dict,
    normalize: str,
    pca_components: int | None,
    use_one_vs_all: bool,
    augment: int = 0,
    augment_strength: str = "mild",
):
    """Train on MNIST training split, evaluate on challenge data."""

    # Data augmentation (before normalization/PCA)
    if augment > 0:
        X_train, y_train = augment_training_data(
            X_train, y_train, n_augment=augment, strength=augment_strength
        )

    # Normalize (fit on train, apply to both)
    if normalize != "none":
        X_train, X_challenge = normalize_features(X_train, X_challenge, method=normalize)

    # PCA (fit on train, transform both)
    if pca_components is not None:
        pca = PCATransform(n_components=pca_components)
        X_train = pca.fit_transform(X_train)
        X_challenge = pca.transform(X_challenge)

    # Train classifier
    if use_one_vs_all:
        base_clf = lambda **kwargs: get_classifier(classifier_name, **kwargs)
        clf = OneVsAllClassifier(base_clf, n_classes=10)
        clf.fit(X_train, y_train, **classifier_params)
    else:
        clf = get_classifier(classifier_name, **classifier_params)
        clf.fit(X_train, y_train)

    # Evaluate on challenge
    y_pred = clf.predict(X_challenge)
    accuracy = evaluate_accuracy(y_challenge, y_pred)
    return accuracy


def main(output_dir: Path = None):
    if not CHALLENGE_MAT.exists():
        print(f"Challenge data not found: {CHALLENGE_MAT}")
        return
    if not MNIST_MAT.exists():
        print(f"MNIST data not found: {MNIST_MAT}")
        return

    mnist_data = load_digits_data(MNIST_MAT)
    X_challenge, y_challenge = load_challenge_data(CHALLENGE_MAT)

    print("=" * 90)
    print("CHALLENGE DATASET EVALUATION")
    print("=" * 90)
    print(f"Challenge samples: {len(y_challenge)}")
    print(f"Reference (1-NN):  0.683")
    print("=" * 90)

    results_summary = []

    for name, clf_name, params, norm, pca, ova, aug, aug_s in EXPERIMENTS:
        trial_accs = []

        for trial in [0, 1]:
            X_train, _, y_train, _ = get_train_test_split(mnist_data, trial)
            acc = train_and_evaluate(
                X_train,
                y_train,
                X_challenge,
                y_challenge,
                clf_name,
                params,
                norm,
                pca,
                ova,
                augment=aug,
                augment_strength=aug_s,
            )
            trial_accs.append(acc)

        mean_acc = np.mean(trial_accs)
        std_acc = np.std(trial_accs)

        results_summary.append(
            {
                "name": name,
                "trial_1": trial_accs[0],
                "trial_2": trial_accs[1],
                "mean": mean_acc,
                "std": std_acc,
            }
        )

        marker = "🏆 " if mean_acc > 0.75 else "   "
        print(
            f"{marker}{name:<22} Trial1: {trial_accs[0]:.4f}  Trial2: {trial_accs[1]:.4f}  "
            f"Mean: {mean_acc:.4f} ± {std_acc:.4f}"
        )

    print("=" * 90)
    best = max(results_summary, key=lambda x: x["mean"])
    print(
        f"🏆 BEST: {best['name']} = {best['mean']:.4f} ± {best['std']:.4f}"
    )
    print(f"   Reference (1-NN): 0.683")
    print(f"   Improvement: +{(best['mean'] - 0.683) * 100:.2f} percentage points")
    print("=" * 90)

    if output_dir is not None:
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        report_path = logs_dir / "challenge_results.txt"
        with open(report_path, "w") as f:
            f.write("=" * 90 + "\n")
            f.write("CHALLENGE DATASET EVALUATION\n")
            f.write("=" * 90 + "\n\n")
            for r in results_summary:
                f.write(
                    f"{r['name']:<25} Trial1: {r['trial_1']:.4f}  Trial2: {r['trial_2']:.4f}  "
                    f"Mean: {r['mean']:.4f} ± {r['std']:.4f}\n"
                )
            f.write("\n" + "=" * 90 + "\n")
            f.write(f"BEST: {best['name']} = {best['mean']:.4f} ± {best['std']:.4f}\n")
            f.write(f"Reference (1-NN): 0.683\n")
            f.write(f"Improvement: +{(best['mean'] - 0.683) * 100:.2f} percentage points\n")
        print(f"\n✅ Results saved to {report_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None, help="Result directory (auto-generated if omitted)")
    args = parser.parse_args()
    output_dir = resolve_run_dir(args.run_dir)
    with setup_run_logging(output_dir):
        main(output_dir)
