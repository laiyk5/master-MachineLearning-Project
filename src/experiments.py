"""Main experiment runner for digit classification."""

import numpy as np
from pathlib import Path
import json
from typing import Dict, List
import time

from sklearn.model_selection import StratifiedKFold

from .data_loader import load_digits_data, get_train_test_split, normalize_features, evaluate_accuracy
from .classifiers import OneVsAllClassifier, PCATransform, get_classifier
from .augmentation import augment_training_data


def _fit_and_score(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classifier_name: str,
    classifier_params: Dict = None,
    normalize: str = "none",
    pca_components: int = None,
    use_one_vs_all: bool = True,
    augment: int = 0,
    augment_strength: str = "mild",
) -> float:
    """Fit classifier pipeline and return accuracy on test set.

    Pipeline: augment (train only) → normalize → PCA → fit → predict → score.
    All preprocessing is fit on train and applied to test.
    """
    if classifier_params is None:
        classifier_params = {}

    X_tr = X_train.copy()
    y_tr = y_train.copy()

    # Data augmentation (before normalization/PCA)
    if augment > 0:
        X_tr, y_tr = augment_training_data(
            X_tr, y_tr, n_augment=augment, strength=augment_strength
        )

    # Normalize
    if normalize != "none":
        X_tr, X_te = normalize_features(X_tr, X_test, method=normalize)
    else:
        X_te = X_test.copy()

    # Apply PCA if requested
    if pca_components is not None:
        pca = PCATransform(n_components=pca_components)
        X_tr = pca.fit_transform(X_tr)
        X_te = pca.transform(X_te)

    # Train classifier
    if use_one_vs_all:
        base_clf = lambda **kwargs: get_classifier(classifier_name, **kwargs)
        clf = OneVsAllClassifier(base_clf, n_classes=10)
        clf.fit(X_tr, y_tr, **classifier_params)
    else:
        clf = get_classifier(classifier_name, **classifier_params)
        clf.fit(X_tr, y_tr)

    # Evaluate
    y_pred = clf.predict(X_te)
    return evaluate_accuracy(y_test, y_pred)


class Experiment:
    """Run digit classification experiments."""

    def __init__(self, data_path: str = None):
        """
        Args:
            data_path: Path to digits4000.mat file
        """
        self.data = load_digits_data(data_path)
        self.results = []

    def cross_validate_on_trial(
        self,
        trial: int,
        classifier_name: str,
        classifier_params: Dict = None,
        normalize: str = "none",
        pca_components: int = None,
        use_one_vs_all: bool = True,
        augment: int = 0,
        augment_strength: str = "mild",
        n_folds: int = 3,
        random_state: int = 42,
    ) -> Dict:
        """Run stratified k-fold CV on a single trial's training data.

        Args:
            trial: Trial number (0 or 1)
            classifier_name: Name of classifier
            classifier_params: Parameters for classifier
            normalize: Normalization method
            pca_components: Number of PCA components (None to skip)
            use_one_vs_all: Use one-vs-all strategy for multi-class
            augment: Number of augmented copies per training sample
            augment_strength: Augmentation intensity
            n_folds: Number of CV folds
            random_state: Random seed for fold splitting

        Returns:
            Dictionary with cv_mean, cv_std, and per-fold accuracies.
        """
        X_train, _, y_train, _ = get_train_test_split(self.data, trial)

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        fold_accs = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            X_tr_fold = X_train[train_idx]
            y_tr_fold = y_train[train_idx]
            X_val_fold = X_train[val_idx]
            y_val_fold = y_train[val_idx]

            acc = _fit_and_score(
                X_tr_fold, y_tr_fold, X_val_fold, y_val_fold,
                classifier_name=classifier_name,
                classifier_params=classifier_params,
                normalize=normalize,
                pca_components=pca_components,
                use_one_vs_all=use_one_vs_all,
                augment=augment,
                augment_strength=augment_strength,
            )
            fold_accs.append(acc)

        return {
            'trial': trial,
            'classifier': classifier_name,
            'classifier_params': classifier_params,
            'normalize': normalize,
            'pca_components': pca_components,
            'use_one_vs_all': use_one_vs_all,
            'augment': augment,
            'augment_strength': augment_strength,
            'n_folds': n_folds,
            'fold_accuracies': fold_accs,
            'cv_mean': float(np.mean(fold_accs)),
            'cv_std': float(np.std(fold_accs)),
        }

    def run_trial(self,
                  trial: int,
                  classifier_name: str,
                  classifier_params: Dict = None,
                  normalize: str = "none",
                  pca_components: int = None,
                  use_one_vs_all: bool = True,
                  augment: int = 0,
                  augment_strength: str = "mild") -> Dict:
        """Run a single trial (train on train split, test on test split).

        Args:
            trial: Trial number (0 or 1)
            classifier_name: Name of classifier
            classifier_params: Parameters for classifier
            normalize: Normalization method
            pca_components: Number of PCA components (None to skip)
            use_one_vs_all: Use one-vs-all strategy for multi-class
            augment: Number of augmented copies per training sample
            augment_strength: Augmentation intensity

        Returns:
            Dictionary with results
        """
        if classifier_params is None:
            classifier_params = {}

        X_train, X_test, y_train, y_test = get_train_test_split(self.data, trial)

        start_time = time.time()
        accuracy = _fit_and_score(
            X_train, y_train, X_test, y_test,
            classifier_name=classifier_name,
            classifier_params=classifier_params,
            normalize=normalize,
            pca_components=pca_components,
            use_one_vs_all=use_one_vs_all,
            augment=augment,
            augment_strength=augment_strength,
        )
        train_time = time.time() - start_time

        result = {
            'trial': trial,
            'classifier': classifier_name,
            'classifier_params': classifier_params,
            'normalize': normalize,
            'pca_components': pca_components,
            'use_one_vs_all': use_one_vs_all,
            'augment': augment,
            'augment_strength': augment_strength,
            'accuracy': accuracy,
            'train_time': train_time,
            'n_train': len(y_train),
            'n_test': len(y_test),
        }

        self.results.append(result)
        return result

    def run_experiment(self,
                       classifier_name: str,
                       classifier_params: Dict = None,
                       normalize: str = "none",
                       pca_components: int = None,
                       use_one_vs_all: bool = True,
                       augment: int = 0,
                       augment_strength: str = "mild") -> Dict:
        """Run both trials and compute mean accuracy.

        Returns:
            Summary with mean/std accuracy across trials
        """
        accuracies = []

        for trial in [0, 1]:
            result = self.run_trial(
                trial=trial,
                classifier_name=classifier_name,
                classifier_params=classifier_params,
                normalize=normalize,
                pca_components=pca_components,
                use_one_vs_all=use_one_vs_all,
                augment=augment,
                augment_strength=augment_strength,
            )
            accuracies.append(result['accuracy'])

        summary = {
            'classifier': classifier_name,
            'params': classifier_params,
            'normalize': normalize,
            'pca_components': pca_components,
            'augment': augment,
            'augment_strength': augment_strength,
            'trial_1_acc': accuracies[0],
            'trial_2_acc': accuracies[1],
            'mean_acc': np.mean(accuracies),
            'std_acc': np.std(accuracies),
        }

        return summary

    def save_results(self, filepath: str):
        """Save all results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)

    def print_summary(self):
        """Print summary of all experiments."""
        print("=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)
        print(f"{'Classifier':<20} {'Norm':<10} {'PCA':<6} {'Trial 1':<10} {'Trial 2':<10} {'Mean±Std':<15}")
        print("-" * 80)

        for r in self.results:
            pca = r['pca_components'] if r['pca_components'] else "None"
            print(f"{r['classifier']:<20} {r['normalize']:<10} {str(pca):<6} "
                  f"{r['accuracy']:.4f}     ...       ...")


def grid_search(experiment: Experiment,
                param_grid: List[Dict]) -> List[Dict]:
    """Run grid search over parameters.

    Args:
        experiment: Experiment instance
        param_grid: List of parameter dictionaries

    Returns:
        List of results
    """
    results = []

    for params in param_grid:
        print(f"\nRunning: {params}")
        result = experiment.run_experiment(**params)
        results.append(result)
        print(f"Mean accuracy: {result['mean_acc']:.4f} ± {result['std_acc']:.4f}")

    return results
