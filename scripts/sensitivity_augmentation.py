#!/usr/bin/env python3
"""Sensitivity analysis for data augmentation intensity.

Runs 3-fold stratified CV on Trial 1 training data for different
augmentation configurations, using the base model:
    SVM-RBF + PCA35 + C=4 + gamma=scale
"""

import sys
from pathlib import Path
import warnings
import json

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.experiments import Experiment
from src.utils import resolve_run_dir, setup_run_logging, get_script_output_dir


CONFIGS = [
    # (name, augment, augment_strength)
    ('Original (no augmentation)', 0, 'mild'),
    ('Mild augmentation ×3', 3, 'mild'),
    ('Medium augmentation ×3', 3, 'medium'),
    ('Strong augmentation ×3', 3, 'strong'),
    ('Medium augmentation ×5', 5, 'medium'),
    ('Medium augmentation ×7', 7, 'medium'),
]

BASE_CONFIG = {
    'classifier_name': 'svm_rbf',
    'classifier_params': {'C': 4, 'gamma': 'scale'},
    'normalize': 'scale255',
    'pca_components': 35,
    'use_one_vs_all': True,
}


def main(output_dir: Path = None):
    print("=" * 80)
    print("Sensitivity Analysis: Data Augmentation Intensity")
    print("Base model: SVM-RBF + PCA35 + C=4")
    print("=" * 80)

    data_path = Path(__file__).parent.parent / "data" / "raw" / "MINIST" / "digits4000.mat"
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return

    exp = Experiment(data_path)

    results = []
    print(f"\n{'Config':<30} {'CV Mean':<10} {'CV Std':<10} {'Folds'}")
    print("-" * 80)

    for name, aug, aug_s in CONFIGS:
        result = exp.cross_validate_on_trial(
            trial=0,
            augment=aug,
            augment_strength=aug_s,
            n_folds=3,
            random_state=42,
            **BASE_CONFIG,
        )
        folds_str = ', '.join(f'{a:.4f}' for a in result['fold_accuracies'])
        print(f"{name:<30} {result['cv_mean']:.4f}    {result['cv_std']:.4f}    {folds_str}")
        results.append({
            'name': name,
            'augment': aug,
            'augment_strength': aug_s,
            'fold_accuracies': result['fold_accuracies'],
            'cv_mean': result['cv_mean'],
            'cv_std': result['cv_std'],
        })

    print("-" * 80)

    # Save results
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "experiments" / "logs"
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    results_path = logs_dir / "sensitivity_augmentation.json"
    with open(results_path, 'w') as f:
        json.dump({'base_config': BASE_CONFIG, 'results': results}, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None)
    args = parser.parse_args()
    base_dir = resolve_run_dir(args.run_dir)
    output_dir = get_script_output_dir(base_dir, Path(__file__).stem)
    with setup_run_logging(output_dir):
        main(output_dir)
