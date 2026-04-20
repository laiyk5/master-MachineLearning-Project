#!/usr/bin/env python3
"""Sensitivity analysis for PCA components and SVM C parameter.

Runs 3-fold stratified CV on Trial 1 training data for a grid of
PCA components and SVM C values, with mild augmentation ×3.
"""

import sys
from pathlib import Path
import warnings
import json

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.experiments import Experiment
from src.utils import resolve_run_dir, setup_run_logging, get_script_output_dir


PCA_COMPONENTS = [20, 25, 30, 35, 40]
C_VALUES = [3, 4, 5, 6, 8]

BASE_CONFIG = {
    'classifier_name': 'svm_rbf',
    'normalize': 'scale255',
    'use_one_vs_all': True,
    'augment': 3,
    'augment_strength': 'mild',
}


def main(output_dir: Path = None):
    print("=" * 80)
    print("Sensitivity Analysis: PCA Components × SVM C")
    print("Base: SVM-RBF + mild aug ×3 + gamma=scale")
    print("=" * 80)

    data_path = Path(__file__).parent.parent / "data" / "raw" / "MINIST" / "digits4000.mat"
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return

    exp = Experiment(data_path)

    # Header
    header = "PCA\\C  " + "".join(f"{c:>8}" for c in C_VALUES)
    print(f"\n{header}")
    print("-" * 60)

    grid_results = []
    for pca in PCA_COMPONENTS:
        row_label = f"{pca:>4}   "
        row_values = []
        for C in C_VALUES:
            result = exp.cross_validate_on_trial(
                trial=0,
                classifier_params={'C': C, 'gamma': 'scale'},
                pca_components=pca,
                n_folds=3,
                random_state=42,
                **BASE_CONFIG,
            )
            row_values.append(result['cv_mean'])
            grid_results.append({
                'pca': pca,
                'C': C,
                'cv_mean': result['cv_mean'],
                'cv_std': result['cv_std'],
                'fold_accuracies': result['fold_accuracies'],
            })
        row_str = row_label + "".join(f"{v:>8.4f}" for v in row_values)
        print(row_str)

    print("-" * 60)

    # Find best
    best = max(grid_results, key=lambda x: x['cv_mean'])
    print(f"\nBest: PCA={best['pca']}, C={best['C']} → CV={best['cv_mean']:.4f} ± {best['cv_std']:.4f}")

    # Save
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "experiments" / "logs"
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    results_path = logs_dir / "sensitivity_pca_c.json"
    with open(results_path, 'w') as f:
        json.dump({'base_config': BASE_CONFIG, 'grid_results': grid_results}, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None)
    args = parser.parse_args()
    base_dir = resolve_run_dir(args.run_dir)
    output_dir = get_script_output_dir(base_dir, Path(__file__).stem)
    with setup_run_logging(output_dir):
        main(output_dir)
