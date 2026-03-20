#!/usr/bin/env python3
"""Main script to run digit classification experiments."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.experiments import Experiment, grid_search


def main():
    print("=" * 80)
    print("CS5487 - Digit Classification Experiments")
    print("=" * 80)
    
    # Initialize experiment
    data_path = Path(__file__).parent / "data" / "raw" / "digits4000.mat"
    
    if not data_path.exists():
        print(f"\n⚠️  Data file not found: {data_path}")
        print("Please download digits4000.mat and place it in data/raw/")
        return
    
    exp = Experiment(data_path)
    
    # Baseline: 1-NN (target: ~91.6%)
    print("\n" + "=" * 80)
    print("1. BASELINE: 1-NN with Euclidean distance")
    print("=" * 80)
    
    result = exp.run_experiment(
        classifier_name='knn',
        classifier_params={'n_neighbors': 1},
        normalize='none',
        use_one_vs_all=False
    )
    print(f"Trial 1: {result['trial_1_acc']:.4f}")
    print(f"Trial 2: {result['trial_2_acc']:.4f}")
    print(f"Mean ± Std: {result['mean_acc']:.4f} ± {result['std_acc']:.4f}")
    
    # Example: SVM with RBF kernel
    print("\n" + "=" * 80)
    print("2. SVM with RBF Kernel")
    print("=" * 80)
    
    result = exp.run_experiment(
        classifier_name='svm_rbf',
        classifier_params={'C': 1.0, 'gamma': 'scale'},
        normalize='scale255',
        use_one_vs_all=True
    )
    print(f"Trial 1: {result['trial_1_acc']:.4f}")
    print(f"Trial 2: {result['trial_2_acc']:.4f}")
    print(f"Mean ± Std: {result['mean_acc']:.4f} ± {result['std_acc']:.4f}")
    
    # Example: Logistic Regression
    print("\n" + "=" * 80)
    print("3. Logistic Regression")
    print("=" * 80)
    
    result = exp.run_experiment(
        classifier_name='logistic',
        classifier_params={'C': 1.0, 'max_iter': 1000},
        normalize='standard',
        use_one_vs_all=True
    )
    print(f"Trial 1: {result['trial_1_acc']:.4f}")
    print(f"Trial 2: {result['trial_2_acc']:.4f}")
    print(f"Mean ± Std: {result['mean_acc']:.4f} ± {result['std_acc']:.4f}")
    
    # Save results
    results_path = Path(__file__).parent / "experiments" / "logs" / "results.json"
    exp.save_results(results_path)
    print(f"\n✅ Results saved to {results_path}")


if __name__ == "__main__":
    main()
