#!/usr/bin/env python3
"""Main script to run digit classification experiments.

Procedure:
1. Cross-validate all candidate methods on Trial 1 training data (3-fold stratified)
   to select the best model and hyperparameters.
2. Evaluate the selected model on both trials (train on full train, test on test).
3. Report the mean of the two trial test accuracies.
"""

import sys
from pathlib import Path
import warnings
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.experiments import Experiment
from src.utils import resolve_run_dir, setup_run_logging, get_script_output_dir


EXPERIMENTS = [
    # (Name, Classifier, Params, Normalization, PCA, OneVsAll, Augment, AugStrength)
    ('1-NN Baseline', 'knn', {'n_neighbors': 1}, 'none', None, False, 0, 'mild'),
    ('3-NN', 'knn', {'n_neighbors': 3}, 'none', None, False, 0, 'mild'),
    ('5-NN', 'knn', {'n_neighbors': 5}, 'none', None, False, 0, 'mild'),
    ('LDA', 'lda', {}, 'none', None, False, 0, 'mild'),
    ('QDA + PCA50', 'qda', {'solver': 'svd'}, 'none', 50, False, 0, 'mild'),
    ('SVM-Linear', 'svm_linear', {'C': 1.0}, 'scale255', None, True, 0, 'mild'),
    ('SVM-RBF', 'svm_rbf', {'C': 1.0, 'gamma': 'scale'}, 'scale255', None, True, 0, 'mild'),
    ('SVM-RBF + PCA50', 'svm_rbf', {'C': 1.0, 'gamma': 'scale'}, 'scale255', 50, True, 0, 'mild'),
    ('Logistic Regression', 'logistic', {'C': 1.0, 'max_iter': 1000}, 'standard', None, True, 0, 'mild'),
    ('SVM-RBF + PCA35 + Aug', 'svm_rbf', {'C': 4, 'gamma': 'scale'}, 'scale255', 35, True, 3, 'mild'),
    ('SVM-RBF + PCA50 + Aug', 'svm_rbf', {'C': 4, 'gamma': 'scale'}, 'scale255', 50, True, 3, 'mild'),
]


def run_cv_selection(exp: Experiment):
    """Run 3-fold stratified CV on Trial 1 training data for all configs."""
    cv_results = []

    print("\n" + "=" * 90)
    print("PHASE A: CROSS-VALIDATION ON TRIAL 1 TRAINING DATA (3-fold stratified)")
    print("=" * 90)

    for i, (name, clf, params, norm, pca, ova, aug, aug_s) in enumerate(EXPERIMENTS, 1):
        print(f"\n[{i}/{len(EXPERIMENTS)}] CV: {name}...")

        try:
            result = exp.cross_validate_on_trial(
                trial=0,
                classifier_name=clf,
                classifier_params=params,
                normalize=norm,
                pca_components=pca,
                use_one_vs_all=ova,
                augment=aug,
                augment_strength=aug_s,
                n_folds=3,
                random_state=42,
            )
            cv_results.append({
                'name': name,
                'classifier': clf,
                'params': params,
                'normalize': norm,
                'pca': pca if pca else '-',
                'use_one_vs_all': ova,
                'augment': aug,
                'fold_accuracies': result['fold_accuracies'],
                'cv_mean': result['cv_mean'],
                'cv_std': result['cv_std'],
            })
            print(f"    ✓ CV: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}  "
                  f"(folds: {', '.join(f'{a:.4f}' for a in result['fold_accuracies'])})")

        except Exception as e:
            print(f"    ✗ Error: {e}")
            cv_results.append({
                'name': name,
                'classifier': clf,
                'params': params,
                'normalize': norm,
                'pca': pca if pca else '-',
                'use_one_vs_all': ova,
                'augment': aug,
                'fold_accuracies': [],
                'cv_mean': 0.0,
                'cv_std': 0.0,
            })

    # Print CV summary table
    print("\n" + "=" * 90)
    print("CV RESULTS SUMMARY (Trial 1 train, 3-fold stratified)")
    print("=" * 90)
    print(f"{'Method':<28} {'Norm':<10} {'PCA':<6} {'CV Mean±Std':<20} {'Folds'}")
    print("-" * 90)

    for r in cv_results:
        marker = "🏆 " if r['cv_mean'] > 0.95 else "   "
        folds_str = ', '.join(f'{a:.4f}' for a in r['fold_accuracies'])
        print(f"{marker}{r['name']:<25} {r['normalize']:<10} {str(r['pca']):<6} "
              f"{r['cv_mean']:.4f}±{r['cv_std']:.4f}     {folds_str}")

    print("=" * 90)

    # Select best config
    best = max(cv_results, key=lambda x: x['cv_mean'])
    print(f"\n🏆 SELECTED: {best['name']} = {best['cv_mean']:.4f} ± {best['cv_std']:.4f}")

    return cv_results, best


def run_final_evaluation(exp: Experiment, best_config: dict):
    """Evaluate the selected model on both trials."""
    print("\n" + "=" * 90)
    print("PHASE B: FINAL EVALUATION ON BOTH TRIALS (selected model only)")
    print("=" * 90)

    result = exp.run_experiment(
        classifier_name=best_config['classifier'],
        classifier_params=best_config['params'],
        normalize=best_config['normalize'],
        pca_components=best_config['pca'] if best_config['pca'] != '-' else None,
        use_one_vs_all=best_config.get('use_one_vs_all', True),
        augment=best_config['augment'],
        augment_strength='mild',
    )

    print(f"\n🏆 {best_config['name']}")
    print(f"   Trial 1: {result['trial_1_acc']:.4f}")
    print(f"   Trial 2: {result['trial_2_acc']:.4f}")
    print(f"   Mean:    {result['mean_acc']:.4f} ± {result['std_acc']:.4f}")
    print(f"   Improvement over 1-NN baseline: +{(result['mean_acc'] - 0.9160)*100:.2f}%")

    return result


def main(output_dir: Path = None):
    print("=" * 90)
    print("CS5487 - Digit Classification Experiments")
    print("Team: Yikai LAI, Zhang Yao")
    print("Procedure: CV on Trial 1 train → select best → evaluate on both trials")
    print("=" * 90)

    # Initialize experiment
    data_path = Path(__file__).parent.parent / "data" / "raw" / "MINIST" / "digits4000.mat"

    if not data_path.exists():
        print(f"\n⚠️  Data file not found: {data_path}")
        print("Please download digits4000.mat and place it in data/raw/MINIST/")
        return

    exp = Experiment(data_path)

    # Phase A: CV on Trial 1 training data to select best model
    cv_results, best_config = run_cv_selection(exp)

    # Phase B: Final evaluation on both trials
    final_result = run_final_evaluation(exp, best_config)

    # Save results
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "experiments" / "logs"
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Save full results JSON
    results_path = logs_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'cv_results': cv_results,
            'selected_config': best_config,
            'final_evaluation': final_result,
        }, f, indent=2)
    print(f"\n✅ Detailed results saved to {results_path}")

    # Save summary text
    summary_path = logs_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("CS5487 - Digit Classification Results\n")
        f.write("Team: Yikai LAI, Zhang Yao\n")
        f.write("=" * 90 + "\n\n")
        f.write("PHASE A: Cross-Validation on Trial 1 Training Data (3-fold stratified)\n")
        f.write("-" * 90 + "\n")
        for r in cv_results:
            f.write(f"{r['name']:<28} {r['cv_mean']:.4f} ± {r['cv_std']:.4f}\n")
        f.write("\n")
        f.write(f"SELECTED: {best_config['name']} (CV = {best_config['cv_mean']:.4f} ± {best_config['cv_std']:.4f})\n\n")
        f.write("PHASE B: Final Evaluation on Both Trials\n")
        f.write("-" * 90 + "\n")
        f.write(f"Trial 1: {final_result['trial_1_acc']:.4f}\n")
        f.write(f"Trial 2: {final_result['trial_2_acc']:.4f}\n")
        f.write(f"Mean:    {final_result['mean_acc']:.4f} ± {final_result['std_acc']:.4f}\n")
    print(f"✅ Summary saved to {summary_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None, help="Result directory (auto-generated if omitted)")
    args = parser.parse_args()
    base_dir = resolve_run_dir(args.run_dir)
    output_dir = get_script_output_dir(base_dir, Path(__file__).stem)
    with setup_run_logging(output_dir):
        main(output_dir)
