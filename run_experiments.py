#!/usr/bin/env python3
"""Main script to run digit classification experiments."""

import sys
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.experiments import Experiment


def run_all_experiments(exp: Experiment):
    """Run all experiments and print summary table."""
    
    experiments = [
        # (Name, Classifier, Params, Normalization, PCA, OneVsAll)
        ('1-NN Baseline', 'knn', {'n_neighbors': 1}, 'none', None, False),
        ('3-NN', 'knn', {'n_neighbors': 3}, 'none', None, False),
        ('5-NN', 'knn', {'n_neighbors': 5}, 'none', None, False),
        ('LDA', 'lda', {}, 'none', None, False),
        ('QDA + PCA50', 'qda', {'solver': 'svd'}, 'none', 50, False),
        ('SVM-Linear', 'svm_linear', {'C': 1.0}, 'scale255', None, True),
        ('SVM-RBF', 'svm_rbf', {'C': 1.0, 'gamma': 'scale'}, 'scale255', None, True),
        ('SVM-RBF + PCA50', 'svm_rbf', {'C': 1.0, 'gamma': 'scale'}, 'scale255', 50, True),
        ('Logistic Regression', 'logistic', {'C': 1.0, 'max_iter': 1000}, 'standard', None, True),
    ]
    
    results_summary = []
    
    print("\n" + "=" * 90)
    print("RUNNING ALL EXPERIMENTS")
    print("=" * 90)
    
    for i, (name, clf, params, norm, pca, ova) in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Running: {name}...")
        
        try:
            result = exp.run_experiment(
                classifier_name=clf,
                classifier_params=params,
                normalize=norm,
                pca_components=pca,
                use_one_vs_all=ova
            )
            
            results_summary.append({
                'name': name,
                'norm': norm,
                'pca': pca if pca else '-',
                'trial_1': result['trial_1_acc'],
                'trial_2': result['trial_2_acc'],
                'mean': result['mean_acc'],
                'std': result['std_acc'],
            })
            
            print(f"    ✓ Accuracy: {result['mean_acc']:.4f} ± {result['std_acc']:.4f}")
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            results_summary.append({
                'name': name,
                'norm': norm,
                'pca': pca if pca else '-',
                'trial_1': 0.0,
                'trial_2': 0.0,
                'mean': 0.0,
                'std': 0.0,
            })
    
    # Print summary table
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Method':<25} {'Norm':<12} {'PCA':<8} {'Trial 1':<10} {'Trial 2':<10} {'Mean±Std':<15}")
    print("-" * 90)
    
    for r in results_summary:
        marker = "🏆 " if r['mean'] > 0.94 else "   "
        print(f"{marker}{r['name']:<22} {r['norm']:<12} {str(r['pca']):<8} "
              f"{r['trial_1']:<10.4f} {r['trial_2']:<10.4f} {r['mean']:.4f}±{r['std']:.4f}")
    
    print("=" * 90)
    
    # Best result
    best = max(results_summary, key=lambda x: x['mean'])
    print(f"\n🏆 BEST: {best['name']} = {best['mean']:.4f} ± {best['std']:.4f}")
    print(f"   Beats baseline by: +{(best['mean'] - 0.9160)*100:.2f}%")
    
    return results_summary


def main():
    print("=" * 90)
    print("CS5487 - Digit Classification Experiments")
    print("Team: Yikai LAI, Zhang Yao")
    print("=" * 90)
    
    # Initialize experiment
    data_path = Path(__file__).parent / "data" / "raw" / "digits4000.mat"
    
    if not data_path.exists():
        print(f"\n⚠️  Data file not found: {data_path}")
        print("Please download digits4000.mat and place it in data/raw/")
        return
    
    exp = Experiment(data_path)
    
    # Run all experiments
    run_all_experiments(exp)
    
    # Save results
    results_path = Path(__file__).parent / "experiments" / "logs" / "results.json"
    exp.save_results(results_path)
    print(f"\n✅ Detailed results saved to {results_path}")
    
    # Also save summary
    summary_path = Path(__file__).parent / "experiments" / "logs" / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("CS5487 - Digit Classification Results\n")
        f.write("Team: Yikai LAI, Zhang Yao\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"Baseline (1-NN): 0.9160 ± 0.0035\n")
        f.write(f"Best Result: See experiments above\n")
    print(f"✅ Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
