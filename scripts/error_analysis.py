"""Error analysis and confusion matrix visualization for digit classification."""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple

from src.data_loader import load_digits_data, get_train_test_split, normalize_features, display_image
from src.classifiers import OneVsAllClassifier, PCATransform, get_classifier
from src.utils import resolve_run_dir, setup_run_logging, get_script_output_dir


class ErrorAnalyzer:
    """Analyze classification errors and generate visualizations."""
    
    def __init__(self, data_path: str = None):
        self.data = load_digits_data(data_path)
        self.results = {}
        
    def run_classifier(self,
                       classifier_name: str,
                       classifier_params: Dict = None,
                       normalize: str = "none",
                       pca_components: int = None,
                       use_one_vs_all: bool = True) -> Dict:
        """Run classifier and collect predictions for both trials."""
        if classifier_params is None:
            classifier_params = {}
            
        all_predictions = []
        all_true_labels = []
        trial_accuracies = []
        
        for trial in [0, 1]:
            X_train, X_test, y_train, y_test = get_train_test_split(self.data, trial)
            
            # Normalize
            if normalize != "none":
                X_train, X_test = normalize_features(X_train, X_test, method=normalize)
            
            # Apply PCA if requested
            if pca_components is not None:
                pca = PCATransform(n_components=pca_components)
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)
            
            # Train and predict
            if use_one_vs_all:
                base_clf = lambda **kwargs: get_classifier(classifier_name, **kwargs)
                clf = OneVsAllClassifier(base_clf, n_classes=10)
                clf.fit(X_train, y_train, **classifier_params)
            else:
                clf = get_classifier(classifier_name, **classifier_params)
                clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            
            all_predictions.extend(y_pred)
            all_true_labels.extend(y_test)
            trial_accuracies.append(np.mean(y_pred == y_test))
        
        return {
            'y_true': np.array(all_true_labels),
            'y_pred': np.array(all_predictions),
            'accuracy': np.mean(trial_accuracies),
            'classifier': classifier_name,
            'params': classifier_params,
            'normalize': normalize,
            'pca': pca_components
        }
    
    def compute_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute confusion matrix."""
        n_classes = 10
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        return cm
    
    def analyze_errors(self, result: Dict) -> Dict:
        """Analyze errors for a classifier result."""
        y_true = result['y_true']
        y_pred = result['y_pred']
        
        cm = self.compute_confusion_matrix(y_true, y_pred)
        
        # Per-digit accuracy
        per_digit_correct = np.diag(cm)
        per_digit_total = cm.sum(axis=1)
        per_digit_accuracy = per_digit_correct / per_digit_total
        
        # Most confused pairs
        errors = []
        for true_digit in range(10):
            for pred_digit in range(10):
                if true_digit != pred_digit and cm[true_digit, pred_digit] > 0:
                    errors.append({
                        'true': true_digit,
                        'pred': pred_digit,
                        'count': cm[true_digit, pred_digit],
                        'rate': cm[true_digit, pred_digit] / per_digit_total[true_digit]
                    })
        
        errors.sort(key=lambda x: x['count'], reverse=True)
        
        return {
            'confusion_matrix': cm,
            'per_digit_accuracy': per_digit_accuracy,
            'top_confusions': errors[:10],
            'total_errors': np.sum(cm) - np.trace(cm)
        }
    
    def plot_confusion_matrix(self, result: Dict, save_path: str = None):
        """Plot confusion matrix heatmap."""
        analysis = self.analyze_errors(result)
        cm = analysis['confusion_matrix']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Normalize for percentage view
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=range(10), yticklabels=range(10),
                    ax=ax, cbar_kws={'label': 'Proportion'})
        
        ax.set_xlabel('Predicted Digit', fontsize=12)
        ax.set_ylabel('True Digit', fontsize=12)
        
        title = f"{result['classifier']}"
        if result['pca']:
            title += f" (PCA={result['pca']})"
        title += f"\nAccuracy: {result['accuracy']:.4f}"
        ax.set_title(title, fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_error_comparison(self, results: List[Dict], save_path: str = None):
        """Compare error rates across classifiers."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, result in enumerate(results[:6]):  # Max 6 classifiers
            ax = axes[idx]
            analysis = self.analyze_errors(result)
            cm = analysis['confusion_matrix']
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Reds',
                        xticklabels=range(10), yticklabels=range(10),
                        ax=ax, cbar=False, vmin=0, vmax=1)
            
            title = f"{result['classifier']}"
            if result.get('pca'):
                title += f" (PCA={result['pca']})"
            title += f"\n{result['accuracy']:.3f}"
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('Predicted', fontsize=9)
            ax.set_ylabel('True', fontsize=9)
        
        # Hide unused subplots
        for idx in range(len(results), 6):
            axes[idx].axis('off')
        
        plt.suptitle('Confusion Matrices Comparison', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_per_digit_accuracy(self, results: List[Dict], save_path: str = None):
        """Plot per-digit accuracy comparison."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(10)
        width = 0.15
        
        for idx, result in enumerate(results):
            analysis = self.analyze_errors(result)
            acc = analysis['per_digit_accuracy']
            label = f"{result['classifier']}"
            if result.get('pca'):
                label += f"(PCA{result['pca']})"
            ax.bar(x + idx * width, acc, width, label=label, alpha=0.8)
        
        ax.set_xlabel('Digit', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Per-Digit Accuracy Comparison', fontsize=14)
        ax.set_xticks(x + width * (len(results) - 1) / 2)
        ax.set_xticklabels(range(10))
        ax.legend(loc='lower left', fontsize=9)
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_most_confused_pairs(self, result: Dict, top_n: int = 10, save_path: str = None):
        """Plot most commonly confused digit pairs."""
        analysis = self.analyze_errors(result)
        top_confusions = analysis['top_confusions'][:top_n]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        labels = [f"{c['true']}→{c['pred']}" for c in top_confusions]
        counts = [c['count'] for c in top_confusions]
        
        bars = ax.barh(labels, counts, color='coral')
        ax.set_xlabel('Number of Misclassifications', fontsize=12)
        ax.set_ylabel('True → Predicted', fontsize=12)
        ax.set_title(f'Top {top_n} Confused Digit Pairs - {result["classifier"]}', fontsize=14)
        ax.invert_yaxis()
        
        # Add value labels
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                    str(count), va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def generate_error_report(self, results: List[Dict], save_path: str = None) -> str:
        """Generate a text report of error analysis."""
        lines = []
        lines.append("=" * 80)
        lines.append("DIGIT CLASSIFICATION - ERROR ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        for result in results:
            analysis = self.analyze_errors(result)
            
            lines.append("-" * 80)
            lines.append(f"Classifier: {result['classifier']}")
            if result.get('pca'):
                lines.append(f"PCA Components: {result['pca']}")
            lines.append(f"Overall Accuracy: {result['accuracy']:.4f}")
            lines.append(f"Total Errors: {analysis['total_errors']} / 4000")
            lines.append("")
            
            lines.append("Per-Digit Accuracy:")
            for digit in range(10):
                acc = analysis['per_digit_accuracy'][digit]
                lines.append(f"  Digit {digit}: {acc:.4f}")
            lines.append("")
            
            lines.append("Top 5 Most Confused Pairs:")
            for i, conf in enumerate(analysis['top_confusions'][:5], 1):
                lines.append(f"  {i}. {conf['true']} → {conf['pred']}: "
                           f"{conf['count']} errors ({conf['rate']:.2%})")
            lines.append("")
        
        lines.append("=" * 80)
        
        report = "\n".join(lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Saved report: {save_path}")
        
        return report


def main(output_dir: Path = None):
    """Run error analysis on all classifiers."""
    import os

    # Setup paths
    data_path = Path(__file__).parent.parent / "data" / "raw" / "MINIST" / "digits4000.mat"
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results" / "figures"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    analyzer = ErrorAnalyzer(data_path)

    # Define classifiers to analyze (based on existing results)
    configs = [
        ('knn', {'n_neighbors': 1}, 'none', None, False),
        ('knn', {'n_neighbors': 3}, 'none', None, False),
        ('svm_rbf', {'C': 1.0, 'gamma': 'scale'}, 'scale255', None, True),
        ('svm_rbf', {'C': 1.0, 'gamma': 'scale'}, 'scale255', 50, True),
        ('qda', {'solver': 'svd'}, 'none', 50, False),
        ('logistic', {'C': 1.0, 'max_iter': 1000}, 'standard', None, True),
    ]

    print("Running error analysis...")
    results = []

    for clf_name, params, norm, pca, ova in configs:
        print(f"  Analyzing {clf_name} (pca={pca})...")
        result = analyzer.run_classifier(clf_name, params, norm, pca, ova)
        results.append(result)

        # Individual confusion matrix
        safe_name = f"{clf_name}_pca{pca if pca else 'none'}"
        analyzer.plot_confusion_matrix(
            result,
            save_path=figures_dir / f"confusion_matrix_{safe_name}.png"
        )

        # Most confused pairs
        analyzer.plot_most_confused_pairs(
            result,
            top_n=10,
            save_path=figures_dir / f"confused_pairs_{safe_name}.png"
        )

    # Comparison plots
    print("\nGenerating comparison plots...")
    analyzer.plot_error_comparison(results, save_path=figures_dir / "confusion_comparison.png")
    analyzer.plot_per_digit_accuracy(results, save_path=figures_dir / "per_digit_accuracy.png")

    # Generate report
    print("\nGenerating report...")
    report = analyzer.generate_error_report(results, save_path=figures_dir / "error_report.txt")
    print("\n" + report)

    print(f"\nAll figures saved to: {figures_dir}")
    # plt.show()  # Disabled for non-interactive mode


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None, help="Result directory (auto-generated if omitted)")
    args = parser.parse_args()
    base_dir = resolve_run_dir(args.run_dir)
    output_dir = get_script_output_dir(base_dir, Path(__file__).stem)
    with setup_run_logging(output_dir):
        main(output_dir)
