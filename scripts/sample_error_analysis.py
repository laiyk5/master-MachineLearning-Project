"""Analyze whether errors are sample-specific (hard samples) or model-specific."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import pickle

from src.data_loader import load_digits_data, get_train_test_split, normalize_features
from src.classifiers import OneVsAllClassifier, PCATransform, get_classifier
from src.utils import resolve_run_dir, setup_run_logging


class SampleErrorAnalyzer:
    """Analyze errors at the sample level across multiple models."""
    
    def __init__(self, data_path: str = None):
        self.data = load_digits_data(data_path)
        self.model_predictions = {}  # model_name -> predictions per trial
        self.test_indices = {}  # trial -> test indices
        
    def run_classifier_collect_predictions(self,
                                           name: str,
                                           classifier_name: str,
                                           classifier_params: Dict = None,
                                           normalize: str = "none",
                                           pca_components: int = None,
                                           use_one_vs_all: bool = True):
        """Run classifier and collect per-sample predictions."""
        if classifier_params is None:
            classifier_params = {}
        
        all_predictions = []
        all_true = []
        all_indices = []
        
        for trial in [0, 1]:
            X_train, X_test, y_train, y_test = get_train_test_split(self.data, trial)
            
            # Store test indices for later image retrieval
            train_idx = self.data['trainset'][trial, :] - 1
            test_idx = self.data['testset'][trial, :] - 1
            self.test_indices[trial] = test_idx
            
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
            all_true.extend(y_test)
            all_indices.extend(test_idx)
        
        self.model_predictions[name] = {
            'predictions': np.array(all_predictions),
            'true_labels': np.array(all_true),
            'indices': np.array(all_indices),
            'correct': np.array(all_predictions) == np.array(all_true)
        }
    
    def find_hard_samples(self) -> Dict:
        """Find samples that multiple models get wrong."""
        n_models = len(self.model_predictions)
        if n_models == 0:
            return {}
        
        # Get first model's data as reference
        first_model = list(self.model_predictions.values())[0]
        n_samples = len(first_model['predictions'])
        
        # Count how many models get each sample wrong
        error_counts = np.zeros(n_samples, dtype=int)
        error_details = [[] for _ in range(n_samples)]
        
        for model_name, pred_data in self.model_predictions.items():
            incorrect = ~pred_data['correct']
            for idx in range(n_samples):
                if incorrect[idx]:
                    error_counts[idx] += 1
                    error_details[idx].append({
                        'model': model_name,
                        'predicted': pred_data['predictions'][idx],
                        'true': pred_data['true_labels'][idx]
                    })
        
        # Categorize samples
        hard_samples = []  # Wrong by all models
        medium_samples = []  # Wrong by some models
        easy_samples = []  # Wrong by few/no models
        
        for idx in range(n_samples):
            true_label = first_model['true_labels'][idx]
            sample_idx = first_model['indices'][idx]
            
            sample_info = {
                'sample_index': int(sample_idx),
                'position_in_test': idx,
                'true_label': int(true_label),
                'error_count': int(error_counts[idx]),
                'n_models': n_models,
                'error_details': error_details[idx]
            }
            
            if error_counts[idx] == n_models:
                hard_samples.append(sample_info)
            elif error_counts[idx] > 0:
                medium_samples.append(sample_info)
            else:
                easy_samples.append(sample_info)
        
        return {
            'hard': hard_samples,
            'medium': medium_samples,
            'easy': easy_samples,
            'total_samples': n_samples
        }
    
    def visualize_hard_samples(self, max_samples: int = 20, save_path: str = None):
        """Visualize samples that all models get wrong."""
        analysis = self.find_hard_samples()
        hard_samples = analysis['hard'][:max_samples]
        
        if not hard_samples:
            print("No hard samples found (all models agree on correct predictions)")
            return
        
        n_samples = min(len(hard_samples), max_samples)
        fig, axes = plt.subplots(n_samples, 2, figsize=(8, n_samples * 2.5))
        
        if n_samples == 1:
            axes = axes.reshape(1, 2)
        
        for i, sample in enumerate(hard_samples):
            sample_idx = sample['sample_index']
            true_label = sample['true_label']
            
            # Get the image - transpose because MATLAB stores column-major
            img_vec = self.data['digits_vec'][:, sample_idx]
            img = img_vec.reshape(28, 28).T  # Transpose to fix orientation
            
            # Show image
            axes[i, 0].imshow(img, cmap='gray')
            axes[i, 0].set_title(f'Sample #{sample_idx} | True: {true_label}')
            axes[i, 0].axis('off')
            
            # Show prediction distribution
            predictions = [d['predicted'] for d in sample['error_details']]
            pred_counts = {}
            for p in predictions:
                pred_counts[p] = pred_counts.get(p, 0) + 1
            
            labels = list(pred_counts.keys())
            counts = list(pred_counts.values())
            
            axes[i, 1].bar(labels, counts, color='coral')
            axes[i, 1].set_xlabel('Predicted Label')
            axes[i, 1].set_ylabel('# Models')
            axes[i, 1].set_title('Model Predictions')
            axes[i, 1].set_xticks(range(10))
            
        plt.suptitle(f'"Hard" Samples - All {analysis["hard"][0]["n_models"]} Models Wrong', 
                     fontsize=14, y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig, hard_samples
    
    def visualize_medium_samples(self, max_samples: int = 15, save_path: str = None):
        """Visualize samples that some models get right, some wrong."""
        analysis = self.find_hard_samples()
        # Sort medium samples by how many models got them wrong
        medium_samples = sorted(analysis['medium'], 
                               key=lambda x: x['error_count'], 
                               reverse=True)[:max_samples]
        
        if not medium_samples:
            print("No medium samples found")
            return
        
        n_samples = min(len(medium_samples), max_samples)
        fig, axes = plt.subplots(n_samples, 2, figsize=(8, n_samples * 2.5))
        
        if n_samples == 1:
            axes = axes.reshape(1, 2)
        
        for i, sample in enumerate(medium_samples):
            sample_idx = sample['sample_index']
            true_label = sample['true_label']
            error_count = sample['error_count']
            n_models = sample['n_models']
            
            # Get the image - transpose because MATLAB stores column-major
            img_vec = self.data['digits_vec'][:, sample_idx]
            img = img_vec.reshape(28, 28).T  # Transpose to fix orientation
            
            # Show image
            axes[i, 0].imshow(img, cmap='gray')
            axes[i, 0].set_title(f'Sample #{sample_idx} | True: {true_label}')
            axes[i, 0].axis('off')
            
            # Show prediction distribution
            predictions = [d['predicted'] for d in sample['error_details']]
            pred_counts = {}
            for p in predictions:
                pred_counts[p] = pred_counts.get(p, 0) + 1
            
            labels = list(pred_counts.keys())
            counts = list(pred_counts.values())
            
            axes[i, 1].bar(labels, counts, color='orange', alpha=0.7)
            axes[i, 1].axhline(y=n_models - error_count, color='green', 
                              linestyle='--', label=f'{n_models - error_count} correct')
            axes[i, 1].set_xlabel('Predicted Label')
            axes[i, 1].set_ylabel('# Models')
            axes[i, 1].set_title(f'{error_count}/{n_models} models wrong')
            axes[i, 1].set_xticks(range(10))
            axes[i, 1].legend()
            
        plt.suptitle('"Medium" Samples - Some Models Correct, Some Wrong', 
                     fontsize=14, y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig, medium_samples
    
    def generate_sample_error_report(self, save_path: str = None) -> str:
        """Generate detailed report on sample-level errors."""
        analysis = self.find_hard_samples()
        
        lines = []
        lines.append("=" * 80)
        lines.append("SAMPLE-LEVEL ERROR ANALYSIS")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Total test samples: {analysis['total_samples']}")
        lines.append(f"Number of models compared: {len(self.model_predictions)}")
        lines.append("")
        
        # Summary statistics
        hard_count = len(analysis['hard'])
        medium_count = len(analysis['medium'])
        easy_count = len(analysis['easy'])
        
        lines.append("ERROR DISTRIBUTION:")
        lines.append(f"  Hard samples (all models wrong):   {hard_count:4d} ({100*hard_count/analysis['total_samples']:.1f}%)")
        lines.append(f"  Medium samples (some wrong):       {medium_count:4d} ({100*medium_count/analysis['total_samples']:.1f}%)")
        lines.append(f"  Easy samples (all models correct): {easy_count:4d} ({100*easy_count/analysis['total_samples']:.1f}%)")
        lines.append("")
        
        # Hard samples details
        if analysis['hard']:
            lines.append("-" * 80)
            lines.append("HARD SAMPLES (All models wrong - likely ambiguous or mislabeled):")
            lines.append("-" * 80)
            
            # Group by true label
            by_true_label = {}
            for s in analysis['hard']:
                label = s['true_label']
                if label not in by_true_label:
                    by_true_label[label] = []
                by_true_label[label].append(s)
            
            for label in sorted(by_true_label.keys()):
                samples = by_true_label[label]
                lines.append(f"\n  True Label = {label} ({len(samples)} samples):")
                
                for s in samples[:5]:  # Show first 5 per label
                    predictions = [d['predicted'] for d in s['error_details']]
                    pred_str = ', '.join([f"{p}({predictions.count(p)})" for p in set(predictions)])
                    lines.append(f"    Sample #{s['sample_index']}: predicted as {pred_str}")
                
                if len(samples) > 5:
                    lines.append(f"    ... and {len(samples) - 5} more")
        
        # Medium samples - show breakdown
        if analysis['medium']:
            lines.append("")
            lines.append("-" * 80)
            lines.append("MEDIUM SAMPLES (Inconsistent across models - model capability issue):")
            lines.append("-" * 80)
            
            # Sort by how many models got it wrong
            sorted_medium = sorted(analysis['medium'], 
                                  key=lambda x: x['error_count'], 
                                  reverse=True)
            
            for s in sorted_medium[:20]:  # Top 20
                correct = s['n_models'] - s['error_count']
                predictions = [d['predicted'] for d in s['error_details']]
                pred_str = ', '.join([f"{p}({predictions.count(p)})" for p in set(predictions)])
                lines.append(f"  Sample #{s['sample_index']} (true={s['true_label']}): "
                           f"{correct}/{s['n_models']} models correct, "
                           f"wrong predictions: {pred_str}")
        
        # Agreement analysis
        lines.append("")
        lines.append("-" * 80)
        lines.append("MODEL AGREEMENT ANALYSIS:")
        lines.append("-" * 80)
        
        # For each medium sample, check if models agree on the wrong prediction
        agreement_on_wrong = 0
        disagreement_on_wrong = 0
        
        for s in analysis['medium']:
            wrong_predictions = [d['predicted'] for d in s['error_details']]
            if len(set(wrong_predictions)) == 1:
                agreement_on_wrong += 1
            else:
                disagreement_on_wrong += 1
        
        lines.append(f"  Models agree on wrong prediction: {agreement_on_wrong} samples")
        lines.append(f"  Models disagree on wrong prediction: {disagreement_on_wrong} samples")
        lines.append("")
        
        # Conclusion
        lines.append("=" * 80)
        lines.append("CONCLUSION:")
        lines.append("=" * 80)
        
        if hard_count > 0:
            hard_pct = 100 * hard_count / analysis['total_samples']
            lines.append(f"• {hard_pct:.1f}% of samples are 'hard' - these are likely genuinely")
            lines.append("  ambiguous, poorly written, or potentially mislabeled digits.")
            lines.append("  Improving models won't help much on these.")
            lines.append("")
        
        if medium_count > 0:
            medium_pct = 100 * medium_count / analysis['total_samples']
            lines.append(f"• {medium_pct:.1f}% of samples are 'medium' - different models")
            lines.append("  disagree on these. This suggests model capability differences.")
            lines.append("  Better models could improve here.")
        
        report = "\n".join(lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Saved report: {save_path}")
        
        return report


def main(output_dir: Path = None):
    """Run sample-level error analysis."""
    # Setup paths
    data_path = Path(__file__).parent.parent / "data" / "raw" / "MINIST" / "digits4000.mat"
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results" / "figures"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    analyzer = SampleErrorAnalyzer(data_path)

    # Run multiple classifiers
    print("Running classifiers to collect predictions...")

    configs = [
        ('KNN-1', 'knn', {'n_neighbors': 1}, 'none', None, False),
        ('KNN-3', 'knn', {'n_neighbors': 3}, 'none', None, False),
        ('SVM-RBF', 'svm_rbf', {'C': 1.0, 'gamma': 'scale'}, 'scale255', None, True),
        ('SVM-RBF-PCA50', 'svm_rbf', {'C': 1.0, 'gamma': 'scale'}, 'scale255', 50, True),
        ('QDA-PCA50', 'qda', {'solver': 'svd'}, 'none', 50, False),
    ]

    for name, clf, params, norm, pca, ova in configs:
        print(f"  Running {name}...")
        analyzer.run_classifier_collect_predictions(name, clf, params, norm, pca, ova)

    print("\nAnalyzing sample-level errors...")

    # Generate report
    report = analyzer.generate_sample_error_report(
        save_path=figures_dir / "sample_error_report.txt"
    )
    print("\n" + report)

    # Visualize hard samples
    print("\nVisualizing hard samples...")
    analyzer.visualize_hard_samples(
        max_samples=15,
        save_path=figures_dir / "hard_samples.png"
    )

    # Visualize medium samples
    print("Visualizing medium samples...")
    analyzer.visualize_medium_samples(
        max_samples=12,
        save_path=figures_dir / "medium_samples.png"
    )

    print(f"\nAll outputs saved to: {figures_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None, help="Result directory (auto-generated if omitted)")
    args = parser.parse_args()
    output_dir = resolve_run_dir(args.run_dir)
    with setup_run_logging(output_dir):
        main(output_dir)
