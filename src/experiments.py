"""Main experiment runner for digit classification."""

import numpy as np
from pathlib import Path
import json
from typing import Dict, List
import time

from .data_loader import load_digits_data, get_train_test_split, normalize_features, evaluate_accuracy
from .classifiers import OneVsAllClassifier, PCATransform, get_classifier


class Experiment:
    """Run digit classification experiments."""
    
    def __init__(self, data_path: str = None):
        """
        Args:
            data_path: Path to digits4000.mat file
        """
        self.data = load_digits_data(data_path)
        self.results = []
        
    def run_trial(self, 
                  trial: int,
                  classifier_name: str,
                  classifier_params: Dict = None,
                  normalize: str = "none",
                  pca_components: int = None,
                  use_one_vs_all: bool = True) -> Dict:
        """Run a single trial.
        
        Args:
            trial: Trial number (0 or 1)
            classifier_name: Name of classifier
            classifier_params: Parameters for classifier
            normalize: Normalization method ('standard', 'minmax', 'scale255', 'none')
            pca_components: Number of PCA components (None to skip)
            use_one_vs_all: Use one-vs-all strategy for multi-class
            
        Returns:
            Dictionary with results
        """
        if classifier_params is None:
            classifier_params = {}
            
        # Load data
        X_train, X_test, y_train, y_test = get_train_test_split(self.data, trial)
        
        # Normalize
        if normalize != "none":
            X_train, X_test = normalize_features(X_train, X_test, method=normalize)
        
        # Apply PCA if requested
        if pca_components is not None:
            pca = PCATransform(n_components=pca_components)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
            explained_var = pca.explained_variance_ratio()
        else:
            explained_var = None
        
        # Train classifier
        start_time = time.time()
        
        if use_one_vs_all:
            base_clf = lambda **kwargs: get_classifier(classifier_name, **kwargs)
            clf = OneVsAllClassifier(base_clf, n_classes=10)
            clf.fit(X_train, y_train, **classifier_params)
        else:
            clf = get_classifier(classifier_name, **classifier_params)
            clf.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        # Evaluate
        y_pred = clf.predict(X_test)
        accuracy = evaluate_accuracy(y_test, y_pred)
        
        result = {
            'trial': trial,
            'classifier': classifier_name,
            'classifier_params': classifier_params,
            'normalize': normalize,
            'pca_components': pca_components,
            'explained_variance': explained_var,
            'use_one_vs_all': use_one_vs_all,
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
                       use_one_vs_all: bool = True) -> Dict:
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
                use_one_vs_all=use_one_vs_all
            )
            accuracies.append(result['accuracy'])
        
        summary = {
            'classifier': classifier_name,
            'params': classifier_params,
            'normalize': normalize,
            'pca_components': pca_components,
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
