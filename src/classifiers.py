"""Classifier implementations for digit classification."""

import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from typing import Optional, Tuple


class OneVsAllClassifier:
    """One-vs-All classifier wrapper for multi-class classification.
    
    Trains one binary classifier per class.
    """
    
    def __init__(self, base_classifier, n_classes: int = 10):
        """
        Args:
            base_classifier: Scikit-learn classifier class (not instance)
            n_classes: Number of classes (10 for digits 0-9)
        """
        self.base_classifier = base_classifier
        self.n_classes = n_classes
        self.classifiers = []
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Train one classifier per class.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
        """
        self.classifiers = []
        
        for digit in range(self.n_classes):
            # Create binary labels: +1 for current digit, -1 for others
            y_binary = np.where(y == digit, 1, -1)
            
            # Train classifier
            clf = self.base_classifier(**kwargs)
            clf.fit(X, y_binary)
            self.classifiers.append(clf)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.
        
        Args:
            X: Test features
            
        Returns:
            Predicted labels (n_samples,)
        """
        scores = np.zeros((X.shape[0], self.n_classes))
        
        for digit, clf in enumerate(self.classifiers):
            # Get confidence scores
            if hasattr(clf, 'decision_function'):
                scores[:, digit] = clf.decision_function(X)
            elif hasattr(clf, 'predict_proba'):
                scores[:, digit] = clf.predict_proba(X)[:, 1]
            else:
                scores[:, digit] = clf.predict(X)
        
        # Select class with highest score
        return np.argmax(scores, axis=1)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy."""
        return np.mean(self.predict(X) == y)


class PCATransform:
    """PCA wrapper that keeps transformer for test data."""
    
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.fitted = False
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit PCA and transform training data."""
        X_transformed = self.pca.fit_transform(X)
        self.fitted = True
        return X_transformed
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform test data using fitted PCA."""
        if not self.fitted:
            raise ValueError("PCA not fitted yet!")
        return self.pca.transform(X)
    
    def explained_variance_ratio(self) -> float:
        """Return cumulative explained variance ratio."""
        return np.sum(self.pca.explained_variance_ratio_)


classifiers = {
    'knn': KNeighborsClassifier,
    'svm_linear': lambda **kwargs: SVC(kernel='linear', **kwargs),
    'svm_rbf': lambda **kwargs: SVC(kernel='rbf', **kwargs),
    'svm_poly': lambda **kwargs: SVC(kernel='poly', degree=3, **kwargs),
    'logistic': LogisticRegression,
    'lda': LinearDiscriminantAnalysis,
    'qda': QuadraticDiscriminantAnalysis,
    'linear_svc': LinearSVC,
}


def get_classifier(name: str, **kwargs):
    """Get a classifier by name.
    
    Args:
        name: Classifier name
        **kwargs: Parameters for the classifier
        
    Returns:
        Classifier instance
    """
    if name not in classifiers:
        raise ValueError(f"Unknown classifier: {name}. Available: {list(classifiers.keys())}")
    
    return classifiers[name](**kwargs)
