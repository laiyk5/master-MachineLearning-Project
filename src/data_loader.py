"""Digit Classification Course Project - CS5487

MNIST Digit Classification (Default Project)
- Dataset: 4000 images (400 per class, digits 0-9)
- Feature dimension: 784 (28x28 images)
- Task: Multi-class classification
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Tuple, Dict


def load_digits_data(mat_file: str = None) -> Dict:
    """Load the digits dataset from .mat file.
    
    Args:
        mat_file: Path to digits4000.mat file
        
    Returns:
        Dictionary with keys:
            - digits_vec: (784, 4000) feature matrix
            - digits_labels: (4000,) labels (0-9)
            - trainset: (2, 2000) train indices for 2 trials
            - testset: (2, 2000) test indices for 2 trials
    """
    if mat_file is None:
        # Try to find in data directory
        mat_file = Path(__file__).parent.parent / "data" / "raw" / "digits4000.mat"
    
    data = sio.loadmat(mat_file)
    
    return {
        'digits_vec': data['digits_vec'],           # (784, 4000)
        'digits_labels': data['digits_labels'].flatten(),  # (4000,)
        'trainset': data['trainset'],               # (2, 2000)
        'testset': data['testset']                  # (2, 2000)
    }


def get_train_test_split(data: Dict, trial: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get train/test split for a specific trial.
    
    Args:
        data: Dictionary from load_digits_data
        trial: Trial number (0 or 1)
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    train_idx = data['trainset'][trial, :] - 1  # Convert to 0-indexed
    test_idx = data['testset'][trial, :] - 1
    
    X_train = data['digits_vec'][:, train_idx].T  # (2000, 784)
    X_test = data['digits_vec'][:, test_idx].T    # (2000, 784)
    y_train = data['digits_labels'][train_idx]    # (2000,)
    y_test = data['digits_labels'][test_idx]      # (2000,)
    
    return X_train, X_test, y_train, y_test


def normalize_features(X_train: np.ndarray, X_test: np.ndarray, method: str = "standard") -> Tuple[np.ndarray, np.ndarray]:
    """Normalize features.
    
    Args:
        X_train: Training features
        X_test: Test features
        method: 'standard' (zero mean, unit var), 'minmax' (0-1), 'none'
        
    Returns:
        Normalized X_train, X_test
    """
    if method == "standard":
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8
        X_train_norm = (X_train - mean) / std
        X_test_norm = (X_test - mean) / std
    elif method == "minmax":
        min_val = X_train.min(axis=0)
        max_val = X_train.max(axis=0)
        range_val = max_val - min_val + 1e-8
        X_train_norm = (X_train - min_val) / range_val
        X_test_norm = (X_test - min_val) / range_val
    elif method == "scale255":
        # Just divide by 255
        X_train_norm = X_train / 255.0
        X_test_norm = X_test / 255.0
    else:
        X_train_norm = X_train
        X_test_norm = X_test
    
    return X_train_norm, X_test_norm


def one_vs_all_labels(y: np.ndarray, digit: int) -> np.ndarray:
    """Convert labels to binary for one-vs-all classification.
    
    Args:
        y: Original labels (0-9)
        digit: The digit to classify as +1
        
    Returns:
        Binary labels (+1 for digit, -1 for others)
    """
    return np.where(y == digit, 1, -1)


def evaluate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate classification accuracy.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy (0-1)
    """
    return np.mean(y_true == y_pred)


def display_image(image_vec: np.ndarray, ax=None):
    """Display a 784-dim vector as 28x28 image.
    
    Args:
        image_vec: (784,) vector
        ax: Matplotlib axis (optional)
    """
    import matplotlib.pyplot as plt
    
    # Transpose because MATLAB stores column-major order
    img = image_vec.reshape(28, 28).T
    
    if ax is None:
        plt.figure(figsize=(3, 3))
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.show()
    else:
        ax.imshow(img, cmap='gray')
        ax.axis('off')
