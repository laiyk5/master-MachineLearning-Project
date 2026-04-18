"""Machine Learning Course Project - Digit Classification

Modules:
    data_loader: Load and preprocess MNIST digits data
    classifiers: Classification algorithms (SVM, Logistic Regression, etc.)
    experiments: Experiment runner and evaluation
    augmentation: Geometric data augmentation for digit images
    preprocessing: Data cleaning utilities
    features: Feature engineering
    utils: Result directory management and logging utilities
"""

from . import data_loader
from . import classifiers
from . import experiments
from . import augmentation
from . import preprocessing
from . import features
from . import utils

__all__ = [
    "data_loader",
    "classifiers",
    "experiments",
    "augmentation",
    "preprocessing",
    "features",
    "utils",
]
