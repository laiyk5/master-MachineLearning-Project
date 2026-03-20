"""Feature engineering utilities."""

import pandas as pd
import numpy as np
from typing import List, Optional


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features from raw data.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dataframe with extracted features
    """
    # TODO: Implement feature extraction
    return df


def select_features(
    X: pd.DataFrame, 
    y: pd.Series, 
    n_features: int = 10
) -> List[str]:
    """Select top features using statistical methods.
    
    Args:
        X: Feature matrix
        y: Target variable
        n_features: Number of features to select
        
    Returns:
        List of selected feature names
    """
    from sklearn.feature_selection import SelectKBest, f_classif
    
    selector = SelectKBest(score_func=f_classif, k=n_features)
    selector.fit(X, y)
    
    return X.columns[selector.get_support()].tolist()
