"""Data preprocessing utilities."""

import pandas as pd
import numpy as np
from typing import Optional, Tuple


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess raw data.
    
    Args:
        df: Raw input dataframe
        
    Returns:
        Cleaned dataframe
    """
    # TODO: Implement data cleaning
    return df


def split_data(
    df: pd.DataFrame, 
    target_col: str, 
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets.
    
    Args:
        df: Input dataframe
        target_col: Name of target column
        test_size: Proportion of test set
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
