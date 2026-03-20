import pytest
import pandas as pd
import numpy as np


def test_import():
    """Test that all modules can be imported."""
    from src import preprocessing, features
    assert True


def test_clean_data():
    """Test data cleaning function."""
    from src.preprocessing import clean_data
    
    df = pd.DataFrame({
        'A': [1, 2, None, 4],
        'B': ['a', 'b', 'c', 'd']
    })
    
    result = clean_data(df)
    assert isinstance(result, pd.DataFrame)
