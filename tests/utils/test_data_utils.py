import os
import joblib
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from utils import data

@pytest.fixture
def test_df():
    """Simple DataFrame fixture for preprocessing and split tests."""
    return pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2021-01-01 10:00:00", 
            "2021-01-02 11:00:00", 
            "2021-01-03 12:00:00", 
            "2021-01-04 13:00:00"
        ]),
        "ip": ["192.168.1.1", "10.0.0.1", "172.16.0.1", "192.168.1.2"],
        "amount": [10.5, 20.7, 15.3, 8.2],
        "label": [0, 1, 0, 1]
    })

def test_preprocess_transforms_columns(test_df):
    processed = data.preprocess(test_df)
    assert "timestamp" not in processed.columns
    
    for col in ["hour", "day", "month"]:
        assert col in processed.columns