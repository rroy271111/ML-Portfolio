import os
import sys
import joblib
import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from features.feature_builder import build_features

@pytest.fixture
def raw_df():
    """A small realistic dataset to test feature building"""
    return pd.DataFrame({
        "transaction_id": [1, 2, 3, 4],
        "card_token": ["A123", "B235", "A123", "C999"],
        "amount": [10, 500, 0, 10000],  # includes zero + large amount
        "timestamp": [
            "2024-01-01 10:05:00",
            "2024-01-01 23:30:00",
            None,
            "invalid"                           
        ],
        "merchant_id": [101, 102, 103, 104],
        "device_id": ["dev1", "dev2", "dev3", "dev1"],
        "ip": ["1.1.1.1", "2.2.2.2","3.3.3.3", "1.1.1.1"]
    })

# Tests
def test_build_features_output_columns(raw_df):
    """Ensure returned DataFrame has exactly the expected feature columns"""
    expected_cols = [
        'amount_log', 'hour', 'dow', 'merchant_id', 
        'card_token', 'device_id', 'ip', 'is_large_amount'
    ]

    result = build_features(raw_df)
    assert list(result.columns) == expected_cols