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
        "timestamp": pd.date_range(
            "2021-01-01 10:00:00",
            periods=10,
            freq="D" 
            
        ),
        "ip": [f"192.168.1.1.{i}" for i in range(10)],
        "amount": np.random.uniform(10, 100, 10),
        "label": [0, 1] * 5
    })

def test_preprocess_transforms_columns(test_df):
    processed = data.preprocess(test_df)
    assert "timestamp" not in processed.columns
    
    for col in ["hour", "day", "month"]:
        assert col in processed.columns

    # ip column should be encoded as integers
    assert pd.api.types.is_integer_dtype(processed["ip"])


# Test for train_val_split()
def test_train_val_split_returns_expected_shapes(test_df):
    cfg = {"target_col": "label", "test_size":0.25, "random_state":42}

    X_train, X_val, y_train, y_val = data.train_val_split(test_df, cfg)

    # total_rows = len(X_train) + len(X_val)
    total_rows = len(X_train) + len(X_val)
    assert total_rows == len(test_df)

    # shape consistency
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)


def test_train_val_split_small_dataset_no_stratify():
    df = pd.DataFrame({
        "feat": range(6),
        "label": [0,1, 0, 1, 0, 1]
    })

    cfg = {"target_col": "label", "test_size":0.33, "random_state":42}

    X_train, X_val, y_train, y_val = data.train_val_split(df, cfg)

    assert len(X_train) + len(X_val) == len(df)
    assert len(y_train) == len(X_train)
    assert len(y_val) == len(X_val)

def test_train_val_split_returns_expected_shapes(test_df):
    cfg = {"target_col": "label", "test_size": 0.25, "random_state": 42}

    X_train, X_val, y_train, y_val = data.train_val_split(test_df, cfg)

    # row counts match original
    assert len(X_train) + len(X_val) == len(test_df)

    # features and labels align
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val) 

def test_train_val_split_small_dataset_no_stratify():
    df = pd.DataFrame({
        "feat": range(6),
        "label": [0, 1, 0, 1, 0, 1]
    })

    cfg = {"target_col": "label", "test_size": 0.33, "random_state": 42}

    # should not raise even though statify would fail
    X_train, X_val, y_train, y_val = data.train_val_split(df, cfg)

    assert len(X_train) + len(X_val) == len(df)
    assert len(y_train) == len(X_train)
    assert len(y_val) == len(X_val)

# single class dataset (stratify must be disabled)
def test_train_val_split_single_class():
    df = pd.DataFrame({
        "feat": range(10),
        "label": [1] * 10
    })

    cfg = {"target_col": "label", "test_size": 0.2, "random_state": 42}

    X_train, X_val, y_train, y_val = data.train_val_split(df, cfg)

    # row counts correct
    assert len(X_train) + len(X_val) == 10

    # all labels still 1
    assert (y_train == 1).all()
    assert (y_val == 1).all()

# missing target column - KeyError
def test_train_val_split_missing_target_column():
    df = pd.DataFrame({
        "amount": [10, 20, 30],
        "labelX": [0, 1, 0]
    })

    cfg = {"target_col": "label", "test_size": 0.2}

    with pytest.raises(KeyError):
        data.train_val_split(df, cfg)   

# invalid test_size - sklearn should error
def test_train_val_split_invalid_test_size():
    df = pd.DataFrame({
        "feat": range(10),
        "label":[0, 1] * 5
    })

    # invalid
    cfg = {"target_col": "label", "test_size": 5.0}

    with pytest.raises(ValueError):
        data.sk_train_test_split(df,cfg)

# stratify fails due to extreme imbalance (must fallback)
def test_train_val_split_stratify_fallback_on_imbalance():
   df = pd.DataFrame({
       "feat": range(10),
       "label": [0] * 9 + [1]   # 9:1 imbalance (stratified split will fail)
   })

   cfg = {"target_col": "label", "test_size": 0.2, "random_state": 42}

   X_train, X_val, y_train, y_val = data.train_val_split(df,cfg)

   assert len(X_train) + len(X_val) == 10
   assert len(X_train) == len(y_train)
   assert len(X_val) == len(y_val)