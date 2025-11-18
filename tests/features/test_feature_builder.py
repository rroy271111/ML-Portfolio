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

def test_should_not_produce_missing_values( raw_df):
    result = build_features(raw_df)
    assert not result.isna().any().any()

def test_should_extract_hour_and_dow_correctly(raw_df):
    result = build_features(raw_df)

    # valid timestamp (expected hour/dow)
    assert result.loc[0,"hour"] == 10
    assert result.loc[0, "dow"] == 0
    assert result.loc[1, "hour"] == 23
    assert result.loc[1, "dow"] ==  0

# valid timestamp (expected hour/dow)
def test_should_assign_zero_hour_and_dow_for_invalid_timestamps(raw_df):
    result = build_features(raw_df)

    assert result.loc[2, "hour"] == 0
    assert result.loc[2, "dow"] == 0
    assert result.loc[3, "hour"] == 0
    assert result.loc[3, "dow"] == 0


# amount feature test
def test_should_compute_log1p_on_amount(raw_df):
    result = build_features(raw_df)
    expected = np.log1p(raw_df["amount"].clip(lower=0))
    np.testing.assert_array_equal(result["amount_log"].values, expected.values)

# amount feature test
def test_should_flag_large_amounts_using_99th_percentile(raw_df):
    result = build_features(raw_df)
    q99 = raw_df["amount"].quantile(0.99)
    expected = (raw_df["amount"] > q99).astype(int)
    np.testing.assert_array_equal(result["is_large_amount"].values, expected.values)

# categorical encoding
def test_should_encode_categorical_features_as_integers(raw_df):
    result = build_features(raw_df)

    for col in ["card_token", "device_id", "ip"]:
        assert pd.api.types.is_integer_dtype(result[col])

# edge case
def test_should_fail_when_required_columns_missing():
    bad_df = pd.DataFrame({"amount": [1, 2, 3]}) # missing many fields

    with pytest.raises(KeyError):
        build_features(bad_df)

# edge case
def test_should_handle_empty_dataframe():
    empty_df = pd.DataFrame(columns=[
        "transaction_id", "card_token", "amount",
        "timestamp", "merchant_id", "device_id", "ip"
    ])

    # should not crash (returns empty DataFrame with correct columns)
    result = build_features(empty_df)

    assert len(result) == 0
    expected_cols = [
        "amount_log", "hour", "dow",
        "merchant_id", "card_token",
        "device_id", "ip", "is_large_amount"
    ]

    assert list(result.columns) == expected_cols