"""
Shared data utilities:
- get_data: load from file or call synthetic generator
- train_val_split: performs safe split and returns X_train, X_val, y_train, y_val
- save_model: joblib.dump wrapper
"""

from __future__ import annotations
from typing import Any, Dict, Tuple

import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split as sk_train_test_split
import joblib


# Synthetic data generator loader


def __generator_synthetic(num_rows: int) -> pd.DataFrame:
    """Helper to import and call the synthetic data generator."""
    try:
        from data.synthetic_data_generator import generate_synthetic_transactions
    except Exception as exc:
        raise ImportError("Synthetic data generator not available: " + str(exc))
    return generate_synthetic_transactions(num_rows=num_rows)


# Preprocessing before training


def preprocess(df: pd.DataFrame, logger=None) -> pd.DataFrame:
    """
    Convert non-numeric columns (timestamp, ip, etc.) to usable features.
    - Extracts hour/day/month from timestamp columns
    - Encodes string/object columns as categorical integer codes
    """
    df = df.copy()

    # Convert timestamp to useful numeric features
    if "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(
        df["timestamp"]
    ):
        if logger:
            logger.debug("Extracting timestamp features (hour, day, month).")
        df["hour"] = df["timestamp"].dt.hour
        df["day"] = df["timestamp"].dt.day
        df["month"] = df["timestamp"].dt.month
        df = df.drop(columns=["timestamp"])

    # Encode object columns as categorical codes
    for col in df.select_dtypes(include=["object"]).columns:
        if logger:
            logger.debug(f"Encoding object column '{col}' as categorical codes.")
        df[col] = df[col].astype("category").cat.codes

    return df


# Data loading utility


def get_data(data_config: Dict[str, Any], logger=None) -> pd.DataFrame:
    """
    Load data from a file or generate synthetic data.

    data_config keys:
      - mode: "synthetic" or "file"
      - num_rows: (synthetic)
      - file_path: (file)
    """
    mode = data_config.get("mode", "synthetic")

    if mode == "synthetic":
        num_rows = int(data_config.get("num_rows", 5000))
        if logger:
            logger.info("Generating synthetic data: num_rows=%s", num_rows)
        df = __generator_synthetic(num_rows=num_rows)

    elif mode == "file":
        path = Path(data_config.get("file_path"))
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        if logger:
            logger.info("Loading data from file: %s", str(path))
        df = pd.read_csv(path)

    else:
        raise ValueError(f"Unknown data.mode: {mode}")

    return df


# Train / Validation Split


def train_val_split(
    df: pd.DataFrame, data_config: Dict[str, Any], logger=None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs a safe train/validation split.

    Automatically disables stratification when:
    - dataset too small (<8 rows)
    - single-class target
    - any class has fewer than 2 samples
    - sklearn's stratified split raises ValueError

    Returns:
        X_train, X_val, y_train, y_val
    """
    target = data_config.get("target_col", "label")
    test_size = float(data_config.get("test_size", 0.2))
    random_state = int(data_config.get("random_state", 42))

    if target not in df.columns:
        raise KeyError(
            f"Target column '{target}' not found in dataframe columns: {list(df.columns)}"
        )

    X = df.drop(columns=[target])
    y = df[target]

    # Decide whether to use stratify
    stratify_arg = y
    unique_classes = y.nunique()
    class_counts = y.value_counts()

    if unique_classes < 2 or len(df) < 8 or class_counts.min() < 2:
        if logger:
            logger.warning(
                "Stratification disabled due to insufficient class counts or dataset size."
            )
        stratify_arg = None

    try:
        X_train, X_val, y_train, y_val = sk_train_test_split(
            X, y, test_size=test_size, stratify=stratify_arg, random_state=random_state
        )
    except ValueError as e:
        # Fallback: disable stratify if sklearn fails
        if logger:
            logger.warning(f"Stratified split failed ({e}); retrying without stratify.")
        X_train, X_val, y_train, y_val = sk_train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    if logger:
        logger.debug(
            "train_val_split shapes: X_train=%s, X_val=%s", X_train.shape, X_val.shape
        )

    return X_train, X_val, y_train, y_val


# Model saving


def save_model(model: Any, out_path: str, logger=None) -> None:
    """Saves model to disk using joblib."""
    out_path = Path(out_path)
    os.makedirs(out_path.parent, exist_ok=True)
    joblib.dump(model, out_path)
    if logger:
        logger.info("Model written with joblib to %s", out_path)


def load_train_val_split(features_path: str, data_config: dict, logger=None):

    if logger:
        logger.info("Loading features from %s", features_path)

    df = pd.read_parquet(features_path)
    return train_val_split(
        df,
        data_config=data_config,
        logger=logger,
    )
