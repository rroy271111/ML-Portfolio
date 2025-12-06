# Train pipeline
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any

import mlflow
from mlflow import MlflowClient

from utils.logger import get_logger
from utils import data as data_utils
from utils import metrics as metrics_utils

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def load_training_data():
    """
    Docstring for load_training_data
    """
    X_train, X_val, y_train, y_val = data_utils.load_train_val_split(
        features_path=str(DATA_DIR / "features" / "features.parquet")
    )

    return X_train, X_val, y_train, y_val


def train_model() -> Tuple[Any, Dict[str, Any]]:
    """
    Docstring for train_model

    :return: Description
    :rtype: Tuple[Any, Dict[str, Any]]
    """
    X_train, X_val, y_train, y_val = data_utils.load_train_val_split(
        features_path=str(DATA_DIR / "features" / "features.parquet")
    )

    return X_train, X_val, y_train, y_val


def main():
    pass


if __name__ == "__main__":
    main()
