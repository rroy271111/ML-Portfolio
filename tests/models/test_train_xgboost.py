import os
import shutil
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from models.train_xgboost import (
    load_config,
    train_and_save,
)


# Fixture
@pytest.fixture
def fake_config(tmp_path):
    return {
        "data": {
            "test_size": 0.2,
            "random_state": 42,
            "num_rows": 100,
        },
        "model": {
            "out_path": str(tmp_path / "model" / "clf.pkl"),
            "params": {
                "n_estimators": 10,
                "max_depth": 3,
            },
        },
    }


@pytest.fixture
def fake_df():
    return pd.DataFrame(
        {
            "amount": [100, 200, 300, 150, 130, 160, 170, 140, 210, 220],
            "merchant": [1, 0, 0, 1, 1, 0, 1, 0, 1, 0],
            "label": [0, 1, 0, 1, 0, 0, 1, 0, 1, 1],
        }
    )


@pytest.fixture
def fake_logger():
    logger = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    return logger


# test: loading_config
def test_load_config(tmp_path):
    file = tmp_path / "cfg.yaml"
    file.write_text(
        """
model:
                    name: xgboost
data:
                    rows: 10

"""
    )
    cfg = load_config(str(file))
    assert cfg["model"]["name"] == "xgboost"
    assert cfg["data"]["rows"] == 10


# test : train_and_save
@patch("models.train_xgboost.build_features")
@patch("models.train_xgboost.joblib.dump")
def test_train_and_save_success(
    mock_dump, mock_build_features, fake_df, fake_config, fake_logger
):
    # Fake build_features return DataFrame with same rows
    mock_build_features.return_value = fake_df[["amount", "merchant"]]

    pr = train_and_save(fake_df, fake_config, fake_logger)

    # AUC range sanity check
    assert isinstance(pr, float)
    assert 0.0 <= pr <= 1.0

    # ensure model was saved
    mock_dump.assert_called_once()

    out_path = fake_config["model"]["out_path"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
