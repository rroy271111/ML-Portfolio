import os
import sys
import shutil
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from models import train_xgboost


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


def fake_predict_proba(X):
    return np.array([[0.2, 0.8]] * len(X))


# test: ensure correct arguments passed to XGBClassifier
# @patch("models.train_xgboost.generate_synthetic_transactions")
@patch("models.train_xgboost.joblib.dump")
@patch("models.train_xgboost.XGBClassifier")
@patch("models.train_xgboost.build_features")
def test_xgb_parameters_passed(
    mock_build_features, mock_model, mock_dump, fake_df, fake_config, fake_logger
):
    mock_build_features.return_value = fake_df[["amount", "merchant"]]

    instance = MagicMock()
    mock_model.return_value = instance
    instance.predict_proba.return_value = np.array([[0.2, 0.8]])
    instance.predict_proba.side_effect = fake_predict_proba

    train_and_save(fake_df, fake_config, fake_logger)

    # mock_model.assert_called_once()
    args, kwargs = mock_model.call_args

    # check our custom hyperparameters were passed
    assert kwargs["n_estimators"] == 10
    assert kwargs["max_depth"] == 3
    assert kwargs["random_state"] == 42


# basic CLI test using capsys ( captures stdout/stderr)
@patch("models.train_xgboost.generate_synthetic_transactions")
@patch("models.train_xgboost.train_and_save")
@patch("models.train_xgboost.load_config")
# @patch("models.train_xgboost.get_logger")
def test_cli_runs(
    mock_load_config,
    mock_train_and_save,
    mock_gen,
    caplog,
):
    caplog.set_level("INFO", logger="train_xgboost")

    mock_load_config.return_value = {
        "data": {"num_rows": 10, "random_state": 42, "test_size": 0.2},
        "model": {"out_path": "model.pkl", "params": {}},
    }

    mock_gen.return_value = MagicMock()
    mock_train_and_save.return_value = 0.87

    # simulate CLI
    test_args = ["progname", "--config", "configs/test.yaml"]
    with patch.object(sys, "argv", test_args):
        train_xgboost.main()

    mock_load_config.assert_called_once()
    mock_gen.assert_called_once()
    mock_train_and_save.assert_called_once()

    # verify the log was captured
    assert "Training complete" in caplog.text
