import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from xgboost import XGBClassifier

from trainers.xgb_trainer import train


@pytest.fixture
def dummy_data():
    # simple small binary classification dataset
    X_train = np.random.rand(20, 4)
    y_train = np.random.randint(0, 2, size=20)
    X_val = np.random.rand(10, 4)
    y_val = np.random.randint(0, 2, size=10)
    return X_train, y_train, X_val, y_val


@pytest.fixture
def dummy_config():
    return {
        "model": {
            "params": {
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.2,
                "subsample": 0.5,
                "colsample_bytree": 0.5,
                "gamma": 1,
                "reg_alpha": 0.1,
                "early_stopping_rounds": 5,
                "random_state": 123,
                "eval_metric": "logloss",
            }
        }
    }


@pytest.fixture
def fake_logger():
    return MagicMock()


# Path smoke test
def test_train_path(dummy_data, dummy_config, fake_logger):
    X_train, y_train, X_val, y_val = dummy_data

    model, metrics = train(
        X_train, y_train, X_val, y_val, config=dummy_config, logger=fake_logger
    )

    assert isinstance(model, XGBClassifier)
    assert "val_pr_auc" in metrics
    assert "best_iteration" in metrics
    fake_logger.info.assert_called()


# default params when config is empty
def test_train_uses_default_params(dummy_data, fake_logger):
    X_train, y_train, X_val, y_val = dummy_data

    model, metrics = train(
        X_train, y_train, X_val, y_val, config={}, logger=fake_logger
    )

    assert model.get_params()["n_estimators"] == 200
    assert model.get_params()["max_depth"] == 6
    assert metrics["val_pr_auc"] is not None


# test: XGBoost.fit() is claled with parameters
def test_fit_called_with_eval_set(dummy_data, dummy_config, fake_logger):
    X_train, y_train, X_val, y_val = dummy_data

    with patch.object(XGBClassifier, "fit", return_value=None) as mock_fit:
        train(X_train, y_train, X_val, y_val, dummy_config, fake_logger)

    mock_fit.assert_called_once()

    # check eval_set parameters
    args, kwargs = mock_fit.call_args
    assert "eval_set" in kwargs
    assert isinstance(kwargs["eval_set"], list)
    assert len(kwargs["eval_set"]) == 1


# test logging messages
def test_trainer_logs_steps(dummy_data, dummy_config, fake_logger):
    train(*dummy_data, dummy_config, fake_logger)

    assert fake_logger.info.call_count >= 2
    fake_logger.info.assert_any_call("Fitting XGBoost model...")


# test: PR-AUC failure should not raise
def test_metrics_fallback_when_predict_proba_fails(
    dummy_data, dummy_config, fake_logger
):
    X_train, y_train, X_val, y_val = dummy_data

    with patch.object(XGBClassifier, "predict_proba", side_effect=Exception("boom")):
        model, metrics = train(
            X_train, y_train, X_val, y_val, config=dummy_config, logger=fake_logger
        )
        assert metrics["val_pr_auc"] is None
        fake_logger.warning.assert_called()


# predict_proba should return valid shape
def test_model_probability_shape(dummy_data, dummy_config, fake_logger):
    X_train, y_train, X_val, y_val = dummy_data

    model, _ = train(
        X_train, y_train, X_val, y_val, config=dummy_config, logger=fake_logger
    )

    proba = model.predict_proba(X_val)
    assert proba.shape == (10, 2)


# early stopping is passed down
def test_early_stopping_rounds_passed(dummy_data, dummy_config, fake_logger):
    X_train, y_train, X_val, y_val = dummy_data

    with patch.object(XGBClassifier, "fit") as mock_fit:
        train(X_train, y_train, X_val, y_val, config=dummy_config, logger=fake_logger)

    _, kwargs = mock_fit.call_args
    assert kwargs["early_stopping_rounds"] == 5
