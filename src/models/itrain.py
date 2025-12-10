# ITrain - Orchestrator

from __future__ import annotations
import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from sklearn.model_selection import train_test_split
from features.feature_builder import build_features
from utils.logger import get_logger
from utils.data import get_data, save_model
from utils.metrics import compute_pr_auc
from utils.mlflow_utils import mlflow_start_run_if_enabled


def load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r") as fh:
        cfg = yaml.safe_load(fh)
        if not isinstance(cfg, dict):
            raise ValueError(f"Invalid config format: expected dict, got {type(cfg)}")
        return cfg


def import_trainer(trainer_path: str):
    try:
        mod = importlib.import_module(trainer_path)
    except ModuleNotFoundError as e:
        raise ImportError(f"Trainer module not found: {trainer_path}") from e

    if not hasattr(mod, "train"):
        raise ImportError(f"Trainer {trainer_path} must expose train(X_train, y_train, X_val, y_val, config, logger)")
    return getattr(mod, "train")


def run(cfg_path: str) -> Dict[str, Any]:
    config = load_config(cfg_path)
    logger = get_logger("itrain", cfg=config.get("logging", {}))
    logger.info("Starting itrain")

    trainer_path = config.get("model", {}).get("trainer_path")
    if not trainer_path:
        logger.error("model.trainer_path missing in config")
        raise KeyError("model.trainer_path")

    data_cfg = config.get("data", {})
    df = get_data(data_cfg, logger=logger)
    logger.info("Loaded data shape: %s", getattr(df, "shape", None))

    # Build features from the raw data
    X = build_features(df, include_label=False)
    logger.info("Built features shape: %s", getattr(X, "shape", None))  
    logger.info(f"Feature columns: {list(X.columns)}")

    # Get label from the raw data
    y = df[data_cfg.get("target_col", "label")]

    #Split the features (X) and label(y)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        stratify=y,
        test_size=data_cfg.get("test_size", 0.2),
        random_state=data_cfg.get("random_state", 42)
    )
    
    logger.info("Split shapes: train=%s val=%s", getattr(X_train, "shape", None), getattr(X_val, "shape", None)
    )

    trainer_fn = import_trainer(trainer_path)

    metrics: Dict[str, Any] = {}
    with mlflow_start_run_if_enabled(config.get("mlflow", {}), logger=logger):
        model, metrics = trainer_fn(X_train, y_train, X_val, y_val, config=config, logger=logger)

        # Compute fallback PR AUC if trainer didn't return
        if "pr_auc" not in metrics:
            try:
                preds = None
                try:
                    preds = model.predict_proba(X_val)[:, 1]
                except Exception:
                    logger.debug("Model has no predict_proba")
                if preds is not None:
                    metrics["pr_auc"] = compute_pr_auc(y_val, preds)
            except Exception:
                logger.exception("Fallback metric computation failed")

        logger.info("Metrics: %s", metrics)

        # Save model if path configured
        out_path = config.get("model", {}).get("out_path")
        if out_path and model is not None:
            save_model(model, out_path, logger=logger)
            logger.info("Model saved to %s", out_path)
        else:
            logger.warning("Model not saved (no out_path or no model)")

    logger.info("Training complete")
    return metrics


def parse_args(argv=None):
    p = argparse.ArgumentParser("itrain")
    p.add_argument("-c", "--config", default="configs/default.yaml", help="Path to YAML config")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    try:
        run(args.config)
    except Exception as e:
        logger = get_logger("itrain")
        logger.exception("Fatal: %s", e)
        sys.exit(2)
