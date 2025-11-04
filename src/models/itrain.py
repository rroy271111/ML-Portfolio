# ITrain - Orchestrator

from __future__ import annotations
import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from utils.logger import get_logger
from utils.data import get_data, train_test_split, save_model
from utils.metrics import compute_pr_auc
from utils.mlflow_utils import mlflow_start_run_if_enabled

def load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r") as fh:
        return yaml.safe_load(fh)

def import_trainer(trainer_path: str):
    mod = importlib.import_module(trainer_path)
    if not hasattr(mod, "train"):
        raise ImportError(f"Trainer {trainer_path} must expose train(df, config, logger)")
    return getattr(mod, "train")

def run(cfg_path: str) -> None:
    config = load_config(cfg_path)
    logger = get_logger("itrain", cfg=config.get("logging", {}))
    logger.info("Starting itrain")

    trainer_path = config.get("model", {}).get("trainer_path")
    if not trainer_path:
        logger.error("model.trainer_path missing in config")
        raise KeyError("model.trainer_path")
    
    df = get_data(config.get("data", {}), logger=logger)
    logger.info("Loaded data shape: %s", getattr(df, "shape", None))    

    X_train, X_val, y_train, y_val = train_val_split(df, config.get("data", {}), logger=logger)
    logger.info("Split shapes: train=%s val=%s", getattr(X_train,"shape", None), getattr(X_val, "shape", None))

    trainer_fn = import_trainer(trainer_path)

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

def parse_args(argv=None):
    p = argparse.ArgumentParser("itrain")
    p.add_argument(" config", "-c", default="configs/default.yaml")
    return p.parse_args(argv)

if __name__ == "__main__":
    args = parse_args()
    try:
        run(args.config)
    except Exception as e:
        logger = get_logger("itrain")
        logger.exception("Fatal: %s", e)
        sys.exit(2)
