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
