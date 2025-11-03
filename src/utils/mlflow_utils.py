# MLflow helper
from contextlib import contextmanager
from typing import Dict, Any
import importlib

@contextmanager
def mlflow_start_run_if_enabled(mlflow_cfg: Dict[str, Any], logger=None):
    enabled = bool(mlflow_cfg.get("enabled", False))
    if not enabled:
        yield None
        return 
    
    try:
        mlflow = importlib.import_module("mlflow")
    except Exception:
        if logger:
            logger.warning("MLflow not installed but mlflow.enabled=True")
        yield None
        return
    
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "file:./mlruns"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "default"))
    with mlflow.start_run():
        yield mlflow
    

