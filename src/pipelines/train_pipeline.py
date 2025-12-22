# Train pipeline
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    precision_recall_curve,
    auc,
)

import mlflow
from mlflow import MlflowClient
import mlflow.sklearn

from utils.logger import get_logger
from utils import data as data_utils
from utils import metrics as metrics_utils
import yaml
from trainers.xgb_trainer import train

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
print("PROJECT_ROOT =", PROJECT_ROOT)
DATA_DIR = PROJECT_ROOT / "src" / "data"


def load_training_data():
    """
    Pipeline level data loading orchestration.
    """
    with open(PROJECT_ROOT / "configs" / "default.yaml", "r") as f:
        config = yaml.safe_load(f)

    data_config = config.get("data", {})

    X_train, X_val, y_train, y_val = data_utils.load_train_val_split(
        features_path=str(DATA_DIR / "features" / "features.parquet"),
        data_config=data_config,
        logger=logger,
    )

    return X_train, X_val, y_train, y_val


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Docstring for train_model
    Call into XGBoost trainer and normalize output for MLflow.
    :return: Description
    :rtype: Tuple[Any, Dict[str, Any]]
    """
    logger = get_logger("train_pipeline")

    # Load config
    with open(PROJECT_ROOT / "configs" / "default.yaml", "r") as f:
        config = yaml.safe_load(f)

    logger.info("Starting XGBoost training via trainer")
    model, metrics = train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        config=config,
        logger=logger,
    )

    training_info = {
        "params": config.get("model", {}).get("params", {}),
        "metrics": metrics,
    }

    logger.info("Training finished")
    return model, training_info


def evaluate(model, X_val, y_val) -> Dict[str, float]:
    probs = model.predict_proba(X_val)[:, 1]
    preds = (probs > 0.5).astype(int)

    return {
        "roc_auc": metrics_utils.compute_pr_auc(y_val, probs),
        "pr_auc": average_precision_score(y_val, probs),
        "recall": recall_score(y_val, preds),
        "precision": precision_score(y_val, preds),
        "f1": f1_score(y_val, preds),
    }


def find_best_threshold(y_true, probs):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-8)
    best_idx = f1_scores.argmax()
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    return {
        "best_threshold": float(best_threshold),
        "best_precision": float(precision[best_idx]),
        "best_recall": float(recall[best_idx]),
        "best_f1": float(f1_scores[best_idx]),
        "pr_auc_curve": auc(recall, precision),
    }


def train_and_log(
    model_name: str = "credit_fraud_xgb",
    experiment_name: str = "credit_card_fraud_experiments",
):
    X_train, X_val, y_train, y_val = load_training_data()

    model, training_info = train_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    mlflow.set_tracking_uri(
        # overridden by env MLFLOW_TRACKING_URI in Docker/Airflow
        "http://localhost:5000"
    )

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(
        run_name=f"{model_name} - {datetime.utcnow().isoformat(timespec='seconds')}"
    ):
        # model, training_info = train_model()

        # log params
        params = training_info.get("params", {})
        for k, v in params.items():
            mlflow.log_param(k, v)

        # eval
        eval_metrics = evaluate(model, X_val, y_val)
        for k, v in eval_metrics.items():
            mlflow.log_metric(f"val_{k}", v)

        # threshold tuning
        threshold_info = find_best_threshold(y_val, model.predict_proba(X_val)[:, 1])

        for k, v in threshold_info.items():
            if k != "pr_auc_curve":
                mlflow.log_metric(f"th_{k}", v)

        mlflow.log_metric("val_pr_auc_curve", threshold_info["pr_auc_curve"])
        mlflow.log_param("selected_threshold", threshold_info["best_threshold"])

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name,
        )

        run_id = mlflow.active_run().info.run_id
        logger.info("Logged run to MLflow: run_id=%s", run_id)
        return run_id, eval_metrics


def promote_to_production(
    run_id: str,
    registered_name: str = "credit_fraud_xgb",
    alias: str = "production",
):
    client = MlflowClient()

    # find model version created by 'run'
    versions = client.search_model_versions(f"name='{registered_name}'")

    version = None
    for v in versions:
        if v.run_id == run_id:
            version = v
            break

    if version is None:
        raise RuntimeError(f"No model version found for run_id={run_id}")

    # set alias
    client.set_registered_model_alias(
        name=registered_name,
        alias=alias,
        version=version.version,
    )

    logger.info(
        "Promoted model via alias=%s, name=%s, versions=%s",
        alias,
        registered_name,
        version.version,
    )

    return version.version


def main():
    run_id, _ = train_and_log()
    promote_to_production(run_id)


if __name__ == "__main__":
    main()
