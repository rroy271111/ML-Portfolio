import os
import yaml
import joblib
import argparse
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc

from data.synthetic_data_generator import generate_synthetic_transactions
from features.feature_builder import build_features
from utils.logger import get_logger


def load_config(path: str) -> dict:
    """
    Load YAML configuration
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_and_save(df: pd.DataFrame, config: dict, logger):
    """
    Train an XGBoost model using parameters from config and save it.
    """
    # Build features
    df_feat = build_features(df)
    X = df_feat
    y = df["label"]

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, stratify=y, test_size=config["data"]["test_size"], random_state=config["data"]["random_state"]
    )
    logger.info(f"Data split: train={X_train.shape}, val={X_val.shape}")
    logger.info(f"Feature columns: {list(X_train.columns)}")

    # Model parameters from config
    model_config = config["model"]
    params = model_config.get("params", {})

    logger.info(f"Model configuration: {model_config}")

    classifier = XGBClassifier(
        n_estimators=params.get("n_estimators", 200),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.05),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        reg_lambda=params.get("reg_lambda", 1.0),
        eval_metric=params.get("eval_metric", "logloss"),
        use_label_encoder=False,
        #enable_categorical=True,
        random_state=config["data"]["random_state"]
    )

    # Train
    classifier.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=params.get("early_stopping_rounds", 20),
        verbose=False,
    )

    # Compute validation metrics
    y_val_pred = classifier.predict_proba(X_val)[:, 1]
    precision, recall, _ = precision_recall_curve(y_val, y_val_pred)
    pr_auc = auc(recall, precision)

    logger.info(f"Validation PR AUC: {pr_auc:.4f}")
    logger.info(f"Precision (sample): {precision[:5]}")
    logger.info(f"Recall (sample): {recall[:5]}")

    # Save the model
    out_path = model_config["out_path"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(classifier, out_path)
    logger.info(f"Model saved to {out_path}")

    return pr_auc


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost model from config")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    args = parser.parse_args()

    logger = get_logger("train_xgb")
    logger.info(f"Loading configuration from {args.config}")

    config = load_config(args.config)

    # Log config summary
    logger.info("-------Configuration Summary----------")
    for section, params in config.items():
        logger.info(f"[{section}]")
        for key, value in params.items():
            logger.info(f" {key}: {value}")

    logger.info("Generating synthetic data...")
    df = generate_synthetic_transactions(num_rows=config["data"]["num_rows"])
    logger.info(f"Generated dataset with shape {df.shape}")

    pr_auc = train_and_save(df, config, logger)
    logger.info(f"Training complete. Final PR AUC = {pr_auc:.4f}")


if __name__ == "__main__":
    main()
