# Batch Scoring

from pathlib import Path

import pandas as pd
import mlflow.pyfunc

from utils.logger import get_logger
from features.feature_builder import build_features
from validation.schemas.raw_transactions import raw_transactions_schema
from validation.schemas.features import feature_schema
import pandera.pandas as pa

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def load_new_transactions() -> pd.DataFrame:
    input_path = DATA_DIR / "scoring" / "input.parquet"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Batch scoring input not found: {input_path}. "
            "Upstream ingestion must produce this file."
        )
    logger.info("Loading new transactions from %s", input_path)
    return pd.read_parquet(input_path)


def write_predictions(preds: pd.DataFrame):
    output_path = DATA_DIR / "scoring" / "predictions.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing predictions to %s", output_path)
    preds.to_parquet(output_path, index=False)


def batch_score(model_name: str = "credit_fraud_xgb", alias: str = "production"):
    uri = f"models:/{model_name}@{alias}"
    logger.info("Loading MLflow model from %s", uri)
    model = mlflow.pyfunc.load_model(uri)

    # load and validate raw scoring input
    df_raw = load_new_transactions()
    try:
        df_raw = raw_transactions_schema.validate(df_raw)
    except pa.errors.SchemaError as e:
        logger.error("Invalid scoring batch: %s", e)
        raise

    # build and validate features
    X = build_features(df_raw, include_label=False)
    X = feature_schema.validate(X)

    # predict
    preds = model.predict(X)

    # persist scores with transaction_id
    out_df = df_raw[["transaction_id"]].assign(fraud_score=preds)

    # out_df = pd.DataFrame({e"fraud_score": preds})
    write_predictions(out_df)


def main():
    batch_score()


if __name__ == "__main__":
    main()
