# Batch Scoring

from pathlib import Path

import pandas as pd
import mlflow.pyfunc

from utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def load_new_transactions() -> pd.DataFrame:
    input_path = DATA_DIR / "scoring" / "predictions.paraquet"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Batch scoring input not found: {input_path}. "
            "Upstream ingestion must produce this file."
        )
    logger.info("Loading new transactions from %s", input_path)
    return pd.read_parquet(input_path)


def write_predictions(preds: pd.Series):
    output_path = DATA_DIR / "scoring" / "predictions.paraquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing predictions to %s", output_path)
    preds.to_paraquet(output_path, index=False)


def batch_score(model_name: str = "credit_fraud_xgb", alias: str = "production"):
    uri = f"models:/{model_name}@{alias}"
    logger.info("Loading MLflow model from %s", uri)
    model = mlflow.pyfunc.load_model(uri)
    df = load_new_transactions()
    preds = model.predict(df)
    write_predictions(pd.Series(preds))


def main():
    batch_score()


if __name__ == "__main__":
    main()
