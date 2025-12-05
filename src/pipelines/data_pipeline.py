# Data pipeline

"""
Thin wrappers to orchestrate data ingestion & feature building.

"""

from pathlib import Path

from utils.logger import get_logger
from data.synthetic_data_generator import generate_synthetic_transactions
from features.feature_builder import build_features_from_path as fb_build_features

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def ingest_raw_data():
    """
    Call synthetic generator
    """

    raw_path = DATA_DIR / "raw" / "transactions.parquet"

    if raw_path.exists():
        logger.info("Raw data exists, skipping ingestion: %s", raw_path)
        return

    raw_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Generating synthetic raw data at %s", raw_path)
    generate_synthetic_transactions(output_path=str(raw_path))
    logger.info("Raw data generated.")


def build_features():
    """
    Call feature builder on the raw data and write features.
    """

    raw_path = DATA_DIR / "raw" / "transactions.parquet"
    feat_path = DATA_DIR / "features" / "features.parquet"

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data missing: {raw_path}")

    feat_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Building features from %s -> %s", raw_path, feat_path)
    fb_build_features(input_path=str(raw_path), output_path=str(feat_path))
    logger.info("Features built.")


def main():
    ingest_raw_data()
    build_features()


if __name__ == "__main__":
    main()
