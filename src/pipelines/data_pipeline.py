# Data pipeline

"""
Thin wrappers to orchestrate data ingestion & feature building.

"""

from pathlib import Path

from utils.logger import get_logger
from data.synthetic_data_generator import generate_synthetic_transactions

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def ingest_raw_data():
    """
    Call synthetic generator
    """

    raw_path = DATA_DIR / "raw" / "transactions.parquet"
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Generating synthetic raw data at %s", raw_path)
    generate_synthetic_transactions(output_path=str(raw_path))
    logger.info("Raw data generated.")


def main():
    ingest_raw_data()


if __name__ == "__main__":
    main()
