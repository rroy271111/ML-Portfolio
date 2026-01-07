from utils.logger import get_logger
from pipelines.data_pipeline import ingest_raw_data, build_features

from pipelines.train_pipeline import train_and_log, promote_to_production
from pipelines.batch_scoring import batch_score

logger = get_logger("pipeline-runner")


def run_train():
    logger.info("--PIPELINE: TRAIN START--")

    ingest_raw_data()
    build_features()

    run_id, metrics = train_and_log()
    logger.info("Training metrics: %s", metrics)

    promote_to_production(run_id)

    logger.info("-- PIPELINE: TRAIN COMPLETE")


def run_score():
    logger.info("--PIPELINE: SCORE START--")
    batch_score()
    logger.info("-- PIPELINE: SCORE COMPLETE--")


def main(mode: str):
    if mode == "train":
        run_train()
    elif mode == "score":
        run_score()
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("pipeline-runner")
    parser.add_argument(
        "--mode",
        choices=["train", "score"],
        required=True,
        help="Pipeline mode to run",
    )

    args = parser.parse_args()
    main(args.mode)
