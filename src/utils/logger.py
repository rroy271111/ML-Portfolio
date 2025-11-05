
import logging
import os
from datetime import datetime

def get_logger(name: str, cfg: dict | None = None, log_dir: str = "logs"):
    """
    Return a configured logger writing both to console and file.
    Optionally accepts a logging config dictionary.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(cfg.get("level", "INFO") if cfg else logging.INFO)
    logger.handlers.clear()  # Avoid duplicate logs on re-import

    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.addHandler(console_handler)

    return logger
