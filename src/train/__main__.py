"""Execução pelo terminal: ``python -m src.train``."""

import logging

import numpy as np

from src.logging_config import configure_logging
from src.preprocess import RANDOM_STATE
from src.train.runner import run_training

logger = logging.getLogger("src.train")

if __name__ == "__main__":
    configure_logging()
    np.random.seed(RANDOM_STATE)
    try:
        run_training()
    except (FileNotFoundError, KeyError, ValueError, RuntimeError, OSError) as exc:
        logger.error("Falha no treinamento: %s", exc)
        raise SystemExit(1) from exc
