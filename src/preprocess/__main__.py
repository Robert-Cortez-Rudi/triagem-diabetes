"""Execução pelo terminal: ``python -m src.preprocess``."""

import logging

from src.logging_config import configure_logging
from src.preprocess.pipeline import run_preprocessing

logger = logging.getLogger("src.preprocess")

if __name__ == "__main__":
    configure_logging()
    try:
        run_preprocessing()
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.error("Falha no pré-processamento: %s", exc)
        raise SystemExit(1) from exc
