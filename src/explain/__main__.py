"""Execução pelo terminal: ``python -m src.explain``."""

import json
import logging

from src.explain.explanation import get_prediction_explanation
from src.explain.patient import EXAMPLE_PATIENT
from src.logging_config import configure_logging

logger = logging.getLogger("src.explain")

if __name__ == "__main__":
    configure_logging()
    logger.info("Paciente de teste: idoso (70-74 anos), IMC 34, hipertenso, sem atividade física.")
    try:
        result = get_prediction_explanation(EXAMPLE_PATIENT)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Falha na explicação: %s", exc)
        raise SystemExit(1) from exc
    print(json.dumps(result, indent=2, ensure_ascii=False))
