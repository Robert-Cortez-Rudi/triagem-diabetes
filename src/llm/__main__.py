"""Execução pelo terminal: ``python -m src.llm``."""

import logging

from src.explain import EXAMPLE_PATIENT, get_prediction_explanation
from src.llm.config import OLLAMA_MODEL
from src.llm.service import generate_patient_explanation_with_source
from src.logging_config import configure_logging

logger = logging.getLogger("src.llm")

if __name__ == "__main__":
    configure_logging()
    try:
        technical = get_prediction_explanation(EXAMPLE_PATIENT)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Falha ao gerar a explicação técnica: %s", exc)
        raise SystemExit(1) from exc

    text, source = generate_patient_explanation_with_source(technical)
    origem = f"LLM ({OLLAMA_MODEL})" if source == "llm" else "modo offline (templates)"
    print("=" * 70)
    print(f"Explicação gerada por: {origem}")
    print("=" * 70)
    print(text)
