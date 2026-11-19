"""Função principal: tenta a LLM e cai para o modo offline se ela falhar."""

from __future__ import annotations

import logging
from typing import Any, Mapping

from src.llm.config import LLM_DISABLED, OLLAMA_MODEL, Source
from src.llm.facts import extract_facts
from src.llm.fallback import fallback_explanation
from src.llm.ollama_client import LLMUnavailableError, call_ollama, check_ollama
from src.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from src.llm.texts import DISCLAIMER, GENERIC_TEXT
from src.llm.validation import strip_instruction_echo, validate_llm_text

logger = logging.getLogger(__name__)


def generate_patient_explanation_with_source(
    explanation_dict: Mapping[str, Any],
) -> tuple[str, Source]:
    """Gera a explicação e informa se veio da LLM ou do fallback.

    Útil para o Streamlit mostrar ao usuário qual modo foi usado.

    Args:
        explanation_dict: Saída de ``get_prediction_explanation``.

    Returns:
        Tupla ``(texto em Markdown, "llm" ou "fallback")``. Nunca lança exceção.
    """
    try:
        if LLM_DISABLED:
            raise LLMUnavailableError("LLM desativada por LLM_DISABLED=1.")
        facts = extract_facts(explanation_dict)
        check_ollama()
        logger.info("Gerando explicação com o Ollama (%s)...", OLLAMA_MODEL)
        text = strip_instruction_echo(call_ollama(SYSTEM_PROMPT, build_user_prompt(facts)))
        problems = validate_llm_text(text)
        if problems:
            raise LLMUnavailableError(f"Texto da LLM reprovado: {'; '.join(problems)}")
        return f"{text}\n\n{DISCLAIMER}", "llm"
    except LLMUnavailableError as exc:
        logger.warning("%s Usando o modo offline (templates).", exc)
    except ValueError:
        pass  # entrada inválida: o fallback abaixo registra e trata
    except Exception:  # noqa: BLE001 - a interface nunca pode quebrar
        logger.exception("Erro inesperado na LLM. Usando o modo offline (templates).")

    try:
        return fallback_explanation(explanation_dict), "fallback"
    except ValueError as exc:
        logger.error("Resultado inválido para explicar: %s", exc)
    except Exception:  # noqa: BLE001 - última barreira
        logger.exception("Falha no fallback. Devolvendo texto genérico.")
    return GENERIC_TEXT, "fallback"


def generate_patient_explanation(explanation_dict: dict) -> str:
    """Gera a explicação para o paciente em linguagem simples.

    Tenta a LLM local (Ollama) e, se ela não estiver disponível ou falhar,
    usa o fallback com templates. Nunca lança exceção.

    Args:
        explanation_dict: Saída de ``src.explain.get_prediction_explanation``.

    Returns:
        Texto em Markdown com resumo, fatores, recomendações e aviso de isenção.
    """
    text, _ = generate_patient_explanation_with_source(explanation_dict)
    return text
