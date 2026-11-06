"""Modo offline: explicação montada só com templates, sem LLM."""

from __future__ import annotations

from typing import Any, Mapping

from src.llm.facts import extract_facts
from src.llm.texts import DISCLAIMER, SECTION_ACTIONS, SECTION_FACTORS, SECTION_SUMMARY


def fallback_explanation(explanation_dict: Mapping[str, Any]) -> str:
    """Gera a explicação só com templates, sem LLM.

    Usada quando o Ollama não está disponível ou falha. Segue a mesma
    estrutura do texto da LLM.

    Args:
        explanation_dict: Saída de ``get_prediction_explanation``.

    Returns:
        Texto em Markdown com resumo, fatores, recomendações e aviso.
    """
    facts = extract_facts(explanation_dict)
    parts = [SECTION_SUMMARY, f"**{facts.headline}.** {facts.opening}", ""]

    parts.append(SECTION_FACTORS)
    if facts.risk_phrases:
        intro = (
            "Entre as informações que você forneceu, estas foram as que mais "
            "pesaram no resultado:"
        )
        parts.append(intro)
        parts.extend(f"{i}. {p}" for i, p in enumerate(facts.risk_phrases, start=1))
    else:
        parts.append(
            "Nenhuma das informações fornecidas se destacou como fator de risco "
            "importante."
        )
    if facts.protective_phrases:
        parts.append("")
        parts.append(
            "Pontos a seu favor: " + "; ".join(facts.protective_phrases) + "."
        )
    parts.append("")

    parts.append(SECTION_ACTIONS)
    parts.extend(f"- {r}" for r in facts.recommendations)
    parts.append("")
    parts.append(DISCLAIMER)
    return "\n".join(parts)
