"""Conversão do resultado técnico em fatos simples para o texto do paciente."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.llm.config import MAX_FACTORS, MIN_IMPACT
from src.llm.phrases import RISK_PHRASES
from src.llm.recommendations import GENERAL_RECOMMENDATIONS, RECOMMENDATIONS
from src.llm.texts import CLASS_HEADLINES


def _relevant(factors: Any, sign: int) -> list[dict[str, Any]]:
    """Fatores com impacto relevante no sentido pedido (+1 risco, -1 proteção)."""
    if not isinstance(factors, list):
        return []
    out = []
    for f in factors:
        try:
            impact = float(f["shap_impact"]) * sign
        except (KeyError, TypeError, ValueError):
            continue
        if impact >= MIN_IMPACT and f.get("feature"):
            out.append(f)
    return out[:MAX_FACTORS]


def _risk_phrase(factor: Mapping[str, Any]) -> str:
    builder = RISK_PHRASES.get(str(factor.get("feature")))
    try:
        if builder is not None:
            return builder(factor)
    except (KeyError, TypeError, ValueError, AttributeError):
        pass
    desc = factor.get("description", factor.get("feature", "Um dos fatores"))
    val = factor.get("value_description", factor.get("value", ""))
    return f"{desc} ({val}) contribuiu para o resultado."


def _protective_phrase(factor: Mapping[str, Any]) -> str:
    desc = str(factor.get("description", factor.get("feature", "")))
    val = str(factor.get("value_description", factor.get("value", "")))
    return f"{desc}: {val.lower()}"


@dataclass(frozen=True)
class PatientFacts:
    """Fatos do resultado já em linguagem simples, sem números técnicos."""

    class_code: int
    headline: str
    opening: str
    risk_factors: list[dict[str, Any]]
    risk_phrases: list[str]
    protective_phrases: list[str]
    recommendations: list[str]


def extract_facts(explanation_dict: Mapping[str, Any]) -> PatientFacts:
    """Converte o dicionário do ``src.explain`` em fatos para o texto do paciente.

    Args:
        explanation_dict: Saída de ``get_prediction_explanation``.

    Returns:
        :class:`PatientFacts`. Listas de fatores ausentes ou inválidas viram
        listas vazias.

    Raises:
        ValueError: Se a entrada não for um dicionário ou não trouxer uma
            classe predita válida (0, 1 ou 2). Sem classe não há o que
            explicar, e inventar uma seria pior que avisar.
    """
    if not isinstance(explanation_dict, Mapping):
        raise ValueError(f"Esperado dict; recebido {type(explanation_dict).__name__}.")
    try:
        code = int(explanation_dict["predicted_class"]["code"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Dicionário sem 'predicted_class.code' válido.") from exc
    if code not in CLASS_HEADLINES:
        raise ValueError(f"Classe predita desconhecida: {code}.")
    headline, opening = CLASS_HEADLINES[code]

    risks = _relevant(explanation_dict.get("top_risk_factors"), +1)
    protective = _relevant(explanation_dict.get("protective_factors"), -1)

    recs = [RECOMMENDATIONS[f["feature"]] for f in risks if f["feature"] in RECOMMENDATIONS]
    recs.append(GENERAL_RECOMMENDATIONS[code])

    return PatientFacts(
        class_code=code,
        headline=headline,
        opening=opening,
        risk_factors=risks,
        risk_phrases=[_risk_phrase(f) for f in risks],
        protective_phrases=[_protective_phrase(f) for f in protective],
        recommendations=list(dict.fromkeys(recs)),
    )
