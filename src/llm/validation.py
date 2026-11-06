"""Checagem do texto devolvido pela LLM antes de mostrá-lo ao paciente."""

from __future__ import annotations

from src.llm.config import FORBIDDEN_RE
from src.llm.texts import SECTION_ACTIONS, SECTION_FACTORS, SECTION_SUMMARY


def validate_llm_text(text: str) -> list[str]:
    """Lista os problemas do texto da LLM (lista vazia = texto aprovado).

    Reprova texto curto demais, sem as três seções ou com jargão técnico.
    """
    problems = []
    if len(text) < 200:
        problems.append("texto curto demais")
    for section in (SECTION_SUMMARY, SECTION_FACTORS, SECTION_ACTIONS):
        title = section.lstrip("# ").lower()
        if title not in text.lower():
            problems.append(f"seção ausente: {title}")
    found = sorted({m.group(0).lower() for m in FORBIDDEN_RE.finditer(text)})
    if found:
        problems.append(f"termos técnicos: {found}")
    return problems
