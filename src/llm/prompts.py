"""Prompts enviados à LLM: regras do sistema e fatos do paciente."""

from __future__ import annotations

from src.llm.facts import PatientFacts
from src.llm.texts import SECTION_ACTIONS, SECTION_FACTORS, SECTION_SUMMARY

SYSTEM_PROMPT: str = f"""Você é um assistente de saúde que explica o resultado de uma \
ferramenta de TRIAGEM de diabetes para um paciente leigo, em português do Brasil.

Regras obrigatórias:
- Linguagem clara, acolhedora, direta e respeitosa. Frases curtas. Trate o paciente por "você".
- NUNCA use termos técnicos ou estatísticos: SHAP, log-odds, coeficiente, amostra, \
modelo, algoritmo, probabilidade, percentual, variável ou feature.
- NUNCA diga que a pessoa TEM ou NÃO TEM diabetes. O resultado é de triagem, não é \
diagnóstico. Use expressões como "sinais compatíveis com" ou "pontos em comum com".
- Use SOMENTE os fatos fornecidos. Não invente sintomas, exames, valores ou fatores.
- Não prescreva remédios nem doses.
- Não escreva aviso legal no final; ele será adicionado pelo sistema.

Responda em Markdown, exatamente com estas três seções e nesta ordem:
{SECTION_SUMMARY}
(2 a 3 frases sobre o resultado, de forma acolhedora, citando o título do resultado.)

{SECTION_FACTORS}
(Uma lista numerada explicando, com palavras simples, cada fator de risco fornecido \
e por que ele importa. Se houver pontos a favor, cite-os em uma frase curta no final.)

{SECTION_ACTIONS}
(3 a 5 itens em lista com ações práticas ligadas aos fatores, sempre incluindo \
procurar um profissional de saúde.)"""


def build_user_prompt(facts: PatientFacts) -> str:
    """Monta a mensagem com os fatos do paciente para a LLM.

    Args:
        facts: Fatos já filtrados e em linguagem simples.

    Returns:
        Texto do prompt do usuário.
    """
    lines = [
        f"Título do resultado: {facts.headline}",
        f"Contexto do resultado: {facts.opening}",
        "",
        "Fatores de risco (do mais importante para o menos):",
    ]
    if facts.risk_phrases:
        lines.extend(f"- {p}" for p in facts.risk_phrases)
    else:
        lines.append("- Nenhum fator de risco se destacou.")
    lines.append("")
    lines.append("Pontos a favor do paciente:")
    if facts.protective_phrases:
        lines.extend(f"- {p}" for p in facts.protective_phrases)
    else:
        lines.append("- Nenhum se destacou.")
    lines.append("")
    lines.append("Recomendações sugeridas (adapte com suas palavras):")
    lines.extend(f"- {r}" for r in facts.recommendations)
    lines.append("")
    lines.append("Escreva a explicação para o paciente seguindo as regras.")
    return "\n".join(lines)
