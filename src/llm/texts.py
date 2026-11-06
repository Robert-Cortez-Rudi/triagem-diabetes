"""Textos fixos: títulos das seções, aviso de isenção e abertura de cada classe."""

from __future__ import annotations

SECTION_SUMMARY = "### Resumo do resultado"
SECTION_FACTORS = "### Principais fatores identificados"
SECTION_ACTIONS = "### Recomendações práticas"

DISCLAIMER: str = (
    "> **Importante:** este resultado vem de uma ferramenta de triagem e apoio "
    "investigativo. Ele **não é um diagnóstico médico**. Só um profissional de "
    "saúde, com exames como glicemia de jejum e hemoglobina glicada, pode "
    "confirmar ou descartar diabetes ou pré-diabetes."
)

# Título acolhedor de cada classe e a frase de abertura do resumo.
CLASS_HEADLINES: dict[int, tuple[str, str]] = {
    0: (
        "Risco baixo",
        "As informações que você forneceu se parecem mais com as de pessoas "
        "que não têm diabetes. É um bom sinal, e vale continuar cuidando da saúde.",
    ),
    1: (
        "Sinais compatíveis com pré-diabetes",
        "As informações que você forneceu têm pontos em comum com pessoas que "
        "estão em fase de pré-diabetes. Essa fase costuma ser reversível com "
        "mudanças de hábito, por isso identificá-la cedo é uma ótima notícia.",
    ),
    2: (
        "Indicadores associados ao diabetes",
        "As informações que você forneceu têm pontos em comum com pessoas que "
        "têm diabetes. Isso não significa que você tenha a doença, mas é um "
        "sinal de que vale procurar avaliação médica em breve.",
    ),
}

# Último recurso, quando nem o fallback consegue montar o texto.
GENERIC_TEXT: str = (
    f"{SECTION_SUMMARY}\nNão foi possível detalhar o resultado da triagem agora. "
    "Tente novamente em instantes.\n\n"
    f"{SECTION_ACTIONS}\n- Se você tem dúvidas sobre diabetes, procure uma Unidade "
    "Básica de Saúde (UBS) ou um médico para fazer exames de glicemia.\n\n"
    f"{DISCLAIMER}"
)
