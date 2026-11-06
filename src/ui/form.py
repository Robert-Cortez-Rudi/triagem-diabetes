"""Formulário do paciente na barra lateral e checagem de coerência das respostas."""

from __future__ import annotations

import streamlit as st

from src.explain import VALUE_DESCRIPTIONS
from src.ui.constants import BINARY_FIELDS, BMI_DEFAULT, BMI_MAX, BMI_MIN, NO, PHYS_HLTH_MAX, YES


def _yes_no(value: str) -> int:
    """Converte a resposta da tela no código binário do BRFSS (Sim=1, Não=0)."""
    return 1 if value == YES else 0


def _choice(label: str, feature: str, index: int, **kwargs) -> int:
    """Caixa de seleção com as opções e os textos de ``VALUE_DESCRIPTIONS``."""
    mapping = VALUE_DESCRIPTIONS[feature]
    return st.selectbox(label, options=list(mapping), index=index, format_func=mapping.get, **kwargs)


def render_form() -> dict[str, float] | None:
    """Desenha o formulário na barra lateral.

    Returns:
        Dicionário ``{feature: código BRFSS}`` quando o botão é clicado; senão ``None``.
    """
    with st.sidebar.form("patient_form"):
        st.header("📋 Dados do paciente")

        st.subheader("Perfil")
        age = _choice("Faixa etária", "Age", index=6)
        bmi = st.number_input(
            "IMC (Índice de Massa Corporal)",
            min_value=BMI_MIN, max_value=BMI_MAX, value=BMI_DEFAULT, step=0.5,
            help="IMC = peso (kg) ÷ altura² (m). Ex.: 80 kg e 1,75 m → 26,1.",
        )
        gen_map = VALUE_DESCRIPTIONS["GenHlth"]
        gen_hlth = st.select_slider(
            "Como você avalia sua saúde geral?",
            options=list(gen_map), value=3, format_func=gen_map.get,
        )
        phys_hlth = st.slider(
            "Dias com a saúde física ruim no último mês",
            min_value=0, max_value=PHYS_HLTH_MAX, value=0,
            help="Doença ou lesão física nos últimos 30 dias.",
        )

        st.subheader("Saúde e hábitos")
        answers: dict[str, int] = {}
        for feature, question, help_text in BINARY_FIELDS:
            choice = st.radio(
                question, options=[NO, YES], horizontal=True,
                key=f"bin_{feature}", help=help_text or None,
            )
            answers[feature] = _yes_no(choice)

        st.subheader("Socioeconômico")
        education = _choice("Escolaridade", "Education", index=3)
        income = _choice(
            "Renda familiar anual", "Income", index=4,
            help="Faixas originais da pesquisa BRFSS (EUA), em dólares.",
        )

        submitted = st.form_submit_button(
            "🔍 Realizar Triagem de Risco", type="primary", use_container_width=True
        )

    if not submitted:
        return None

    # Fruits e Veggies seguem separados: explain.prepare_patient cria
    # InNaturaFood com a mesma regra do treino.
    return {
        "Age": int(age),
        "BMI": float(bmi),
        "GenHlth": int(gen_hlth),
        "PhysHlth": int(phys_hlth),
        "Education": int(education),
        "Income": int(income),
        **answers,
    }


def validate_input(patient: dict[str, float]) -> list[str]:
    """Checa a coerência das respostas. Devolve avisos (não bloqueiam a triagem)."""
    warnings = []
    bmi = patient["BMI"]
    if bmi < 15 or bmi > 60:
        warnings.append(
            f"O IMC informado ({bmi:.1f}) é muito incomum. Confira peso e altura."
        )
    if patient["DiffWalk"] == 1 and patient["PhysActivity"] == 1 and patient["GenHlth"] == 1:
        warnings.append(
            "Você marcou dificuldade para andar e saúde excelente ao mesmo tempo. "
            "Confira se as respostas estão corretas."
        )
    return warnings
