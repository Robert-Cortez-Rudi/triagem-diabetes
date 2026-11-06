"""Interface Streamlit do Sistema Inteligente de Triagem de Diabetes.

Fluxo:
    1. O usuário preenche o formulário na barra lateral.
    2. As respostas viram os códigos numéricos do BRFSS 2015 (ver
       :data:`src.explain.VALUE_DESCRIPTIONS`, fonte única das codificações).
    3. ``src.explain`` prediz a classe, as probabilidades e os fatores SHAP.
    4. ``src.llm`` gera a explicação em linguagem simples (LLM local ou fallback).

Os componentes da tela ficam em ``src/ui/``.

Execução::

    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from src.logging_config import configure_logging
from src.ui.factors import render_factors
from src.ui.form import render_form, validate_input
from src.ui.page import render_header, run_triage, warm_up_model
from src.ui.results import render_explanation, render_prediction


def main() -> None:
    """Ponto de entrada da aplicação Streamlit."""
    st.set_page_config(
        page_title="Triagem de Diabetes", page_icon="🩺", layout="wide"
    )
    render_header()

    try:
        warm_up_model()
    except FileNotFoundError as exc:
        st.error(f"Modelo não encontrado. Rode `python -m src.train` antes.\n\n{exc}")
        st.stop()
    except ValueError as exc:
        st.error(f"O modelo salvo é incompatível com a aplicação: {exc}")
        st.stop()

    patient = render_form()
    if patient is not None:
        for message in validate_input(patient):
            st.warning(message)
        run_triage(patient)

    triage = st.session_state.get("triage")
    if triage is None:
        st.info("👈 Preencha o formulário e clique em **Realizar Triagem de Risco**.")
        return

    render_prediction(triage["result"])
    st.divider()
    render_factors(triage["result"])
    st.divider()
    render_explanation(triage["text"], triage["source"])


if __name__ == "__main__":
    configure_logging()
    main()
