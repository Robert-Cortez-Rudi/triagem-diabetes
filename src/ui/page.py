"""Cabeçalho da página, carga do modelo e execução da triagem."""

from __future__ import annotations

import logging

import streamlit as st

from src.explain import get_prediction_explanation, load_explainer
from src.llm import generate_patient_explanation_with_source

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Carregando o modelo...")
def warm_up_model() -> str:
    """Carrega o modelo e o explainer uma única vez por processo."""
    return load_explainer().model_name


def render_header() -> None:
    """Título, subtítulo e aviso de que a ferramenta não faz diagnóstico."""
    st.title("🩺 Sistema Inteligente de Triagem de Diabetes")
    st.markdown(
        "Preencha os dados na barra lateral e clique em **Realizar Triagem de Risco**. "
        "O sistema estima o perfil (sem diabetes, pré-diabetes ou diabetes), mostra os "
        "fatores que mais pesaram e explica o resultado em linguagem simples."
    )
    st.warning(
        "**Atenção:** esta é uma ferramenta de **triagem e apoio**, não de diagnóstico "
        "médico. Só um profissional de saúde, com exames como glicemia de jejum e "
        "hemoglobina glicada, pode confirmar ou descartar diabetes.",
        icon="⚠️",
    )


def run_triage(patient: dict[str, float]) -> None:
    """Executa predição + explicação e guarda o resultado na sessão."""
    st.session_state.pop("triage", None)  # nunca exibir resultado de outro paciente
    try:
        with st.spinner("Calculando o risco..."):
            result = get_prediction_explanation(patient)
    except FileNotFoundError as exc:
        st.error(f"Modelo não encontrado. Rode `python -m src.train` antes.\n\n{exc}")
        return
    except ValueError as exc:
        st.error(f"Dados inválidos para a triagem: {exc}")
        return
    except Exception:  # noqa: BLE001 - a interface não pode quebrar
        logger.exception("Erro inesperado na predição.")
        st.error("Ocorreu um erro inesperado ao calcular o risco. Tente novamente.")
        return

    with st.spinner("Gerando a explicação em linguagem simples (pode levar alguns segundos)..."):
        text, source = generate_patient_explanation_with_source(result)

    st.session_state["triage"] = {"result": result, "text": text, "source": source}
