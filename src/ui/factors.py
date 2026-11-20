"""Seção 2 do resultado: fatores que aumentaram e que reduziram o risco."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.llm import MIN_IMPACT


def _render_how_calculated(result: dict[str, Any], risks: list[dict[str, Any]]) -> None:
    """Expansor com a explicação técnica do SHAP e a tabela de impactos."""
    with st.expander("ℹ️ Como esses fatores são calculados"):
        basis = result.get("explanation_basis", {})
        st.markdown(
            "Os fatores vêm do método **SHAP**, que mede quanto cada resposta "
            "empurrou o resultado em relação a um paciente médio.\n\n"
            f"- Classe explicada: {basis.get('explained_class', '-')}\n"
            f"- Escala: {basis.get('scale', '-')}\n"
            f"- Referência: {basis.get('reference', '-')}\n"
            f"- Observação: {basis.get('note', '-')}"
        )
        if risks:
            st.table(
                {
                    "Fator": [f["description"] for f in risks],
                    "Impacto SHAP (log-odds)": [f"{f['shap_impact']:+.4f}" for f in risks],
                }
            )


def render_factors(result: dict[str, Any]) -> None:
    """Os 3 fatores que mais aumentaram o risco, com barra de impacto relativo."""
    st.subheader("2. Fatores determinantes")
    # Mesmo corte do src.llm: impacto abaixo de MIN_IMPACT é ruído e não é citado.
    risks: list[dict[str, Any]] = [
        f for f in result.get("top_risk_factors", []) if float(f["shap_impact"]) >= MIN_IMPACT
    ]

    if not risks:
        st.info("Nenhum fator se destacou aumentando o risco deste paciente.")
    else:
        st.caption("Os fatores que mais aumentaram o risco, do maior para o menor impacto.")
        max_impact = max(float(f["shap_impact"]) for f in risks) or 1.0
        cols = st.columns(len(risks))
        for rank, (col, factor) in enumerate(zip(cols, risks), start=1):
            impact = float(factor["shap_impact"])
            with col, st.container(border=True):
                st.markdown(f"**#{rank} · {factor['description']}**")
                st.markdown(f"Resposta: `{factor['value_description']}`")
                st.progress(
                    min(impact / max_impact, 1.0),
                    text=f"Impacto relativo: {impact / max_impact:.0%}",
                )

    # Um ponto de atenção (ex.: álcool em excesso) nunca é exibido como proteção,
    # mesmo quando o modelo o associa a menor risco (artefato do BRFSS).
    attention = {p["feature"] for p in result.get("attention_points", [])}
    protective: list[dict[str, Any]] = [
        f for f in result.get("protective_factors", [])
        if -float(f["shap_impact"]) >= MIN_IMPACT and f["feature"] not in attention
    ]
    if protective:
        with st.expander("🛡️ Fatores que reduziram o risco"):
            for factor in protective:
                st.markdown(f"- **{factor['description']}**: {factor['value_description']}")

    _render_how_calculated(result, risks)
