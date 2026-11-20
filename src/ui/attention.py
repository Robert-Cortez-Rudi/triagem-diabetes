"""Seção 3 do resultado: pontos de atenção, independentes do modelo."""

from __future__ import annotations

from typing import Any

import streamlit as st

_GROUPS: tuple[tuple[str, str, str], ...] = (
    ("habit", "🔄 Hábitos que você pode mudar", "#f9a825"),
    ("condition", "🩺 Condições para acompanhar", "#1565c0"),
)


def render_attention(result: dict[str, Any]) -> None:
    """Hábitos e condições de risco presentes nas respostas, em dois grupos."""
    st.subheader("3. Pontos de atenção")
    points: list[dict[str, str]] = result.get("attention_points", [])
    if not points:
        st.success(
            "Nenhuma das suas respostas é um fator de risco conhecido para o diabetes. "
            "Continue cuidando da saúde e faça exames de rotina."
        )
        return

    st.caption(
        "Independentemente do resultado acima, estas respostas são fatores de risco "
        "conhecidos para o desenvolvimento do diabetes tipo 2. Os hábitos podem ser "
        "mudados, e as condições merecem acompanhamento."
    )
    groups = [(title, color, [p for p in points if p["kind"] == kind])
              for kind, title, color in _GROUPS]
    groups = [g for g in groups if g[2]]
    for col, (title, color, items) in zip(st.columns(len(groups)), groups):
        with col, st.container(border=True):
            st.markdown(f"**{title}**")
            for p in items:
                st.markdown(
                    f"<div style='border-left:4px solid {color}; padding-left:0.6rem;"
                    f" margin:0.6rem 0;'><b>{p['title']}</b><br>"
                    f"<span style='font-size:0.9rem; opacity:0.85;'>{p['detail']}</span></div>",
                    unsafe_allow_html=True,
                )
