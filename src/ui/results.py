"""Seções 1 e 3 do resultado: classe predita e explicação em linguagem simples."""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

from src.llm import OLLAMA_MODEL
from src.preprocess import CLASS_LABELS
from src.ui.constants import CLASS_STYLES


def render_prediction(result: dict[str, Any]) -> None:
    """Cartão com a classe predita e barras com as probabilidades."""
    code = int(result["predicted_class"]["code"])
    label = html.escape(result["predicted_class"]["label"])
    bg, border, icon = CLASS_STYLES.get(code, ("#f5f5f5", "#616161", "ℹ️"))

    st.subheader("1. Resultado da triagem")
    st.markdown(
        f"""
        <div style="background:{bg}; border-left:8px solid {border};
                    padding:1.2rem 1.5rem; border-radius:10px; margin-bottom:1rem;">
          <div style="font-size:0.9rem; color:#424242;">Perfil identificado</div>
          <div style="font-size:2rem; font-weight:700; color:{border};">{icon} {label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Probabilidade estimada de cada perfil**")
    probabilities: dict[str, float] = result["probabilities"]
    cols = st.columns(len(CLASS_LABELS))
    for col, (cls_code, cls_label) in zip(cols, CLASS_LABELS.items()):
        p = float(probabilities.get(cls_label, 0.0))
        with col:
            st.metric(cls_label, f"{p:.1%}")
            st.progress(min(max(p, 0.0), 1.0))
    st.caption(f"Modelo: {result.get('model', 'desconhecido')}")


def render_explanation(text: str, source: str) -> None:
    """Explicação em linguagem natural e a origem do texto (LLM ou fallback)."""
    st.subheader("3. Explicação personalizada")
    if source == "llm":
        st.success(f"🤖 Texto gerado pela LLM local (Ollama · `{OLLAMA_MODEL}`).")
    else:
        st.info(
            "📝 Texto gerado no modo offline (templates), porque a LLM local não "
            "está disponível ou não respondeu a tempo."
        )
    with st.container(border=True):
        st.markdown(text)
