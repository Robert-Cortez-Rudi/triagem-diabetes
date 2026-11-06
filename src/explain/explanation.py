"""Predição de um paciente e os fatores SHAP que mais pesaram na decisão."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.explain.descriptions import build_factor
from src.explain.loader import load_explainer
from src.explain.patient import prepare_patient
from src.preprocess import CLASS_LABELS

TOP_N: int = 3
HEALTHY_CLASS: int = 0


def _risk_contributions(
    shap_values: np.ndarray, pred_idx: int, pred_class: int
) -> tuple[np.ndarray, str]:
    """Contribuição de cada feature para o risco e a nota que explica o sinal."""
    label = CLASS_LABELS[pred_class]
    if pred_class == HEALTHY_CLASS:
        note = (
            f"SHAP da classe '{label}' com sinal invertido: "
            "impacto positivo = reduziu a chance de estar sem diabetes"
        )
        return -shap_values[:, pred_idx], note
    note = (
        f"SHAP da classe predita '{label}': "
        "impacto positivo = aumentou a chance dessa classe"
    )
    return shap_values[:, pred_idx], note


def get_prediction_explanation(
    patient_data_dict: Mapping[str, Any] | pd.DataFrame,
    top_n: int = TOP_N,
) -> dict[str, Any]:
    """Prediz a classe de um paciente e explica a decisão com SHAP.

    Args:
        patient_data_dict: Dados do paciente (dict ou DataFrame de 1 linha)
            com as 15 features do modelo, ou ``Fruits``/``Veggies`` no lugar
            de ``InNaturaFood``.
        top_n: Quantos fatores devolver em cada lista.

    Returns:
        Dicionário com ``predicted_class`` (``{"code", "label"}``),
        ``probabilities`` (``{rótulo: probabilidade}``), ``top_risk_factors``
        (até ``top_n``, do maior para o menor impacto), ``protective_factors``
        (até ``top_n``, impacto negativo, do mais forte para o mais fraco),
        ``explanation_basis`` e ``model``. Cada fator traz ``feature``,
        ``description``, ``value``, ``value_description`` e ``shap_impact``
        (em log-odds).

    Raises:
        FileNotFoundError: Se os artefatos do modelo não existirem.
        ValueError: Se os dados do paciente forem inválidos.
    """
    bundle = load_explainer()
    patient = prepare_patient(patient_data_dict, bundle.feature_names)

    proba = bundle.pipeline.predict_proba(patient)[0]
    pred_idx = int(np.argmax(proba))
    pred_class = bundle.classes[pred_idx]

    scaled = bundle.pipeline.named_steps["scaler"].transform(patient)
    shap_values = bundle.explainer(scaled).values[0]  # (n_features, n_classes)
    risk_contrib, basis = _risk_contributions(shap_values, pred_idx, pred_class)

    raw = patient.iloc[0]
    order = np.argsort(risk_contrib)
    risk_idx = [i for i in order[::-1] if risk_contrib[i] > 0][:top_n]
    protect_idx = [i for i in order if risk_contrib[i] < 0][:top_n]

    def build(indices: list[int]) -> list[dict[str, Any]]:
        return [
            build_factor(bundle.feature_names[i], raw.iloc[i], risk_contrib[i]) for i in indices
        ]

    return {
        "predicted_class": {"code": pred_class, "label": CLASS_LABELS[pred_class]},
        "probabilities": {
            CLASS_LABELS[c]: round(float(p), 4) for c, p in zip(bundle.classes, proba)
        },
        "top_risk_factors": build(risk_idx),
        "protective_factors": build(protect_idx),
        "explanation_basis": {
            "explained_class": CLASS_LABELS[pred_class],
            "scale": "log-odds (logit) da Regressão Logística",
            "reference": "média do conjunto de treino",
            "note": basis,
        },
        "model": bundle.model_name,
    }
