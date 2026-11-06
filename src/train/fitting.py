"""Treino de um candidato e validação cruzada estratificada."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from src.preprocess import RANDOM_STATE
from src.train.config import CV_FOLDS
from src.train.metrics import evaluate_model
from src.train.models import ModelSpec

logger = logging.getLogger(__name__)


def train_model(spec: ModelSpec, X_train: pd.DataFrame, y_train: pd.Series) -> BaseEstimator:
    """Instancia e treina um modelo candidato.

    Args:
        spec: Receita do modelo.
        X_train: Features de treino.
        y_train: Target de treino.

    Returns:
        Estimador treinado.

    Raises:
        RuntimeError: Se o treinamento falhar.
    """
    model = spec.factory()
    fit_params: dict[str, Any] = {}
    if spec.needs_sample_weight:
        fit_params["sample_weight"] = compute_sample_weight("balanced", y_train)

    try:
        model.fit(X_train, y_train, **fit_params)
    except Exception as exc:  # noqa: BLE001 - repassa com contexto do modelo
        raise RuntimeError(f"Falha ao treinar {spec.name}: {exc}") from exc
    return model


def cross_validate_model(
    spec: ModelSpec,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = CV_FOLDS,
) -> dict[str, float]:
    """Validação cruzada estratificada, com o mesmo tratamento de pesos do treino.

    Os pesos do XGBoost são recalculados em cada dobra, só com os dados dela.

    Args:
        spec: Receita do modelo.
        X: Features de treino.
        y: Target de treino.
        n_splits: Número de dobras.

    Returns:
        Médias e desvios das métricas usadas na seleção.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict[str, float]] = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        model = train_model(spec, X.iloc[tr_idx], y.iloc[tr_idx])
        ev = evaluate_model(model, X.iloc[va_idx], y.iloc[va_idx])
        rows.append(
            {
                "f1_macro": ev.f1_macro,
                "recall_class1": ev.recall_class1,
                "f1_class1": ev.f1_class1,
                "selection_score": ev.selection_score,
            }
        )
        logger.info(
            "    dobra %d/%d: F1 Macro=%.4f | Recall C1=%.4f | score=%.4f",
            fold,
            n_splits,
            ev.f1_macro,
            ev.recall_class1,
            ev.selection_score,
        )

    df = pd.DataFrame(rows)
    summary: dict[str, float] = {}
    for col in df.columns:
        summary[f"{col}_mean"] = float(df[col].mean())
        summary[f"{col}_std"] = float(df[col].std(ddof=0))
    return summary
