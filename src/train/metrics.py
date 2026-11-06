"""Métricas de avaliação e o score de seleção dos modelos."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.preprocess import CLASS_LABELS
from src.train.config import CLASSES, PREDIABETES


def harmonic_mean(a: float, b: float) -> float:
    """Média harmônica de dois valores não negativos (0 se ambos forem 0)."""
    return 2 * a * b / (a + b) if (a + b) > 0 else 0.0


@dataclass
class EvaluationResult:
    """Métricas de um modelo em um conjunto de dados."""

    accuracy: float
    f1_macro: float
    f1_weighted: float
    recall_class1: float
    precision_class1: float
    f1_class1: float
    recall_per_class: dict[str, float] = field(default_factory=dict)
    confusion_matrix: list[list[int]] = field(default_factory=list)

    @property
    def selection_score(self) -> float:
        """Média harmônica entre F1 Macro e Recall da Classe 1."""
        return harmonic_mean(self.f1_macro, self.recall_class1)


def evaluate_model(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
) -> EvaluationResult:
    """Calcula as métricas de avaliação de um modelo treinado.

    Args:
        model: Estimador já treinado.
        X: Features do conjunto avaliado.
        y: Target verdadeiro.

    Returns:
        :class:`EvaluationResult` com acurácia, F1 (macro/weighted), métricas
        da Classe 1 e matriz de confusão.
    """
    y_pred = model.predict(X)

    recalls = recall_score(y, y_pred, labels=CLASSES, average=None, zero_division=0)
    precisions = precision_score(y, y_pred, labels=CLASSES, average=None, zero_division=0)
    f1s = f1_score(y, y_pred, labels=CLASSES, average=None, zero_division=0)
    idx = CLASSES.index(PREDIABETES)

    return EvaluationResult(
        accuracy=float(accuracy_score(y, y_pred)),
        f1_macro=float(f1_score(y, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        f1_weighted=float(
            f1_score(y, y_pred, labels=CLASSES, average="weighted", zero_division=0)
        ),
        recall_class1=float(recalls[idx]),
        precision_class1=float(precisions[idx]),
        f1_class1=float(f1s[idx]),
        recall_per_class={CLASS_LABELS[c]: float(r) for c, r in zip(CLASSES, recalls)},
        confusion_matrix=confusion_matrix(y, y_pred, labels=CLASSES).tolist(),
    )
