"""Modelos candidatos: Regressão Logística, Random Forest e XGBoost."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.preprocess import RANDOM_STATE
from src.train.config import CLASSES


@dataclass(frozen=True)
class ModelSpec:
    """Receita de um modelo candidato.

    Attributes:
        name: Nome exibido nos logs e no relatório.
        factory: Função que devolve um estimador novo, ainda não treinado.
        needs_sample_weight: Se o desbalanceamento é tratado por pesos de
            amostra no ``fit`` (caso do XGBoost), em vez de ``class_weight``.
    """

    name: str
    factory: Callable[[], BaseEstimator]
    needs_sample_weight: bool = False


def _logistic_regression() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _random_forest() -> RandomForestClassifier:
    # Profundidade e folha mínima limitadas: árvores completas decoram a classe
    # majoritária e quase nunca chegam a prever Pré-diabetes.
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def _xgboost() -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=len(CLASSES),
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("Regressão Logística", _logistic_regression),
    ModelSpec("Random Forest", _random_forest),
    ModelSpec("XGBoost", _xgboost, needs_sample_weight=True),
)
