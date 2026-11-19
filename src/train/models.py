"""Modelos candidatos: Regressão Logística, Árvore de Decisão, k-NN, Random Forest e XGBoost."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.base import BaseEstimator

from src.train import estimators


@dataclass(frozen=True)
class ModelSpec:
    """Receita de um modelo candidato.

    Attributes:
        name: Nome exibido nos logs e no relatório.
        factory: Função que devolve um estimador novo, ainda não treinado.
        needs_sample_weight: Se o desbalanceamento é tratado por pesos de
            amostra no ``fit`` (caso do XGBoost), em vez de ``class_weight``.
        deployable: Se o modelo pode ir para o app. Só a Regressão Logística:
            ``src.explain`` (SHAP linear), a interface e a LLM dependem dos
            seus coeficientes, e a interpretabilidade é requisito do sistema.
    """

    name: str
    factory: Callable[[], BaseEstimator]
    needs_sample_weight: bool = False
    deployable: bool = False


# Os quatro primeiros são exigidos pelo documento de requisitos; o XGBoost entra
# como representante avançado dos métodos baseados em árvores. Todos são
# comparados; o modelo publicado é o melhor entre os ``deployable``.
MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("Regressão Logística", estimators.logistic_regression, deployable=True),
    ModelSpec("Árvore de Decisão", estimators.decision_tree),
    ModelSpec("k-NN", estimators.knn),
    ModelSpec("Random Forest", estimators.random_forest),
    ModelSpec("XGBoost", estimators.xgboost, needs_sample_weight=True),
)
