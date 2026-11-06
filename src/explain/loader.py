"""Carga do modelo salvo e montagem do explainer SHAP (uma vez por processo)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import shap
from sklearn.pipeline import Pipeline

from src.preprocess import PROJECT_ROOT

logger = logging.getLogger(__name__)

MODELS_DIR: Path = PROJECT_ROOT / "models"
MODEL_PATH: Path = MODELS_DIR / "best_model.joblib"
METRICS_PATH: Path = MODELS_DIR / "metrics.json"


@dataclass(frozen=True)
class ExplainerBundle:
    """Modelo, explainer SHAP e metadados carregados de ``models/``."""

    pipeline: Pipeline
    explainer: shap.LinearExplainer
    feature_names: list[str]
    classes: list[int]
    model_name: str


@lru_cache(maxsize=1)
def load_explainer(
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
) -> ExplainerBundle:
    """Carrega o pipeline e a ordem das features e monta o ``LinearExplainer``.

    Args:
        model_path: Caminho do ``best_model.joblib``.
        metrics_path: Caminho do ``metrics.json`` gerado por ``src.train``.

    Returns:
        :class:`ExplainerBundle` pronto para uso.

    Raises:
        FileNotFoundError: Se algum artefato não existir (rode ``python -m src.train``).
        ValueError: Se o modelo salvo não for o Pipeline scaler + Regressão Logística.
    """
    for path in (model_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Artefato não encontrado: {path}. Rode 'python -m src.train' antes."
            )

    pipeline = joblib.load(model_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    feature_names: list[str] = metrics["feature_names"]

    if not isinstance(pipeline, Pipeline) or {"scaler", "clf"} - set(pipeline.named_steps):
        raise ValueError(
            "O explain.py espera um Pipeline com as etapas 'scaler' e 'clf' "
            f"(Regressão Logística); encontrado: {type(pipeline).__name__}."
        )

    clf = pipeline.named_steps["clf"]
    n_features = len(feature_names)
    if clf.coef_.shape[1] != n_features:
        raise ValueError(
            f"O modelo tem {clf.coef_.shape[1]} features e o metrics.json lista {n_features}."
        )

    # Background = média do treino. No espaço padronizado ela é o vetor zero.
    masker = (np.zeros(n_features), np.eye(n_features))
    explainer = shap.LinearExplainer(clf, masker)

    logger.info("Modelo '%s' e explainer SHAP carregados.", metrics.get("best_model", "?"))
    return ExplainerBundle(
        pipeline=pipeline,
        explainer=explainer,
        feature_names=feature_names,
        classes=[int(c) for c in clf.classes_],
        model_name=metrics.get("best_model", "desconhecido"),
    )
