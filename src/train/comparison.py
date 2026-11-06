"""Comparação dos candidatos e escolha do vencedor pela validação cruzada."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from sklearn.base import BaseEstimator

from src.preprocess import PreprocessResult
from src.train.config import CV_FOLDS
from src.train.fitting import cross_validate_model, train_model
from src.train.metrics import EvaluationResult, evaluate_model
from src.train.models import MODEL_SPECS, ModelSpec
from src.train.reporting import log_evaluation

logger = logging.getLogger(__name__)


@dataclass
class ModelReport:
    """Resultado completo de um modelo candidato."""

    name: str
    cv: dict[str, float]
    test: EvaluationResult
    train_seconds: float
    model: BaseEstimator = field(repr=False)

    @property
    def cv_score(self) -> float:
        return self.cv["selection_score_mean"]


def compare_models(
    data: PreprocessResult,
    specs: tuple[ModelSpec, ...] = MODEL_SPECS,
    cv_folds: int = CV_FOLDS,
) -> tuple[list[ModelReport], ModelReport]:
    """Treina, valida e avalia todos os candidatos e escolhe o vencedor.

    Para cada modelo: validação cruzada no treino, treino final em todo o
    treino e avaliação no teste. O vencedor é o de maior score de seleção
    médio na validação cruzada.

    Args:
        data: Conjuntos de treino e teste vindos de ``src.preprocess``.
        specs: Modelos a comparar.
        cv_folds: Número de dobras da validação cruzada.

    Returns:
        Tupla ``(relatórios de todos os modelos, relatório do vencedor)``.

    Raises:
        RuntimeError: Se nenhum modelo puder ser treinado.
    """
    reports: list[ModelReport] = []
    for spec in specs:
        logger.info("=" * 70)
        logger.info("Modelo: %s", spec.name)
        try:
            logger.info("  Validação cruzada estratificada (%d dobras):", cv_folds)
            cv = cross_validate_model(spec, data.X_train, data.y_train, cv_folds)
            logger.info(
                "  CV: F1 Macro=%.4f±%.4f | Recall C1=%.4f±%.4f | score=%.4f",
                cv["f1_macro_mean"],
                cv["f1_macro_std"],
                cv["recall_class1_mean"],
                cv["recall_class1_std"],
                cv["selection_score_mean"],
            )

            start = time.perf_counter()
            model = train_model(spec, data.X_train, data.y_train)
            elapsed = time.perf_counter() - start
            logger.info("  Treino final concluído em %.1fs", elapsed)

            test_ev = evaluate_model(model, data.X_test, data.y_test)
            log_evaluation(spec.name, test_ev)
        except RuntimeError as exc:
            logger.error("  %s descartado: %s", spec.name, exc)
            continue

        reports.append(ModelReport(spec.name, cv, test_ev, elapsed, model))

    if not reports:
        raise RuntimeError("Nenhum modelo foi treinado com sucesso.")

    best = max(reports, key=lambda r: r.cv_score)
    return reports, best
