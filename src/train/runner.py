"""Fluxo completo do treino: pré-processamento, comparação e salvamento."""

from __future__ import annotations

from src.preprocess import run_preprocessing
from src.train.comparison import ModelReport, compare_models
from src.train.persistence import save_model
from src.train.reporting import log_summary


def run_training() -> ModelReport:
    """Executa o fluxo completo: pré-processamento, comparação e salvamento.

    Returns:
        Relatório do modelo vencedor.
    """
    data = run_preprocessing()
    reports, best = compare_models(data)
    log_summary(reports, best)
    save_model(best, reports, data.feature_names)
    return best
