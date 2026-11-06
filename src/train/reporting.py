"""Relatórios no log: métricas de teste, tabela comparativa e resumo final."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from src.preprocess import CLASS_LABELS
from src.train.config import CLASSES
from src.train.metrics import EvaluationResult

if TYPE_CHECKING:  # evita import circular com src.train.comparison
    from src.train.comparison import ModelReport

logger = logging.getLogger(__name__)


def log_evaluation(name: str, ev: EvaluationResult) -> None:
    """Escreve no log as métricas de teste e a matriz de confusão de um modelo."""
    logger.info("  [%s] métricas no conjunto de TESTE:", name)
    logger.info("    Acurácia ............ %.4f", ev.accuracy)
    logger.info("    F1 Macro ............ %.4f", ev.f1_macro)
    logger.info("    F1 Weighted ......... %.4f", ev.f1_weighted)
    logger.info("    Recall Classe 1 ..... %.4f   <-- métrica chave", ev.recall_class1)
    logger.info("    Precisão Classe 1 ... %.4f", ev.precision_class1)
    logger.info("    F1 Classe 1 ......... %.4f   <-- métrica chave", ev.f1_class1)

    labels = [f"{c}-{CLASS_LABELS[c]}" for c in CLASSES]
    cm = pd.DataFrame(
        ev.confusion_matrix,
        index=[f"real {lab}" for lab in labels],
        columns=[f"prev {c}" for c in CLASSES],
    )
    logger.info("    Matriz de confusão (linhas = real, colunas = previsto):")
    for line in cm.to_string().splitlines():
        logger.info("      %s", line)


def summary_table(reports: list[ModelReport], best: ModelReport) -> pd.DataFrame:
    """Monta a tabela comparativa dos modelos (CV e teste)."""
    rows = []
    for r in reports:
        rows.append(
            {
                "Modelo": ("* " if r is best else "  ") + r.name,
                "CV Score": r.cv_score,
                "CV F1 Macro": r.cv["f1_macro_mean"],
                "CV Recall C1": r.cv["recall_class1_mean"],
                "Acurácia": r.test.accuracy,
                "F1 Macro": r.test.f1_macro,
                "F1 Weighted": r.test.f1_weighted,
                "Recall C1": r.test.recall_class1,
                "F1 C1": r.test.f1_class1,
            }
        )
    return pd.DataFrame(rows).sort_values("CV Score", ascending=False).reset_index(drop=True)


def log_summary(reports: list[ModelReport], best: ModelReport) -> None:
    """Escreve no log a tabela comparativa e o modelo escolhido."""
    table = summary_table(reports, best)
    logger.info("=" * 70)
    logger.info("RESUMO COMPARATIVO (* = modelo escolhido; CV = validação cruzada no treino)")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        for line in table.to_string(index=False).splitlines():
            logger.info("  %s", line)

    logger.info("=" * 70)
    logger.info(
        "Modelo escolhido: %s (score de seleção na CV = %.4f)", best.name, best.cv_score
    )
    logger.info(
        "No teste: F1 Macro=%.4f | Recall Pré-diabetes=%.4f | F1 Pré-diabetes=%.4f",
        best.test.f1_macro,
        best.test.recall_class1,
        best.test.f1_class1,
    )
