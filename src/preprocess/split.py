"""Distribuição das classes e divisão treino/teste estratificada."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocess.config import (
    CLASS_LABELS,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
    PreprocessResult,
)

logger = logging.getLogger(__name__)


def log_class_distribution(y: pd.Series, title: str) -> pd.DataFrame:
    """Registra no log a contagem e a proporção de cada classe do target.

    Args:
        y: Série com o target.
        title: Rótulo do conjunto (ex.: "Dataset limpo", "Treino").

    Returns:
        DataFrame com as colunas ``count`` e ``proportion`` por classe.
    """
    counts = y.value_counts().sort_index()
    dist = pd.DataFrame({"count": counts, "proportion": counts / counts.sum()})

    logger.info("Distribuição do target - %s (n=%d):", title, len(y))
    for cls, row in dist.iterrows():
        marker = "  <-- classe minoritária (atenção)" if cls == 1 else ""
        logger.info(
            "  %d - %-13s %8d  (%6.2f%%)%s",
            cls,
            CLASS_LABELS.get(int(cls), "?"),
            row["count"],
            row["proportion"] * 100,
            marker,
        )
    return dist


def split_dataset(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> PreprocessResult:
    """Divide em treino e teste com amostragem estratificada pelo target.

    Args:
        df: DataFrame limpo, contendo o target.
        test_size: Fração destinada ao teste (padrão 0.20).
        random_state: Semente de aleatoriedade (padrão 42).

    Returns:
        :class:`PreprocessResult` com X/y de treino e teste.
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    logger.info(
        "Divisão estratificada: treino=%d (%.0f%%), teste=%d (%.0f%%)",
        len(X_train),
        (1 - test_size) * 100,
        len(X_test),
        test_size * 100,
    )
    return PreprocessResult(X_train, X_test, y_train, y_test)
