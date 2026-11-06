"""Limpeza: duplicatas, descarte de colunas e a feature ``InNaturaFood``."""

from __future__ import annotations

import logging

import pandas as pd

from src.preprocess.config import (
    DROP_COLUMN_ALIASES,
    DROP_COLUMNS,
    FOOD_SOURCE_COLUMNS,
    NEW_FEATURE,
)

logger = logging.getLogger(__name__)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas duplicadas, considerando todas as colunas originais.

    Args:
        df: DataFrame bruto.

    Returns:
        DataFrame sem duplicatas e com índice reiniciado.
    """
    n_dup = int(df.duplicated().sum())
    pct = n_dup / len(df) if len(df) else 0.0
    logger.info("Registros duplicados encontrados: %d (%.2f%%)", n_dup, pct * 100)
    return df.drop_duplicates().reset_index(drop=True)


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove as 5 variáveis descartadas pelas regras de negócio.

    Aceita as grafias alternativas de ``AnyHealthcare``. Uma coluna que não
    aparece no DataFrame gera aviso, não erro.

    Args:
        df: DataFrame com as colunas originais.

    Returns:
        DataFrame sem as colunas descartadas.
    """
    to_drop: list[str] = []
    for col in DROP_COLUMNS:
        candidates = DROP_COLUMN_ALIASES.get(col, (col,))
        found = [c for c in candidates if c in df.columns]
        if found:
            to_drop.extend(found)
        else:
            logger.warning("Coluna a descartar não encontrada: %s", col)

    logger.info("Colunas descartadas (%d): %s", len(to_drop), to_drop)
    return df.drop(columns=to_drop)


def add_in_natura_food(df: pd.DataFrame) -> pd.DataFrame:
    """Cria ``InNaturaFood`` a partir de ``Fruits`` e ``Veggies`` e remove as originais.

    ``InNaturaFood`` vale 1 se a pessoa consome frutas OU vegetais diariamente,
    e 0 se não consome nenhum dos dois. Serve tanto para o treino quanto para
    a entrada do usuário no ``app.py``.

    Args:
        df: DataFrame contendo ``Fruits`` e ``Veggies``.

    Returns:
        DataFrame com ``InNaturaFood`` e sem ``Fruits``/``Veggies``.

    Raises:
        KeyError: Se ``Fruits`` ou ``Veggies`` estiverem ausentes.
    """
    missing = [c for c in FOOD_SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Colunas necessárias para {NEW_FEATURE} ausentes: {missing}")

    out = df.copy()
    fruits, veggies = FOOD_SOURCE_COLUMNS
    out[NEW_FEATURE] = ((out[fruits] == 1) | (out[veggies] == 1)).astype(int)
    out = out.drop(columns=list(FOOD_SOURCE_COLUMNS))
    logger.info(
        "Feature criada: %s (%.2f%% = 1); removidas: %s",
        NEW_FEATURE,
        out[NEW_FEATURE].mean() * 100,
        list(FOOD_SOURCE_COLUMNS),
    )
    return out
