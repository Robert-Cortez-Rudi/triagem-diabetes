"""Pipeline completo de pré-processamento: limpeza, verificação e divisão."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.preprocess.cleaning import add_in_natura_food, drop_unused_columns, remove_duplicates
from src.preprocess.config import DATA_PATH, DROP_COLUMNS, NEW_FEATURE, TARGET, PreprocessResult
from src.preprocess.loading import load_raw_data
from src.preprocess.split import log_class_distribution, split_dataset

logger = logging.getLogger(__name__)


def clean_and_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica limpeza, descarte de colunas e feature engineering.

    Ordem: duplicatas -> descarte de colunas -> ``InNaturaFood`` -> target como int.
    As duplicatas são removidas sobre as colunas originais, antes do descarte.

    Args:
        df: DataFrame bruto vindo de :func:`load_raw_data`.

    Returns:
        DataFrame limpo, pronto para a divisão treino/teste.
    """
    out = remove_duplicates(df)
    out = drop_unused_columns(out)
    out = add_in_natura_food(out)
    out[TARGET] = out[TARGET].astype(int)

    logger.info("Shape após limpeza: %s linhas x %s colunas", *out.shape)
    return out


def run_preprocessing(path: Path | str = DATA_PATH) -> PreprocessResult:
    """Executa o pipeline completo de pré-processamento.

    Args:
        path: Caminho do CSV bruto.

    Returns:
        :class:`PreprocessResult` com os conjuntos de treino e teste.
    """
    raw = load_raw_data(path)
    clean = clean_and_transform(raw)

    removed = [c for c in DROP_COLUMNS if c not in clean.columns]
    logger.info(
        "Verificação: %d/%d colunas descartadas ausentes; %s presente: %s",
        len(removed),
        len(DROP_COLUMNS),
        NEW_FEATURE,
        NEW_FEATURE in clean.columns,
    )

    log_class_distribution(clean[TARGET], "Dataset limpo")
    result = split_dataset(clean)

    logger.info("Features finais (%d): %s", len(result.feature_names), result.feature_names)

    train_dist = log_class_distribution(result.y_train, "Treino")
    test_dist = log_class_distribution(result.y_test, "Teste")
    max_diff = (train_dist["proportion"] - test_dist["proportion"]).abs().max()
    logger.info("Maior diferença de proporção entre treino e teste: %.4f p.p.", max_diff * 100)

    return result
