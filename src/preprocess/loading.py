"""Carga do CSV bruto do BRFSS 2015."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.preprocess.config import DATA_PATH, TARGET

logger = logging.getLogger(__name__)


def load_raw_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Carrega o CSV bruto do BRFSS 2015.

    Args:
        path: Caminho do arquivo CSV.

    Returns:
        DataFrame com os dados originais.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se o arquivo estiver vazio ou sem a coluna target.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset não encontrado em: {path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"O arquivo está vazio: {path}") from exc

    if TARGET not in df.columns:
        raise ValueError(f"Coluna target '{TARGET}' ausente no dataset.")

    logger.info("Dataset carregado de %s", path.name)
    logger.info("Shape original: %s linhas x %s colunas", *df.shape)
    return df
