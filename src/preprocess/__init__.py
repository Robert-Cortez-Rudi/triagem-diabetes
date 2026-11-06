"""Pré-processamento do dataset de triagem de diabetes (BRFSS 2015).

Etapas:
    1. Carga do CSV bruto (``loading``).
    2. Remoção de registros duplicados (``cleaning``).
    3. Descarte das variáveis não utilizadas (``cleaning``).
    4. Feature engineering: ``InNaturaFood`` = Fruits OU Veggies (``cleaning``).
    5. Divisão treino/teste estratificada, 80/20, ``random_state=42`` (``split``).

Constantes em ``config``; orquestração em ``pipeline``.

Uso pelo terminal::

    python -m src.preprocess
"""

from src.preprocess.cleaning import add_in_natura_food, drop_unused_columns, remove_duplicates
from src.preprocess.config import (
    CLASS_LABELS,
    DATA_PATH,
    DROP_COLUMN_ALIASES,
    DROP_COLUMNS,
    FOOD_SOURCE_COLUMNS,
    NEW_FEATURE,
    PROJECT_ROOT,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
    PreprocessResult,
)
from src.preprocess.loading import load_raw_data
from src.preprocess.pipeline import clean_and_transform, run_preprocessing
from src.preprocess.split import log_class_distribution, split_dataset

__all__ = [
    "CLASS_LABELS",
    "DATA_PATH",
    "DROP_COLUMN_ALIASES",
    "DROP_COLUMNS",
    "FOOD_SOURCE_COLUMNS",
    "NEW_FEATURE",
    "PROJECT_ROOT",
    "RANDOM_STATE",
    "TARGET",
    "TEST_SIZE",
    "PreprocessResult",
    "add_in_natura_food",
    "clean_and_transform",
    "drop_unused_columns",
    "load_raw_data",
    "log_class_distribution",
    "remove_duplicates",
    "run_preprocessing",
    "split_dataset",
]
