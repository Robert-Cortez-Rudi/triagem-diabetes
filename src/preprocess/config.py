"""Constantes do pré-processamento e o contêiner do resultado."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DATA_PATH: Path = PROJECT_ROOT / "data" / "diabetes_012_health_indicators_BRFSS2015.csv"

TARGET: str = "Diabetes_012"
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20

CLASS_LABELS: dict[int, str] = {
    0: "Sem diabetes",
    1: "Pré-diabetes",
    2: "Diabetes",
}

# O CSV original grafa "AnyHealthcare"; as outras grafias cobrem variações do dataset.
DROP_COLUMNS: tuple[str, ...] = (
    "MentHlth",
    "Sex",
    "CholCheck",
    "AnyHealthcare",
    "NoDocbcCost",
)
DROP_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "AnyHealthcare": ("AnyHealthcare", "AnyHealthCare", "AnyHealtcare"),
}

FOOD_SOURCE_COLUMNS: tuple[str, str] = ("Fruits", "Veggies")
NEW_FEATURE: str = "InNaturaFood"


@dataclass(frozen=True)
class PreprocessResult:
    """Saída do pipeline de pré-processamento."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    @property
    def feature_names(self) -> list[str]:
        """Lista ordenada das features usadas pelo modelo."""
        return list(self.X_train.columns)
