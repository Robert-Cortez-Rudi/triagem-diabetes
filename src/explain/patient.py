"""Entrada do paciente: validação, ``InNaturaFood`` e ordem das colunas."""

from __future__ import annotations

import logging
from typing import Any, Mapping

import pandas as pd

from src.preprocess import FOOD_SOURCE_COLUMNS, NEW_FEATURE

logger = logging.getLogger(__name__)

# Paciente de teste usado por ``python -m src.explain`` e ``python -m src.llm``.
EXAMPLE_PATIENT: dict[str, float] = {
    "HighBP": 1, "HighChol": 1, "BMI": 34.0, "Smoker": 1, "Stroke": 0,
    "HeartDiseaseorAttack": 0, "PhysActivity": 0, "Fruits": 0, "Veggies": 1,
    "HvyAlcoholConsump": 0, "GenHlth": 4, "PhysHlth": 10, "DiffWalk": 1,
    "Age": 11, "Education": 4, "Income": 4,
}


def prepare_patient(
    patient_data: Mapping[str, Any] | pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """Converte a entrada do paciente num DataFrame de 1 linha, na ordem do modelo.

    Aceita ``InNaturaFood`` pronto ou ``Fruits`` + ``Veggies``; nesse caso a
    feature é calculada com a mesma regra do treino. Colunas extras são
    ignoradas com aviso.

    Args:
        patient_data: Dicionário ``{feature: valor}`` ou DataFrame de 1 linha.
        feature_names: Ordem das features esperada pelo modelo.

    Returns:
        DataFrame ``float`` com 1 linha e as colunas em ``feature_names``.

    Raises:
        ValueError: Se houver mais de uma linha, faltar feature ou houver valor
            não numérico.
    """
    if isinstance(patient_data, pd.DataFrame):
        df = patient_data.copy()
    elif isinstance(patient_data, Mapping):
        df = pd.DataFrame([dict(patient_data)])
    else:
        raise ValueError(
            f"Entrada deve ser dict ou DataFrame; recebido {type(patient_data).__name__}."
        )

    if len(df) != 1:
        raise ValueError(f"Esperado 1 paciente (1 linha); recebido {len(df)}.")

    if NEW_FEATURE not in df.columns and all(c in df.columns for c in FOOD_SOURCE_COLUMNS):
        fruits, veggies = FOOD_SOURCE_COLUMNS
        df[NEW_FEATURE] = ((df[fruits] == 1) | (df[veggies] == 1)).astype(int)

    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Features ausentes nos dados do paciente: {missing}")

    extra = sorted(set(df.columns) - set(feature_names) - set(FOOD_SOURCE_COLUMNS))
    if extra:
        logger.warning("Colunas ignoradas (não usadas pelo modelo): %s", extra)

    df = df[feature_names]
    try:
        df = df.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Todos os valores do paciente devem ser numéricos: {exc}") from exc

    if df.isna().any(axis=None):
        raise ValueError(f"Valores ausentes em: {df.columns[df.isna().iloc[0]].tolist()}")
    return df
