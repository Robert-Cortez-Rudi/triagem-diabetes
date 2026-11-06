"""Constantes do treinamento: destino dos artefatos, classes e validação cruzada."""

from __future__ import annotations

from pathlib import Path

from src.preprocess import CLASS_LABELS, PROJECT_ROOT

MODELS_DIR: Path = PROJECT_ROOT / "models"
MODEL_FILENAME: str = "best_model.joblib"
METRICS_FILENAME: str = "metrics.json"

CLASSES: list[int] = sorted(CLASS_LABELS)
PREDIABETES: int = 1
CV_FOLDS: int = 5
