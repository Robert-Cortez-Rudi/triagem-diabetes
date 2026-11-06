"""Gravação do modelo vencedor e do ``metrics.json`` em ``models/``."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib

from src.preprocess import CLASS_LABELS, RANDOM_STATE
from src.train.comparison import ModelReport
from src.train.config import CV_FOLDS, METRICS_FILENAME, MODEL_FILENAME, MODELS_DIR

logger = logging.getLogger(__name__)


def save_model(
    best: ModelReport,
    reports: list[ModelReport],
    feature_names: list[str],
    models_dir: Path = MODELS_DIR,
) -> tuple[Path, Path]:
    """Salva o modelo vencedor e o arquivo de métricas em ``models/``.

    Args:
        best: Relatório do modelo vencedor.
        reports: Relatórios de todos os modelos (vão para o JSON).
        feature_names: Ordem das colunas esperada pelo modelo.
        models_dir: Pasta de destino.

    Returns:
        Tupla ``(caminho do modelo, caminho das métricas)``.

    Raises:
        OSError: Se não for possível gravar os arquivos.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / MODEL_FILENAME
    metrics_path = models_dir / METRICS_FILENAME

    joblib.dump(best.model, model_path)

    payload = {
        "best_model": best.name,
        "selection_criterion": (
            "média harmônica entre F1 Macro e Recall da Classe 1 (Pré-diabetes), "
            "média da validação cruzada estratificada no treino"
        ),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_state": RANDOM_STATE,
        "cv_folds": CV_FOLDS,
        "feature_names": feature_names,
        "class_labels": {str(k): v for k, v in CLASS_LABELS.items()},
        "models": {
            r.name: {
                "cv": r.cv,
                "test": {**asdict(r.test), "selection_score": r.test.selection_score},
                "train_seconds": round(r.train_seconds, 2),
            }
            for r in reports
        },
    }
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Modelo salvo em: %s", model_path)
    logger.info("Métricas salvas em: %s", metrics_path)
    return model_path, metrics_path
