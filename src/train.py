"""Treinamento, avaliação e comparação de modelos de triagem de diabetes.

Modelos comparados:
    1. Regressão Logística (baseline), dentro de um Pipeline com StandardScaler.
    2. Random Forest.
    3. XGBoost.

Desbalanceamento: a Classe 1 (Pré-diabetes) tem cerca de 2% da base. A
Regressão Logística e a Random Forest usam ``class_weight="balanced"``; o
XGBoost não tem esse parâmetro para multiclasse e recebe pesos por amostra
equivalentes (``compute_sample_weight("balanced")``).

Seleção: cada modelo passa por validação cruzada estratificada no conjunto de
treino. O vencedor é o de maior *score de seleção*, a média harmônica entre o
F1 Macro e o Recall da Classe 1 na validação cruzada. A média harmônica pune o
modelo que sacrifica uma métrica pela outra. O conjunto de teste só é usado
para o relatório final, nunca para escolher o modelo.

Artefatos salvos em ``models/``:
    - ``best_model.joblib``: estimador vencedor, já treinado em todo o treino
      (no caso da Regressão Logística, o Pipeline inclui o scaler).
    - ``metrics.json``: métricas de todos os modelos, modelo escolhido, ordem
      das features e rótulos das classes.

Uso pelo terminal::

    python -m src.train
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.preprocess import (
    CLASS_LABELS,
    PROJECT_ROOT,
    RANDOM_STATE,
    PreprocessResult,
    run_preprocessing,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------------- #
MODELS_DIR: Path = PROJECT_ROOT / "models"
MODEL_FILENAME: str = "best_model.joblib"
METRICS_FILENAME: str = "metrics.json"

CLASSES: list[int] = sorted(CLASS_LABELS)
PREDIABETES: int = 1
CV_FOLDS: int = 5


# --------------------------------------------------------------------------- #
# Modelos
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    """Receita de um modelo candidato.

    Attributes:
        name: Nome exibido nos logs e no relatório.
        factory: Função que devolve um estimador novo, ainda não treinado.
        needs_sample_weight: Se o desbalanceamento é tratado por pesos de
            amostra no ``fit`` (caso do XGBoost), em vez de ``class_weight``.
    """

    name: str
    factory: Callable[[], BaseEstimator]
    needs_sample_weight: bool = False


def _logistic_regression() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _random_forest() -> RandomForestClassifier:
    # Profundidade e folha mínima limitadas: árvores completas decoram a classe
    # majoritária e quase nunca chegam a prever Pré-diabetes.
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def _xgboost() -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=len(CLASSES),
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("Regressão Logística", _logistic_regression),
    ModelSpec("Random Forest", _random_forest),
    ModelSpec("XGBoost", _xgboost, needs_sample_weight=True),
)


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
@dataclass
class EvaluationResult:
    """Métricas de um modelo em um conjunto de dados."""

    accuracy: float
    f1_macro: float
    f1_weighted: float
    recall_class1: float
    precision_class1: float
    f1_class1: float
    recall_per_class: dict[str, float] = field(default_factory=dict)
    confusion_matrix: list[list[int]] = field(default_factory=list)

    @property
    def selection_score(self) -> float:
        """Média harmônica entre F1 Macro e Recall da Classe 1."""
        return harmonic_mean(self.f1_macro, self.recall_class1)


def harmonic_mean(a: float, b: float) -> float:
    """Média harmônica de dois valores não negativos (0 se ambos forem 0)."""
    return 2 * a * b / (a + b) if (a + b) > 0 else 0.0


def train_model(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> BaseEstimator:
    """Instancia e treina um modelo candidato.

    Args:
        spec: Receita do modelo.
        X_train: Features de treino.
        y_train: Target de treino.

    Returns:
        Estimador treinado.

    Raises:
        RuntimeError: Se o treinamento falhar.
    """
    model = spec.factory()
    fit_params: dict[str, Any] = {}
    if spec.needs_sample_weight:
        fit_params["sample_weight"] = compute_sample_weight("balanced", y_train)

    try:
        model.fit(X_train, y_train, **fit_params)
    except Exception as exc:  # noqa: BLE001 - repassa com contexto do modelo
        raise RuntimeError(f"Falha ao treinar {spec.name}: {exc}") from exc
    return model


def evaluate_model(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
) -> EvaluationResult:
    """Calcula as métricas de avaliação de um modelo treinado.

    Args:
        model: Estimador já treinado.
        X: Features do conjunto avaliado.
        y: Target verdadeiro.

    Returns:
        :class:`EvaluationResult` com acurácia, F1 (macro/weighted), métricas
        da Classe 1 e matriz de confusão.
    """
    y_pred = model.predict(X)

    recalls = recall_score(y, y_pred, labels=CLASSES, average=None, zero_division=0)
    precisions = precision_score(y, y_pred, labels=CLASSES, average=None, zero_division=0)
    f1s = f1_score(y, y_pred, labels=CLASSES, average=None, zero_division=0)
    idx = CLASSES.index(PREDIABETES)

    return EvaluationResult(
        accuracy=float(accuracy_score(y, y_pred)),
        f1_macro=float(f1_score(y, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        f1_weighted=float(
            f1_score(y, y_pred, labels=CLASSES, average="weighted", zero_division=0)
        ),
        recall_class1=float(recalls[idx]),
        precision_class1=float(precisions[idx]),
        f1_class1=float(f1s[idx]),
        recall_per_class={CLASS_LABELS[c]: float(r) for c, r in zip(CLASSES, recalls)},
        confusion_matrix=confusion_matrix(y, y_pred, labels=CLASSES).tolist(),
    )


def cross_validate_model(
    spec: ModelSpec,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = CV_FOLDS,
) -> dict[str, float]:
    """Validação cruzada estratificada, com o mesmo tratamento de pesos do treino.

    Os pesos do XGBoost são recalculados em cada dobra, só com os dados dela.

    Args:
        spec: Receita do modelo.
        X: Features de treino.
        y: Target de treino.
        n_splits: Número de dobras.

    Returns:
        Médias e desvios das métricas usadas na seleção.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict[str, float]] = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        model = train_model(spec, X.iloc[tr_idx], y.iloc[tr_idx])
        ev = evaluate_model(model, X.iloc[va_idx], y.iloc[va_idx])
        rows.append(
            {
                "f1_macro": ev.f1_macro,
                "recall_class1": ev.recall_class1,
                "f1_class1": ev.f1_class1,
                "selection_score": ev.selection_score,
            }
        )
        logger.info(
            "    dobra %d/%d: F1 Macro=%.4f | Recall C1=%.4f | score=%.4f",
            fold,
            n_splits,
            ev.f1_macro,
            ev.recall_class1,
            ev.selection_score,
        )

    df = pd.DataFrame(rows)
    summary: dict[str, float] = {}
    for col in df.columns:
        summary[f"{col}_mean"] = float(df[col].mean())
        summary[f"{col}_std"] = float(df[col].std(ddof=0))
    return summary


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


# --------------------------------------------------------------------------- #
# Comparação
# --------------------------------------------------------------------------- #
@dataclass
class ModelReport:
    """Resultado completo de um modelo candidato."""

    name: str
    cv: dict[str, float]
    test: EvaluationResult
    train_seconds: float
    model: BaseEstimator = field(repr=False)

    @property
    def cv_score(self) -> float:
        return self.cv["selection_score_mean"]


def compare_models(
    data: PreprocessResult,
    specs: tuple[ModelSpec, ...] = MODEL_SPECS,
    cv_folds: int = CV_FOLDS,
) -> tuple[list[ModelReport], ModelReport]:
    """Treina, valida e avalia todos os candidatos e escolhe o vencedor.

    Para cada modelo: validação cruzada no treino, treino final em todo o
    treino e avaliação no teste. O vencedor é o de maior score de seleção
    médio na validação cruzada.

    Args:
        data: Conjuntos de treino e teste vindos de ``src.preprocess``.
        specs: Modelos a comparar.
        cv_folds: Número de dobras da validação cruzada.

    Returns:
        Tupla ``(relatórios de todos os modelos, relatório do vencedor)``.

    Raises:
        RuntimeError: Se nenhum modelo puder ser treinado.
    """
    reports: list[ModelReport] = []
    for spec in specs:
        logger.info("=" * 70)
        logger.info("Modelo: %s", spec.name)
        try:
            logger.info("  Validação cruzada estratificada (%d dobras):", cv_folds)
            cv = cross_validate_model(spec, data.X_train, data.y_train, cv_folds)
            logger.info(
                "  CV: F1 Macro=%.4f±%.4f | Recall C1=%.4f±%.4f | score=%.4f",
                cv["f1_macro_mean"],
                cv["f1_macro_std"],
                cv["recall_class1_mean"],
                cv["recall_class1_std"],
                cv["selection_score_mean"],
            )

            start = time.perf_counter()
            model = train_model(spec, data.X_train, data.y_train)
            elapsed = time.perf_counter() - start
            logger.info("  Treino final concluído em %.1fs", elapsed)

            test_ev = evaluate_model(model, data.X_test, data.y_test)
            log_evaluation(spec.name, test_ev)
        except RuntimeError as exc:
            logger.error("  %s descartado: %s", spec.name, exc)
            continue

        reports.append(ModelReport(spec.name, cv, test_ev, elapsed, model))

    if not reports:
        raise RuntimeError("Nenhum modelo foi treinado com sucesso.")

    best = max(reports, key=lambda r: r.cv_score)
    return reports, best


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


# --------------------------------------------------------------------------- #
# Persistência
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Execução
# --------------------------------------------------------------------------- #
def run_training() -> ModelReport:
    """Executa o fluxo completo: pré-processamento, comparação e salvamento.

    Returns:
        Relatório do modelo vencedor.
    """
    data = run_preprocessing()
    reports, best = compare_models(data)

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

    save_model(best, reports, data.feature_names)
    return best


def _configure_logging() -> None:
    """Configura o log do console para a execução direta do módulo."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


if __name__ == "__main__":
    _configure_logging()
    np.random.seed(RANDOM_STATE)
    try:
        run_training()
    except (FileNotFoundError, KeyError, ValueError, RuntimeError, OSError) as exc:
        logger.error("Falha no treinamento: %s", exc)
        raise SystemExit(1) from exc
