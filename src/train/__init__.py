"""Treinamento, avaliação e comparação de modelos de triagem de diabetes.

Modelos comparados (``models``; estimadores em ``estimators``):
    1. Regressão Logística multinomial, dentro de um Pipeline com StandardScaler.
    2. Árvore de Decisão.
    3. k-NN, com StandardScaler e subamostragem (Pipeline do imbalanced-learn).
    4. Random Forest.
    5. XGBoost (opcional no requisito; representante avançado de árvores).

Desbalanceamento: a Classe 1 (Pré-diabetes) tem cerca de 2% da base. A
Regressão Logística, a Árvore de Decisão e a Random Forest usam
``class_weight="balanced"``; o XGBoost não tem esse parâmetro para multiclasse e
recebe pesos por amostra equivalentes (``compute_sample_weight("balanced")``);
o k-NN não aceita pesos e usa ``RandomUnderSampler`` só durante o ``fit``.

Seleção (``comparison``): cada modelo passa por validação cruzada estratificada
no conjunto de treino. O vencedor é o de maior *score de seleção*, a média
harmônica entre o F1 Macro e o Recall da Classe 1 na validação cruzada. A média
harmônica pune o modelo que sacrifica uma métrica pela outra. O conjunto de
teste só é usado para o relatório final, nunca para escolher o modelo.
Todos os modelos entram no ranking, mas só os ``deployable`` (hoje, a
Regressão Logística) podem ser publicados: o SHAP linear de ``src.explain``,
a interface e a LLM dependem da interpretabilidade por coeficientes.

Artefatos salvos em ``models/`` (``persistence``):
    - ``best_model.joblib``: estimador vencedor, já treinado em todo o treino
      (no caso da Regressão Logística, o Pipeline inclui o scaler).
    - ``metrics.json``: métricas de todos os modelos, modelo escolhido, ordem
      das features e rótulos das classes.

Uso pelo terminal::

    python -m src.train
"""

from src.train.comparison import ModelReport, compare_models
from src.train.config import (
    CLASSES,
    CV_FOLDS,
    METRICS_FILENAME,
    MODEL_FILENAME,
    MODELS_DIR,
    PREDIABETES,
)
from src.train.fitting import cross_validate_model, train_model
from src.train.metrics import EvaluationResult, evaluate_model, harmonic_mean
from src.train.models import MODEL_SPECS, ModelSpec
from src.train.persistence import save_model
from src.train.reporting import log_evaluation, log_summary, summary_table
from src.train.runner import run_training

__all__ = [
    "CLASSES", "CV_FOLDS", "METRICS_FILENAME", "MODEL_FILENAME", "MODELS_DIR",
    "MODEL_SPECS", "PREDIABETES", "EvaluationResult", "ModelReport", "ModelSpec",
    "compare_models", "cross_validate_model", "evaluate_model", "harmonic_mean",
    "log_evaluation", "log_summary", "run_training", "save_model", "summary_table",
    "train_model",
]
