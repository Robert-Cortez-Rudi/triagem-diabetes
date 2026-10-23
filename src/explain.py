"""Explicabilidade técnica (SHAP) da predição de um paciente.

Recebe os dados de um paciente, aplica o modelo salvo em ``models/`` e devolve
a classe predita, as probabilidades e os fatores que mais pesaram na decisão,
num dicionário pronto para a LLM (``src/llm.py``) e para o Streamlit.

Como o SHAP é calculado:
    O modelo é um Pipeline ``StandardScaler`` -> ``LogisticRegression``
    multinomial. O paciente é padronizado com o scaler salvo e o
    ``shap.LinearExplainer`` calcula as contribuições na escala de log-odds
    (logit) de cada classe. A referência (background) é a média do treino. No
    espaço padronizado essa média é exatamente zero, então não é preciso
    carregar o dataset na hora da predição. Para modelo linear com features
    independentes, o SHAP de uma feature é ``coef * (x_padronizado - 0)``.

O que conta como "risco":
    Fator de risco é o que empurra o paciente PARA uma condição de diabetes;
    fator de proteção é o que o afasta dela.
    - Classe predita 1 ou 2: usa o SHAP da própria classe predita.
    - Classe predita 0: usa o SHAP da classe "Sem diabetes" com o sinal
      invertido. O que reduz a chance de estar saudável conta como risco.
    Assim, ``top_risk_factors`` significa sempre "aumentou o risco", qualquer
    que seja a classe predita.

Uso pelo terminal::

    python -m src.explain
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.preprocess import CLASS_LABELS, FOOD_SOURCE_COLUMNS, NEW_FEATURE, PROJECT_ROOT

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------------- #
MODELS_DIR: Path = PROJECT_ROOT / "models"
MODEL_PATH: Path = MODELS_DIR / "best_model.joblib"
METRICS_PATH: Path = MODELS_DIR / "metrics.json"

TOP_N: int = 3
HEALTHY_CLASS: int = 0

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "HighBP": "Pressão Alta",
    "HighChol": "Colesterol Alto",
    "BMI": "Índice de Massa Corporal (IMC)",
    "Smoker": "Histórico de Tabagismo",
    "Stroke": "Histórico de AVC",
    "HeartDiseaseorAttack": "Doença Cardíaca ou Infarto",
    "PhysActivity": "Atividade Física",
    "HvyAlcoholConsump": "Consumo Excessivo de Álcool",
    "GenHlth": "Autoavaliação da Saúde Geral",
    "PhysHlth": "Dias com Saúde Física Ruim (último mês)",
    "DiffWalk": "Dificuldade para Caminhar",
    "Age": "Faixa Etária",
    "Education": "Escolaridade",
    "Income": "Faixa de Renda",
    NEW_FEATURE: "Consumo Diário de Frutas ou Verduras",
}

# Textos dos códigos do BRFSS 2015, para a LLM falar em termos concretos.
_YES_NO = {0: "Não", 1: "Sim"}
VALUE_DESCRIPTIONS: dict[str, dict[int, str]] = {
    "HighBP": _YES_NO,
    "HighChol": _YES_NO,
    "Smoker": {0: "Não", 1: "Sim (fumou 100+ cigarros na vida)"},
    "Stroke": _YES_NO,
    "HeartDiseaseorAttack": _YES_NO,
    "PhysActivity": {0: "Não praticou nos últimos 30 dias", 1: "Praticou nos últimos 30 dias"},
    "HvyAlcoholConsump": _YES_NO,
    "DiffWalk": _YES_NO,
    NEW_FEATURE: {0: "Não consome", 1: "Consome diariamente"},
    "GenHlth": {1: "Excelente", 2: "Muito boa", 3: "Boa", 4: "Regular", 5: "Ruim"},
    "Age": {
        1: "18 a 24 anos", 2: "25 a 29 anos", 3: "30 a 34 anos", 4: "35 a 39 anos",
        5: "40 a 44 anos", 6: "45 a 49 anos", 7: "50 a 54 anos", 8: "55 a 59 anos",
        9: "60 a 64 anos", 10: "65 a 69 anos", 11: "70 a 74 anos", 12: "75 a 79 anos",
        13: "80 anos ou mais",
    },
    "Education": {
        1: "Nunca frequentou a escola", 2: "Fundamental incompleto",
        3: "Fundamental completo / médio incompleto", 4: "Ensino médio completo",
        5: "Superior incompleto / técnico", 6: "Superior completo",
    },
    "Income": {
        1: "Menos de US$ 10 mil", 2: "US$ 10 a 15 mil", 3: "US$ 15 a 20 mil",
        4: "US$ 20 a 25 mil", 5: "US$ 25 a 35 mil", 6: "US$ 35 a 50 mil",
        7: "US$ 50 a 75 mil", 8: "US$ 75 mil ou mais",
    },
}


# --------------------------------------------------------------------------- #
# Carregamento (feito uma vez por processo)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExplainerBundle:
    """Modelo, explainer SHAP e metadados carregados de ``models/``."""

    pipeline: Pipeline
    explainer: shap.LinearExplainer
    feature_names: list[str]
    classes: list[int]
    model_name: str


@lru_cache(maxsize=1)
def load_explainer(
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
) -> ExplainerBundle:
    """Carrega o pipeline e a ordem das features e monta o ``LinearExplainer``.

    Args:
        model_path: Caminho do ``best_model.joblib``.
        metrics_path: Caminho do ``metrics.json`` gerado por ``src.train``.

    Returns:
        :class:`ExplainerBundle` pronto para uso.

    Raises:
        FileNotFoundError: Se algum artefato não existir (rode ``python -m src.train``).
        ValueError: Se o modelo salvo não for o Pipeline scaler + Regressão Logística.
    """
    for path in (model_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Artefato não encontrado: {path}. Rode 'python -m src.train' antes."
            )

    pipeline = joblib.load(model_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    feature_names: list[str] = metrics["feature_names"]

    if not isinstance(pipeline, Pipeline) or {"scaler", "clf"} - set(pipeline.named_steps):
        raise ValueError(
            "O explain.py espera um Pipeline com as etapas 'scaler' e 'clf' "
            f"(Regressão Logística); encontrado: {type(pipeline).__name__}."
        )

    clf = pipeline.named_steps["clf"]
    n_features = len(feature_names)
    if clf.coef_.shape[1] != n_features:
        raise ValueError(
            f"O modelo tem {clf.coef_.shape[1]} features e o metrics.json lista {n_features}."
        )

    # Background = média do treino. No espaço padronizado ela é o vetor zero.
    masker = (np.zeros(n_features), np.eye(n_features))
    explainer = shap.LinearExplainer(clf, masker)

    logger.info("Modelo '%s' e explainer SHAP carregados.", metrics.get("best_model", "?"))
    return ExplainerBundle(
        pipeline=pipeline,
        explainer=explainer,
        feature_names=feature_names,
        classes=[int(c) for c in clf.classes_],
        model_name=metrics.get("best_model", "desconhecido"),
    )


# --------------------------------------------------------------------------- #
# Entrada do paciente
# --------------------------------------------------------------------------- #
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


def describe_value(feature: str, value: float) -> str:
    """Traduz o código de uma feature para texto legível (ex.: Age 10 -> "65 a 69 anos")."""
    if feature == "BMI":
        return f"{value:.1f}"
    if feature == "PhysHlth":
        return f"{int(value)} dia(s)"
    mapping = VALUE_DESCRIPTIONS.get(feature)
    if mapping is not None and float(value).is_integer():
        return mapping.get(int(value), str(int(value)))
    return f"{value:g}"


# --------------------------------------------------------------------------- #
# Explicação
# --------------------------------------------------------------------------- #
def _factor(feature: str, value: float, impact: float) -> dict[str, Any]:
    return {
        "feature": feature,
        "description": FEATURE_DESCRIPTIONS.get(feature, feature),
        "value": int(value) if float(value).is_integer() else round(float(value), 2),
        "value_description": describe_value(feature, value),
        "shap_impact": round(float(impact), 4),
    }


def get_prediction_explanation(
    patient_data_dict: Mapping[str, Any] | pd.DataFrame,
    top_n: int = TOP_N,
) -> dict[str, Any]:
    """Prediz a classe de um paciente e explica a decisão com SHAP.

    Args:
        patient_data_dict: Dados do paciente (dict ou DataFrame de 1 linha)
            com as 15 features do modelo, ou ``Fruits``/``Veggies`` no lugar
            de ``InNaturaFood``.
        top_n: Quantos fatores devolver em cada lista.

    Returns:
        Dicionário com:
            - ``predicted_class``: ``{"code": int, "label": str}``.
            - ``probabilities``: ``{rótulo: probabilidade}`` das 3 classes.
            - ``top_risk_factors``: até ``top_n`` fatores que aumentaram o risco,
              do maior para o menor impacto.
            - ``protective_factors``: até ``top_n`` fatores que reduziram o
              risco, do mais forte para o mais fraco (impacto negativo).
            - ``explanation_basis``: classe e escala usadas no SHAP.
            - ``model``: nome do modelo.
        Cada fator traz ``feature``, ``description``, ``value``,
        ``value_description`` e ``shap_impact`` (em log-odds).

    Raises:
        FileNotFoundError: Se os artefatos do modelo não existirem.
        ValueError: Se os dados do paciente forem inválidos.
    """
    bundle = load_explainer()
    patient = prepare_patient(patient_data_dict, bundle.feature_names)

    proba = bundle.pipeline.predict_proba(patient)[0]
    pred_idx = int(np.argmax(proba))
    pred_class = bundle.classes[pred_idx]

    scaled = bundle.pipeline.named_steps["scaler"].transform(patient)
    shap_values = bundle.explainer(scaled).values[0]  # (n_features, n_classes)

    if pred_class == HEALTHY_CLASS:
        risk_contrib = -shap_values[:, pred_idx]
        basis = (
            f"SHAP da classe '{CLASS_LABELS[pred_class]}' com sinal invertido: "
            "impacto positivo = reduziu a chance de estar sem diabetes"
        )
    else:
        risk_contrib = shap_values[:, pred_idx]
        basis = (
            f"SHAP da classe predita '{CLASS_LABELS[pred_class]}': "
            "impacto positivo = aumentou a chance dessa classe"
        )

    raw = patient.iloc[0]
    order = np.argsort(risk_contrib)
    risk_idx = [i for i in order[::-1] if risk_contrib[i] > 0][:top_n]
    protect_idx = [i for i in order if risk_contrib[i] < 0][:top_n]

    def build(indices: list[int]) -> list[dict[str, Any]]:
        return [
            _factor(bundle.feature_names[i], raw.iloc[i], risk_contrib[i]) for i in indices
        ]

    return {
        "predicted_class": {"code": pred_class, "label": CLASS_LABELS[pred_class]},
        "probabilities": {
            CLASS_LABELS[c]: round(float(p), 4) for c, p in zip(bundle.classes, proba)
        },
        "top_risk_factors": build(risk_idx),
        "protective_factors": build(protect_idx),
        "explanation_basis": {
            "explained_class": CLASS_LABELS[pred_class],
            "scale": "log-odds (logit) da Regressão Logística",
            "reference": "média do conjunto de treino",
            "note": basis,
        },
        "model": bundle.model_name,
    }


# --------------------------------------------------------------------------- #
# Execução de teste
# --------------------------------------------------------------------------- #
EXAMPLE_PATIENT: dict[str, float] = {
    "HighBP": 1,
    "HighChol": 1,
    "BMI": 34.0,
    "Smoker": 1,
    "Stroke": 0,
    "HeartDiseaseorAttack": 0,
    "PhysActivity": 0,
    "Fruits": 0,
    "Veggies": 1,
    "HvyAlcoholConsump": 0,
    "GenHlth": 4,
    "PhysHlth": 10,
    "DiffWalk": 1,
    "Age": 11,
    "Education": 4,
    "Income": 4,
}


def _configure_logging() -> None:
    """Configura o log do console para a execução direta do módulo."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


if __name__ == "__main__":
    _configure_logging()
    logger.info("Paciente de teste: idoso (70-74 anos), IMC 34, hipertenso, sem atividade física.")
    try:
        result = get_prediction_explanation(EXAMPLE_PATIENT)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Falha na explicação: %s", exc)
        raise SystemExit(1) from exc
    print(json.dumps(result, indent=2, ensure_ascii=False))
