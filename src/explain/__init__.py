"""Explicabilidade técnica (SHAP) da predição de um paciente.

Recebe os dados de um paciente, aplica o modelo salvo em ``models/`` e devolve
a classe predita, as probabilidades e os fatores que mais pesaram na decisão,
num dicionário pronto para a LLM (``src/llm``) e para o Streamlit.

Como o SHAP é calculado (``loader`` e ``explanation``):
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

from src.explain.descriptions import (
    FEATURE_DESCRIPTIONS,
    VALUE_DESCRIPTIONS,
    build_factor,
    describe_value,
)
from src.explain.explanation import HEALTHY_CLASS, TOP_N, get_prediction_explanation
from src.explain.loader import (
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    ExplainerBundle,
    load_explainer,
)
from src.explain.patient import EXAMPLE_PATIENT, prepare_patient

__all__ = [
    "EXAMPLE_PATIENT",
    "FEATURE_DESCRIPTIONS",
    "HEALTHY_CLASS",
    "METRICS_PATH",
    "MODEL_PATH",
    "MODELS_DIR",
    "TOP_N",
    "VALUE_DESCRIPTIONS",
    "ExplainerBundle",
    "build_factor",
    "describe_value",
    "get_prediction_explanation",
    "load_explainer",
    "prepare_patient",
]
