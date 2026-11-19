"""Fábricas dos estimadores candidatos, cada uma devolvendo um modelo novo."""

from __future__ import annotations

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.preprocess import RANDOM_STATE
from src.train.config import CLASSES


def logistic_regression() -> Pipeline:
    """Regressão Logística multinomial; a padronização torna os coeficientes comparáveis."""
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


def decision_tree() -> DecisionTreeClassifier:
    """Árvore de Decisão; a profundidade limitada mantém as regras legíveis e evita decorar."""
    return DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def knn() -> ImbPipeline:
    """k-NN sobre dados padronizados, com subamostragem aleatória das classes maiores.

    O k-NN não aceita ``class_weight``: sem balancear, os vizinhos de quase todo
    paciente seriam da classe majoritária. A subamostragem (só no ``fit``) iguala
    as classes e ainda reduz a base de busca, o que torna a predição viável.
    """
    return ImbPipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("undersample", RandomUnderSampler(random_state=RANDOM_STATE)),
            ("clf", KNeighborsClassifier(n_neighbors=25, n_jobs=-1)),
        ]
    )


def random_forest() -> RandomForestClassifier:
    """Random Forest; árvores completas decoram a classe majoritária, então são limitadas."""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def xgboost() -> XGBClassifier:
    """XGBoost multiclasse; o balanceamento vem dos pesos por amostra no ``fit``."""
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
