"""Nomes legíveis das features e textos dos códigos do BRFSS 2015."""

from __future__ import annotations

from typing import Any

from src.preprocess import NEW_FEATURE

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


def build_factor(feature: str, value: float, impact: float) -> dict[str, Any]:
    """Monta o dicionário de um fator: nome legível, resposta e impacto SHAP."""
    return {
        "feature": feature,
        "description": FEATURE_DESCRIPTIONS.get(feature, feature),
        "value": int(value) if float(value).is_integer() else round(float(value), 2),
        "value_description": describe_value(feature, value),
        "shap_impact": round(float(impact), 4),
    }
