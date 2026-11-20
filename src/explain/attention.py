"""Pontos de atenção: respostas que são fatores de risco conhecidos para o diabetes.

Diferente do SHAP, aqui não entra o modelo. São regras fixas, baseadas nos
fatores de risco reconhecidos pelas diretrizes de saúde (como as da Sociedade
Brasileira de Diabetes). Assim, um hábito de risco aparece para o paciente
mesmo quando pesou pouco na predição; por exemplo, o sedentarismo de quem
recebeu "risco baixo".
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

HABIT, CONDITION = "habit", "condition"
AGE_SCREENING_CODE = 4  # faixa 35 a 39 anos: idade de início do rastreio de glicemia


def _point(feature: str, kind: str, title: str, detail: str) -> dict[str, str]:
    return {"feature": feature, "kind": kind, "title": title, "detail": detail}


def _habits(p: Mapping[str, float]) -> list[dict[str, str]]:
    out = []
    bmi = p.get("BMI", 0)
    if bmi >= 25:
        faixa = "obesidade" if bmi >= 30 else "sobrepeso"
        out.append(_point("BMI", HABIT, f"IMC {bmi:.1f}, na faixa de {faixa}",
            "O excesso de peso é o principal fator de risco do diabetes tipo 2 que pode "
            "ser mudado. Perder de 5% a 7% do peso já reduz bastante o risco."))
    if p.get("PhysActivity") == 0:
        out.append(_point("PhysActivity", HABIT, "Sem atividade física no último mês",
            "A atividade física ajuda o corpo a usar melhor a insulina. A meta é chegar "
            "a 150 minutos por semana, como 30 minutos de caminhada em 5 dias."))
    fruits, veggies = p.get("Fruits"), p.get("Veggies")
    if fruits == 0 or veggies == 0 or p.get("InNaturaFood") == 0:
        falta = ("frutas nem verduras" if fruits == 0 and veggies == 0 or fruits is None
                 else "frutas" if fruits == 0 else "verduras ou legumes")
        out.append(_point("InNaturaFood", HABIT, f"Não come {falta} todos os dias",
            "Frutas, verduras e legumes trazem fibras que ajudam a controlar o açúcar "
            "no sangue. O ideal é incluí-los nas refeições todos os dias."))
    if p.get("Smoker") == 1:
        out.append(_point("Smoker", HABIT, "Histórico de tabagismo",
            "O cigarro aumenta a resistência à insulina. Se ainda fuma, o SUS oferece "
            "tratamento gratuito para parar."))
    if p.get("HvyAlcoholConsump") == 1:
        out.append(_point("HvyAlcoholConsump", HABIT, "Consumo excessivo de álcool",
            "O excesso de álcool favorece o ganho de peso e atrapalha o controle do "
            "açúcar no sangue."))
    return out


def _conditions(p: Mapping[str, float]) -> list[dict[str, str]]:
    out = []
    if p.get("HighBP") == 1:
        out.append(_point("HighBP", CONDITION, "Pressão alta",
            "Pressão alta e diabetes costumam aparecer juntas. Manter a pressão "
            "controlada protege o coração e os rins."))
    if p.get("HighChol") == 1:
        out.append(_point("HighChol", CONDITION, "Colesterol alto",
            "Colesterol alto, pressão alta e excesso de peso juntos aumentam muito o "
            "risco de diabetes."))
    if p.get("HeartDiseaseorAttack") == 1 or p.get("Stroke") == 1:
        feature = "HeartDiseaseorAttack" if p.get("HeartDiseaseorAttack") == 1 else "Stroke"
        out.append(_point(feature, CONDITION, "Doença cardíaca, infarto ou AVC",
            "Quem já teve problemas no coração ou na circulação deve medir a glicemia "
            "com regularidade."))
    if p.get("GenHlth", 0) >= 4:
        out.append(_point("GenHlth", CONDITION, "Saúde geral regular ou ruim",
            "Vale conversar com um médico sobre o que tem afetado sua saúde."))
    if p.get("Age", 0) >= AGE_SCREENING_CODE:
        out.append(_point("Age", CONDITION, "Idade a partir de 35 anos",
            "A partir dessa idade, recomenda-se medir a glicemia periodicamente, mesmo "
            "sem sintomas."))
    return out


def get_attention_points(patient_data: Mapping[str, Any] | pd.DataFrame) -> list[dict[str, str]]:
    """Lista os pontos de atenção do paciente, hábitos primeiro e depois condições.

    Args:
        patient_data: Respostas do paciente (dict ou DataFrame de 1 linha), com
            ``Fruits``/``Veggies`` ou ``InNaturaFood``.

    Returns:
        Lista de dicionários com ``feature``, ``kind`` (``"habit"`` ou
        ``"condition"``), ``title`` e ``detail``. Lista vazia se nada se aplicar.
    """
    raw = patient_data.iloc[0].to_dict() if isinstance(patient_data, pd.DataFrame) else patient_data
    p = {k: float(v) for k, v in raw.items() if pd.api.types.is_number(v)}
    return _habits(p) + _conditions(p)
