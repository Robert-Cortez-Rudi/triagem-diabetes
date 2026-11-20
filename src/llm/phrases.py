"""Frase em linguagem simples de cada fator quando ele AUMENTA o risco."""

from __future__ import annotations

from typing import Any


def _bmi_phrase(value: float) -> str:
    if value >= 30:
        faixa = "na faixa de obesidade"
    elif value >= 25:
        faixa = "na faixa de sobrepeso"
    elif value < 18.5:
        faixa = "abaixo do peso considerado adequado"
    else:
        faixa = "na faixa considerada adequada"
    return (
        f"Seu Índice de Massa Corporal (IMC) é {value:.1f}, {faixa}. O excesso de "
        "peso dificulta o uso da insulina pelo corpo e é um dos fatores mais "
        "ligados ao diabetes tipo 2."
    )


# Cada função recebe o fator no formato do src.explain.
RISK_PHRASES: dict[str, Any] = {
    "BMI": lambda f: _bmi_phrase(float(f["value"])),
    "HighBP": lambda f: (
        "Você informou ter pressão alta. Pressão alta e diabetes costumam "
        "aparecer juntas e uma agrava a outra."
    ),
    "HighChol": lambda f: (
        "Você informou ter colesterol alto, que costuma andar junto com "
        "alterações no açúcar do sangue."
    ),
    "Age": lambda f: (
        f"Sua faixa etária ({f['value_description']}) pesa no resultado: o risco "
        "de diabetes aumenta naturalmente com a idade."
    ),
    # "Boa" (3) fica abaixo da média da base e também pesa como risco, mas não
    # indica problema; só "regular" e "ruim" (4 e 5) recebem o texto de alerta.
    "GenHlth": lambda f: (
        f"Você avaliou sua saúde geral como \"{f['value_description'].lower()}\". "
        + ("Essa percepção costuma refletir problemas de saúde que merecem atenção."
           if int(f["value"]) >= 4 else
           "Quem avalia a saúde como muito boa ou excelente costuma ter risco menor, "
           "por isso esse ponto pesou um pouco.")
    ),
    "PhysActivity": lambda f: (
        "Você não praticou atividade física no último mês. O exercício ajuda o "
        "corpo a usar melhor o açúcar do sangue."
    ),
    "DiffWalk": lambda f: (
        "Você relatou dificuldade para caminhar ou subir escadas, o que costuma "
        "reduzir a movimentação no dia a dia."
    ),
    "Smoker": lambda f: (
        "Você tem histórico de tabagismo. O cigarro aumenta a resistência à "
        "insulina e o risco de complicações."
    ),
    "HvyAlcoholConsump": lambda f: (
        "Você relatou consumo elevado de álcool, que pode desregular o açúcar do "
        "sangue e sobrecarregar o fígado."
    ),
    "HeartDiseaseorAttack": lambda f: (
        "Você tem histórico de doença no coração ou infarto, condição que "
        "costuma estar ligada ao diabetes."
    ),
    "Stroke": lambda f: (
        "Você tem histórico de AVC, condição que costuma estar ligada a "
        "alterações de pressão e de açúcar no sangue."
    ),
    "PhysHlth": lambda f: (
        f"Você teve {f['value_description']} com a saúde física ruim no último "
        "mês, o que pode indicar algo que precisa ser investigado."
    ),
    "InNaturaFood": lambda f: (
        "Você não consome frutas nem verduras todos os dias. Esses alimentos "
        "ajudam a controlar o açúcar do sangue e o peso."
    ),
    "Education": lambda f: (
        f"Sua escolaridade ({f['value_description'].lower()}) aparece entre os "
        "fatores. Na população, ela está ligada ao acesso a informação e a "
        "serviços de saúde."
    ),
    "Income": lambda f: (
        f"Sua faixa de renda ({f['value_description']}) aparece entre os "
        "fatores. Na população, ela está ligada ao acesso a alimentação "
        "saudável e a serviços de saúde."
    ),
}
