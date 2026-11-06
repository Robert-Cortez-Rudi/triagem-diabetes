"""Constantes da interface: limites dos campos, cores e perguntas Sim/Não."""

from __future__ import annotations

BMI_MIN, BMI_MAX, BMI_DEFAULT = 12.0, 98.0, 25.0  # faixa observada no BRFSS 2015
PHYS_HLTH_MAX = 30

YES, NO = "Sim", "Não"

# Cor de cada classe: (fundo, borda, ícone).
CLASS_STYLES: dict[int, tuple[str, str, str]] = {
    0: ("#e8f5e9", "#2e7d32", "✅"),
    1: ("#fff8e1", "#f9a825", "⚠️"),
    2: ("#ffebee", "#c62828", "🔴"),
}

# Campos binários: (feature, pergunta, ajuda).
BINARY_FIELDS: list[tuple[str, str, str]] = [
    ("HighBP", "Tem pressão alta?", "Algum profissional de saúde já disse que você tem pressão alta?"),
    ("HighChol", "Tem colesterol alto?", "Algum profissional de saúde já disse que seu colesterol é alto?"),
    ("Smoker", "Já fumou pelo menos 100 cigarros na vida?", "Cerca de 5 maços ao longo da vida."),
    ("Stroke", "Já teve AVC (derrame)?", ""),
    ("HeartDiseaseorAttack", "Tem doença cardíaca ou já teve infarto?", ""),
    ("PhysActivity", "Praticou atividade física nos últimos 30 dias?", "Fora do trabalho."),
    ("HvyAlcoholConsump", "Consome bebida alcoólica em excesso?",
     "Homens: mais de 14 doses por semana. Mulheres: mais de 7 doses por semana."),
    ("DiffWalk", "Tem dificuldade séria para andar ou subir escadas?", ""),
    ("Fruits", "Come frutas pelo menos 1 vez por dia?", ""),
    ("Veggies", "Come verduras ou legumes pelo menos 1 vez por dia?", ""),
]
