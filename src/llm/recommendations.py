"""Recomendações práticas ligadas a cada fator de risco e a cada classe."""

from __future__ import annotations

RECOMMENDATIONS: dict[str, str] = {
    "BMI": "Busque reduzir o peso aos poucos, com alimentação equilibrada e "
           "acompanhamento de um nutricionista, se possível.",
    "HighBP": "Meça a pressão com regularidade e siga o tratamento indicado pelo "
              "seu médico. Reduzir o sal ajuda bastante.",
    "HighChol": "Faça exames de colesterol periodicamente e reduza frituras, "
                "embutidos e alimentos ultraprocessados.",
    "Age": "Faça check-ups anuais, incluindo exames de glicemia de jejum e "
           "hemoglobina glicada.",
    "GenHlth": "Converse com um médico sobre o que tem afetado sua saúde no dia a dia.",
    "PhysActivity": "Tente chegar a 150 minutos de atividade física por semana, "
                    "como 30 minutos de caminhada em 5 dias.",
    "DiffWalk": "Peça orientação a um profissional sobre atividades adaptadas, "
                "como hidroginástica ou exercícios sentados.",
    "Smoker": "Se ainda fuma, procure apoio para parar: o SUS oferece tratamento "
              "gratuito para quem quer deixar o cigarro.",
    "HvyAlcoholConsump": "Reduza o consumo de bebidas alcoólicas.",
    "HeartDiseaseorAttack": "Mantenha o acompanhamento com o cardiologista em dia.",
    "Stroke": "Mantenha o acompanhamento médico em dia e controle pressão e glicemia.",
    "PhysHlth": "Procure avaliação médica para investigar os dias de mal-estar físico.",
    "InNaturaFood": "Inclua frutas, verduras e legumes nas refeições todos os dias.",
    "Education": "A Unidade Básica de Saúde (UBS) do seu bairro pode tirar dúvidas "
                 "e orientar sobre prevenção de forma gratuita.",
    "Income": "A Unidade Básica de Saúde (UBS) oferece consultas e exames de "
              "glicemia gratuitos pelo SUS.",
}

GENERAL_RECOMMENDATIONS: dict[int, str] = {
    0: "Mantenha os bons hábitos e faça exames de rotina conforme a orientação "
       "do seu médico.",
    1: "Agende uma consulta para fazer exames de glicemia. Na pré-diabetes, "
       "mudanças de hábito podem fazer os níveis voltarem ao normal.",
    2: "Agende uma consulta médica o quanto antes para fazer exames de "
       "glicemia e receber orientação adequada.",
}
