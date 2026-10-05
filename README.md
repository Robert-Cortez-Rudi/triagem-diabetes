# Triagem de Diabetes — Projeto Final CARCDSI

Sistema inteligente de apoio à triagem que classifica o perfil de diabetes em três categorias (sem diabetes, pré-diabetes ou diabetes) e indica os fatores associados à predição. É uma ferramenta de apoio à triagem e não substitui diagnóstico médico.

## Autores

- Emerson Soares da Silva (CG3033741)
- Robert Cortez Rudi (CG3035735)

## Dataset

[Diabetes Health Indicators (BRFSS 2015)](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

Baixe o arquivo `diabetes_012_health_indicators_BRFSS2015.csv` e salve em `data/`.

## Estrutura do projeto

```
.
├── data/              # dataset em CSV (não versionado)
├── notebooks/         # notebooks de análise exploratória e experimentos
├── models/            # modelos treinados
├── src/
│   ├── __init__.py    # marca src como pacote Python
│   ├── preprocess.py  # carrega o CSV, trata duplicatas e divide treino e teste
│   ├── train.py       # Pipeline, validação cruzada, comparação de modelos e salvamento
│   ├── explain.py     # fatores associados a cada predição
│   └── llm.py         # explicação em linguagem simples com um LLM local
├── app.py             # interface Streamlit
├── requirements.txt   # dependências do projeto
├── .gitignore         # arquivos e pastas fora do versionamento
└── README.md          # este arquivo
```

## Como executar

Crie o ambiente virtual:

```
python -m venv .venv
```

Ative o ambiente:

```
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate
```

Instale as dependências:

```
pip install -r requirements.txt
```

## Status

Em desenvolvimento: fase de Machine Learning
