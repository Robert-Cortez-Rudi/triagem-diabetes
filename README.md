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
├── src/               # cada pasta é um pacote; arquivos com no máximo 100 linhas
│   ├── __init__.py    # marca src como pacote Python
│   ├── logging_config.py  # formato do log no terminal
│   ├── preprocess/    # carrega o CSV, trata duplicatas e divide treino e teste
│   ├── train/         # Pipeline, validação cruzada, comparação de modelos e salvamento
│   ├── explain/       # fatores associados a cada predição (SHAP)
│   ├── llm/           # explicação em linguagem simples (Ollama, com modo offline)
│   └── ui/            # componentes da interface Streamlit
├── app.py             # ponto de entrada da interface Streamlit
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

### Instalar o Ollama (LLM local)

O texto explicativo para o paciente é escrito por uma LLM que roda no seu
computador pelo [Ollama](https://ollama.com). O Ollama é um programa à parte:
ele **não** é instalado pelo `requirements.txt`, então cada computador que
for executar o projeto precisa instalá-lo e baixar o modelo.

1. Instale o Ollama.

   ```
   # Windows (PowerShell)
   winget install Ollama.Ollama

   # Linux
   curl -fsSL https://ollama.com/install.sh | sh

   # macOS: baixe o instalador em https://ollama.com/download
   ```

   Depois da instalação, feche e abra o terminal de novo.

2. Baixe o modelo usado pelo projeto (cerca de 2 GB):

   ```
   ollama pull qwen2.5:3b
   ```

3. Confira se está funcionando:

   ```
   ollama run qwen2.5:3b "Explique em duas frases o que é pré-diabetes."
   ```

No Windows e no macOS o Ollama fica rodando em segundo plano depois de
instalado. No Linux, se ele não estiver ativo, rode `ollama serve` em outro
terminal. O projeto se conecta a ele em `http://127.0.0.1:11434`.

O modelo `qwen2.5:3b` foi escolhido por escrever bem em português e rodar
sem placa de vídeo. Sem GPU, cada explicação leva cerca de 30 a 60 segundos.

> **Sem Ollama o projeto continua funcionando.** Se o Ollama não estiver
> instalado, estiver desligado ou não tiver o modelo, o sistema usa um modo
> offline: a explicação é montada com textos prontos a partir dos mesmos
> fatores. Você só deixa de ter o texto escrito pela LLM. O terminal mostra
> um aviso indicando o modo usado.

#### Configuração opcional

O comportamento da LLM pode ser ajustado por variáveis de ambiente:

| Variável | Padrão | Para que serve |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5:3b` | Usar outro modelo baixado no Ollama |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Usar um Ollama que roda em outro computador da rede |
| `OLLAMA_TIMEOUT` | `120` | Tempo máximo, em segundos, para a LLM responder |
| `LLM_DISABLED` | `0` | `1` força o modo offline, sem chamar a LLM |

Exemplo no PowerShell: `$env:OLLAMA_MODEL = "llama3.2:3b"`.

Para usar o Ollama de outro computador, ele precisa estar liberado na rede
(`OLLAMA_HOST=0.0.0.0` no computador que roda o Ollama). O Ollama não tem
senha, então faça isso só em rede confiável, nunca exposto na internet.

### Executar as etapas

Rode os comandos a partir da raiz do projeto, com o ambiente virtual ativo:

```
python -m src.preprocess   # limpa os dados e mostra a divisão treino/teste
python -m src.train        # compara os modelos e salva o melhor em models/
python -m src.explain      # explica a predição de um paciente de teste (SHAP)
python -m src.llm          # gera o texto para o paciente (Ollama ou modo offline)
```

O `src.train` precisa rodar antes dos dois últimos, porque eles usam
`models/best_model.joblib` e `models/metrics.json`.

## Status

Em desenvolvimento: módulos de ML, explicabilidade e LLM prontos; interface
Streamlit (`app.py`) pendente.
