# Triagem de Diabetes — Projeto Final CARCDSI

Sistema inteligente de apoio à triagem que classifica o perfil de diabetes em três categorias (sem diabetes, pré-diabetes ou diabetes) e indica os fatores associados à predição. É uma ferramenta de apoio à triagem e não substitui diagnóstico médico.

## Autores

- Emerson Soares da Silva (CG3033741)
- Robert Cortez Rudi (CG3035735)

## Dataset

[Diabetes Health Indicators (BRFSS 2015)](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

Baixe o arquivo `diabetes_012_health_indicators_BRFSS2015.csv` e salve em `data/`.

## Arquitetura em 3 camadas

1. **Aprendizado e construção do conhecimento** (`src/preprocess/`, `src/train/`):
   carga do CSV, remoção de duplicatas, descarte de `MentHlth`, `Sex`, `CholCheck`,
   `AnyHealthcare` e `NoDocbcCost`, criação da feature `InNaturaFood` (frutas OU
   vegetais diariamente), divisão estratificada 80/20, tratamento do desbalanceamento
   e comparação de cinco algoritmos.
2. **Inferência inteligente e evidências** (`app.py`, `src/ui/`, `src/explain/`): o
   paciente preenche o formulário no Streamlit, o modelo prediz a classe e as
   probabilidades, e o SHAP aponta os fatores que mais aumentaram e reduziram o risco.
3. **Explicação e justificativa** (`src/llm/`): uma LLM local e privada (Ollama,
   `qwen2.5:3b`) transforma o resultado e os fatores SHAP em um texto simples e
   empático. Sem a LLM, um modo offline monta o texto com templates.

## Modelos e resultados

Algoritmos comparados, todos com `random_state = 42`:

| Algoritmo | Tratamento do desbalanceamento |
|---|---|
| Regressão Logística multinomial (com `StandardScaler`) | `class_weight="balanced"` |
| Árvore de Decisão | `class_weight="balanced"` |
| k-NN, k = 25 (com `StandardScaler`) | subamostragem aleatória (`imbalanced-learn`) |
| Random Forest | `class_weight="balanced"` |
| XGBoost (opcional, árvores com *boosting*) | pesos por amostra balanceados |

A métrica principal é o **F1 Macro**, acompanhada do **recall por classe** e da
**matriz de confusão**. Para escolher o modelo, usamos a média harmônica entre o F1
Macro e o Recall do Pré-diabetes na validação cruzada estratificada de 5 dobras. O
conjunto de teste nunca é usado na escolha.

Resultados no conjunto de teste (45.957 registros; CV = validação cruzada no treino):

| Modelo | Score de seleção (CV) | F1 Macro | Recall Sem diabetes | Recall Pré-diabetes | Recall Diabetes | Acurácia |
|---|---|---|---|---|---|---|
| Árvore de Decisão | 0,3551 | 0,4019 | 0,581 | 0,289 | 0,611 | 0,5793 |
| k-NN | 0,3430 | 0,3937 | 0,594 | 0,337 | 0,519 | 0,5773 |
| **Regressão Logística** (publicada) | 0,3416 | **0,4263** | 0,650 | 0,283 | 0,597 | 0,6346 |
| XGBoost | 0,2984 | 0,4229 | 0,638 | 0,251 | 0,635 | 0,6295 |
| Random Forest | 0,2481 | 0,4287 | 0,659 | 0,191 | 0,675 | 0,6522 |

A Árvore de Decisão lidera o score de seleção, mas tem F1 Macro bem menor que a
Regressão Logística (0,395 contra 0,422 na CV). Publicamos a **Regressão Logística**
porque ela tem o maior F1 Macro entre as três primeiras e é interpretável por
coeficientes, o que permite a explicação SHAP exata usada na interface e na LLM. As
métricas completas e as matrizes de confusão ficam em `models/metrics.json`, e as
figuras do artigo em `notebooks/`.

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
│   ├── train/         # 5 modelos, validação cruzada, comparação e salvamento
│   ├── explain/       # fatores associados a cada predição (SHAP)
│   ├── llm/           # explicação em linguagem simples (Ollama, com modo offline)
│   └── ui/            # componentes da interface Streamlit
├── app.py             # ponto de entrada da interface Streamlit
├── CLAUDE.md          # escopo, dicionário de dados e regras técnicas do projeto
├── requirements.txt   # dependências do projeto (versões fixadas)
├── .gitignore         # arquivos e pastas fora do versionamento
└── README.md          # este arquivo
```

## Como executar

Requer Python 3.11 ou superior (testado no 3.13). Crie o ambiente virtual:

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
`models/best_model.joblib` e `models/metrics.json`. O modelo treinado já vem no
repositório, então esse passo só é necessário para reproduzir o treino (cerca de
3 minutos).

### Abrir a interface

```
streamlit run app.py
```

O navegador abre em `http://localhost:8501`. Preencha o formulário na barra lateral
e clique em **Realizar Triagem de Risco**. A tela mostra o perfil estimado com as
probabilidades, os fatores que mais pesaram e a explicação em linguagem simples.

### Notebook

`notebooks/01_analise_exploratoria_e_metricas.ipynb` gera as figuras e a tabela
comparativa usadas no artigo (distribuição do target, correlação, matriz de confusão
e SHAP). Ele lê o modelo salvo em `models/`.

## Aviso

Esta é uma ferramenta de **apoio à triagem**. Ela não faz diagnóstico e não substitui
a consulta médica. Só um profissional de saúde, com exames como glicemia de jejum e
hemoglobina glicada, pode confirmar ou descartar diabetes.
