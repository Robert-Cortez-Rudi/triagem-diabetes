# CLAUDE.md — Diretrizes Técnicas e Regras do Projeto

## 1. Informações Gerais do Projeto
- **Nome do Sistema**: Sistema Inteligente de Triagem de Diabetes
- **Repositório**: `Robert-Cortez-Rudi/triagem-diabetes`
- **Autores**: Emerson Soares da Silva (CG3033741) e Robert Cortez Rudi (CG3035735)
- **Contexto**: Projeto Final da disciplina CARCDSI
- **Escopo**: Aplicação de Machine Learning e LLM local para classificação do perfil de diabetes em 3 níveis (Sem diabetes, Pré-diabetes, Diabetes), com geração de evidências explicáveis (SHAP) e justificativa amigável em linguagem natural.
- **Natureza**: Ferramenta de **apoio e triagem prévia**. Nunca substitui consulta ou diagnóstico médico; toda saída ao paciente deve deixar isso claro.

---

## 2. Arquitetura em 3 Camadas

1. **Aprendizado e Construção do Conhecimento** (`src/preprocess/`, `src/train/`): carga, limpeza, descarte de variáveis, `InNaturaFood`, tratamento do desbalanceamento, treino e comparação dos modelos.
2. **Inferência Inteligente & Evidências** (`app.py`, `src/ui/`, `src/explain/`): entrada do paciente no Streamlit, predição e valores SHAP.
3. **Explicação e Justificativa** (`src/llm/`): LLM local (Ollama, `qwen2.5:3b`) traduz resultado e fatores SHAP em linguagem simples, com modo *fallback* offline por templates.

---

## 3. Estrutura de Pastas e Módulos

Cada pasta de `src/` é um pacote executável com `python -m src.<pacote>`. Arquivos com no máximo 100 linhas.

```
.
├── data/              # Dataset original em CSV (não versionado)
├── notebooks/         # Análise exploratória, métricas e figuras do artigo
├── models/            # best_model.joblib e metrics.json
├── src/
│   ├── preprocess/    # Carga, limpeza, InNaturaFood e divisão estratificada
│   ├── train/         # Estimadores, validação cruzada, comparação e exportação
│   ├── explain/       # SHAP: fatores de risco e de proteção de cada predição
│   ├── llm/           # Ollama + fallback offline, prompts e validação do texto
│   └── ui/            # Componentes da interface Streamlit
├── app.py             # Ponto de entrada da interface Streamlit
├── requirements.txt   # Dependências com versões fixadas
└── README.md          # Documentação principal e instruções de execução
```

---

## 4. Dicionário de Dados e Regras de Negócio

### Target (`Diabetes_012`)
- `0`: Sem diabetes ou apenas diabetes gestacional.
- `1`: Pré-diabetes (Classe minoritária e crítica — requer atenção ao desbalanceamento).
- `2`: Diabetes diagnosticado.

### Variáveis Utilizadas
- **Muito Importantes**: `BMI` (IMC), `HighBP` (Hipertensão), `HighChol` (Colesterol alto), `Age` (Faixa etária), `GenHlth` (Autoavaliação da saúde).
- **Importantes / Hábitos de Saúde**: `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`, `Stroke`, `DiffWalk`, `Smoker`, `PhysHlth`, `Education`, `Income`.

### Feature Engineering Obrigatório
- Criar a variável `InNaturaFood` a partir da união de `Fruits` e `Veggies` (`1` se consome frutas OU vegetais diariamente, `0` se não consome nenhum).

### Variáveis Descartadas (Remover no Pré-processamento)
- `MentHlth` (Saúde mental)
- `Sex` (Gênero)
- `CholCheck` (Verificação de colesterol nos últimos 5 anos)
- `AnyHealthCare` (Plano de saúde; o CSV grafa `AnyHealthcare`)
- `NoDocbcCost` (Deixou de ir ao médico por custo)

---

## 5. Algoritmos Comparados

1. **Regressão Logística Multinomial** — interpretabilidade via coeficientes.
2. **Árvore de Decisão** — classificação por regras e condições.
3. **k-NN** — classificação pela similaridade entre indivíduos.
4. **Random Forest** — ganho de desempenho com múltiplas árvores.
5. **XGBoost** (opcional) — representante avançado dos métodos de árvores.

Desbalanceamento: `class_weight="balanced"` (Regressão Logística, Árvore, Random Forest), pesos por amostra (XGBoost) e subamostragem com `imbalanced-learn` (k-NN).

---

## 6. Diretrizes de Código e Padrões Técnicos

1. **Linguagem e Bibliotecas**:
   - Python 3.11+
   - Manipulação de Dados: `pandas`, `numpy`
   - Machine Learning: `scikit-learn`, `imbalanced-learn`, `xgboost`
   - Explicabilidade: `shap`
   - Interface: `streamlit`
   - LLM local: Ollama via `requests`
   - Reprodutibilidade: Fixar `random_state = 42` em todas as operações estocásticas.

2. **Qualidade do Código**:
   - Modularização estrita dentro da pasta `src/`.
   - Funções bem documentadas com *docstrings* e *type hints*.
   - Tratamento defensivo de exceções e *logs* de execução limpos.

3. **Métricas de Avaliação**:
   - **Métrica principal**: F1-Score Macro.
   - Recall por classe e matriz de confusão.
   - **Métrica Chave**: Recall e F1-Score específicos para a **Classe 1 (Pré-diabetes)** devido à importância médica do diagnóstico precoce.
   - Acurácia e F1 Weighted como complemento.
   - Seleção do modelo: média harmônica entre F1 Macro e Recall da Classe 1 na validação cruzada estratificada (5 dobras); o teste nunca é usado para escolher. Todos os modelos entram no ranking, mas só a Regressão Logística é publicada no app (`deployable=True`), porque o SHAP linear, a interface e a LLM dependem dos seus coeficientes.

---

## 7. Fluxo de Execução do Sistema (`app.py`)
1. **Entrada de Dados**: Usuário preenche seus dados na interface do Streamlit.
2. **Pré-processamento**: Aplica as mesmas transformações do treino (`InNaturaFood`, descarte de colunas e escalonamento se necessário).
3. **Predição**: Modelo carregado de `models/` gera a probabilidade e a classe predita (`0`, `1` ou `2`).
4. **Evidências (SHAP)**: Identifica os 3 fatores que mais impulsionaram a classificação daquele paciente.
5. **Explicação (LLM Local)**: LLM recebe o resultado e os fatores do SHAP e constrói um texto amigável em linguagem simples para o paciente.
