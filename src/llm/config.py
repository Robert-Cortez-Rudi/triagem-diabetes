"""Configuração da LLM local (variáveis de ambiente) e limites do texto."""

from __future__ import annotations

import os
import re
from typing import Literal

# 127.0.0.1 em vez de localhost: no Windows, localhost tenta IPv6 antes e
# demora alguns segundos para desistir quando o Ollama está desligado.
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_TIMEOUT: float = float(os.getenv("OLLAMA_TIMEOUT", "120"))
HEALTHCHECK_TIMEOUT: float = 3.0
LLM_DISABLED: bool = os.getenv("LLM_DISABLED", "0") == "1"

# Impacto mínimo (em log-odds) para um fator ser citado ao paciente. Abaixo
# disso a contribuição é ruído do modelo e confundiria a leitura.
MIN_IMPACT: float = 0.05
MAX_FACTORS: int = 3

LLM_SEED: int = 42
LLM_TEMPERATURE: float = 0.3

# Termos técnicos que não podem chegar ao paciente.
FORBIDDEN_TERMS: tuple[str, ...] = (
    "shap", "log-odds", "logodds", "logit", "coeficiente", "amostra",
    "regressão logística", "modelo estatístico", "feature",
)
FORBIDDEN_RE = re.compile("|".join(re.escape(t) for t in FORBIDDEN_TERMS), re.IGNORECASE)

Source = Literal["llm", "fallback"]
