"""Explicação em linguagem simples para o paciente, com LLM local e modo offline.

Recebe o dicionário técnico de ``src.explain`` e devolve um texto em
português claro, com três seções (resumo, fatores e recomendações) e o aviso
de que o sistema é de triagem, não de diagnóstico.

Fluxo (``service``):
    1. Filtra os fatores com impacto relevante (``MIN_IMPACT``) e os traduz em
       frases simples (``facts``). A LLM nunca vê números do SHAP, então não
       tem como repeti-los.
    2. Tenta gerar o texto com o Ollama (``ollama_client``).
    3. Se o Ollama estiver desligado, sem o modelo, lento demais ou devolver
       texto com jargão técnico (``validation``), usa o fallback (``fallback``).
    4. O aviso de isenção é sempre anexado pelo código, nunca depende da LLM.

:func:`generate_patient_explanation` nunca lança exceção: no pior caso,
devolve o texto do fallback.

Configuração por variáveis de ambiente, todas opcionais (``config``):
    - ``OLLAMA_HOST``: endereço do servidor (padrão ``http://127.0.0.1:11434``).
    - ``OLLAMA_MODEL``: modelo usado (padrão ``qwen2.5:3b``).
    - ``OLLAMA_TIMEOUT``: tempo máximo de geração em segundos (padrão 120).
    - ``LLM_DISABLED``: ``1`` para pular a LLM e usar só o fallback.

Uso pelo terminal::

    python -m src.llm
"""

from src.llm.config import (
    FORBIDDEN_TERMS, HEALTHCHECK_TIMEOUT, LLM_DISABLED, LLM_SEED, LLM_TEMPERATURE,
    MAX_FACTORS, MIN_IMPACT, OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_TIMEOUT, Source,
)
from src.llm.facts import PatientFacts, extract_facts
from src.llm.fallback import fallback_explanation
from src.llm.ollama_client import LLMUnavailableError, call_ollama, check_ollama
from src.llm.phrases import RISK_PHRASES
from src.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from src.llm.recommendations import GENERAL_RECOMMENDATIONS, RECOMMENDATIONS
from src.llm.service import generate_patient_explanation, generate_patient_explanation_with_source
from src.llm.texts import (
    CLASS_HEADLINES, DISCLAIMER, GENERIC_TEXT, SECTION_ACTIONS, SECTION_FACTORS,
    SECTION_SUMMARY,
)
from src.llm.validation import validate_llm_text

__all__ = [
    "CLASS_HEADLINES", "DISCLAIMER", "FORBIDDEN_TERMS", "GENERAL_RECOMMENDATIONS",
    "GENERIC_TEXT", "HEALTHCHECK_TIMEOUT", "LLM_DISABLED", "LLM_SEED", "LLM_TEMPERATURE",
    "LLMUnavailableError", "MAX_FACTORS", "MIN_IMPACT", "OLLAMA_HOST", "OLLAMA_MODEL",
    "OLLAMA_TIMEOUT", "RECOMMENDATIONS", "RISK_PHRASES", "SECTION_ACTIONS",
    "SECTION_FACTORS", "SECTION_SUMMARY", "SYSTEM_PROMPT", "PatientFacts", "Source",
    "build_user_prompt", "call_ollama", "check_ollama", "extract_facts",
    "fallback_explanation", "generate_patient_explanation",
    "generate_patient_explanation_with_source", "validate_llm_text",
]
