"""Comunicação com o servidor Ollama: checagem rápida e geração do texto."""

from __future__ import annotations

import requests

from src.llm.config import (
    HEALTHCHECK_TIMEOUT,
    LLM_SEED,
    LLM_TEMPERATURE,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
)


class LLMUnavailableError(RuntimeError):
    """O servidor da LLM não respondeu ou não tem o modelo configurado."""


def check_ollama(host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL) -> None:
    """Confere rapidamente se o Ollama está no ar e tem o modelo baixado.

    Evita esperar o timeout longo de geração quando o servidor está desligado.

    Raises:
        LLMUnavailableError: Se o servidor não responder ou o modelo não existir.
    """
    try:
        resp = requests.get(f"{host}/api/tags", timeout=HEALTHCHECK_TIMEOUT)
        resp.raise_for_status()
        names = {m.get("name", "") for m in resp.json().get("models", [])}
    except (requests.RequestException, ValueError) as exc:
        raise LLMUnavailableError(f"Ollama não respondeu em {host}: {exc}") from exc

    wanted = model if ":" in model else f"{model}:latest"
    if wanted not in names:
        raise LLMUnavailableError(
            f"Modelo '{model}' não encontrado no Ollama. Rode: ollama pull {model}"
        )


def call_ollama(
    system: str,
    user: str,
    host: str = OLLAMA_HOST,
    model: str = OLLAMA_MODEL,
    timeout: float = OLLAMA_TIMEOUT,
) -> str:
    """Envia o prompt ao Ollama (``/api/chat``) e devolve o texto gerado.

    Raises:
        LLMUnavailableError: Em erro de rede, timeout ou resposta vazia.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": LLM_TEMPERATURE, "seed": LLM_SEED},
    }
    try:
        resp = requests.post(f"{host}/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
        text = resp.json().get("message", {}).get("content", "")
    except (requests.RequestException, ValueError) as exc:
        raise LLMUnavailableError(f"Falha na geração pelo Ollama: {exc}") from exc

    text = text.strip()
    if not text:
        raise LLMUnavailableError("O Ollama devolveu uma resposta vazia.")
    return text
