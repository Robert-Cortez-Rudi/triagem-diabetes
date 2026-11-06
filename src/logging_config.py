"""Configuração do log de console usada na execução direta dos módulos."""

from __future__ import annotations

import logging


def configure_logging() -> None:
    """Configura o log do console para a execução direta de um módulo."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
