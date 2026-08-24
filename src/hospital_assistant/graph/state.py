"""Shared LangGraph state, per ESTRATEGIA.md §5 (already fully specified there)."""

from __future__ import annotations

from typing import TypedDict


class AssistantState(TypedDict):
    paciente_id: str | None
    pergunta: str
    exames_pendentes: list[dict]
    contexto_rag: list[dict]
    sugestao_llm: str
    flags_seguranca: list[str]
    alerta: str | None
    status: str  # "pendente" | "aprovado" | "rejeitado"
