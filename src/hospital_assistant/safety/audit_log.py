"""Audit trail for every graph run. Placeholder for Bloco 3 (Pessoa C).

Replace with the real SQLite-backed audit log described in ESTRATEGIA.md §6:
table `auditoria` (id, timestamp, pergunta, paciente_id, fontes_rag JSON,
resposta_llm, flags_seguranca JSON, status, aprovador, timestamp_aprovacao),
persisted, and queried from Tela 3. Until then, Tela 3 (Pessoa D, Bloco 1)
reads `mock_audit_rows()` below.
"""

from __future__ import annotations

from typing import TypedDict


class AuditRow(TypedDict):
    id: int
    timestamp: str
    pergunta: str
    paciente_id: str | None
    fontes_rag: list[dict]
    resposta_llm: str
    flags_seguranca: list[str]
    status: str
    aprovador: str | None
    timestamp_aprovacao: str | None


def mock_audit_rows() -> list[AuditRow]:
    """Fake audit history, standing in for the real SQLite table."""
    return [
        {
            "id": 1,
            "timestamp": "2026-08-24T09:00:00",
            "pergunta": "[MOCK] Qual o protocolo para dor torácica aguda?",
            "paciente_id": "1",
            "fontes_rag": [{"source": "mock_doc_1.md", "score": 0.83}],
            "resposta_llm": "[MOCK LLM] Sugestão de exemplo.",
            "flags_seguranca": [],
            "status": "pendente",
            "aprovador": None,
            "timestamp_aprovacao": None,
        }
    ]


def log_interaction(row: AuditRow) -> None:
    """TODO(Bloco 3 — Pessoa C): persist to SQLite per ESTRATEGIA.md §6."""
    raise NotImplementedError("Audit persistence not implemented yet.")
