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
    """Fake audit history, standing in for the real SQLite table.

    Spans multiple patients, dates and statuses so Tela 3's status/paciente/data
    filters (Bloco 1, Pessoa D) have something to actually filter.
    """
    return [
        {
            "id": 1,
            "timestamp": "2026-08-20T09:00:00",
            "pergunta": "[MOCK] Qual o protocolo para dor torácica aguda?",
            "paciente_id": "1",
            "fontes_rag": [{"source": "mock_doc_1.md", "score": 0.83}],
            "resposta_llm": "[MOCK LLM] Sugestão de exemplo.",
            "flags_seguranca": [],
            "status": "pendente",
            "aprovador": None,
            "timestamp_aprovacao": None,
        },
        {
            "id": 2,
            "timestamp": "2026-08-21T10:30:00",
            "pergunta": "[MOCK] Paciente com histórico de hipertensão, ajustar dosagem?",
            "paciente_id": "2",
            "fontes_rag": [{"source": "mock_doc_2.md", "score": 0.77}],
            "resposta_llm": "[MOCK LLM] Sugestão de exemplo.",
            "flags_seguranca": [],
            "status": "aprovado",
            "aprovador": "Dr. Souza",
            "timestamp_aprovacao": "2026-08-21T11:00:00",
        },
        {
            "id": 3,
            "timestamp": "2026-08-21T14:15:00",
            "pergunta": "[MOCK] Interação medicamentosa suspeita?",
            "paciente_id": "1",
            "fontes_rag": [{"source": "mock_doc_3.md", "score": 0.65}],
            "resposta_llm": "[MOCK LLM] Sugestão de exemplo.",
            "flags_seguranca": ["interacao_medicamentosa"],
            "status": "rejeitado",
            "aprovador": "Dra. Lima",
            "timestamp_aprovacao": "2026-08-21T14:45:00",
        },
        {
            "id": 4,
            "timestamp": "2026-08-22T08:00:00",
            "pergunta": "[MOCK] Protocolo para febre pós-operatória?",
            "paciente_id": "3",
            "fontes_rag": [{"source": "mock_doc_4.md", "score": 0.71}],
            "resposta_llm": "[MOCK LLM] Sugestão de exemplo.",
            "flags_seguranca": [],
            "status": "pendente",
            "aprovador": None,
            "timestamp_aprovacao": None,
        },
        {
            "id": 5,
            "timestamp": "2026-08-23T16:20:00",
            "pergunta": "[MOCK] Reavaliação de exames pendentes?",
            "paciente_id": "2",
            "fontes_rag": [{"source": "mock_doc_5.md", "score": 0.88}],
            "resposta_llm": "[MOCK LLM] Sugestão de exemplo.",
            "flags_seguranca": [],
            "status": "aprovado",
            "aprovador": "Dr. Souza",
            "timestamp_aprovacao": "2026-08-23T16:50:00",
        },
    ]


def filter_audit_rows(
    rows: list[AuditRow],
    status: str = "todos",
    paciente_id: str = "todos",
    data: str = "todas",
) -> list[AuditRow]:
    """Apply Tela 3's status/paciente/data filters to a list of audit rows.

    `data` matches against the date portion (YYYY-MM-DD) of `timestamp`.
    """
    if status != "todos":
        rows = [r for r in rows if r["status"] == status]
    if paciente_id != "todos":
        rows = [r for r in rows if r["paciente_id"] == paciente_id]
    if data != "todas":
        rows = [r for r in rows if r["timestamp"].startswith(data)]
    return rows


def log_interaction(row: AuditRow) -> None:
    """TODO(Bloco 3 — Pessoa C): persist to SQLite per ESTRATEGIA.md §6."""
    raise NotImplementedError("Audit persistence not implemented yet.")
