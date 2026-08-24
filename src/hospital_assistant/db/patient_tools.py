"""Parameterized patient-data tools exposed to LangChain. Placeholder for Bloco 1 (Pessoa B).

Deliberately NOT a free-text SQL agent (ESTRATEGIA.md §4/§1) — keep these as
narrow, parameterized functions to eliminate the risk of a hallucinated query
over clinical data. Replace the bodies with real SQLite lookups against
data/patients_mock.db once schema.sql has real columns.
"""

from __future__ import annotations

from typing import TypedDict


class ExamRecord(TypedDict):
    id: int
    status: str


class PatientHistory(TypedDict):
    paciente_id: str
    nome: str
    exames: list[ExamRecord]


def get_pending_exams(paciente_id: str) -> list[ExamRecord]:
    """TODO(Bloco 1 — Pessoa B): query data/patients_mock.db for real pending exams."""
    return [{"id": 1, "status": "pendente"}]


def get_patient_history(paciente_id: str) -> PatientHistory:
    """TODO(Bloco 1 — Pessoa B): query data/patients_mock.db for real patient history."""
    return {
        "paciente_id": paciente_id,
        "nome": "Paciente Mock",
        "exames": get_pending_exams(paciente_id),
    }
