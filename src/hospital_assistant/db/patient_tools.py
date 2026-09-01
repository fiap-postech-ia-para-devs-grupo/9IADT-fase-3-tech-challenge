"""Parameterized patient-data tools exposed to LangChain, per ESTRATEGIA.md §4.

Deliberately NOT a free-text SQL agent (ESTRATEGIA.md §1/§4) — narrow,
parameterized functions only, to eliminate the risk of a hallucinated query
over clinical data.
"""

from __future__ import annotations

import sqlite3
from typing import TypedDict

from hospital_assistant.paths import PATIENTS_DB


class ExamRecord(TypedDict):
    id: int
    tipo: str
    status: str
    data_solicitacao: str
    data_resultado: str | None
    resultado: str | None


class MedicationRecord(TypedDict):
    id: int
    nome: str
    dosagem: str
    frequencia: str
    data_inicio: str


class AlertRecord(TypedDict):
    id: int
    descricao: str
    severidade: str
    data: str
    resolvido: bool


class PatientHistory(TypedDict):
    paciente_id: str
    nome: str
    data_nascimento: str
    prontuario: str
    exames: list[ExamRecord]
    medicacoes: list[MedicationRecord]
    alertas: list[AlertRecord]


class PatientSummary(TypedDict):
    id: str
    nome: str
    prontuario: str


_EXAM_COLUMNS = "id, tipo, status, data_solicitacao, data_resultado, resultado"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(PATIENTS_DB)
    conn.row_factory = sqlite3.Row
    return conn


def list_patients() -> list[PatientSummary]:
    """Pacientes disponíveis para o seletor da Tela 1 — não expõe dados clínicos."""
    conn = _connect()
    try:
        rows = conn.execute("SELECT id, nome, prontuario FROM pacientes ORDER BY nome").fetchall()
        return [{"id": str(row["id"]), "nome": row["nome"], "prontuario": row["prontuario"]} for row in rows]
    finally:
        conn.close()


def get_pending_exams(paciente_id: str) -> list[ExamRecord]:
    conn = _connect()
    try:
        rows = conn.execute(
            f"SELECT {_EXAM_COLUMNS} FROM exames WHERE paciente_id = ? AND status = 'pendente' "
            "ORDER BY data_solicitacao",
            (int(paciente_id),),
        ).fetchall()
        return [dict(row) for row in rows]  # type: ignore[misc]
    finally:
        conn.close()


def get_patient_history(paciente_id: str) -> PatientHistory:
    pid = int(paciente_id)
    conn = _connect()
    try:
        paciente = conn.execute(
            "SELECT nome, data_nascimento, prontuario FROM pacientes WHERE id = ?",
            (pid,),
        ).fetchone()
        if paciente is None:
            raise ValueError(f"paciente {paciente_id!r} não encontrado")

        exames = conn.execute(
            f"SELECT {_EXAM_COLUMNS} FROM exames WHERE paciente_id = ? ORDER BY data_solicitacao",
            (pid,),
        ).fetchall()
        medicacoes = conn.execute(
            "SELECT id, nome, dosagem, frequencia, data_inicio FROM medicacoes WHERE paciente_id = ?",
            (pid,),
        ).fetchall()
        alertas = conn.execute(
            "SELECT id, descricao, severidade, data, resolvido FROM alertas WHERE paciente_id = ?",
            (pid,),
        ).fetchall()

        return {
            "paciente_id": paciente_id,
            "nome": paciente["nome"],
            "data_nascimento": paciente["data_nascimento"],
            "prontuario": paciente["prontuario"],
            "exames": [dict(row) for row in exames],  # type: ignore[misc]
            "medicacoes": [dict(row) for row in medicacoes],  # type: ignore[misc]
            "alertas": [
                {**dict(row), "resolvido": bool(row["resolvido"])} for row in alertas
            ],  # type: ignore[misc]
        }
    finally:
        conn.close()
