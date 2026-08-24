"""Smoke tests for the base scaffolding — not the per-ticket test suites.

Guardrail/retriever/patient_tools test coverage is Bloco 2 (Pessoa E); this
file only checks the skeleton imports and runs end-to-end against mocks.
"""

from __future__ import annotations

from hospital_assistant.db.patient_tools import get_patient_history
from hospital_assistant.graph.flow import run
from hospital_assistant.rag.retriever import retrieve
from hospital_assistant.safety.audit_log import mock_audit_rows


def test_graph_runs_end_to_end_against_mocks():
    result = run("Qual o protocolo para dor torácica aguda?", paciente_id="1")
    assert result["status"] == "pendente"
    assert result["sugestao_llm"]


def test_retriever_returns_scored_chunks():
    chunks = retrieve("dor torácica")
    assert chunks
    assert 0.0 <= chunks[0]["score"] <= 1.0


def test_patient_tools_returns_history():
    history = get_patient_history("1")
    assert history["paciente_id"] == "1"


def test_mock_audit_rows_shape():
    rows = mock_audit_rows()
    assert rows[0]["status"] == "pendente"
