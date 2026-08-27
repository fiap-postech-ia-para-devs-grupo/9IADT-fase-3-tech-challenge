"""Unit tests for the Tela 3 (Auditoria) status/paciente/data filters.

Bloco 1 — Pessoa D. Exercises `filter_audit_rows` directly against the
mock rows returned by `mock_audit_rows`, so the filter logic in app.py
stays a thin wiring layer over something testable without Streamlit.
"""

from __future__ import annotations

from hospital_assistant.safety.audit_log import filter_audit_rows, mock_audit_rows


def test_no_filters_returns_all_rows():
    rows = mock_audit_rows()
    assert filter_audit_rows(rows) == rows


def test_filter_by_status():
    rows = mock_audit_rows()
    result = filter_audit_rows(rows, status="aprovado")
    assert result
    assert all(r["status"] == "aprovado" for r in result)


def test_filter_by_paciente_id():
    rows = mock_audit_rows()
    result = filter_audit_rows(rows, paciente_id="1")
    assert result
    assert all(r["paciente_id"] == "1" for r in result)


def test_filter_by_data():
    rows = mock_audit_rows()
    target_date = rows[0]["timestamp"][:10]
    result = filter_audit_rows(rows, data=target_date)
    assert result
    assert all(r["timestamp"].startswith(target_date) for r in result)


def test_filters_combine():
    rows = mock_audit_rows()
    target = rows[0]
    assert target["paciente_id"] is not None
    result = filter_audit_rows(
        rows,
        status=target["status"],
        paciente_id=target["paciente_id"],
        data=target["timestamp"][:10],
    )
    assert target in result


def test_filter_with_no_matches_returns_empty():
    rows = mock_audit_rows()
    assert filter_audit_rows(rows, status="nao-existe") == []
