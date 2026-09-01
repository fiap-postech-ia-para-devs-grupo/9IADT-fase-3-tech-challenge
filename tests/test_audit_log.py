import json
import os

import pytest

from hospital_assistant.safety.audit_log import (
    ClinicalAuditLogger,
    apply_decision,
    filter_audit_rows,
    mock_audit_rows,
)


def limpar_log():
    caminho = ClinicalAuditLogger.LOG_ESTRUTURADO_PATH

    if os.path.exists(caminho):
        os.remove(caminho)


def test_registra_evento_jsonl():
    limpar_log()

    state = {
        "paciente_id": "TESTE-001",
        "paciente_idade": 30,
        "categoria_triagem": "geral",
        "pergunta": "Qual a conduta para dor torácica aguda?",
        "resposta_final": ("Procure avaliação presencial."),
        "sinais_alarme_detectados": [],
        "bloqueado_por_seguranca": False,
        "motivo_bloqueio": None,
        "requer_validacao_humana": False,
        "validado_por_humano": False,
        "fontes_citadas": ["Fonte de teste"],
        "passos_processamento": [
            "Router iniciado",
            "Validação concluída",
        ],
    }

    ClinicalAuditLogger.registrar_evento(state)

    assert os.path.exists(ClinicalAuditLogger.LOG_ESTRUTURADO_PATH)

    with open(
        ClinicalAuditLogger.LOG_ESTRUTURADO_PATH,
        encoding="utf-8",
    ) as arquivo:
        linha = arquivo.readline()

    registro = json.loads(linha)

    assert registro["paciente_id"] == "TESTE-001"

    assert registro["categoria_triagem"] == "GERAL"

    assert registro["fontes_citadas"] == ["Fonte de teste"]


def test_identifica_pendencia_medica():
    limpar_log()

    state = {
        "paciente_id": "TESTE-002",
        "categoria_triagem": "geral",
        "pergunta": ("Qual remédio devo prescrever?"),
        "resposta_final": ("Solicitação encaminhada para validação."),
        "requer_validacao_humana": True,
        "validado_por_humano": False,
        "fontes_citadas": ["Fonte de teste"],
    }

    ClinicalAuditLogger.registrar_evento(state)

    pendencias = ClinicalAuditLogger.ler_pendencias()

    assert len(pendencias) == 1

    assert pendencias[0]["paciente_id"] == "TESTE-002"

    assert pendencias[0]["requer_validacao_humana"] is True

    assert pendencias[0]["validado_por_humano"] is False


def test_evento_validado_nao_fica_pendente():
    limpar_log()

    state = {
        "paciente_id": "TESTE-003",
        "categoria_triagem": "geral",
        "pergunta": ("Pergunta validada."),
        "resposta_final": ("Resposta validada."),
        "requer_validacao_humana": True,
        "validado_por_humano": True,
        "fontes_citadas": [],
    }

    ClinicalAuditLogger.registrar_evento(state)

    pendencias = ClinicalAuditLogger.ler_pendencias()

    assert pendencias == []


# =============================================================
# Tela 3 (Auditoria) — filtros sobre mock_audit_rows/filter_audit_rows.
# Contrato consumido por app.py; não deve quebrar com a reconciliação.
# =============================================================


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


# =============================================================
# Tela 2 (Fila de Validação Humana) — apply_decision, per issue #16.
# =============================================================


def test_apply_decision_aprovado_atualiza_status_e_aprovador():
    rows = mock_audit_rows()
    pendente = next(r for r in rows if r["status"] == "pendente")

    result = apply_decision(rows, pendente["id"], "aprovado", aprovador="Dra. Lima")

    atualizado = next(r for r in result if r["id"] == pendente["id"])
    assert atualizado["status"] == "aprovado"
    assert atualizado["aprovador"] == "Dra. Lima"
    assert atualizado["timestamp_aprovacao"] is not None


def test_apply_decision_rejeitado_atualiza_status():
    rows = mock_audit_rows()
    pendente = next(r for r in rows if r["status"] == "pendente")

    result = apply_decision(rows, pendente["id"], "rejeitado", aprovador="Dr. Souza")

    atualizado = next(r for r in result if r["id"] == pendente["id"])
    assert atualizado["status"] == "rejeitado"
    assert atualizado["aprovador"] == "Dr. Souza"


def test_apply_decision_com_resposta_editada_substitui_resposta_llm():
    rows = mock_audit_rows()
    pendente = next(r for r in rows if r["status"] == "pendente")

    result = apply_decision(
        rows,
        pendente["id"],
        "aprovado",
        aprovador="Dra. Lima",
        resposta_editada="Resposta corrigida pelo médico.",
    )

    atualizado = next(r for r in result if r["id"] == pendente["id"])
    assert atualizado["resposta_llm"] == "Resposta corrigida pelo médico."
    assert atualizado["status"] == "aprovado"


def test_apply_decision_nao_afeta_outras_linhas():
    rows = mock_audit_rows()
    pendente = next(r for r in rows if r["status"] == "pendente")
    outros_ids_antes = {r["id"]: r for r in rows if r["id"] != pendente["id"]}

    result = apply_decision(rows, pendente["id"], "aprovado", aprovador="Dra. Lima")

    for row in result:
        if row["id"] != pendente["id"]:
            assert row == outros_ids_antes[row["id"]]


def test_apply_decision_decisao_invalida_levanta_erro():
    rows = mock_audit_rows()
    pendente = next(r for r in rows if r["status"] == "pendente")

    with pytest.raises(ValueError):
        apply_decision(rows, pendente["id"], "invalida", aprovador="Dra. Lima")
