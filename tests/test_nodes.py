"""Unit tests for the real graph node bodies (src/hospital_assistant/graph/nodes.py).

Complements tests/test_integracao_rag_patient_tools.py (which focuses on the
db/rag combination) by covering the guardrails/audit wiring each node adds
on top: flags accumulation, alert emission and audit persistence.
"""

from __future__ import annotations

import json
import os
from typing import cast

from hospital_assistant.graph.nodes import (
    emitir_alerta_se_necessario,
    gerar_sugestao_llm,
    log_auditoria,
    receber_paciente,
    validar_seguranca,
)
from hospital_assistant.graph.state import AssistantState
from hospital_assistant.safety.audit_log import ClinicalAuditLogger


def _state(**overrides) -> AssistantState:
    base: AssistantState = {
        "paciente_id": None,
        "pergunta": "",
        "exames_pendentes": [],
        "contexto_rag": [],
        "sugestao_llm": "",
        "flags_seguranca": [],
        "alerta": None,
        "status": "pendente",
    }
    return cast(AssistantState, {**base, **overrides})


def _limpar_auditoria():
    if os.path.exists(ClinicalAuditLogger.LOG_ESTRUTURADO_PATH):
        os.remove(ClinicalAuditLogger.LOG_ESTRUTURADO_PATH)


def test_receber_paciente_sinaliza_emergencia():
    resultado = receber_paciente(_state(pergunta="Paciente com dor torácica intensa."))

    assert "emergencia_clinica" in resultado["flags_seguranca"]


def test_receber_paciente_pergunta_normal_nao_sinaliza():
    resultado = receber_paciente(_state(pergunta="Qual a conduta para pneumonia?"))

    assert resultado["flags_seguranca"] == []


def test_gerar_sugestao_llm_usa_mock_llm():
    resultado = gerar_sugestao_llm(_state(pergunta="conduta para sepse"))

    assert resultado["sugestao_llm"]


def test_validar_seguranca_marca_flag_quando_exige_validacao():
    estado = _state(pergunta="Qual remédio devo prescrever?", sugestao_llm="Considere o uso do medicamento.")

    resultado = validar_seguranca(estado)

    assert "requer_validacao_humana" in resultado["flags_seguranca"]
    assert "não realiza prescrição" in resultado["sugestao_llm"]


def test_validar_seguranca_nao_marca_flag_para_resposta_segura():
    estado = _state(
        pergunta="Quais os sinais de alerta na sepse?",
        sugestao_llm="Procure avaliação presencial imediatamente.",
    )

    resultado = validar_seguranca(estado)

    assert "requer_validacao_humana" not in resultado["flags_seguranca"]


def test_emitir_alerta_por_exame_pendente():
    resultado = emitir_alerta_se_necessario(_state(exames_pendentes=[{"id": 1, "status": "pendente"}]))

    assert resultado["alerta"] is not None
    assert "exame" in resultado["alerta"].lower()


def test_emitir_alerta_por_emergencia_clinica():
    resultado = emitir_alerta_se_necessario(_state(flags_seguranca=["emergencia_clinica"]))

    assert resultado["alerta"] is not None
    assert "emergência" in resultado["alerta"].lower()


def test_emitir_alerta_nenhum_quando_tudo_normal():
    resultado = emitir_alerta_se_necessario(_state())

    assert resultado["alerta"] is None


def test_log_auditoria_persiste_evento_e_marca_pendente():
    _limpar_auditoria()

    estado = _state(
        paciente_id="1",
        pergunta="conduta para dor torácica aguda",
        sugestao_llm="Sugestão de exemplo.",
        contexto_rag=[{"text": "trecho", "source": "protocolos_sinteticos/dor_toracica_aguda.md", "score": 0.9}],
    )

    resultado = log_auditoria(estado)

    assert resultado["status"] == "pendente"
    assert os.path.exists(ClinicalAuditLogger.LOG_ESTRUTURADO_PATH)

    with open(ClinicalAuditLogger.LOG_ESTRUTURADO_PATH, encoding="utf-8") as arquivo:
        registro = json.loads(arquivo.readline())

    assert registro["paciente_id"] == "1"
    assert registro["fontes_citadas"] == ["protocolos_sinteticos/dor_toracica_aguda.md"]
