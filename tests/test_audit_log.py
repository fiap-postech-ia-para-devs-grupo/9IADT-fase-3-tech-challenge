import json
import os

from hospital_assistant.safety.audit_log import (
    ClinicalAuditLogger,
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
        "categoria_triagem": "ginecologia",
        "pergunta": "Tenho cólicas.",
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

    assert registro["categoria_triagem"] == "GINECOLOGIA"

    assert registro["fontes_citadas"] == ["Fonte de teste"]


def test_identifica_pendencia_medica():
    limpar_log()

    state = {
        "paciente_id": "TESTE-002",
        "categoria_triagem": "ginecologia",
        "pergunta": ("Qual remédio devo tomar?"),
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
        "categoria_triagem": "ginecologia",
        "pergunta": ("Pergunta validada."),
        "resposta_final": ("Resposta validada."),
        "requer_validacao_humana": True,
        "validado_por_humano": True,
        "fontes_citadas": [],
    }

    ClinicalAuditLogger.registrar_evento(state)

    pendencias = ClinicalAuditLogger.ler_pendencias()

    assert pendencias == []
