"""LangGraph node bodies for the real exames/protocolo/alerta flow, per
ESTRATEGIA.md §5.

Each node calls into the real db/rag/llm/safety modules — patient_tools
(SQLite), retriever (Chroma), model_loader (base LLM + adapter) and
ClinicalGuardrails/ClinicalAuditLogger (safety/audit architecture). Node
names and state shape match `flow.py`'s `_NODE_ORDER`, which is also
depended on directly by tests/test_integracao_rag_patient_tools.py.
"""

from __future__ import annotations

from hospital_assistant.db.patient_tools import get_pending_exams
from hospital_assistant.graph.state import AssistantState
from hospital_assistant.llm.model_loader import load_llm
from hospital_assistant.rag.retriever import retrieve
from hospital_assistant.safety.audit_log import ClinicalAuditLogger
from hospital_assistant.safety.guardrails import ClinicalGuardrails

_guardrails = ClinicalGuardrails()


def _add_flag(flags: list[str], flag: str) -> list[str]:
    if flag in flags:
        return flags
    return [*flags, flag]


def receber_paciente(state: AssistantState) -> AssistantState:
    _valido, _mensagem, sinais = _guardrails.validar_input(state)
    if not sinais:
        return state

    flags = list(state.get("flags_seguranca", []))
    for sinal in sinais:
        flags = _add_flag(flags, sinal)
    return {**state, "flags_seguranca": flags}


def verificar_exames_pendentes(state: AssistantState) -> AssistantState:
    paciente_id = state.get("paciente_id")
    exames = [dict(e) for e in get_pending_exams(paciente_id)] if paciente_id else []
    return {**state, "exames_pendentes": exames}


def consultar_protocolo(state: AssistantState) -> AssistantState:
    chunks = retrieve(state["pergunta"], k=3)
    return {**state, "contexto_rag": [dict(c) for c in chunks]}


def gerar_sugestao_llm(state: AssistantState) -> AssistantState:
    llm = load_llm()
    sugestao = llm.generate(state["pergunta"])
    return {**state, "sugestao_llm": sugestao}


def validar_seguranca(state: AssistantState) -> AssistantState:
    resposta_validada, requer_validacao_humana = _guardrails.validar_output(state, state["sugestao_llm"])

    flags = list(state.get("flags_seguranca", []))
    if requer_validacao_humana:
        flags = _add_flag(flags, "requer_validacao_humana")

    return {**state, "sugestao_llm": resposta_validada, "flags_seguranca": flags}


def emitir_alerta_se_necessario(state: AssistantState) -> AssistantState:
    alertas: list[str] = []

    if state.get("exames_pendentes"):
        alertas.append("Exame crítico pendente")

    if "emergencia_clinica" in state.get("flags_seguranca", []):
        alertas.append("Sinal de emergência clínica detectado — encaminhar para atendimento presencial imediato")

    alerta = " | ".join(alertas) if alertas else None
    return {**state, "alerta": alerta}


def log_auditoria(state: AssistantState) -> AssistantState:
    flags = state.get("flags_seguranca", [])
    ClinicalAuditLogger.registrar_evento(
        {
            "paciente_id": state.get("paciente_id"),
            "pergunta": state.get("pergunta", ""),
            "resposta_final": state.get("sugestao_llm", ""),
            "sinais_alarme_detectados": flags,
            "bloqueado_por_seguranca": "emergencia_clinica" in flags,
            "motivo_bloqueio": state.get("alerta"),
            "requer_validacao_humana": "requer_validacao_humana" in flags,
            "fontes_citadas": [c["source"] for c in state.get("contexto_rag", []) if c.get("source")],
            "documentos_retornados": state.get("contexto_rag", []),
        }
    )
    return {**state, "status": "pendente"}
