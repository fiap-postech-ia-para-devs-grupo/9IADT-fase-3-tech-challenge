"""LangGraph node bodies. Placeholder for Bloco 2 (Pessoa C).

Each node currently just calls the mock/stub from its owning module (db,
rag, llm, safety) so `flow.py` runs end-to-end today. Replace each body with
real logic per ESTRATEGIA.md §5 as the underlying modules land — the node
names and state shape should stay the same so `flow.py` doesn't need to change.
"""

from __future__ import annotations

from hospital_assistant.db.patient_tools import get_pending_exams
from hospital_assistant.graph.state import AssistantState
from hospital_assistant.llm.model_loader import load_llm
from hospital_assistant.rag.retriever import retrieve
from hospital_assistant.safety.guardrails import check_guardrails


def receber_paciente(state: AssistantState) -> AssistantState:
    return state


def verificar_exames_pendentes(state: AssistantState) -> AssistantState:
    paciente_id = state.get("paciente_id")
    exames = get_pending_exams(paciente_id) if paciente_id else []
    return {**state, "exames_pendentes": exames}


def consultar_protocolo(state: AssistantState) -> AssistantState:
    chunks = retrieve(state["pergunta"])
    return {**state, "contexto_rag": [dict(c) for c in chunks]}


def gerar_sugestao_llm(state: AssistantState) -> AssistantState:
    llm = load_llm()
    sugestao = llm.generate(state["pergunta"])
    return {**state, "sugestao_llm": sugestao}


def validar_seguranca(state: AssistantState) -> AssistantState:
    flags = check_guardrails(state["sugestao_llm"])
    return {**state, "flags_seguranca": flags}


def emitir_alerta_se_necessario(state: AssistantState) -> AssistantState:
    alerta = "Exame crítico pendente" if state["exames_pendentes"] else None
    return {**state, "alerta": alerta}


def log_auditoria(state: AssistantState) -> AssistantState:
    # TODO(Bloco 3 — Pessoa C): persist via safety.audit_log.log_interaction once implemented.
    return {**state, "status": "pendente"}
