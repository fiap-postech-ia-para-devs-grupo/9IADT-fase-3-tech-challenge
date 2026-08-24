"""Assembles the linear StateGraph per ESTRATEGIA.md §5. Compiled once, reused by app.py."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from hospital_assistant.graph import nodes
from hospital_assistant.graph.state import AssistantState

_NODE_ORDER = (
    "receber_paciente",
    "verificar_exames_pendentes",
    "consultar_protocolo",
    "gerar_sugestao_llm",
    "validar_seguranca",
    "emitir_alerta_se_necessario",
    "log_auditoria",
)


def build_graph():
    graph = StateGraph(AssistantState)
    for name in _NODE_ORDER:
        graph.add_node(name, getattr(nodes, name))

    graph.set_entry_point(_NODE_ORDER[0])
    for a, b in zip(_NODE_ORDER, _NODE_ORDER[1:]):
        graph.add_edge(a, b)
    graph.add_edge(_NODE_ORDER[-1], END)

    return graph.compile()


_compiled = None


def run(pergunta: str, paciente_id: str | None = None) -> AssistantState:
    global _compiled
    if _compiled is None:
        _compiled = build_graph()

    initial: AssistantState = {
        "paciente_id": paciente_id,
        "pergunta": pergunta,
        "exames_pendentes": [],
        "contexto_rag": [],
        "sugestao_llm": "",
        "flags_seguranca": [],
        "alerta": None,
        "status": "pendente",
    }
    return _compiled.invoke(initial)
