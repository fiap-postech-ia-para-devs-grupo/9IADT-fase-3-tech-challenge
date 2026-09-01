from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from hospital_assistant.nodes import ClinicalNodes
from hospital_assistant.state import HospitalAssistantState


def build_hospital_graph():
    """
    Constrói o grafo principal do assistente.

    O grafo é independente das implementações específicas
    do RAG do Vinicius e da LLM do Marcelo.

    """

    nodes = ClinicalNodes()

    workflow = StateGraph(HospitalAssistantState)

    # ========================================================
    # NÓS
    # ========================================================

    workflow.add_node(
        "analisador_router",
        nodes.analisar_entrada_router,
    )

    workflow.add_node(
        "triagem_ginecologia",
        nodes.triar_ginecologia,
    )

    workflow.add_node(
        "triagem_obstetricia",
        nodes.triar_obstetricia,
    )

    workflow.add_node(
        "acolhimento_violencia",
        nodes.acolher_violencia,
    )

    workflow.add_node(
        "validar_seguranca",
        nodes.validar_seguranca,
    )

    workflow.add_node(
        "revisao_humana",
        nodes.revisao_humana,
    )

    workflow.add_node(
        "auditoria_final",
        nodes.registrar_auditoria_final,
    )

    # ========================================================
    # ENTRADA
    # ========================================================

    workflow.set_entry_point("analisador_router")

    # ========================================================
    # ROUTER
    # ========================================================

    def rotear_entrada(
        state: HospitalAssistantState,
    ) -> str:

        if state.get(
            "bloqueado_por_seguranca",
            False,
        ):
            return "validar_seguranca"

        categoria = state.get(
            "categoria_triagem",
            "geral",
        )

        if categoria == "ginecologia":
            return "triagem_ginecologia"

        if categoria == "obstetricia":
            return "triagem_obstetricia"

        if categoria == "violencia_domestica":
            return "acolhimento_violencia"

        return "validar_seguranca"

    workflow.add_conditional_edges(
        "analisador_router",
        rotear_entrada,
        {
            "triagem_ginecologia": ("triagem_ginecologia"),
            "triagem_obstetricia": ("triagem_obstetricia"),
            "acolhimento_violencia": ("acolhimento_violencia"),
            "validar_seguranca": ("validar_seguranca"),
        },
    )

    # ========================================================
    # TRIAGEM → SEGURANÇA
    # ========================================================

    workflow.add_edge(
        "triagem_ginecologia",
        "validar_seguranca",
    )

    workflow.add_edge(
        "triagem_obstetricia",
        "validar_seguranca",
    )

    workflow.add_edge(
        "acolhimento_violencia",
        "validar_seguranca",
    )

    # ========================================================
    # SAÍDA
    # ========================================================

    def rotear_saida(
        state: HospitalAssistantState,
    ) -> str:
        """
        Se a resposta exigir validação humana, NÃO aprova
        automaticamente.

        A pendência é registrada na auditoria para futura
        integração com a fila do Renato.
        """

        requer_validacao = state.get(
            "requer_validacao_humana",
            False,
        )

        validado = state.get(
            "validado_por_humano",
            False,
        )

        if requer_validacao and not validado:
            return "auditoria_final"

        return "auditoria_final"

    workflow.add_conditional_edges(
        "validar_seguranca",
        rotear_saida,
        {
            "auditoria_final": "auditoria_final",
        },
    )

    # ========================================================
    # REVISÃO HUMANA
    # ========================================================
    #
    # Mantido como ponto de integração futuro.
    #
    # Não é chamado automaticamente pelo grafo.
    #
    # O Renato poderá futuramente utilizar este nó ou uma
    # estratégia de interrupt/resume para efetivar a validação.
    # ========================================================

    workflow.add_edge(
        "revisao_humana",
        "auditoria_final",
    )

    # ========================================================
    # AUDITORIA → FIM
    # ========================================================

    workflow.add_edge(
        "auditoria_final",
        END,
    )

    # ========================================================
    # CHECKPOINTER
    # ========================================================

    checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)
