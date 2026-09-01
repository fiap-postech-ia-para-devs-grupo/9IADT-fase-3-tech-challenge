from typing import Any, NotRequired, TypedDict


class HospitalAssistantState(TypedDict):
    """
    Estado compartilhado do LangGraph.

    Apenas a pergunta é obrigatória na entrada.
    Os demais campos podem ser preenchidos progressivamente
    pelos nós do grafo.

    O estado mantém interfaces estáveis para:
    - Guardrails
    - RAG do Vinicius
    - LLM do Marcelo
    - Auditoria
    - futura fila de validação humana
    """

    # =========================================================
    # ENTRADA
    # =========================================================

    pergunta: str

    # =========================================================
    # IDENTIFICAÇÃO / CONTEXTO
    # =========================================================

    paciente_id: NotRequired[str | None]
    paciente_idade: NotRequired[int | None]

    historico: NotRequired[list[dict[str, str]]]

    # =========================================================
    # TRIAGEM
    # =========================================================

    categoria_triagem: NotRequired[str]

    sinais_alarme_detectados: NotRequired[list[str]]

    # =========================================================
    # RAG
    # =========================================================

    documentos_retornados: NotRequired[list[dict[str, Any]]]

    # =========================================================
    # RESPOSTA
    # =========================================================

    resposta_bruta: NotRequired[str]

    resposta_final: NotRequired[str]

    # =========================================================
    # EXPLAINABILITY
    # =========================================================

    fontes_citadas: NotRequired[list[str]]

    # =========================================================
    # SEGURANÇA
    # =========================================================

    bloqueado_por_seguranca: NotRequired[bool]

    motivo_bloqueio: NotRequired[str | None]

    requer_validacao_humana: NotRequired[bool]

    validado_por_humano: NotRequired[bool]

    # =========================================================
    # TRACING
    # =========================================================

    passos_processamento: NotRequired[list[str]]
