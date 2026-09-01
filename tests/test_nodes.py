from hospital_assistant.nodes import (
    ClinicalNodes,
    FallbackRAGProvider,
    MockLLM,
)


def test_mock_llm_tem_interface_invoke():
    llm = MockLLM()

    resposta = llm.invoke("Teste")

    assert hasattr(
        resposta,
        "content",
    )

    assert isinstance(
        resposta.content,
        str,
    )


def test_fallback_rag_funciona():
    rag = FallbackRAGProvider()

    documentos = rag.buscar(
        "Tenho dúvidas sobre menstruação.",
        "ginecologia",
    )

    assert len(documentos) >= 1

    assert "page_content" in documentos[0]

    assert "metadata" in documentos[0]


def test_nodes_inicializa():
    nodes = ClinicalNodes()

    assert nodes.guardrails is not None

    assert nodes.llm is not None

    assert nodes.rag is not None


def test_router_detecta_ginecologia():
    nodes = ClinicalNodes()

    state = {"pergunta": ("Tenho muita cólica durante a menstruação.")}

    resultado = nodes.analisar_entrada_router(state)

    assert resultado["categoria_triagem"] == "ginecologia"

    assert resultado["bloqueado_por_seguranca"] is False


def test_router_detecta_obstetricia():
    nodes = ClinicalNodes()

    state = {"pergunta": ("Estou grávida e quero saber sobre pré-natal.")}

    resultado = nodes.analisar_entrada_router(state)

    assert resultado["categoria_triagem"] == "obstetricia"


def test_router_detecta_violencia():
    nodes = ClinicalNodes()

    state = {"pergunta": ("Meu parceiro me bateu.")}

    resultado = nodes.analisar_entrada_router(state)

    assert resultado["categoria_triagem"] == "violencia_domestica"

    assert resultado["bloqueado_por_seguranca"] is False


def test_router_prioriza_emergencia():
    nodes = ClinicalNodes()

    state = {"pergunta": ("Estou com hemorragia.")}

    resultado = nodes.analisar_entrada_router(state)

    assert "emergencia_clinica" in resultado["sinais_alarme_detectados"]

    assert resultado["bloqueado_por_seguranca"] is True

    assert resultado["motivo_bloqueio"] is not None


def test_ginecologia_usa_fallback_rag():
    nodes = ClinicalNodes()

    # O teste aceita tanto o RAG real quanto fallback.
    resultado = nodes.triar_ginecologia({"pergunta": ("Tenho cólica menstrual.")})

    assert resultado["resposta_bruta"]

    assert len(resultado["fontes_citadas"]) >= 1

    assert len(resultado["documentos_retornados"]) >= 1
