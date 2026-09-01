"""Integração retriever + patient_tools, per issue #8.

Valida que os dois módulos funcionam em conjunto e que o formato de dados que
produzem é o que os nós do LangGraph (nodes.py, Pessoa C) esperam gravar em
AssistantState — não duplica os testes unitários de cada módulo isoladamente
(Pessoa E, Bloco 2), foca na combinação dos dois numa mesma interação.
"""

from __future__ import annotations

from typing import cast

import pytest

from hospital_assistant.db.patient_tools import get_patient_history, get_pending_exams
from hospital_assistant.graph.nodes import consultar_protocolo, verificar_exames_pendentes
from hospital_assistant.graph.state import AssistantState
from hospital_assistant.rag.retriever import retrieve


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


def test_verificar_exames_pendentes_node_com_paciente_com_pendencia():
    resultado = verificar_exames_pendentes(_state(paciente_id="1"))

    assert resultado["exames_pendentes"]
    exame = resultado["exames_pendentes"][0]
    assert exame["status"] == "pendente"
    assert set(exame) == {"id", "tipo", "status", "data_solicitacao", "data_resultado", "resultado"}


def test_verificar_exames_pendentes_node_paciente_sem_pendencia():
    # paciente 2 no seed só tem exame concluído
    resultado = verificar_exames_pendentes(_state(paciente_id="2"))

    assert resultado["exames_pendentes"] == []


def test_consultar_protocolo_node_retorna_contexto_rag_serializavel():
    resultado = consultar_protocolo(_state(pergunta="sintomas de pneumonia"))

    assert resultado["contexto_rag"]
    for chunk in resultado["contexto_rag"]:
        assert isinstance(chunk, dict)  # já convertido de RetrievedChunk, não TypedDict "vivo"
        assert set(chunk) == {"text", "source", "score"}
        assert isinstance(chunk["score"], float)


def test_get_pending_exams_paciente_inexistente_nao_quebra():
    assert get_pending_exams("999") == []


def test_get_patient_history_paciente_inexistente_levanta_erro_explicito():
    with pytest.raises(ValueError):
        get_patient_history("999")


def test_exames_e_rag_coexistem_na_mesma_interacao():
    """Simula o que verificar_exames_pendentes + consultar_protocolo fazem em
    sequência dentro do grafo, pra garantir que uma chamada não interfere na
    outra (conexões sqlite/Chroma independentes)."""
    paciente_id = "3"
    pergunta = "conduta para crise hipertensiva"

    exames = get_pending_exams(paciente_id)
    chunks = retrieve(pergunta, k=3)

    assert exames  # paciente 3 tem exame pendente no seed
    assert chunks
    assert len(chunks) <= 3
    assert all(0.0 <= c["score"] <= 1.0 for c in chunks)
