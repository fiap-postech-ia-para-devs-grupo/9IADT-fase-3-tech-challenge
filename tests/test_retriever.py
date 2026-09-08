"""Testes unitários do retriever RAG, per issue #19 (Bloco 2 — Pessoa E).

Cobre o módulo isolado (formato do retorno, comportamento de `k`); a
combinação com `patient_tools`/nós do grafo já é coberta por
tests/test_integracao_rag_patient_tools.py.
"""

from __future__ import annotations

from hospital_assistant.rag.retriever import retrieve


def test_retrieve_retorna_chunks_com_texto_fonte_e_score():
    chunks = retrieve("crise hipertensiva", k=3)

    assert chunks
    for chunk in chunks:
        assert set(chunk) == {"text", "source", "score"}
        assert isinstance(chunk["text"], str) and chunk["text"]
        assert isinstance(chunk["source"], str) and chunk["source"] != "desconhecida"
        assert isinstance(chunk["score"], float)


def test_retrieve_traz_fonte_esperada_no_top_k():
    # Query catalogada em docs/retrieval-eval.md como acerto confiável de top-3.
    chunks = retrieve("crise hipertensiva", k=3)

    fontes = [c["source"] for c in chunks]
    assert any("crise_hipertensiva" in fonte for fonte in fontes)


def test_retrieve_respeita_k():
    assert len(retrieve("sepse", k=1)) == 1
    assert len(retrieve("sepse", k=3)) <= 3


def test_retrieve_scores_em_faixa_valida_de_similaridade_de_cosseno():
    chunks = retrieve("qual o prazo de liberação de um exame laboratorial marcado como urgente?", k=3)

    assert chunks
    assert all(0.0 <= c["score"] <= 1.0 for c in chunks)
    # Scores decrescentes: top-1 é o mais similar, por definição de top-k.
    scores = [c["score"] for c in chunks]
    assert scores == sorted(scores, reverse=True)
