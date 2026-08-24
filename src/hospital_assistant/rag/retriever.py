"""Top-k retrieval with source + score. Placeholder for Bloco 2 (Pessoa B).

Replace `retrieve` with the real Chroma-backed retriever described in
ESTRATEGIA.md §4 (top-k=3, returns text + origin metadata + similarity
score — used both to build the LLM context and for explainability on Tela 2).
"""

from __future__ import annotations

from typing import TypedDict


class RetrievedChunk(TypedDict):
    text: str
    source: str
    score: float


def mock_chunks(query: str) -> list[RetrievedChunk]:
    """Fake retrieval results, standing in for the real Chroma retriever."""
    return [
        {
            "text": f"[MOCK] Protocolo relevante para: {query[:60]!r}",
            "source": "protocolos_sinteticos/mock_doc_1.md",
            "score": 0.83,
        }
    ]


def retrieve(query: str, k: int = 3) -> list[RetrievedChunk]:
    """TODO(Bloco 2 — Pessoa B): implement per ESTRATEGIA.md §4 (Chroma, k=3)."""
    return mock_chunks(query)[:k]
