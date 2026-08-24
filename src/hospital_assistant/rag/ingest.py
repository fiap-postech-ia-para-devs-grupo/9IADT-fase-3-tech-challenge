"""Chroma indexing. Placeholder for Bloco 2 (Pessoa B).

Replace with the real ingestion pipeline described in ESTRATEGIA.md §4:
embed with sentence-transformers/all-MiniLM-L6-v2, persist a Chroma vector
store at data/chroma/, indexing synthetic protocols + a PubMedQA/MedQuAD
sample.
"""

from __future__ import annotations

from hospital_assistant.paths import CHROMA_DIR


def ingest() -> None:
    """TODO(Bloco 2 — Pessoa B): implement per ESTRATEGIA.md §4."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError("Chroma ingestion not implemented yet.")
