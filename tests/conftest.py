"""Bootstrap de dados compartilhado entre testes.

Uma vez que patient_tools e retriever passaram a consultar dados reais
(SQLite / Chroma) em vez de mocks, `pytest` só passa numa checkout limpa se
esses dados existirem — sem isso, `patient_tools` falha (tabela inexistente)
e `retriever` volta lista vazia silenciosamente. Semeia/indexa uma vez por
sessão, só se ainda não existir, para não repetir o custo em toda run local.
"""

from __future__ import annotations

import pytest

from hospital_assistant.db.seed_mock_data import seed
from hospital_assistant.paths import CHROMA_DIR, PATIENTS_DB
from hospital_assistant.rag.ingest import ingest


def _chroma_populated() -> bool:
    return (CHROMA_DIR / "chroma.sqlite3").exists()


@pytest.fixture(scope="session", autouse=True)
def _bootstrap_data() -> None:
    if not PATIENTS_DB.exists():
        seed()
    if not _chroma_populated():
        ingest()
