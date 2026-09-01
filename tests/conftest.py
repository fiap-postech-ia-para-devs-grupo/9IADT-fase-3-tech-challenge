"""Bootstrap de dados compartilhado entre testes.

Uma vez que patient_tools e retriever passaram a consultar dados reais
(SQLite / Chroma) em vez de mocks, `pytest` só passa numa checkout limpa se
esses dados existirem — sem isso, `patient_tools` falha (tabela inexistente)
e `retriever` volta lista vazia silenciosamente. Semeia/indexa uma vez por
sessão, só se ainda não existir, para não repetir o custo em toda run local.
"""

from __future__ import annotations

import os

import pytest

from hospital_assistant.db.seed_mock_data import seed
from hospital_assistant.paths import CHROMA_DIR, PATIENTS_DB
from hospital_assistant.rag.ingest import ingest
from hospital_assistant.safety.audit_log import ClinicalAuditLogger


def _chroma_populated() -> bool:
    return (CHROMA_DIR / "chroma.sqlite3").exists()


@pytest.fixture(scope="session", autouse=True)
def _bootstrap_data() -> None:
    if not PATIENTS_DB.exists():
        seed()
    if not _chroma_populated():
        ingest()


@pytest.fixture
def limpar_auditoria() -> None:
    """Reset the real `clinical_audit.jsonl` trail before a test that seeds it directly.

    Shared by every test that writes to the audit log (directly via
    `ClinicalAuditLogger.registrar_evento`, or indirectly by running the real
    graph/app) so each test starts from a clean file instead of whatever the
    previous test — or a previous local run — left behind.
    """
    caminho = ClinicalAuditLogger.LOG_ESTRUTURADO_PATH
    if os.path.exists(caminho):
        os.remove(caminho)
