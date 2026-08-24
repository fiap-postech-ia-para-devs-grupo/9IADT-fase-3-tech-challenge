"""Populates data/patients_mock.db with synthetic patients. Placeholder for Bloco 1 (Pessoa B).

Replace with the real seeding described in ESTRATEGIA.md §4 once schema.sql
has real columns — this version just proves the schema loads and can be
queried, using one throwaway row per table.
"""

from __future__ import annotations

import sqlite3

from hospital_assistant.paths import PATIENTS_DB, PROJECT_ROOT

SCHEMA_PATH = PROJECT_ROOT / "src" / "hospital_assistant" / "db" / "schema.sql"


def seed(db_path=PATIENTS_DB) -> None:
    """TODO(Bloco 1 — Pessoa B): expand with real synthetic patients per ESTRATEGIA.md §4."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA_PATH.read_text())
        conn.execute("INSERT INTO pacientes (nome) VALUES ('Paciente Mock')")
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    seed()
