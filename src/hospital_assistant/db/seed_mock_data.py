"""Popula data/patients_mock.db com pacientes sintéticos, per ESTRATEGIA.md §4."""

from __future__ import annotations

import sqlite3

from hospital_assistant.paths import PATIENTS_DB, PROJECT_ROOT

SCHEMA_PATH = PROJECT_ROOT / "src" / "hospital_assistant" / "db" / "schema.sql"

_PACIENTES = [
    # nome, data_nascimento, prontuario
    ("Maria Silva Santos", "1985-03-14", "PRONT-0001"),
    ("João Pedro Almeida", "1990-07-22", "PRONT-0002"),
    ("Ana Beatriz Costa", "1978-11-05", "PRONT-0003"),
]

_EXAMES = [
    # paciente_id, tipo, status, data_solicitacao, data_resultado, resultado
    (1, "Hemograma completo", "pendente", "2026-08-20", None, None),
    (1, "Raio-X tórax", "concluido", "2026-08-10", "2026-08-12", "Sem alterações significativas"),
    (2, "Glicemia em jejum", "concluido", "2026-08-15", "2026-08-16", "Dentro da normalidade"),
    (3, "Ressonância magnética - joelho", "pendente", "2026-08-25", None, None),
]

_MEDICACOES = [
    # paciente_id, nome, dosagem, frequencia, data_inicio
    (1, "Losartana", "50mg", "1x ao dia", "2025-01-10"),
    (2, "Metformina", "850mg", "2x ao dia", "2024-06-01"),
    (3, "Ibuprofeno", "400mg", "a cada 8h se dor", "2026-08-20"),
]

_ALERTAS = [
    # paciente_id, descricao, severidade, data, resolvido
    (3, "Exame crítico pendente há mais de 5 dias", "alta", "2026-08-30", 0),
]


def seed(db_path=PATIENTS_DB) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA_PATH.read_text())
        conn.executemany(
            "INSERT INTO pacientes (nome, data_nascimento, prontuario) VALUES (?, ?, ?)",
            _PACIENTES,
        )
        conn.executemany(
            "INSERT INTO exames (paciente_id, tipo, status, data_solicitacao, data_resultado, resultado) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            _EXAMES,
        )
        conn.executemany(
            "INSERT INTO medicacoes (paciente_id, nome, dosagem, frequencia, data_inicio) VALUES (?, ?, ?, ?, ?)",
            _MEDICACOES,
        )
        conn.executemany(
            "INSERT INTO alertas (paciente_id, descricao, severidade, data, resolvido) VALUES (?, ?, ?, ?, ?)",
            _ALERTAS,
        )
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    seed()
