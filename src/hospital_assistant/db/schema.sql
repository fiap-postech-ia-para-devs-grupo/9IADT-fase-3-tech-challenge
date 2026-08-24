-- Placeholder schema for Bloco 1 (Pessoa B). ESTRATEGIA.md §4 names the four
-- tables below (pacientes, exames, medicacoes, alertas) but leaves column
-- design open — flesh these out for real before seed_mock_data.py depends on them.
--
-- TODO(Bloco 1 — Pessoa B): design real columns per ESTRATEGIA.md §4/§5.

CREATE TABLE IF NOT EXISTS pacientes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nome TEXT NOT NULL
    -- TODO: mais campos (data_nascimento, prontuario, etc.)
);

CREATE TABLE IF NOT EXISTS exames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_id INTEGER NOT NULL REFERENCES pacientes(id),
    status TEXT NOT NULL
    -- TODO: mais campos (tipo, data, resultado, etc.)
);

CREATE TABLE IF NOT EXISTS medicacoes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_id INTEGER NOT NULL REFERENCES pacientes(id)
    -- TODO: mais campos (nome, dosagem, etc.)
);

CREATE TABLE IF NOT EXISTS alertas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_id INTEGER NOT NULL REFERENCES pacientes(id),
    descricao TEXT NOT NULL
    -- TODO: mais campos (severidade, data, etc.)
);
