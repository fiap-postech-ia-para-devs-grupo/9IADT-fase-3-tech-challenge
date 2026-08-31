-- Schema de dados estruturados de pacientes, per ESTRATEGIA.md §4.
-- Consultado exclusivamente via funções parametrizadas em patient_tools.py —
-- nunca via SQL livre sobre dados clínicos (ESTRATEGIA.md §1).

CREATE TABLE IF NOT EXISTS pacientes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nome TEXT NOT NULL,
    data_nascimento TEXT NOT NULL,
    prontuario TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS exames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_id INTEGER NOT NULL REFERENCES pacientes(id),
    tipo TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pendente', 'concluido')),
    data_solicitacao TEXT NOT NULL,
    data_resultado TEXT,
    resultado TEXT
);

CREATE TABLE IF NOT EXISTS medicacoes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_id INTEGER NOT NULL REFERENCES pacientes(id),
    nome TEXT NOT NULL,
    dosagem TEXT NOT NULL,
    frequencia TEXT NOT NULL,
    data_inicio TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alertas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_id INTEGER NOT NULL REFERENCES pacientes(id),
    descricao TEXT NOT NULL,
    severidade TEXT NOT NULL CHECK (severidade IN ('baixa', 'media', 'alta')),
    data TEXT NOT NULL,
    resolvido INTEGER NOT NULL DEFAULT 0
);
