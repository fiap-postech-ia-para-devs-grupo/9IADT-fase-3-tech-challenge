"""Cadastro de médicos, persistido em JSON.

Deliberadamente **fora do SQLite**. O banco de pacientes é consultado pelo
grafo durante o atendimento e coberto por testes de outra trilha; acrescentar
tabela e schema ali para um cadastro administrativo colocaria em risco algo que
já funciona, em troca de nada — este cadastro não é lido pelo assistente, só
pelas telas.

O arquivo fica em `data/medicos.json`, ao lado dos demais artefatos de dados, e
é criado com um conjunto inicial na primeira leitura.
"""

from __future__ import annotations

import json
from typing import TypedDict

from hospital_assistant.paths import DATA_DIR

ARQUIVO = DATA_DIR / "medicos.json"

ESPECIALIDADES = [
    "Clínica Médica",
    "Cardiologia",
    "Infectologia",
    "Pneumologia",
    "Medicina de Emergência",
    "Neurologia",
    "Nefrologia",
    "Cirurgia Geral",
]

_INICIAIS: list[dict] = [
    {"id": 1, "nome": "Marcelo Costa", "crm": "CRM-SP 128450", "especialidade": "Clínica Médica", "ativo": True},
    {"id": 2, "nome": "Vinicius Geizler", "crm": "CRM-SP 131207", "especialidade": "Infectologia", "ativo": True},
    {"id": 3, "nome": "Antonio Bazo", "crm": "CRM-SP 140933", "especialidade": "Cardiologia", "ativo": True},
    {"id": 4, "nome": "Renato Mattos", "crm": "CRM-SP 152018", "especialidade": "Pneumologia", "ativo": True},
    {"id": 5, "nome": "Vinicius Blasque", "crm": "CRM-SP 160744", "especialidade": "Medicina de Emergência", "ativo": True},
]


class Medico(TypedDict):
    id: int
    nome: str
    crm: str
    especialidade: str
    ativo: bool


def listar(apenas_ativos: bool = False) -> list[Medico]:
    """Lê o cadastro, semeando o arquivo na primeira chamada."""
    if not ARQUIVO.exists():
        _gravar(_INICIAIS)  # type: ignore[arg-type]

    try:
        medicos: list[Medico] = json.loads(ARQUIVO.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # Arquivo corrompido não pode derrubar a tela: volta ao conjunto inicial.
        medicos = list(_INICIAIS)  # type: ignore[arg-type]
        _gravar(medicos)

    if apenas_ativos:
        medicos = [m for m in medicos if m.get("ativo", True)]
    return sorted(medicos, key=lambda m: m["nome"])


def _gravar(medicos: list[Medico]) -> None:
    ARQUIVO.parent.mkdir(parents=True, exist_ok=True)
    ARQUIVO.write_text(json.dumps(medicos, ensure_ascii=False, indent=2), encoding="utf-8")


def criar(nome: str, crm: str, especialidade: str) -> Medico:
    """Cadastra um médico. Levanta `ValueError` em campo vazio ou CRM repetido."""
    nome, crm = nome.strip(), crm.strip()
    if not nome or not crm:
        raise ValueError("Nome e CRM são obrigatórios.")

    medicos = listar()
    if any(m["crm"].lower() == crm.lower() for m in medicos):
        raise ValueError(f"Já existe um médico cadastrado com o CRM {crm}.")

    novo: Medico = {
        "id": max((m["id"] for m in medicos), default=0) + 1,
        "nome": nome,
        "crm": crm,
        "especialidade": especialidade,
        "ativo": True,
    }
    _gravar([*medicos, novo])
    return novo


def alternar_ativo(medico_id: int) -> None:
    """Ativa ou inativa um médico.

    Nunca remove a linha: o nome do validador fica gravado na trilha de
    auditoria, e apagar o cadastro deixaria registros apontando para ninguém.
    """
    medicos = listar()
    for medico in medicos:
        if medico["id"] == medico_id:
            medico["ativo"] = not medico.get("ativo", True)
    _gravar(medicos)
