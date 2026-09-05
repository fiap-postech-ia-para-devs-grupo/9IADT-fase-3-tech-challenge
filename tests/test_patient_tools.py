"""Escrita no prontuário a partir do laudo.

`patient_tools` só lia, porque o grafo do assistente só consulta. Estas funções
existem para o laudo: quando o médico conclui um documento com prescrição, o
medicamento precisa chegar ao prontuário — senão o laudo diz uma coisa e o
histórico do paciente diz outra, e é o histórico que a próxima consulta lê.
"""

from __future__ import annotations

import sqlite3

import pytest

from hospital_assistant.db.patient_tools import (
    get_patient_history,
    list_patients,
    registrar_alerta,
    registrar_medicacao,
)

# --- escrita a partir do laudo ----------------------------------------------


def test_medicacao_registrada_aparece_no_historico() -> None:
    """O laudo prescreve e o prontuário precisa refletir: é o prontuário que a
    próxima consulta ao assistente vai ler."""
    paciente = list_patients()[0]

    registrar_medicacao(paciente["id"], "Ceftriaxona", "1 g EV", "12/12h", "2026-09-05")

    nomes = [m["nome"] for m in get_patient_history(paciente["id"])["medicacoes"]]
    assert "Ceftriaxona" in nomes


def test_alerta_registrado_nasce_em_aberto() -> None:
    paciente = list_patients()[0]

    registrar_alerta(paciente["id"], "Reavaliar em 6h", "alta", "2026-09-05")

    alertas = get_patient_history(paciente["id"])["alertas"]
    novo = next(a for a in alertas if a["descricao"] == "Reavaliar em 6h")
    assert novo["resolvido"] is False
    assert novo["severidade"] == "alta"


def test_severidade_invalida_e_recusada_pelo_schema() -> None:
    """O CHECK do banco impede gravar algo que a tela não sabe exibir."""
    paciente = list_patients()[0]

    with pytest.raises(sqlite3.IntegrityError):
        registrar_alerta(paciente["id"], "x", "gravissima", "2026-09-05")
