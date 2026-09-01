"""Tests for patient_tools.list_patients, per issue #15 (Tela 1 · seletor de paciente).

Tela 1 precisa popular um seletor com os pacientes do mock DB em vez de pedir
o paciente_id como texto livre — list_patients() é a função parametrizada que
alimenta esse seletor, seguindo o mesmo padrão de get_pending_exams /
get_patient_history (ESTRATEGIA.md §1/§4: nunca SQL livre sobre dados clínicos).
"""

from __future__ import annotations

from hospital_assistant.db.patient_tools import list_patients


def test_list_patients_retorna_todos_os_pacientes_do_seed():
    pacientes = list_patients()

    assert len(pacientes) == 3
    assert {p["nome"] for p in pacientes} == {
        "Maria Silva Santos",
        "João Pedro Almeida",
        "Ana Beatriz Costa",
    }


def test_list_patients_formato_dos_campos():
    pacientes = list_patients()

    for paciente in pacientes:
        assert set(paciente) == {"id", "nome", "prontuario"}
        assert isinstance(paciente["id"], str)


def test_list_patients_ordenado_por_nome():
    pacientes = list_patients()

    nomes = [p["nome"] for p in pacientes]
    assert nomes == sorted(nomes)
