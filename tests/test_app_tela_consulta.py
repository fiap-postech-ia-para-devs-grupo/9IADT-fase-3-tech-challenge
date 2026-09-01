"""Tela 1 · Consulta ao Assistente, per issue #15.

Roda app.py de ponta a ponta via streamlit.testing.v1.AppTest: formulário de
pergunta + seletor de paciente (mock data), integração real com o grafo
(hospital_assistant.graph.flow.run) e badge de "pendente de validação
humana" no resultado.
"""

from __future__ import annotations

import json

from streamlit.testing.v1 import AppTest

from hospital_assistant.paths import PROJECT_ROOT

# Mirrors app.py's _SEM_PACIENTE — not importable directly since app.py lives
# outside the src/ package root that pytest puts on sys.path.
_SEM_PACIENTE = "Nenhum paciente selecionado"
_APP_PATH = str(PROJECT_ROOT / "app.py")


def _tela_1(at: AppTest) -> AppTest:
    at.sidebar.radio[0].set_value("Tela 1 · Consulta").run()
    return at


def test_seletor_de_paciente_lista_pacientes_do_mock_data():
    at = _tela_1(AppTest.from_file(_APP_PATH).run())

    opcoes = at.selectbox[0].options
    assert opcoes[0] == _SEM_PACIENTE
    assert any("Maria Silva Santos" in opcao for opcao in opcoes)
    assert any("João Pedro Almeida" in opcao for opcao in opcoes)
    assert any("Ana Beatriz Costa" in opcao for opcao in opcoes)


def test_seletor_de_paciente_e_opcional_por_padrao():
    at = _tela_1(AppTest.from_file(_APP_PATH).run())

    assert at.selectbox[0].value == _SEM_PACIENTE


def test_consultar_sem_pergunta_nao_executa_o_grafo():
    at = _tela_1(AppTest.from_file(_APP_PATH).run())

    at.button[0].click().run()

    assert not at.warning
    assert not at.json


def test_consultar_com_pergunta_executa_o_grafo_e_mostra_pendente_de_validacao():
    at = _tela_1(AppTest.from_file(_APP_PATH).run())

    at.text_area[0].set_value("Qual a conduta para dor torácica aguda?")
    at.button[0].click().run(timeout=30)

    assert not at.exception
    assert at.warning[0].value == "Pendente de validação humana"
    assert json.loads(at.json[0].value)["status"] == "pendente"


def test_consultar_com_paciente_selecionado_propaga_paciente_id_ao_grafo():
    at = _tela_1(AppTest.from_file(_APP_PATH).run())

    at.text_area[0].set_value("Quais exames estão pendentes?")
    opcao_maria = next(o for o in at.selectbox[0].options if "Maria Silva Santos" in o)
    at.selectbox[0].set_value(opcao_maria)
    at.button[0].click().run(timeout=30)

    assert not at.exception
    resultado = json.loads(at.json[0].value)
    assert resultado["paciente_id"] == "1"
    assert resultado["exames_pendentes"]
