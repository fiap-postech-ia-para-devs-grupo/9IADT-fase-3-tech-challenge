"""Tela 2 · Fila de Validação Humana, per issue #16.

Roda app.py de ponta a ponta via streamlit.testing.v1.AppTest: lista de
respostas pendentes, expander com fontes RAG, e os botões Aprovar / Rejeitar
/ Editar atualizando `st.session_state.audit_rows` (o mock por trás de
`auditoria.status` até a tabela real existir).
"""

from __future__ import annotations

import json

from streamlit.testing.v1 import AppTest

from hospital_assistant.paths import PROJECT_ROOT

_APP_PATH = str(PROJECT_ROOT / "app.py")


def _tela_2(at: AppTest) -> AppTest:
    at.sidebar.radio[0].set_value("Tela 2 · Validação").run()
    return at


def test_lista_apenas_respostas_pendentes():
    at = _tela_2(AppTest.from_file(_APP_PATH).run())

    rows = at.session_state["audit_rows"]
    pendentes = [r for r in rows if r["status"] == "pendente"]

    assert len(at.expander) == len(pendentes)
    textos = [e.label for e in at.expander]
    for row in pendentes:
        assert row["pergunta"] in textos


def test_expander_mostra_fontes_rag_para_explainability():
    at = _tela_2(AppTest.from_file(_APP_PATH).run())

    rows = at.session_state["audit_rows"]
    pendente = next(r for r in rows if r["status"] == "pendente")

    assert any(pendente["fontes_rag"] == json.loads(j.value) for j in at.json)


def test_aprovar_atualiza_status_para_aprovado():
    at = _tela_2(AppTest.from_file(_APP_PATH).run())

    rows_antes = at.session_state["audit_rows"]
    pendente = next(r for r in rows_antes if r["status"] == "pendente")

    at.text_input(key="aprovador_nome").set_value("Dra. Lima")
    at.button(key=f"aprovar-{pendente['id']}").click().run()

    assert not at.exception
    atualizado = next(r for r in at.session_state["audit_rows"] if r["id"] == pendente["id"])
    assert atualizado["status"] == "aprovado"
    assert atualizado["aprovador"] == "Dra. Lima"


def test_rejeitar_atualiza_status_para_rejeitado():
    at = _tela_2(AppTest.from_file(_APP_PATH).run())

    pendente = next(r for r in at.session_state["audit_rows"] if r["status"] == "pendente")

    at.button(key=f"rejeitar-{pendente['id']}").click().run()

    atualizado = next(r for r in at.session_state["audit_rows"] if r["id"] == pendente["id"])
    assert atualizado["status"] == "rejeitado"


def test_editar_revela_formulario_e_salvar_aprova_com_texto_editado():
    at = _tela_2(AppTest.from_file(_APP_PATH).run())

    pendente = next(r for r in at.session_state["audit_rows"] if r["status"] == "pendente")

    at.button(key=f"editar-{pendente['id']}").click().run()
    assert at.text_area(key=f"edicao-{pendente['id']}")

    at.text_area(key=f"edicao-{pendente['id']}").set_value("Resposta corrigida pelo médico.")
    at.button(key=f"salvar-{pendente['id']}").click().run()

    assert not at.exception
    atualizado = next(r for r in at.session_state["audit_rows"] if r["id"] == pendente["id"])
    assert atualizado["status"] == "aprovado"
    assert atualizado["resposta_llm"] == "Resposta corrigida pelo médico."


def test_aprovacao_reflete_na_tela_3_auditoria():
    at = AppTest.from_file(_APP_PATH).run()
    at = _tela_2(at)

    pendente = next(r for r in at.session_state["audit_rows"] if r["status"] == "pendente")
    at.button(key=f"aprovar-{pendente['id']}").click().run()

    at.sidebar.radio[0].set_value("Tela 3 · Auditoria").run()

    linhas = at.dataframe[0].value
    assert linhas.loc[linhas["id"] == pendente["id"], "status"].iloc[0] == "aprovado"


def test_sem_pendentes_mostra_mensagem_informativa():
    at = _tela_2(AppTest.from_file(_APP_PATH).run())

    for row in list(at.session_state["audit_rows"]):
        if row["status"] == "pendente":
            at.button(key=f"aprovar-{row['id']}").click().run()

    assert not at.expander
    assert at.info[0].value == "Nenhuma resposta pendente de validação."
