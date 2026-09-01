"""Tela 2 · Fila de Validação Humana, per issue #16 (queue/approve UI) and
issue #17 (wired to the real audit trail instead of mock data).

Roda app.py de ponta a ponta via streamlit.testing.v1.AppTest: lista de
respostas pendentes lida do `clinical_audit.jsonl` real via
`audit_log.real_audit_rows`, expander com fontes RAG, e os botões Aprovar /
Rejeitar / Editar atualizando `st.session_state.audit_rows` (a decisão em si
continua session-scoped — persistir de volta no JSONL é uma tabela
`auditoria` separada, fora do escopo desta issue).
"""

from __future__ import annotations

import json

from streamlit.testing.v1 import AppTest

from hospital_assistant.paths import PROJECT_ROOT
from hospital_assistant.safety.audit_log import ClinicalAuditLogger

_APP_PATH = str(PROJECT_ROOT / "app.py")


def _registrar(
    pergunta: str,
    paciente_id: str | None = "1",
    *,
    resposta_final: str = "Resposta de teste.",
    requer_validacao_humana: bool = True,
) -> None:
    ClinicalAuditLogger.registrar_evento(
        {
            "paciente_id": paciente_id,
            "pergunta": pergunta,
            "resposta_final": resposta_final,
            "requer_validacao_humana": requer_validacao_humana,
            "validado_por_humano": False,
            "fontes_citadas": ["protocolo_teste.md"],
            "documentos_retornados": [{"text": "trecho de teste", "source": "protocolo_teste.md", "score": 0.9}],
        }
    )


def _tela_2(at: AppTest) -> AppTest:
    at.sidebar.radio[0].set_value("Tela 2 · Validação").run()
    return at


def test_lista_apenas_respostas_que_requerem_validacao(limpar_auditoria):
    _registrar("Qual remédio devo prescrever para a dor?", "1")
    _registrar("Reavaliação de exames pendentes?", "2", requer_validacao_humana=False)

    at = _tela_2(AppTest.from_file(_APP_PATH).run())

    rows = at.session_state["audit_rows"]
    pendentes = [r for r in rows if r["status"] == "pendente"]

    assert len(pendentes) == 1
    assert len(at.expander) == len(pendentes)
    assert pendentes[0]["pergunta"] in [e.label for e in at.expander]


def test_expander_mostra_fontes_rag_para_explainability(limpar_auditoria):
    _registrar("Qual remédio devo prescrever para a dor?", "1")

    at = _tela_2(AppTest.from_file(_APP_PATH).run())

    pendente = next(r for r in at.session_state["audit_rows"] if r["status"] == "pendente")

    assert any(pendente["fontes_rag"] == json.loads(j.value) for j in at.json)


def test_aprovar_atualiza_status_para_aprovado(limpar_auditoria):
    _registrar("Qual remédio devo prescrever para a dor?", "1")

    at = _tela_2(AppTest.from_file(_APP_PATH).run())

    pendente = next(r for r in at.session_state["audit_rows"] if r["status"] == "pendente")

    at.text_input(key="aprovador_nome").set_value("Dra. Lima")
    at.button(key=f"aprovar-{pendente['id']}").click().run()

    assert not at.exception
    atualizado = next(r for r in at.session_state["audit_rows"] if r["id"] == pendente["id"])
    assert atualizado["status"] == "aprovado"
    assert atualizado["aprovador"] == "Dra. Lima"


def test_rejeitar_atualiza_status_para_rejeitado(limpar_auditoria):
    _registrar("Qual remédio devo prescrever para a dor?", "1")

    at = _tela_2(AppTest.from_file(_APP_PATH).run())

    pendente = next(r for r in at.session_state["audit_rows"] if r["status"] == "pendente")

    at.button(key=f"rejeitar-{pendente['id']}").click().run()

    atualizado = next(r for r in at.session_state["audit_rows"] if r["id"] == pendente["id"])
    assert atualizado["status"] == "rejeitado"


def test_editar_revela_formulario_e_salvar_aprova_com_texto_editado(limpar_auditoria):
    _registrar("Qual remédio devo prescrever para a dor?", "1")

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


def test_aprovacao_reflete_na_tela_3_auditoria(limpar_auditoria):
    _registrar("Qual remédio devo prescrever para a dor?", "1")

    at = AppTest.from_file(_APP_PATH).run()
    at = _tela_2(at)

    pendente = next(r for r in at.session_state["audit_rows"] if r["status"] == "pendente")
    at.button(key=f"aprovar-{pendente['id']}").click().run()

    at.sidebar.radio[0].set_value("Tela 3 · Auditoria").run()

    linhas = at.dataframe[0].value
    assert linhas.loc[linhas["id"] == pendente["id"], "status"].iloc[0] == "aprovado"


def test_sem_pendentes_mostra_mensagem_informativa(limpar_auditoria):
    at = _tela_2(AppTest.from_file(_APP_PATH).run())

    assert not at.expander
    assert at.info[0].value == "Nenhuma resposta pendente de validação."
