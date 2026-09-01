"""Fluxo completo ponta a ponta, per issue #17 (Bloco 4): pergunta → grafo →
fila → aprovação → auditoria, tudo sobre dados reais (nenhum mock_audit_rows
envolvido) — só a LLM em si é mock (Bloco 3, Pessoa A, fora de escopo aqui).

Roda app.py via streamlit.testing.v1.AppTest: Tela 1 dispara o grafo real
(RAG real, guardrails reais), que grava em `clinical_audit.jsonl`; Tela 2 lê
esse arquivo via `audit_log.real_audit_rows` e aprova a resposta; Tela 3
confirma que a decisão aparece na auditoria.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from hospital_assistant.paths import PROJECT_ROOT

_APP_PATH = str(PROJECT_ROOT / "app.py")
_PERGUNTA = "Qual remédio devo prescrever para o paciente?"


def test_pergunta_grafo_fila_aprovacao_auditoria(limpar_auditoria):
    at = AppTest.from_file(_APP_PATH).run()

    # Tela 1 · Consulta — dispara o grafo real, que grava a auditoria.
    at.text_area[0].set_value(_PERGUNTA)
    at.button[0].click().run(timeout=30)

    assert not at.exception
    assert at.warning[0].value == "Pendente de validação humana"

    # Tela 2 · Validação — a mesma pergunta deve estar na fila real.
    at.sidebar.radio[0].set_value("Tela 2 · Validação").run()

    pendente = next(r for r in at.session_state["audit_rows"] if r["pergunta"] == _PERGUNTA)
    assert pendente["status"] == "pendente"

    at.text_input(key="aprovador_nome").set_value("Dra. Lima")
    at.button(key=f"aprovar-{pendente['id']}").click().run()

    # Tela 3 · Auditoria — a aprovação deve refletir no histórico real.
    at.sidebar.radio[0].set_value("Tela 3 · Auditoria").run()

    linhas = at.dataframe[0].value
    linha = linhas.loc[linhas["id"] == pendente["id"]].iloc[0]
    assert linha["status"] == "aprovado"
    assert linha["aprovador"] == "Dra. Lima"
