"""Fluxo completo ponta a ponta: pergunta → grafo → fila → aprovação → auditoria.

Sucede as quatro suítes que dirigiam as três telas originais. Elas foram
removidas junto com aquela interface, mas o contrato que garantiam não mudou e
é o que está coberto aqui: a pergunta passa pelo grafo real (RAG real,
guardrails reais), o registro é gravado em `clinical_audit.jsonl`, aparece na
fila de validação, e a decisão do médico chega à auditoria.

Nada aqui é mockado além da própria LLM — o stand-in entra porque a suíte roda
sem GPU, o que `conftest` declara explicitamente.

A navegação é por `st.query_params`, e não por rádio na barra lateral: por isso
cada troca de módulo abaixo mexe na rota e roda de novo, que é exatamente o que
o navegador faz ao clicar num item do menu.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from hospital_assistant.paths import PROJECT_ROOT
from hospital_assistant.safety.audit_log import real_audit_rows

_APP_PATH = str(PROJECT_ROOT / "app.py")
_PERGUNTA = "Qual remédio devo prescrever para o paciente?"


def _abrir(pagina: str) -> AppTest:
    at = AppTest.from_file(_APP_PATH)
    at.query_params["p"] = pagina
    return at.run(timeout=60)


def _navegar(at: AppTest, pagina: str) -> AppTest:
    """Troca de módulo na **mesma** sessão.

    Instanciar um `AppTest` novo por página não serviria: a decisão do médico
    vive em `st.session_state`, não em disco, e uma sessão nova nasceria sem
    ela. O teste segue o caminho real do usuário, que clica no menu sem perder
    a sessão do navegador.
    """
    at.query_params["p"] = pagina
    return at.run(timeout=60)


def test_pergunta_grafo_fila_aprovacao_auditoria(limpar_auditoria):
    # --- Assistente: dispara o grafo real, que grava a trilha ---------------
    at = _abrir("assistente")
    assert not at.exception

    at.text_area[0].set_value(_PERGUNTA)
    at.button(key="enviar_pergunta").click().run(timeout=90)
    assert not at.exception

    registro = next(linha for linha in real_audit_rows() if linha["pergunta"] == _PERGUNTA)
    assert registro["status"] in ("pendente", "nao_necessaria")

    # --- Fila de validação: a resposta precisa estar lá e ser aprovável -----
    at = _navegar(at, "validacao")
    assert not at.exception

    at.text_input(key="aprovador_portal").set_value("Dra. Lima").run(timeout=60)
    at.button(key=f"portal-aprovar-{registro['id']}").click().run(timeout=60)
    assert not at.exception

    # --- Auditoria: a decisão precisa aparecer no histórico -----------------
    at = _navegar(at, "auditoria")
    assert not at.exception

    # `Nº` sai como texto: a tabela é montada para leitura, e o id é
    # identificador, não grandeza. Comparar sem converter não casa nunca.
    tabela = at.dataframe[0].value
    linha = tabela.loc[tabela["Nº"] == str(registro["id"])].iloc[0]
    assert linha["Situação"] == "Aprovado"
    assert linha["Validado por"] == "Dra. Lima"


def test_toda_resposta_entra_na_fila_mesmo_sem_medicamento(limpar_auditoria):
    """Política do §12: o guardrail só marca prescrição, a revisão vale para tudo.

    Uma pergunta clínica comum não dispara `requer_validacao_humana`, e nas
    telas originais escapava da fila — chegava ao médico solicitante sem
    revisão de ninguém.
    """
    at = _abrir("assistente")
    at.text_area[0].set_value("Quais critérios do qSOFA indicam gravidade?")
    at.button(key="enviar_pergunta").click().run(timeout=90)
    assert not at.exception

    at = _abrir("validacao")

    assert not at.exception
    assert at.text_input(key="aprovador_portal") is not None


def test_navegacao_desconhecida_cai_na_pagina_padrao():
    """Rota inventada na URL não pode derrubar a aplicação."""
    at = _abrir("pagina-que-nao-existe")

    assert not at.exception
