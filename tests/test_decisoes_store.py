"""Persistência das decisões de validação.

Cobre o defeito que motivou o módulo: a decisão do médico vivia em
`st.session_state` e sumia num recarregamento da página. Num sistema cujo
argumento é "nenhuma resposta chega ao médico sem revisão rastreável", perder a
revisão num F5 esvazia a garantia.
"""

from __future__ import annotations

import json

from hospital_assistant.ui import decisoes_store

_DECISAO: decisoes_store.Decisao = {
    "status": "aprovado",
    "aprovador": "Dra. Lima",
    "timestamp_aprovacao": "2026-09-05T10:00:00+00:00",
    "resposta_llm": None,
}


def test_decisao_sobrevive_a_uma_nova_leitura(limpar_auditoria) -> None:
    """O caso do F5: gravar e reler sem nenhum estado em memória no meio."""
    decisoes_store.registrar(7, _DECISAO)

    assert decisoes_store.carregar()[7]["aprovador"] == "Dra. Lima"


def test_gravar_uma_decisao_preserva_as_outras(limpar_auditoria) -> None:
    """Cada aprovação reescreve o arquivo inteiro; sem cuidado, apagaria as demais."""
    decisoes_store.registrar(1, _DECISAO)
    decisoes_store.registrar(2, {**_DECISAO, "status": "rejeitado"})

    guardadas = decisoes_store.carregar()

    assert guardadas[1]["status"] == "aprovado"
    assert guardadas[2]["status"] == "rejeitado"


def test_id_volta_como_inteiro(limpar_auditoria) -> None:
    """JSON só tem chave de texto, mas o id da auditoria é inteiro."""
    decisoes_store.registrar(3, _DECISAO)

    assert 3 in decisoes_store.carregar()


def test_arquivo_ausente_nao_quebra(limpar_auditoria) -> None:
    assert decisoes_store.carregar() == {}


def test_arquivo_corrompido_nao_derruba_a_fila(limpar_auditoria) -> None:
    """Perder o histórico é ruim; impedir a revisão das pendentes é pior."""
    decisoes_store.ARQUIVO.parent.mkdir(parents=True, exist_ok=True)
    decisoes_store.ARQUIVO.write_text("{ isto não é json", encoding="utf-8")

    assert decisoes_store.carregar() == {}


def test_escrita_deixa_json_valido(limpar_auditoria) -> None:
    decisoes_store.registrar(9, _DECISAO)

    conteudo = json.loads(decisoes_store.ARQUIVO.read_text(encoding="utf-8"))

    assert conteudo["9"]["status"] == "aprovado"
