"""Emissão de laudo a partir de resposta validada.

O ponto que o módulo existe para garantir: laudo é o documento que atribui uma
conduta a um médico com nome e CRM. Emitir a partir de uma resposta pendente
inverteria a ordem que o projeto inteiro existe para proteger.
"""

from __future__ import annotations

import pytest

from hospital_assistant.ui import laudo

_APROVADA = {
    "id": 12,
    "timestamp": "2026-09-05T10:00:00+00:00",
    "pergunta": "Qual a conduta inicial na suspeita de sepse?",
    "paciente_id": "1",
    "fontes_rag": [{"source": "protocolos_sinteticos/sepse.md", "score": 0.81}],
    "resposta_llm": "Coletar lactato e hemoculturas antes do antimicrobiano.",
    "flags_seguranca": [],
    "status": "aprovado",
    "aprovador": "Dra. Lima · CRM-SP 128450",
    "timestamp_aprovacao": "2026-09-05T11:30:00+00:00",
}

_PACIENTE = {"id": "1", "nome": "Ana Souza", "prontuario": "PR-0001"}


def test_laudo_traz_paciente_conduta_e_responsavel() -> None:
    documento = laudo.gerar(_APROVADA, _PACIENTE)

    assert "Ana Souza" in documento
    assert "PR-0001" in documento
    assert "Coletar lactato" in documento
    assert "Dra. Lima · CRM-SP 128450" in documento


def test_laudo_cita_a_fundamentacao_pelo_titulo_do_protocolo() -> None:
    """Mesmo critério das outras telas: caminho de arquivo não diz nada ao médico."""
    documento = laudo.gerar(_APROVADA, _PACIENTE)

    assert "Suspeita de sepse" in documento
    assert "sepse.md" not in documento


def test_laudo_sem_paciente_diz_isso_explicitamente() -> None:
    """Omitir o campo sugeriria que houve um paciente identificado."""
    documento = laudo.gerar({**_APROVADA, "paciente_id": None}, None)

    assert "Consulta sem paciente vinculado" in documento


@pytest.mark.parametrize("status", ["pendente", "rejeitado", "nao_necessaria"])
def test_resposta_nao_aprovada_nao_vira_laudo(status: str) -> None:
    with pytest.raises(laudo.RespostaNaoAprovada):
        laudo.gerar({**_APROVADA, "status": status}, _PACIENTE)


def test_mensagem_do_erro_usa_o_status_legivel() -> None:
    """Quem lê o erro é o operador da tela, não quem mantém o código."""
    with pytest.raises(laudo.RespostaNaoAprovada, match="Pendente de validação"):
        laudo.gerar({**_APROVADA, "status": "pendente"}, _PACIENTE)
