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

_ANAMNESE = "Paciente com febre há 2 dias, taquipneica, sem foco evidente."
_PRESCRICAO = "Ceftriaxona 1 g EV 12/12h, conforme avaliação da médica assistente."


def _gerar(**ajustes):
    dados = {"linha": _APROVADA, "paciente": _PACIENTE, "anamnese": _ANAMNESE,
             "prescricao": _PRESCRICAO}
    dados.update(ajustes)
    return laudo.gerar(
        dados["linha"], dados["paciente"], dados["anamnese"], dados["prescricao"]
    )


def test_laudo_traz_paciente_conduta_e_responsavel() -> None:
    documento = _gerar()

    assert "Ana Souza" in documento
    assert "PR-0001" in documento
    assert "Coletar lactato" in documento
    assert "Dra. Lima · CRM-SP 128450" in documento


def test_laudo_cita_a_fundamentacao_pelo_titulo_do_protocolo() -> None:
    """Mesmo critério das outras telas: caminho de arquivo não diz nada ao médico."""
    documento = _gerar()

    assert "Suspeita de sepse" in documento
    assert "sepse.md" not in documento


def test_laudo_sem_paciente_nao_e_emitido() -> None:
    """A consulta pode não ter prontuário vinculado; o laudo, não.

    É um documento sobre alguém — emitir sem identificar o paciente produziria
    uma conduta assinada sem destinatário.
    """
    with pytest.raises(laudo.LaudoIncompleto, match="Escolha o paciente"):
        _gerar(linha={**_APROVADA, "paciente_id": None}, paciente=None)


@pytest.mark.parametrize("status", ["pendente", "rejeitado", "nao_necessaria"])
def test_resposta_nao_aprovada_nao_vira_laudo(status: str) -> None:
    with pytest.raises(laudo.RespostaNaoAprovada):
        _gerar(linha={**_APROVADA, "status": status})


def test_mensagem_do_erro_usa_o_status_legivel() -> None:
    """Quem lê o erro é o operador da tela, não quem mantém o código."""
    with pytest.raises(laudo.RespostaNaoAprovada, match="Pendente de validação"):
        _gerar(linha={**_APROVADA, "status": "pendente"})


# --- anamnese e prescrição são do médico ------------------------------------


def test_laudo_traz_anamnese_e_prescricao() -> None:
    documento = _gerar()

    assert "## Anamnese" in documento
    assert "taquipneica" in documento
    assert "## Prescrição" in documento
    assert "Ceftriaxona" in documento


def test_documento_atribui_a_prescricao_a_quem_assina() -> None:
    """A avaliação mostrou o modelo devolvendo posologia; o laudo precisa deixar
    claro que essa parte não saiu dele."""
    documento = _gerar()

    assert "redigidas" in documento
    assert "profissional que assina" in documento


@pytest.mark.parametrize("faltando", ["anamnese", "prescricao"])
def test_laudo_sem_texto_do_medico_nao_e_emitido(faltando: str) -> None:
    with pytest.raises(laudo.LaudoIncompleto):
        _gerar(**{faltando: "   "})


def test_rascunho_incompleto_pode_ser_salvo(limpar_auditoria) -> None:
    """O médico escreve a anamnese, sai para conferir um exame e volta."""
    laudo.salvar_rascunho(5, _ANAMNESE, "")

    assert laudo.obter_rascunho(5)["anamnese"] == _ANAMNESE
    assert laudo.esta_completo(5) is False


def test_rascunho_completo_libera_a_emissao(limpar_auditoria) -> None:
    laudo.salvar_rascunho(5, _ANAMNESE, _PRESCRICAO, paciente_id="1")

    assert laudo.esta_completo(5) is True


def test_paciente_da_consulta_dispensa_escolher_de_novo(limpar_auditoria) -> None:
    """Quando a consulta já veio com prontuário, não há o que selecionar."""
    laudo.salvar_rascunho(5, _ANAMNESE, _PRESCRICAO)

    assert laudo.esta_completo(5, paciente_id="1") is True
    assert laudo.esta_completo(5) is False


def test_rascunho_inexistente_volta_vazio(limpar_auditoria) -> None:
    assert laudo.obter_rascunho(999) == {
        "anamnese": "",
        "prescricao": "",
        "paciente_id": None,
        "risco": None,
        "alerta": None,
    }


# --- classificação de risco -------------------------------------------------


def test_risco_aparece_no_documento() -> None:
    documento = laudo.gerar(_APROVADA, _PACIENTE, _ANAMNESE, _PRESCRICAO, risco="vermelho")

    assert "Vermelho — emergência" in documento


def test_sem_risco_o_documento_diz_que_nao_foi_classificado() -> None:
    """Campo em branco sugeriria que a classificação foi esquecida na impressão."""
    documento = laudo.gerar(_APROVADA, _PACIENTE, _ANAMNESE, _PRESCRICAO)

    assert "não classificado" in documento


def test_risco_e_alerta_persistem_no_rascunho(limpar_auditoria) -> None:
    laudo.salvar_rascunho(
        8, _ANAMNESE, _PRESCRICAO, paciente_id="1", risco="amarelo", alerta="Reavaliar em 6h"
    )

    guardado = laudo.obter_rascunho(8)

    assert guardado["risco"] == "amarelo"
    assert guardado["alerta"] == "Reavaliar em 6h"
