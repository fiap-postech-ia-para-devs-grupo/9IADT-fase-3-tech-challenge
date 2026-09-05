"""Base de conhecimento que cresce com os atendimentos, e o cache que ela habilita.

O ponto sensível é o critério de reaproveitamento: devolver uma resposta que
ninguém revisou transformaria um erro do modelo em resposta oficial para todos
os atendimentos seguintes.
"""

from __future__ import annotations

from hospital_assistant.ui import conhecimento_store, decisoes_store

_PERGUNTA = "Qual a conduta inicial na suspeita de sepse?"
_RESPOSTA = "Coletar lactato e hemoculturas antes do antimicrobiano."


def _aprovar(audit_id: int, resposta_editada: str | None = None) -> None:
    decisoes_store.registrar(
        audit_id,
        {
            "status": "aprovado",
            "aprovador": "Dra. Lima",
            "timestamp_aprovacao": "2026-09-05T10:00:00+00:00",
            "resposta_llm": resposta_editada,
        },
    )


def test_resposta_aprovada_e_reaproveitada(limpar_auditoria) -> None:
    conhecimento_store.registrar(1, _PERGUNTA, _RESPOSTA)
    _aprovar(1)

    achada = conhecimento_store.buscar_similar("qual a conduta inicial na suspeita de sepse")

    assert achada is not None
    assert achada["resposta"] == _RESPOSTA


def test_resposta_pendente_nao_e_reaproveitada(limpar_auditoria) -> None:
    """Sem revisão, reaproveitar propagaria um erro do modelo como conhecimento."""
    conhecimento_store.registrar(1, _PERGUNTA, _RESPOSTA)

    assert conhecimento_store.buscar_similar(_PERGUNTA) is None


def test_resposta_rejeitada_nao_e_reaproveitada(limpar_auditoria) -> None:
    conhecimento_store.registrar(1, _PERGUNTA, _RESPOSTA)
    decisoes_store.registrar(
        1,
        {
            "status": "rejeitado",
            "aprovador": "Dra. Lima",
            "timestamp_aprovacao": "2026-09-05T10:00:00+00:00",
            "resposta_llm": None,
        },
    )

    assert conhecimento_store.buscar_similar(_PERGUNTA) is None


def test_reaproveita_o_texto_revisado_pelo_medico(limpar_auditoria) -> None:
    """Quando o médico edita antes de aprovar, é a versão dele que vale."""
    conhecimento_store.registrar(1, _PERGUNTA, _RESPOSTA)
    _aprovar(1, resposta_editada="Conduta revisada pela médica responsável.")

    achada = conhecimento_store.buscar_similar(_PERGUNTA)

    assert achada is not None
    assert achada["resposta"] == "Conduta revisada pela médica responsável."


def test_pergunta_clinicamente_distinta_nao_casa(limpar_auditoria) -> None:
    """O limiar alto é o que impede devolver a conduta de um quadro para outro."""
    conhecimento_store.registrar(1, _PERGUNTA, _RESPOSTA)
    _aprovar(1)

    assert conhecimento_store.buscar_similar("Qual a conduta na crise hipertensiva?") is None


def test_nao_atravessa_pacientes(limpar_auditoria) -> None:
    """A mesma pergunta muda de resposta conforme o prontuário."""
    conhecimento_store.registrar(1, _PERGUNTA, _RESPOSTA, paciente_id="1")
    _aprovar(1)

    assert conhecimento_store.buscar_similar(_PERGUNTA, paciente_id="2") is None
    assert conhecimento_store.buscar_similar(_PERGUNTA, paciente_id="1") is not None


def test_acentuacao_e_caixa_nao_atrapalham(limpar_auditoria) -> None:
    assert conhecimento_store.semelhanca("Critérios do qSOFA?", "criterios do qsofa") > 0.9


def test_base_vazia_nao_quebra(limpar_auditoria) -> None:
    assert conhecimento_store.listar() == []
    assert conhecimento_store.buscar_similar("qualquer coisa") is None
