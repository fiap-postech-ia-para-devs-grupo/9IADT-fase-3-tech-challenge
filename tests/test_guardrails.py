from hospital_assistant.safety.guardrails import (
    ClinicalGuardrails,
)


def criar_guardrails():
    return ClinicalGuardrails()


def test_detecta_emergencia():
    guardrails = criar_guardrails()

    state = {"pergunta": ("Paciente com dor torácica intensa e instabilidade hemodinâmica.")}

    valido, mensagem, sinais = guardrails.validar_input(state)

    assert valido is True

    assert "emergencia_clinica" in sinais

    assert "dor torácica intensa" in sinais

    assert "urgência" in mensagem.lower() or "urgencia" in mensagem.lower()


def test_detecta_violencia_domestica():
    guardrails = criar_guardrails()

    state = {"pergunta": ("Paciente relata que o marido a agrediu e está com medo.")}

    valido, mensagem, sinais = guardrails.validar_input(state)

    assert valido is True

    assert "suspeita_violencia_domestica" in sinais

    assert mensagem == "ACOLHIMENTO_VIOLENCIA"


def test_pergunta_normal():
    guardrails = criar_guardrails()

    state = {"pergunta": ("Qual a conduta inicial para suspeita de pneumonia?")}

    valido, mensagem, sinais = guardrails.validar_input(state)

    assert valido is True

    assert mensagem == "OK"

    assert sinais == []


def test_pergunta_vazia():
    guardrails = criar_guardrails()

    state = {"pergunta": ""}

    valido, mensagem, sinais = guardrails.validar_input(state)

    assert valido is False

    assert mensagem == "Pergunta vazia."

    assert sinais == []


def test_medicamento_exige_validacao_humana():
    guardrails = criar_guardrails()

    state = {"pergunta": ("Qual remédio devo prescrever?")}

    resposta = "Você pode conversar com um profissional sobre as opções."

    resposta_final, requer_validacao = guardrails.validar_output(
        state,
        resposta,
    )

    assert requer_validacao is True

    assert "não realiza prescrição" in resposta_final


def test_dosagem_exige_validacao_humana():
    guardrails = criar_guardrails()

    state = {"pergunta": ("Qual a dosagem correta de captopril?")}

    resposta = "A dosagem depende da avaliação clínica."

    resposta_final, requer_validacao = guardrails.validar_output(
        state,
        resposta,
    )

    assert requer_validacao is True


def test_prescricao_direta_exige_validacao():
    guardrails = criar_guardrails()

    state = {"pergunta": ("O que posso fazer?")}

    resposta = "Tome o remédio conforme indicado."

    resposta_final, requer_validacao = guardrails.validar_output(
        state,
        resposta,
    )

    assert requer_validacao is True

    assert "não realiza prescrição" in resposta_final


def test_suaviza_diagnostico():
    guardrails = criar_guardrails()

    state = {"pergunta": ("O que pode ser isso?")}

    resposta = "Você tem uma pneumonia."

    resposta_final, _ = guardrails.validar_output(
        state,
        resposta,
    )

    assert "Você tem" not in resposta_final

    assert "podem ser compatíveis" in resposta_final


def test_adiciona_aviso_de_avaliacao_presencial():
    guardrails = criar_guardrails()

    state = {"pergunta": ("Tenho uma dúvida sobre um protocolo.")}

    resposta = "Esta é uma informação geral."

    resposta_final, _ = guardrails.validar_output(
        state,
        resposta,
    )

    assert "avaliação presencial" in resposta_final.lower() or "avaliacao presencial" in resposta_final.lower()


def test_resposta_sem_medicamento_nao_exige_validacao():
    guardrails = criar_guardrails()

    state = {"pergunta": ("Quais os sinais de alerta na crise hipertensiva?")}

    resposta = "Procure avaliação presencial em caso de lesão de órgão-alvo."

    _, requer_validacao = guardrails.validar_output(
        state,
        resposta,
    )

    assert requer_validacao is False
