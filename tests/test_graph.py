import os

from hospital_assistant.graph import (
    build_hospital_graph,
)


def limpar_auditoria():
    caminho = "clinical_audit.jsonl"

    if os.path.exists(caminho):
        os.remove(caminho)


def test_grafo_compila():
    app = build_hospital_graph()

    assert app is not None


def test_fluxo_ginecologia():
    limpar_auditoria()

    app = build_hospital_graph()

    estado_inicial = {"pergunta": ("Tenho cólicas durante a menstruação.")}

    resultado = app.invoke(
        estado_inicial,
        config={"configurable": {"thread_id": "teste-ginecologia"}},
    )

    assert resultado["categoria_triagem"] == "ginecologia"

    assert resultado["resposta_final"]

    assert "passos_processamento" in resultado


def test_fluxo_emergencia():
    limpar_auditoria()

    app = build_hospital_graph()

    estado_inicial = {"pergunta": ("Estou com hemorragia.")}

    resultado = app.invoke(
        estado_inicial,
        config={"configurable": {"thread_id": "teste-emergencia"}},
    )

    assert resultado["bloqueado_por_seguranca"] is True

    assert "emergência" in resultado["resposta_final"].lower() or "emergencia" in resultado["resposta_final"].lower()


def test_fluxo_violencia():
    limpar_auditoria()

    app = build_hospital_graph()

    estado_inicial = {"pergunta": ("Meu parceiro me agrediu.")}

    resultado = app.invoke(
        estado_inicial,
        config={"configurable": {"thread_id": "teste-violencia"}},
    )

    assert resultado["categoria_triagem"] == "violencia_domestica"

    assert "acolhimento" in " ".join(resultado["passos_processamento"]).lower()


def test_fluxo_medicamento_requer_validacao():
    limpar_auditoria()

    app = build_hospital_graph()

    estado_inicial = {"pergunta": ("Qual remédio devo tomar?")}

    resultado = app.invoke(
        estado_inicial,
        config={"configurable": {"thread_id": "teste-medicamento"}},
    )

    assert resultado["requer_validacao_humana"] is True

    assert (
        resultado.get(
            "validado_por_humano",
            False,
        )
        is False
    )

    assert "não realiza prescrição" in resultado["resposta_final"]
