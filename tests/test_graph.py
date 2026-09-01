from hospital_assistant.graph.flow import build_hospital_graph, run


def test_grafo_compila():
    grafo = build_hospital_graph()

    assert grafo is not None


def test_fluxo_normal_com_exame_pendente(limpar_auditoria):
    # paciente 1 tem um exame pendente no seed (Hemograma completo)
    resultado = run("Qual a conduta para dor torácica aguda?", paciente_id="1")

    assert resultado["status"] == "pendente"
    assert resultado["sugestao_llm"]
    assert resultado["exames_pendentes"]
    assert resultado["alerta"] is not None
    assert "exame" in resultado["alerta"].lower()


def test_fluxo_sem_paciente_nao_consulta_exames(limpar_auditoria):
    resultado = run("Quais os sinais de alerta na sepse?")

    assert resultado["exames_pendentes"] == []


def test_fluxo_emergencia_gera_alerta(limpar_auditoria):
    resultado = run("Paciente com dor torácica intensa e instabilidade hemodinâmica.")

    assert "emergencia_clinica" in resultado["flags_seguranca"]
    assert resultado["alerta"] is not None
    assert "emergência" in resultado["alerta"].lower()


def test_fluxo_medicamento_requer_validacao(limpar_auditoria):
    resultado = run("Qual remédio devo prescrever para o paciente?")

    assert "requer_validacao_humana" in resultado["flags_seguranca"]
    assert "não realiza prescrição" in resultado["sugestao_llm"]


def test_fluxo_consulta_rag_retorna_contexto(limpar_auditoria):
    resultado = run("sintomas de pneumonia")

    assert resultado["contexto_rag"]
