"""Formatação de saída do portal.

Cobre exatamente os defeitos visíveis que motivaram a camada de UI: lista de
dicionários virando `[object Object]` na tabela, timestamp ISO com
microssegundos ocupando meia coluna, status interno (`nao_necessaria`)
aparecendo cru, e nome de campo no cabeçalho em vez de rótulo legível.
"""

from __future__ import annotations

from hospital_assistant.ui import componentes as ui
from hospital_assistant.ui import rotulos

FONTES = [
    {"text": "Coletar lactato em 3h.", "source": "protocolos_sinteticos\\sepse.md", "score": 0.8123},
    {"text": "Hemocultura antes do antimicrobiano.", "source": "sepse.md", "score": 0.61},
]

LINHA = {
    "id": 1,
    "timestamp": "2026-09-04T22:30:46.812498+00:00",
    "pergunta": "Qual a conduta na sepse?",
    "paciente_id": "1",
    "fontes_rag": FONTES,
    "resposta_llm": "Sugiro considerar coleta de lactato.",
    "flags_seguranca": ["requer_validacao_humana"],
    "status": "pendente",
    "aprovador": None,
    "timestamp_aprovacao": None,
}


# --- datas ------------------------------------------------------------------


def test_data_hora_vira_formato_brasileiro() -> None:
    assert ui.formatar_data_hora("2026-09-04T22:30:46.812498+00:00") == "04/09/2026 22:30"


def test_data_hora_vazia_vira_travessao() -> None:
    assert ui.formatar_data_hora("") == "—"


def test_data_hora_invalida_volta_intacta() -> None:
    """Melhor mostrar o dado bruto do que esconder que veio fora do formato."""
    assert ui.formatar_data_hora("ontem") == "ontem"


# --- fontes -----------------------------------------------------------------


def test_fontes_viram_titulo_do_protocolo_e_score() -> None:
    """Este é o campo que aparecia como [object Object] na tabela."""
    resultado = ui.formatar_fontes(FONTES)

    assert "Protocolo interno — Suspeita de sepse (0.81)" in resultado
    assert "[object" not in resultado


def test_fontes_nao_expoem_nome_de_arquivo() -> None:
    """A auditoria é lida por médico: caminho e extensão não dizem nada a ele."""
    resultado = ui.formatar_fontes(FONTES)

    assert "protocolos_sinteticos" not in resultado
    assert ".md" not in resultado


def test_fontes_vazias_viram_travessao() -> None:
    assert ui.formatar_fontes([]) == "—"
    assert ui.formatar_fontes(None) == "—"


def test_fonte_sem_score_nao_quebra() -> None:
    assert ui.formatar_fontes([{"text": "t", "source": "sepse.md"}]) == (
        "Protocolo interno — Suspeita de sepse"
    )


# --- flags e status ---------------------------------------------------------


def test_flags_viram_nomes_legiveis() -> None:
    assert ui.formatar_flags(["emergencia_clinica"]) == "Sinal de emergência"


def test_status_interno_vira_texto_legivel() -> None:
    assert ui.nome_do_status("nao_necessaria") == "Sem validação exigida"
    assert ui.nome_do_status("aprovado") == "Aprovado"


def test_badge_de_status_carrega_cor_e_rotulo() -> None:
    badge = ui.badge_status("pendente")

    assert "Pendente de validação" in badge
    assert "class=\"badge\"" in badge


def test_badge_escapa_html_do_conteudo() -> None:
    """Rótulo desconhecido chega até o HTML; não pode injetar marcação."""
    assert "<script>" not in ui.badge_status("<script>alert(1)</script>")


# --- rótulos ----------------------------------------------------------------


def test_rotulo_troca_nome_de_campo_por_nome_de_negocio() -> None:
    assert rotulos.rotular("paciente_id") == "Paciente"
    assert rotulos.rotular("timestamp") == "Data e hora"
    assert rotulos.rotular("resposta_llm") == "Resposta do assistente"


def test_rotulo_desconhecido_ainda_fica_legivel() -> None:
    assert rotulos.rotular("campo_novo_qualquer") == "Campo novo qualquer"


# --- tabela -----------------------------------------------------------------


def test_tabela_usa_cabecalhos_legiveis() -> None:
    colunas = list(ui.tabela_auditoria([LINHA]).columns)

    assert "Data e hora" in colunas
    assert "Fontes consultadas" in colunas
    assert "paciente_id" not in colunas
    assert "resposta_llm" not in colunas


def test_tabela_formata_os_valores_das_celulas() -> None:
    linha = ui.tabela_auditoria([LINHA]).iloc[0]

    assert linha["Data e hora"] == "04/09/2026 22:30"
    assert linha["Situação"] == "Pendente de validação"
    assert "Suspeita de sepse" in linha["Fontes consultadas"]


def test_tabela_vazia_mantem_as_colunas() -> None:
    """Sem isso a tabela some da tela quando um filtro não casa com nada."""
    assert "Situação" in list(ui.tabela_auditoria([]).columns)


def test_tabela_generica_rotula_as_colunas() -> None:
    exames = [{"tipo": "Hemograma", "status": "pendente", "data_solicitacao": "2026-08-30"}]

    colunas = list(ui.tabela_generica(exames, ["tipo", "status", "data_solicitacao"]).columns)

    assert colunas == ["Exame", "Situação", "Solicitado em"]


# --- paginação --------------------------------------------------------------


def test_paginacao_recorta_a_pagina_pedida() -> None:
    itens, total = ui.paginar(list(range(25)), pagina=2, por_pagina=10)

    assert itens == list(range(10, 20))
    assert total == 3


def test_paginacao_normaliza_pagina_alta() -> None:
    """Um filtro pode encolher o total depois que o widget já avançou a página."""
    itens, total = ui.paginar(list(range(5)), pagina=9, por_pagina=10)

    assert itens == list(range(5))
    assert total == 1


def test_paginacao_de_lista_vazia_nao_quebra() -> None:
    itens, total = ui.paginar([], pagina=1, por_pagina=10)

    assert itens == []
    assert total == 1


# --- base de conhecimento ---------------------------------------------------


def test_faq_tem_todas_as_categorias_declaradas() -> None:
    usadas = {item["categoria"] for item in rotulos.FAQ}

    assert usadas <= set(rotulos.CATEGORIAS)


def test_faq_filtra_por_categoria() -> None:
    itens = rotulos.filtrar_faq(categoria="protocolo")

    assert itens
    assert all(item["categoria"] == "protocolo" for item in itens)


def test_faq_e_so_conteudo_clinico() -> None:
    """A base descrevia o próprio sistema — "por que a resposta não veio do
    modelo treinado" não é conhecimento clínico, e ocupava espaço numa tela que
    o médico consulta durante o atendimento."""
    texto = " ".join(item["pergunta"] + item["resposta"] for item in rotulos.FAQ).lower()

    for termo in ("modelo treinado", "fine-tun", "mock", "placa de vídeo"):
        assert termo not in texto


def test_faq_busca_no_texto_da_resposta() -> None:
    assert rotulos.filtrar_faq(busca="qSOFA")


def test_faq_busca_sem_resultado_devolve_vazio() -> None:
    assert rotulos.filtrar_faq(busca="assunto que nao existe no corpus") == []


def test_toda_pergunta_frequente_cita_a_fonte() -> None:
    """A explicabilidade exigida pelo desafio vale também para a base estática."""
    assert all(item["fonte"] for item in rotulos.FAQ)


# --- procedência ------------------------------------------------------------


def test_nome_da_fonte_le_o_titulo_do_proprio_protocolo() -> None:
    """Evita um mapa arquivo→título, que sairia de sincronia a cada protocolo novo."""
    assert (
        rotulos.nome_da_fonte("protocolos_sinteticos/sepse.md")
        == "Protocolo interno — Suspeita de sepse"
    )


def test_nome_da_fonte_aceita_caminho_do_windows() -> None:
    """O RAG grava a origem com a barra do sistema onde a ingestão rodou."""
    assert (
        rotulos.nome_da_fonte(r"data\raw\protocolos_sinteticos\crise_hipertensiva.md")
        == "Protocolo interno — Crise hipertensiva"
    )


def test_nome_da_fonte_sem_arquivo_degrada_para_rotulo_legivel() -> None:
    """Documento removido do corpus não pode derrubar a tela de auditoria."""
    assert rotulos.nome_da_fonte("protocolo_inexistente.md") == "Protocolo inexistente"


def test_fontes_agrupam_trechos_do_mesmo_protocolo() -> None:
    """O RAG devolve trechos; dois do mesmo documento davam duas linhas iguais.

    Depois que o rótulo virou o título do protocolo, "FAQ interno — Solicitação
    de exames urgentes (0.49)" aparecia repetido, e a lista deixava de informar
    de quantos documentos distintos a resposta saiu.
    """
    resultado = ui.formatar_fontes(
        [
            {"source": "protocolos_sinteticos/sepse.md", "score": 0.49},
            {"source": "protocolos_sinteticos/sepse.md", "score": 0.41},
            {"source": "protocolos_sinteticos/dor_toracica_aguda.md", "score": 0.46},
        ]
    )

    assert resultado.count("Suspeita de sepse") == 1
    assert "Dor torácica aguda" in resultado


def test_fontes_agrupadas_ficam_com_o_melhor_score() -> None:
    """Mostrar o score mais fraco subestimaria a aderência da resposta à fonte."""
    resultado = ui.formatar_fontes(
        [
            {"source": "protocolos_sinteticos/sepse.md", "score": 0.41},
            {"source": "protocolos_sinteticos/sepse.md", "score": 0.81},
        ]
    )

    assert "(0.81)" in resultado
    assert "(0.41)" not in resultado


# --- classificação de risco -------------------------------------------------


def test_cartao_de_risco_traz_cor_e_data() -> None:
    cartao = ui.cartao_risco("vermelho", "2026-09-05T10:00:00+00:00")

    assert "Vermelho" in cartao
    assert "05/09/2026" in cartao


def test_sem_classificacao_o_cartao_diz_isso() -> None:
    """Espaço vazio ao lado de Exames e Alertas parece dado que não carregou."""
    assert "Não avaliado" in ui.cartao_risco(None)


def test_risco_desconhecido_nao_quebra() -> None:
    assert "Não avaliado" in ui.cartao_risco("roxo")
