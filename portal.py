"""Portal Clínico — interface modular do Assistente Virtual Médico.

Entry point **paralelo** ao `app.py`. As três telas originais e suas quatro
suítes de teste continuam intactas: nada aqui as importa, e nada aqui altera
módulo compartilhado. Rodar um não interfere no outro.

    streamlit run portal.py

O que muda em relação às telas originais:

- **Navegação agrupada** em vez de três rádios soltos: Assistente (assistente,
  fila de validação e base de conhecimento), Cadastro e Auditoria. A base de
  conhecimento fica sob Assistente por ser a base que fundamenta as respostas,
  não um módulo paralelo. A rota vive em `st.query_params`, então o endereço é
  compartilhável e o botão voltar funciona.
- **Assistente em formato de conversa**, com histórico na sessão, em vez de
  um formulário que devolve JSON cru.
- **Saídas formatadas**: as fontes do RAG viram cartões com score em barra, e
  a tabela de auditoria mostra datas em pt-BR, situações por extenso e o nome
  do arquivo de origem — não mais `[object Object]`.
- **Tabelas paginadas** com nomes de coluna legíveis.

Duas correções de comportamento, aplicadas **como política do portal** para
não alterar código de outra trilha:

1. Toda resposta entra na fila de validação, e não apenas as que mencionam
   medicamento. O critério da ESTRATEGIA §12 é que nenhuma resposta chegue ao
   médico solicitante sem revisão; o guardrail sozinho só marca quando detecta
   termo de prescrição, então uma pergunta clínica comum escapava da fila.
2. A fila é lida a cada renderização. Nas telas originais os registros são
   semeados uma única vez em `session_state`, então uma consulta feita depois
   de abrir a fila não aparecia até recarregar a página.
"""

from __future__ import annotations

import random
from typing import Any

import streamlit as st

from hospital_assistant.db.patient_tools import get_patient_history, list_patients
from hospital_assistant.graph.flow import run
from hospital_assistant.llm.model_loader import descrever_backend, get_llm
from hospital_assistant.safety.audit_log import AuditRow, apply_decision, real_audit_rows
from hospital_assistant.ui import componentes as ui
from hospital_assistant.ui import rotulos, tema

st.set_page_config(page_title="Portal Clínico · Assistente Médico", layout="wide")

SEM_PACIENTE = "Nenhum paciente selecionado"

# Navegação. O identificador interno (`assistente`, `validacao`, …) é o que
# circula em `st.query_params` e indexa `PAGINAS`; o label é só apresentação.
# Separar os dois permite renomear o menu sem tocar em rota, callback ou estado
# — foi a razão de o rótulo antigo ("Registros") poder virar "Cadastro" sem
# efeito colateral.
#
# "Base de conhecimento" fica sob ASSISTENTE, e não numa seção própria: é a
# base que fundamenta as respostas da IA, não um módulo paralelo.
MENU: list[tuple[str, list[tuple[str, str]]]] = [
    ("Assistente", [
        ("assistente", "Assistente"),
        ("validacao", "Fila de validação"),
        ("conhecimento", "Base de conhecimento"),
    ]),
    ("Cadastro", [
        ("pacientes", "Pacientes"),
    ]),
    ("Auditoria", [
        ("auditoria", "Auditoria"),
    ]),
]

PAGINA_PADRAO = "assistente"


# ---------------------------------------------------------------------------
# Estado
# ---------------------------------------------------------------------------


def _decisoes() -> dict[int, dict[str, Any]]:
    """Decisões de validação tomadas nesta sessão, indexadas pelo id do registro.

    Mantidas à parte das linhas de auditoria — e não sobre uma cópia congelada
    delas — para que a lista possa ser relida do disco a cada renderização sem
    perder o que o médico já decidiu.
    """
    return st.session_state.setdefault("decisoes", {})


def carregar_registros() -> list[AuditRow]:
    """Lê a trilha de auditoria do disco e aplica as decisões da sessão.

    A releitura a cada chamada é intencional: é o que faz uma consulta nova
    aparecer na fila sem recarregar a página.
    """
    linhas = real_audit_rows()
    decisoes = _decisoes()

    for linha in linhas:
        decisao = decisoes.get(linha["id"])
        if decisao is None:
            continue
        linha["status"] = decisao["status"]
        linha["aprovador"] = decisao["aprovador"]
        linha["timestamp_aprovacao"] = decisao["timestamp_aprovacao"]
        if decisao.get("resposta_llm") is not None:
            linha["resposta_llm"] = decisao["resposta_llm"]

    return linhas


def pendentes(linhas: list[AuditRow]) -> list[AuditRow]:
    """Registros que aguardam revisão humana.

    Inclui `nao_necessaria`: o guardrail só exige validação quando detecta
    termo de medicamento, e o critério da ESTRATEGIA §12 é que **nenhuma**
    resposta chegue ao médico sem revisão. Aqui a política é a do §12.
    """
    return [linha for linha in linhas if linha["status"] in ("pendente", "nao_necessaria")]


def registrar_decisao(
    linha: AuditRow,
    decisao: str,
    aprovador: str | None,
    resposta_editada: str | None = None,
) -> None:
    """Grava a decisão na sessão, reaproveitando a regra de `apply_decision`."""
    atualizadas = apply_decision(
        [linha], linha["id"], decisao, aprovador, resposta_editada=resposta_editada
    )
    atualizada = atualizadas[0]
    _decisoes()[linha["id"]] = {
        "status": atualizada["status"],
        "aprovador": atualizada["aprovador"],
        "timestamp_aprovacao": atualizada["timestamp_aprovacao"],
        "resposta_llm": resposta_editada,
    }


# ---------------------------------------------------------------------------
# Módulo: Assistente
# ---------------------------------------------------------------------------


LIMITE_PERGUNTA = 600
SUGESTOES_POR_VEZ = 3


def sugestoes(semente: int) -> list[dict[str, Any]]:
    """Sorteia perguntas da base de conhecimento para oferecer como atalho.

    As sugestões saem da mesma base que fundamenta as respostas, e não de uma
    lista à parte: garante que toda sugestão tenha protocolo indexado por trás,
    e faz a base crescer junto com o assistente — acrescentar um protocolo
    passa a alimentar as duas telas de uma vez.

    Só entram categorias clínicas: a base também explica o funcionamento do
    assistente, e essas entradas pertencem à tela de conhecimento, não ao
    composer de quem está atendendo.

    O sorteio usa semente explícita para que a lista só mude quando o usuário
    pedir, e não a cada rerun do Streamlit.
    """
    itens = [item for item in rotulos.FAQ if item["categoria"] in rotulos.CATEGORIAS_CLINICAS]
    random.Random(semente).shuffle(itens)
    return itens[:SUGESTOES_POR_VEZ]


def _cabecalho_paciente(paciente: dict[str, Any]) -> None:
    """Identificação e indicadores do paciente em atendimento.

    Os dados vêm de `get_patient_history` e não do resumo do seletor:
    `list_patients` devolve apenas id, nome e prontuário — deliberadamente, para
    não expor dado clínico a quem só precisa montar um dropdown.
    """
    historico = get_patient_history(paciente["id"])
    pendentes_exames = [e for e in historico["exames"] if e["status"] == "pendente"]
    alertas_abertos = [a for a in historico["alertas"] if not a["resolvido"]]

    st.markdown(
        f"**{historico['nome']}** &nbsp;·&nbsp; prontuário `{historico['prontuario']}`"
        f" &nbsp;·&nbsp; nascimento "
        f"{ui.formatar_data_hora(historico['data_nascimento']).split(' ')[0]}",
        unsafe_allow_html=True,
    )
    indicadores = st.columns(4)
    indicadores[0].markdown(ui.metrica(len(historico["exames"]), "Exames"), unsafe_allow_html=True)
    indicadores[1].markdown(
        ui.metrica(len(pendentes_exames), "Exames pendentes"), unsafe_allow_html=True
    )
    indicadores[2].markdown(
        ui.metrica(len(historico["medicacoes"]), "Medicações"), unsafe_allow_html=True
    )
    indicadores[3].markdown(
        ui.metrica(len(alertas_abertos), "Alertas abertos"), unsafe_allow_html=True
    )


def _enviar(pergunta: str, paciente_id: str | None) -> None:
    """Roda o grafo e acrescenta o turno ao histórico da conversa."""
    historico: list[dict[str, Any]] = st.session_state.setdefault("conversa", [])
    historico.append({"papel": "user", "texto": pergunta, "fontes": []})

    with st.spinner("Consultando protocolos e prontuário…"):
        resultado = run(pergunta, paciente_id)

    texto = resultado["sugestao_llm"]
    if resultado.get("alerta"):
        texto += f"\n\n> **Alerta emitido para a equipe:** {resultado['alerta']}"

    historico.append(
        {"papel": "assistant", "texto": texto, "fontes": resultado.get("contexto_rag", [])}
    )


def _render_conversa(historico: list[dict[str, Any]]) -> None:
    """Desenha o histórico como conversa.

    As fontes ficam recolhidas num expander por resposta: são a justificativa da
    sugestão, consultadas quando o médico quer conferir a procedência, e abertas
    por padrão empurrariam a próxima pergunta para fora da tela.
    """
    for turno in historico:
        with st.chat_message(turno["papel"]):
            st.markdown(turno["texto"])
            fontes = turno.get("fontes") or []
            if fontes:
                with st.expander(f"Base consultada ({len(fontes)} trechos)"):
                    for posicao, fonte in enumerate(fontes, start=1):
                        st.markdown(ui.cartao_fonte(fonte, posicao), unsafe_allow_html=True)


def _chips_sugestao() -> None:
    """Atalhos para a base de conhecimento, logo abaixo da caixa de pergunta.

    O clique **preenche** o composer em vez de enviar: a sugestão é ponto de
    partida, e o médico costuma querer completar a pergunta com o contexto do
    caso antes de mandar.

    Larguras proporcionais ao rótulo mantêm os chips agrupados à esquerda:
    colunas iguais espalhariam as pílulas por toda a largura e elas deixariam de
    ser lidas como um conjunto.
    """
    semente = st.session_state.setdefault("semente_sugestoes", 0)
    escolhidas = sugestoes(semente)

    # Três chips, e não quatro: com `nowrap` a pílula não encolhe, e a quarta
    # espremia o "Gerar outras" para fora da borda. As larguras seguem o
    # tamanho do rótulo para os chips ficarem agrupados à esquerda.
    rotulos_chip = [ui.resumir_texto(item["pergunta"], limite=32) for item in escolhidas]
    larguras = [max(1.0, len(texto) / 4.2) for texto in rotulos_chip]

    # O container com `key` existe só para o CSS conseguir apertar o `gap` entre
    # as colunas: sem ele os chips ficam com o respiro padrão do Streamlit e
    # deixam de ser lidos como um conjunto.
    with st.container(key="linha_sugestoes"):
        colunas = st.columns([*larguras, 3.4])

        for posicao, (item, rotulo) in enumerate(zip(escolhidas, rotulos_chip, strict=True)):
            # `help` carrega a pergunta inteira: o chip é truncado por desenho,
            # mas o médico precisa poder conferir o texto antes de enviar.
            if colunas[posicao].button(
                rotulo,
                key=f"sugestao-{posicao}",
                help=item["pergunta"],
                use_container_width=True,
            ):
                # O texto não pode ser escrito direto na key do widget: ele já
                # foi instanciado nesta rodada e o Streamlit recusa a alteração.
                # Guardar num rascunho e trocar a key recria o campo já com o
                # conteúdo — o mesmo mecanismo que limpa o composer no envio.
                st.session_state["rascunho_pergunta"] = item["pergunta"]
                st.session_state["ciclo_composer"] = st.session_state["ciclo_composer"] + 1
                st.rerun()

        if colunas[-1].button(
            "🎲 Gerar outras", key="regerar_sugestoes", use_container_width=True
        ):
            st.session_state["semente_sugestoes"] = semente + 1
            st.rerun()


def modulo_assistente() -> None:
    # `st.container(key=…)` em vez de uma `div` aberta por `st.markdown`: aquela
    # é fechada pelo parser no fim do próprio container de markdown, então os
    # widgets seguintes viram irmãos dela e nenhum seletor descendente casa —
    # era por isso que a coluna central e o estilo dos chips não pegavam. Com a
    # key, o Streamlit emite `st-key-bloco_assistente` no wrapper de verdade.
    with st.container(key="bloco_assistente"):
        _assistente_conteudo()


def _assistente_conteudo() -> None:
    historico: list[dict[str, Any]] = st.session_state.setdefault("conversa", [])

    pacientes = list_patients()
    por_rotulo = {SEM_PACIENTE: None} | {
        f"{p['nome']} ({p['prontuario']})": p["id"] for p in pacientes
    }
    if historico:
        _render_conversa(historico)
    else:
        st.markdown(
            '<div class="saudacao"><h2>Como posso ajudar no atendimento?</h2>'
            "<p>Pergunte sobre conduta clínica, protocolo institucional ou exames. "
            "Toda resposta passa por revisão médica antes de valer como conduta.</p></div>",
            unsafe_allow_html=True,
        )

    # Sem `st.form`: o seletor de paciente vive dentro do composer e precisa
    # atualizar o cabeçalho clínico assim que muda. Num form nada é reavaliado
    # até o envio, e o médico escolheria o paciente sem ver os exames pendentes.
    # A limpeza do campo, que o form daria de graça, vem do `ciclo` na key.
    ciclo = st.session_state.setdefault("ciclo_composer", 0)
    pergunta = st.text_area(
        "Pergunta",
        value=st.session_state.get("rascunho_pergunta", ""),
        height=150,
        max_chars=LIMITE_PERGUNTA,
        placeholder="Descreva o caso ou pergunte sobre um protocolo…",
        label_visibility="collapsed",
        key=f"pergunta_{ciclo}",
    )

    if not historico:
        _chips_sugestao()

    escolha = st.selectbox("Paciente em atendimento", list(por_rotulo), key="paciente_conversa")
    paciente_id = por_rotulo[escolha]

    if paciente_id:
        selecionado = next(p for p in pacientes if p["id"] == paciente_id)
        with st.container(border=True):
            _cabecalho_paciente(selecionado)

    # Envio na última linha, alinhado à direita: é a ação que fecha o composer,
    # depois de pergunta, atalhos e contexto do paciente já estarem definidos.
    acoes = st.columns([4, 1])
    enviar = acoes[1].button(
        "Enviar", key="enviar_pergunta", type="primary", use_container_width=True
    )

    if enviar and pergunta.strip():
        _enviar(pergunta.strip(), paciente_id)
        st.session_state["rascunho_pergunta"] = ""
        st.session_state["ciclo_composer"] = ciclo + 1
        st.rerun()

    if historico and st.button("Nova conversa", key="limpar_conversa"):
        st.session_state["conversa"] = []
        st.rerun()



# ---------------------------------------------------------------------------
# Módulo: Fila de validação
# ---------------------------------------------------------------------------


def modulo_validacao() -> None:
    st.markdown("### Fila de validação")
    st.caption(
        "Nenhuma sugestão vale como conduta antes de passar por aqui. Confira as fontes que "
        "fundamentaram a resposta antes de decidir."
    )

    registros = carregar_registros()
    fila = pendentes(registros)

    aprovador = st.text_input("Médico responsável pela validação", key="aprovador_portal")

    if not fila:
        st.success("Nenhuma resposta aguardando validação.")
        return

    st.markdown(f"**{len(fila)}** {'resposta aguardando' if len(fila) == 1 else 'respostas aguardando'} revisão.")

    # Índice de pacientes: a trilha grava só o id, e "Paciente: 1" não diz nada
    # a quem revisa. A resolução acontece aqui, uma vez, e não a cada linha.
    pacientes = {p["id"]: p for p in list_patients()}

    for linha in fila:
        with st.expander(ui.resumir_texto(linha["pergunta"], limite=110)):
            paciente = pacientes.get(linha["paciente_id"] or "")
            cabecalho = st.columns([3, 1])
            if paciente:
                # Só nome e prontuário: `list_patients` não traz data de
                # nascimento, e buscar o histórico completo de cada linha da
                # fila seria uma consulta por item para exibir um campo.
                cabecalho[0].markdown(
                    f"**{paciente['nome']}** &nbsp;·&nbsp; prontuário `{paciente['prontuario']}`",
                    unsafe_allow_html=True,
                )
            else:
                cabecalho[0].markdown("**Consulta sem paciente vinculado**")
            cabecalho[1].markdown(ui.badge_status(linha["status"]), unsafe_allow_html=True)
            st.caption(f"Registrado em {ui.formatar_data_hora(linha['timestamp'])}")

            st.divider()
            st.markdown("##### Resposta sugerida")
            st.markdown(linha["resposta_llm"])

            if linha["flags_seguranca"]:
                st.markdown(
                    f'<div class="aviso-seguranca"><strong>Sinalizações do guardrail:</strong> '
                    f"{ui.formatar_flags(linha['flags_seguranca'])}</div>",
                    unsafe_allow_html=True,
                )

            fontes = linha["fontes_rag"]
            st.markdown(f"**Fontes consultadas** ({len(fontes)})")
            if fontes:
                for posicao, fonte in enumerate(fontes, start=1):
                    st.markdown(ui.cartao_fonte(fonte, posicao), unsafe_allow_html=True)
            else:
                st.caption("Nenhuma fonte recuperada para esta pergunta.")

            editando = st.session_state.get(f"editar_portal_{linha['id']}", False)

            st.divider()
            if editando:
                texto = st.text_area(
                    "Ajuste a resposta antes de aprovar",
                    value=linha["resposta_llm"],
                    key=f"portal-edicao-{linha['id']}",
                    height=220,
                )
                confirmacao = st.columns([1, 1, 3])
                if confirmacao[0].button(
                    "Salvar e aprovar",
                    key=f"portal-salvar-{linha['id']}",
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state[f"editar_portal_{linha['id']}"] = False
                    registrar_decisao(linha, "aprovado", aprovador or None, resposta_editada=texto)
                    st.rerun()
                if confirmacao[1].button(
                    "Cancelar", key=f"portal-cancelar-{linha['id']}", use_container_width=True
                ):
                    st.session_state[f"editar_portal_{linha['id']}"] = False
                    st.rerun()
            else:
                # Larguras desiguais com folga à direita: as três ações não
                # devem ocupar a linha inteira nem ficar do mesmo peso visual —
                # aprovar é a ação principal, editar é a de escape.
                acoes = st.columns([1, 1, 1, 2])
                if acoes[0].button(
                    "Aprovar",
                    key=f"portal-aprovar-{linha['id']}",
                    type="primary",
                    use_container_width=True,
                ):
                    registrar_decisao(linha, "aprovado", aprovador or None)
                    st.rerun()
                if acoes[1].button(
                    "Editar", key=f"portal-editar-{linha['id']}", use_container_width=True
                ):
                    st.session_state[f"editar_portal_{linha['id']}"] = True
                    st.rerun()
                if acoes[2].button(
                    "Rejeitar", key=f"portal-rejeitar-{linha['id']}", use_container_width=True
                ):
                    registrar_decisao(linha, "rejeitado", aprovador or None)
                    st.rerun()


# ---------------------------------------------------------------------------
# Módulo: Base de conhecimento
# ---------------------------------------------------------------------------


def modulo_conhecimento() -> None:
    st.markdown("### Base de conhecimento")
    st.caption(
        "Respostas curtas para as dúvidas recorrentes, com a fonte de cada uma. "
        "Espelha os protocolos indexados no assistente."
    )

    filtros = st.columns([2, 3])
    categoria = filtros[0].selectbox(
        "Categoria",
        ["todas", *rotulos.CATEGORIAS],
        format_func=lambda c: "Todas as categorias" if c == "todas" else rotulos.CATEGORIAS[c],
    )
    busca = filtros[1].text_input("Buscar por palavra", placeholder="sepse, exame urgente, dose…")

    itens = rotulos.filtrar_faq(categoria, busca)

    contagens = st.columns(len(rotulos.CATEGORIAS))
    for coluna, (chave, nome) in zip(contagens, rotulos.CATEGORIAS.items(), strict=True):
        total = len([i for i in rotulos.FAQ if i["categoria"] == chave])
        coluna.markdown(ui.metrica(total, nome), unsafe_allow_html=True)

    st.markdown("")
    if not itens:
        st.info("Nenhuma pergunta encontrada para esse filtro.")
        return

    for item in itens:
        st.markdown(ui.cartao_faq(item), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Módulo: Pacientes
# ---------------------------------------------------------------------------


def modulo_pacientes() -> None:
    st.markdown("### Pacientes")
    st.caption("Consulta ao cadastro e ao prontuário. Somente leitura.")

    pacientes = list_patients()
    if not pacientes:
        st.warning(
            "Nenhum paciente cadastrado. Rode `uv run python -m hospital_assistant.db.seed_mock_data`."
        )
        return

    escolha = st.selectbox(
        "Paciente",
        [p["id"] for p in pacientes],
        format_func=lambda pid: next(
            f"{p['nome']} ({p['prontuario']})" for p in pacientes if p["id"] == pid
        ),
    )

    historico = get_patient_history(escolha)
    indicadores = st.columns(3)
    indicadores[0].markdown(ui.metrica(len(historico["exames"]), "Exames"), unsafe_allow_html=True)
    indicadores[1].markdown(
        ui.metrica(len(historico["medicacoes"]), "Medicações"), unsafe_allow_html=True
    )
    indicadores[2].markdown(
        ui.metrica(len(historico["alertas"]), "Alertas"), unsafe_allow_html=True
    )

    abas = st.tabs(["Exames", "Medicações", "Alertas"])
    with abas[0]:
        st.dataframe(
            ui.tabela_generica(
                historico["exames"],
                ["tipo", "status", "data_solicitacao", "data_resultado", "resultado"],
            ),
            use_container_width=True,
            hide_index=True,
        )
    with abas[1]:
        st.dataframe(
            ui.tabela_generica(
                historico["medicacoes"], ["nome", "dosagem", "frequencia", "data_inicio"]
            ),
            use_container_width=True,
            hide_index=True,
        )
    with abas[2]:
        st.dataframe(
            ui.tabela_generica(
                historico["alertas"], ["descricao", "severidade", "data", "resolvido"]
            ),
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# Módulo: Auditoria
# ---------------------------------------------------------------------------


def modulo_auditoria() -> None:
    st.markdown("### Auditoria")
    st.caption("Trilha completa de execuções do assistente, com filtros e paginação.")

    registros = carregar_registros()
    if not registros:
        st.info("Nenhuma execução registrada ainda. Faça uma consulta no módulo Assistente.")
        return

    filtros = st.columns([2, 2, 2, 1])
    situacao = filtros[0].selectbox(
        "Situação",
        ["todas", *tema.CORES_STATUS],
        format_func=lambda s: "Todas" if s == "todas" else ui.nome_do_status(s),
    )
    ids_pacientes = sorted({r["paciente_id"] for r in registros if r["paciente_id"]})
    paciente = filtros[1].selectbox("Paciente", ["todos", *ids_pacientes])
    datas = sorted({r["timestamp"][:10] for r in registros if r["timestamp"]})
    data = filtros[2].selectbox(
        "Data",
        ["todas", *datas],
        format_func=lambda d: "Todas" if d == "todas" else ui.formatar_data_hora(d).split(" ")[0],
    )
    por_pagina = filtros[3].selectbox("Por página", [10, 25, 50], index=0)

    filtrados = [
        r
        for r in registros
        if (situacao == "todas" or r["status"] == situacao)
        and (paciente == "todos" or r["paciente_id"] == paciente)
        and (data == "todas" or r["timestamp"].startswith(data))
    ]

    if not filtrados:
        st.info("Nenhum registro para esse filtro.")
        return

    indicadores = st.columns(4)
    for coluna, status in zip(indicadores, tema.CORES_STATUS, strict=False):
        coluna.markdown(
            ui.metrica(len([r for r in filtrados if r["status"] == status]), ui.nome_do_status(status)),
            unsafe_allow_html=True,
        )

    pagina = st.session_state.get("pagina_auditoria", 1)
    itens, total_paginas = ui.paginar(filtrados, pagina, por_pagina)

    st.dataframe(ui.tabela_auditoria(itens), use_container_width=True, hide_index=True)

    navegacao = st.columns([1, 2, 1])
    if navegacao[0].button("← Anterior", disabled=pagina <= 1, key="auditoria_anterior"):
        st.session_state["pagina_auditoria"] = pagina - 1
        st.rerun()
    navegacao[1].markdown(
        f"<div style='text-align:center;color:{tema.TEXTO_TENUE};font-size:.85rem;padding-top:.4rem'>"
        f"Página {min(pagina, total_paginas)} de {total_paginas} · {len(filtrados)} registros</div>",
        unsafe_allow_html=True,
    )
    if navegacao[2].button("Próxima →", disabled=pagina >= total_paginas, key="auditoria_proxima"):
        st.session_state["pagina_auditoria"] = pagina + 1
        st.rerun()


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

PAGINAS = {
    "assistente": modulo_assistente,
    "validacao": modulo_validacao,
    "conhecimento": modulo_conhecimento,
    "pacientes": modulo_pacientes,
    "auditoria": modulo_auditoria,
}


def pagina_atual() -> str:
    """Identificador da página ativa, lido da URL.

    A rota vive em `st.query_params` (e não só em `session_state`) para que o
    endereço seja compartilhável e o botão voltar do navegador funcione. Um
    identificador desconhecido — link antigo, digitação — cai no padrão em vez
    de estourar `KeyError`.
    """
    pedido = st.query_params.get("p", PAGINA_PADRAO)
    return pedido if pedido in PAGINAS else PAGINA_PADRAO


def _item_menu(chave: str, label: str, ativo: bool, contagem: int = 0) -> str:
    """Item de navegação como âncora estilizada.

    Âncora e não `st.button` porque o botão do Streamlit não aceita indicador
    lateral, ícone inline nem estado ativo sem herdar a aparência de botão —
    e a navegação precisa parecer menu, não formulário. O `target="_self"`
    evita que o clique abra uma aba nova.
    """
    classes = "nav-item ativo" if ativo else "nav-item"
    selo = f'<span class="nav-contagem">{contagem}</span>' if contagem else ""
    return (
        f'<a class="{classes}" href="?p={chave}" target="_self">'
        f"{tema.icone(chave)}<span>{label}</span>{selo}</a>"
    )


def barra_lateral(pendencias: int) -> None:
    """Marca, navegação agrupada e status operacional."""
    atual = pagina_atual()
    st.markdown(tema.cabecalho_marca(), unsafe_allow_html=True)

    for grupo, itens in MENU:
        st.markdown(f'<div class="grupo-menu">{grupo}</div>', unsafe_allow_html=True)
        st.markdown(
            "".join(
                _item_menu(
                    chave,
                    label,
                    ativo=chave == atual,
                    contagem=pendencias if chave == "validacao" else 0,
                )
                for chave, label in itens
            ),
            unsafe_allow_html=True,
        )

    # O rótulo do modelo é escrito para quem opera a tela, não para quem
    # mantém o código: "MockLLM" é nome de classe. O estado é o mesmo, a
    # leitura é que muda — e a distinção entre demonstração e modelo treinado
    # continua explícita, que é o que importa para não demonstrar o stand-in
    # achando que é o modelo.
    em_demonstracao = "Mock" in descrever_backend(get_llm())
    if em_demonstracao:
        cor_ia, fundo_ia, rotulo_ia = tema.PENDENTE, tema.PENDENTE_FUNDO, "Modo demonstração"
        modelo = "sem placa de vídeo dedicada"
    else:
        cor_ia, fundo_ia, rotulo_ia = tema.APROVADO, tema.APROVADO_FUNDO, "IA operacional"
        modelo = "Llama 3.2 · ajustado"

    chips = [
        f'<span class="chip" style="color:{cor_ia};background:{fundo_ia}">'
        f'<span class="badge" style="padding:0"></span>{rotulo_ia}</span>',
        f'<span class="chip-modelo">Modelo: {modelo}</span>',
    ]
    if pendencias:
        rotulo = "1 aguardando validação" if pendencias == 1 else f"{pendencias} aguardando validação"
        chips.insert(
            1,
            f'<span class="chip" style="color:{tema.PENDENTE};background:{tema.PENDENTE_FUNDO}">'
            f"{rotulo}</span>",
        )

    st.markdown(
        f'<div class="rodape-status">{"".join(chips)}</div>', unsafe_allow_html=True
    )


def main() -> None:
    st.markdown(tema.css(), unsafe_allow_html=True)

    pendencias = len(pendentes(carregar_registros()))
    with st.sidebar:
        barra_lateral(pendencias)

    PAGINAS[pagina_atual()]()


if __name__ == "__main__":
    main()
