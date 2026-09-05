"""Portal Clínico — interface do Assistente Virtual Médico.

Entry point único da aplicação:

    streamlit run app.py

Substitui as três telas originais (consulta, validação e auditoria), mantendo
todas as funcionalidades delas — aprovar, rejeitar, editar antes de aprovar,
e filtrar a auditoria por situação, paciente e data — e acrescentando:

- **Navegação agrupada** em vez de três rádios soltos: Assistente (assistente,
  fila de validação e base de conhecimento), Cadastro e Auditoria. A base de
  conhecimento fica sob Assistente por ser a base que fundamenta as respostas,
  não um módulo paralelo. A rota vive em `st.query_params`, então o endereço é
  compartilhável e o botão voltar funciona.
- **Assistente em formato de conversa**, com histórico na sessão, em vez de um
  formulário que devolve JSON cru.
- **Saídas formatadas**: as fontes do RAG viram cartões com score em barra, e a
  tabela de auditoria mostra datas em pt-BR, situações por extenso e o título
  do protocolo de origem — não mais `[object Object]`.
- **Tabelas paginadas** com nomes de coluna legíveis.

Duas correções de comportamento em relação às telas originais:

1. Toda resposta entra na fila de validação, e não apenas as que mencionam
   medicamento. O critério da ESTRATEGIA §12 é que nenhuma resposta chegue ao
   médico solicitante sem revisão; o guardrail sozinho só marca quando detecta
   termo de prescrição, então uma pergunta clínica comum escapava da fila.
2. A fila é lida a cada renderização. Nas telas originais os registros eram
   semeados uma única vez em `session_state`, então uma consulta feita depois
   de abrir a fila não aparecia até recarregar a página.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import streamlit as st

from hospital_assistant.db.patient_tools import (
    get_patient_history,
    list_patients,
    registrar_alerta,
    registrar_medicacao,
)
from hospital_assistant.graph.flow import run
from hospital_assistant.llm.model_loader import (
    AmbienteSemModelo,
    descrever_backend,
    get_llm,
)
from hospital_assistant.safety.audit_log import (
    AuditRow,
    apply_decision,
    filter_audit_rows,
    real_audit_rows,
)
from hospital_assistant.ui import componentes as ui
from hospital_assistant.ui import (
    conhecimento_store,
    decisoes_store,
    laudo,
    medicos_store,
    rotulos,
    tema,
)

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
        ("laudos", "Prontuário eletrônico"),
    ]),
    ("Cadastro", [
        ("pacientes", "Pacientes"),
        ("medicos", "Médicos"),
    ]),
    ("Auditoria", [
        ("auditoria", "Auditoria"),
    ]),
]

PAGINA_PADRAO = "assistente"


# ---------------------------------------------------------------------------
# Estado
# ---------------------------------------------------------------------------


def _decisoes() -> dict[int, Any]:
    """Decisões de validação já tomadas, indexadas pelo id do registro.

    Vêm do disco, e não de `session_state`: a revisão de um médico é o registro
    que dá validade clínica à resposta, e precisa sobreviver a um recarregamento
    da página, a outra aba e à próxima sessão.

    Mantidas à parte das linhas de auditoria — e não sobre uma cópia congelada
    delas — para que a lista possa ser relida do disco a cada renderização sem
    perder o que o médico já decidiu.
    """
    return decisoes_store.carregar()


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
    """Persiste a decisão, reaproveitando a regra de `apply_decision`."""
    atualizadas = apply_decision(
        [linha], linha["id"], decisao, aprovador, resposta_editada=resposta_editada
    )
    atualizada = atualizadas[0]
    decisoes_store.registrar(
        linha["id"],
        {
            "status": atualizada["status"],
            "aprovador": atualizada["aprovador"],
            "timestamp_aprovacao": atualizada["timestamp_aprovacao"],
            "resposta_llm": resposta_editada,
        },
    )


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

    O sorteio usa semente explícita para que a lista só mude quando o usuário
    pedir, e não a cada rerun do Streamlit.
    """
    itens: list[dict[str, Any]] = list(rotulos.FAQ)

    # A base cresce com o uso: perguntas já respondidas e validadas viram
    # atalho, senão o assistente ofereceria para sempre as mesmas cinco
    # perguntas do conjunto estático. Só as aprovadas, pelo mesmo motivo do
    # cache — sugerir uma pergunta cuja resposta ninguém revisou empurraria o
    # médico para um caminho não validado.
    aprovadas = {
        registro_id
        for registro_id, decisao in decisoes_store.carregar().items()
        if decisao.get("status") == "aprovado"
    }
    for entrada in conhecimento_store.listar():
        if entrada["audit_id"] in aprovadas:
            itens.append(
                {
                    "pergunta": entrada["pergunta"],
                    "resposta": entrada["resposta"],
                    "categoria": "protocolo",
                    "fonte": "Atendimento validado nº " + str(entrada["audit_id"]),
                }
            )

    random.Random(semente).shuffle(itens)
    return itens[:SUGESTOES_POR_VEZ]


def _enviar(pergunta: str, paciente_id: str | None) -> None:
    """Responde a pergunta e acrescenta o turno ao histórico da conversa.

    Antes de acionar o modelo, procura na base de conhecimento uma pergunta
    semelhante **já aprovada por um médico** para o mesmo paciente. O ganho é
    grande — dezenas de segundos viram milissegundos numa GPU, minutos viram
    milissegundos em CPU — e a resposta reaproveitada é a revisada, não a que o
    modelo produziu, porque o médico pode tê-la editado antes de aprovar.
    """
    historico: list[dict[str, Any]] = st.session_state.setdefault("conversa", [])
    historico.append({"papel": "user", "texto": pergunta, "fontes": []})

    conhecida = conhecimento_store.buscar_similar(pergunta, paciente_id)
    if conhecida is not None:
        historico.append(
            {
                "papel": "assistant",
                "texto": (
                    conhecida["resposta"]
                    + "\n\n> Resposta recuperada da base de conhecimento: já foi produzida "
                    "para uma pergunta equivalente e validada por um médico responsável."
                ),
                "fontes": [],
            }
        )
        return

    with st.spinner("Consultando protocolos e prontuário…"):
        resultado = run(pergunta, paciente_id)

    texto = resultado["sugestao_llm"]
    if resultado.get("alerta"):
        texto += f"\n\n> **Alerta emitido para a equipe:** {resultado['alerta']}"

    historico.append(
        {"papel": "assistant", "texto": texto, "fontes": resultado.get("contexto_rag", [])}
    )

    # O id vem da trilha recém-gravada pelo grafo: é ele que liga a entrada da
    # base à decisão do médico, e sem essa ligação não haveria como saber se a
    # resposta pode ser reaproveitada.
    registros = real_audit_rows()
    if registros:
        conhecimento_store.registrar(
            registros[-1]["id"],
            pergunta,
            resultado["sugestao_llm"],
            paciente_id,
            medico=st.session_state.get("medico_solicitante"),
        )


def _render_conversa(historico: list[dict[str, Any]]) -> None:
    """Desenha o histórico como conversa.

    As fontes ficam recolhidas num expander por resposta: são a justificativa da
    sugestão, consultadas quando o médico quer conferir a procedência, e abertas
    por padrão empurrariam a próxima pergunta para fora da tela.
    """
    for turno in historico:
        with st.chat_message(turno["papel"]):
            if turno["papel"] == "user":
                st.markdown(turno["texto"])
                continue

            # A resposta do modelo é texto corrido. As seções abaixo são de
            # apresentação, não de geração: nomeiam o que já existe — a análise,
            # a fundamentação e o destino da sugestão — para o revisor saber o
            # que está lendo e o que ainda falta acontecer.
            st.markdown("**Análise e conduta sugerida**")
            st.markdown(turno["texto"])

            # Mesmo critério do prompt: o que não entrou no contexto do modelo
            # não pode aparecer como fundamentação da resposta dele.
            fontes = ui.fontes_relevantes(turno.get("fontes"))
            st.markdown("**Fundamentação**")
            if fontes:
                with st.expander(f"Protocolos consultados ({len(fontes)} trechos)"):
                    for posicao, fonte in enumerate(fontes, start=1):
                        st.markdown(ui.cartao_fonte(fonte, posicao), unsafe_allow_html=True)
            else:
                st.caption(
                    "Nenhum protocolo institucional do hospital cobre esta pergunta. A "
                    "orientação acima é conhecimento clínico geral e não conduta "
                    "padronizada desta instituição."
                )

            st.markdown("**Encaminhamento**")
            st.info(
                "Esta sugestão está na fila de validação e só vale como conduta depois de "
                "aprovada por um médico responsável. O laudo é emitido a partir daí.",
                icon="🩺",
            )


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

    contexto = st.columns(2)
    with contexto[0]:
        escolha = st.selectbox(
            "Paciente em atendimento", list(por_rotulo), key="paciente_conversa"
        )
    with contexto[1]:
        # Quem pergunta e quem valida são pessoas diferentes, e a trilha só
        # guardava a segunda. Sem isto o laudo não tem como dizer de quem partiu
        # a análise.
        _seletor_de_medico("Médico solicitante", "medico_solicitante")
    paciente_id = por_rotulo[escolha]

    # Sem os cartões de exames e alertas aqui: eles pertencem ao prontuário, em
    # Cadastro › Pacientes, e no composer só empurravam a caixa de pergunta e o
    # botão de enviar para fora da primeira dobra.

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

    aprovador = _seletor_de_medico()

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

    _atendimentos_na_base(busca)


def _atendimentos_na_base(busca: str) -> None:
    """O que a base aprendeu com os atendimentos, separado do conjunto curado.

    Ficam em seção própria porque têm origem e autoridade diferentes: o FAQ é
    conteúdo revisado e versionado; isto é o que o assistente respondeu num
    atendimento. A situação de validação aparece em cada um justamente para essa
    distinção não se perder.
    """
    entradas = conhecimento_store.listar()
    if busca.strip():
        termo = busca.strip().lower()
        entradas = [e for e in entradas if termo in e["pergunta"].lower()]
    if not entradas:
        return

    decisoes = decisoes_store.carregar()

    st.markdown("#### Aprendido nos atendimentos")
    st.caption(
        "Respostas produzidas durante o uso. As validadas viram atalho no assistente e "
        "podem ser reaproveitadas; as demais ficam aqui como histórico."
    )

    for entrada in entradas[:20]:
        decisao = decisoes.get(entrada["audit_id"]) or {}
        situacao = decisao.get("status", "pendente")
        with st.container(border=True):
            cabecalho = st.columns([4, 1])
            cabecalho[0].markdown(f"**{entrada['pergunta']}**")
            cabecalho[1].markdown(ui.badge_status(situacao), unsafe_allow_html=True)
            st.markdown(ui.resumir_texto(entrada["resposta"], limite=260))
            st.caption(
                f"Atendimento nº {entrada['audit_id']}"
                + (f" · solicitado por {entrada['medico']}" if entrada.get("medico") else "")
            )


# ---------------------------------------------------------------------------
# Módulo: Pacientes
# ---------------------------------------------------------------------------


PACIENTES_POR_PAGINA = 10
DETALHE = "paciente_em_detalhe"


def _risco_atual(paciente_id: str) -> tuple[str | None, str | None]:
    """Classificação de risco do atendimento mais recente do paciente.

    Vem do laudo, e não do cadastro: risco é avaliação de um momento. Devolve
    também a data, porque uma classificação de meses atrás não diz o mesmo que a
    de hoje — e sem a data quem lê não tem como saber a diferença.
    """
    atendimentos = [
        linha
        for linha in carregar_registros()
        if linha["paciente_id"] == paciente_id and linha["status"] == "aprovado"
    ]
    for linha in sorted(atendimentos, key=lambda r: r["timestamp"], reverse=True):
        rascunho = laudo.obter_rascunho(linha["id"])
        if rascunho["risco"]:
            return rascunho["risco"], linha["timestamp"]
    return None, None


def _evolucao(historico: dict[str, Any]) -> list[dict[str, str]]:
    """Eventos do prontuário em ordem cronológica, do mais recente ao mais antigo.

    As abas separam por tipo, o que responde "quais exames ele tem" mas não
    "o que aconteceu com ele". A evolução responde a segunda: exame pedido,
    medicação iniciada e alerta aberto na mesma linha do tempo, que é como o
    caso de fato se desenrolou.
    """
    eventos: list[dict[str, str]] = []

    for exame in historico["exames"]:
        concluido = exame["status"] == "concluido"
        eventos.append(
            {
                "data": exame["data_resultado"] if concluido else exame["data_solicitacao"],
                "tipo": "Exame",
                "icone": "🧪",
                "descricao": (
                    f"{exame['tipo']} — {exame['resultado']}"
                    if concluido and exame["resultado"]
                    else f"{exame['tipo']} solicitado"
                ),
            }
        )

    for medicacao in historico["medicacoes"]:
        eventos.append(
            {
                "data": medicacao["data_inicio"],
                "tipo": "Medicação",
                "icone": "💊",
                "descricao": (
                    f"{medicacao['nome']} {medicacao['dosagem']} — {medicacao['frequencia']}"
                ),
            }
        )

    for alerta in historico["alertas"]:
        eventos.append(
            {
                "data": alerta["data"],
                "tipo": "Alerta",
                "icone": "✅" if alerta["resolvido"] else "⚠️",
                "descricao": (
                    f"{alerta['descricao']} ({alerta['severidade']})"
                    + (" — resolvido" if alerta["resolvido"] else "")
                ),
            }
        )

    return sorted(eventos, key=lambda e: e["data"] or "", reverse=True)


def modulo_pacientes() -> None:
    st.markdown("### Pacientes")

    em_detalhe = st.session_state.get(DETALHE)
    if em_detalhe is None:
        _lista_pacientes()
    else:
        _detalhe_paciente(em_detalhe)


def _lista_pacientes() -> None:
    """Grid de pacientes com o risco de cada um, paginada."""
    st.caption("Cadastro de pacientes com a classificação de risco do último atendimento.")

    pacientes = list_patients()
    if not pacientes:
        # Sem comando de terminal: quem opera a tela não administra o banco, e
        # o texto anterior expunha o módulo interno de carga de dados.
        st.warning("Nenhum paciente cadastrado. Acione a equipe técnica para carregar a base.")
        return

    # O risco é resolvido antes dos filtros porque é por ele que se filtra, e
    # calculá-lo depois obrigaria a percorrer a lista duas vezes.
    riscos = {p["id"]: _risco_atual(p["id"]) for p in pacientes}

    filtros = st.columns([3, 2])
    busca = filtros[0].text_input("Buscar", placeholder="nome ou prontuário")
    risco_filtro = filtros[1].selectbox(
        "Classificação de risco",
        ["todos", *laudo.RISCOS, "sem_classificacao"],
        format_func=lambda chave: {
            "todos": "Todas",
            "sem_classificacao": "Sem classificação",
        }.get(chave, laudo.RISCOS.get(chave, chave)),
    )

    if busca.strip():
        termo = busca.strip().lower()
        pacientes = [
            p for p in pacientes if termo in p["nome"].lower() or termo in p["prontuario"].lower()
        ]
    if risco_filtro == "sem_classificacao":
        pacientes = [p for p in pacientes if riscos[p["id"]][0] is None]
    elif risco_filtro != "todos":
        pacientes = [p for p in pacientes if riscos[p["id"]][0] == risco_filtro]

    if not pacientes:
        st.info("Nenhum paciente encontrado para esses filtros.")
        return

    pagina = st.session_state.get("pagina_pacientes", 1)
    itens, total_paginas = ui.paginar(pacientes, pagina, PACIENTES_POR_PAGINA)

    for paciente in itens:
        risco, avaliado_em = riscos[paciente["id"]]
        # Alertas só da página visível: são uma consulta por paciente, e fazer
        # isso para o cadastro inteiro custaria caro para exibir dez linhas.
        abertos = [
            alerta
            for alerta in get_patient_history(paciente["id"])["alertas"]
            if not alerta["resolvido"]
        ]

        with st.container(border=True):
            colunas = st.columns([3, 2, 2, 2, 2])

            # Todas as colunas seguem "legenda + um valor". Misturar colunas de
            # uma linha com colunas de duas fazia cada cartão ter uma altura, e
            # a lista ficava serrilhada.
            colunas[0].caption("Paciente")
            colunas[0].markdown(f"**{paciente['nome']}**")

            colunas[1].caption("Prontuário")
            colunas[1].markdown(f"`{paciente['prontuario']}`")

            colunas[2].caption("Classificação de risco")
            colunas[2].markdown(ui.badge_risco(risco, avaliado_em), unsafe_allow_html=True)

            colunas[3].caption("Alertas abertos")
            colunas[3].markdown(ui.badge_alertas(abertos), unsafe_allow_html=True)

            colunas[4].caption("&nbsp;", unsafe_allow_html=True)
            if colunas[4].button(
                "Ver detalhes", key=f"detalhe-{paciente['id']}", use_container_width=True
            ):
                st.session_state[DETALHE] = paciente["id"]
                st.rerun()

    if total_paginas > 1:
        navegacao = st.columns([1, 2, 1])
        if navegacao[0].button("← Anteriores", disabled=pagina <= 1):
            st.session_state["pagina_pacientes"] = pagina - 1
            st.rerun()
        navegacao[1].markdown(
            f"<div style='text-align:center;color:{tema.TEXTO_TENUE};font-size:.82rem'>"
            f"Página {min(pagina, total_paginas)} de {total_paginas}</div>",
            unsafe_allow_html=True,
        )
        if navegacao[2].button("Próximos →", disabled=pagina >= total_paginas):
            st.session_state["pagina_pacientes"] = pagina + 1
            st.rerun()


def _detalhe_paciente(paciente_id: str) -> None:
    """Prontuário completo de um paciente: indicadores, evolução e as tabelas."""
    if st.button("← Voltar para os pacientes"):
        st.session_state[DETALHE] = None
        st.rerun()

    historico = get_patient_history(paciente_id)
    risco, avaliado_em = _risco_atual(paciente_id)

    st.markdown(
        f"#### {historico['nome']} &nbsp;·&nbsp; prontuário `{historico['prontuario']}`",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Nascimento em {ui.formatar_data_hora(historico['data_nascimento']).split(' ')[0]}"
    )

    indicadores = st.columns(4)
    indicadores[0].markdown(ui.metrica(len(historico["exames"]), "Exames"), unsafe_allow_html=True)
    indicadores[1].markdown(
        ui.metrica(len(historico["medicacoes"]), "Medicações"), unsafe_allow_html=True
    )
    indicadores[2].markdown(
        ui.metrica(
            len([a for a in historico["alertas"] if not a["resolvido"]]), "Alertas abertos"
        ),
        unsafe_allow_html=True,
    )
    indicadores[3].markdown(ui.cartao_risco(risco, avaliado_em), unsafe_allow_html=True)

    abas = st.tabs(["Evolução", "Exames", "Medicações", "Alertas"])

    with abas[0]:
        eventos = _evolucao(historico)
        if not eventos:
            st.info("Nenhum evento registrado para este paciente.")
        for evento in eventos:
            with st.container(border=True):
                colunas = st.columns([1, 2, 7])
                colunas[0].markdown(evento["icone"])
                colunas[1].caption(ui.formatar_data_hora(evento["data"]).split(" ")[0])
                colunas[2].markdown(f"**{evento['tipo']}** · {evento['descricao']}")

    with abas[1]:
        st.dataframe(
            ui.tabela_generica(
                historico["exames"],
                ["tipo", "status", "data_solicitacao", "data_resultado", "resultado"],
            ),
            use_container_width=True,
            hide_index=True,
        )
    with abas[2]:
        st.dataframe(
            ui.tabela_generica(
                historico["medicacoes"], ["nome", "dosagem", "frequencia", "data_inicio"]
            ),
            use_container_width=True,
            hide_index=True,
        )
    with abas[3]:
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


def _enriquecer(linhas: list[AuditRow]) -> list[dict[str, Any]]:
    """Acrescenta às linhas da trilha o que ela não guarda.

    A auditoria grava o **id** do paciente e o nome de quem aprovou. Quem
    perguntou e como o paciente se chama estão em outros lugares — o cadastro e
    o registro do atendimento —, e sem eles a tela mostrava "1" na coluna
    Paciente e nada sobre o solicitante.
    """
    pacientes = {p["id"]: p for p in list_patients()}

    enriquecidas: list[dict[str, Any]] = []
    for linha in linhas:
        paciente = pacientes.get(linha["paciente_id"] or "")
        consulta = conhecimento_store.obter(linha["id"])
        enriquecidas.append(
            {
                **linha,
                "paciente": (
                    f"{paciente['nome']} ({paciente['prontuario']})" if paciente else None
                ),
                "medico_solicitante": consulta["medico"] if consulta else None,
                # O laudo é o desfecho da análise, não um evento separado na
                # trilha: marcá-lo aqui evita duplicar a linha só para dizer que
                # o documento saiu.
                "tipo_operacao": (
                    "Análise + laudo" if laudo.esta_completo(linha["id"]) else "Análise"
                ),
            }
        )
    return enriquecidas


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

    # Filtro da trilha de segurança, não reimplementado aqui: a regra de quais
    # linhas a auditoria mostra é contrato testado em test_audit_log.py, e
    # duplicá-la faria as duas versões divergirem silenciosamente.
    filtrados = filter_audit_rows(
        registros,
        status="todos" if situacao == "todas" else situacao,
        paciente_id=paciente,
        data=data,
    )

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

    st.dataframe(
        ui.tabela_auditoria(_enriquecer(itens)), use_container_width=True, hide_index=True
    )

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

def _seletor_de_medico(rotulo: str = "Médico responsável pela validação",
                       chave: str = "aprovador_portal") -> str | None:
    """Um médico do cadastro, identificado por nome e CRM.

    O campo era texto livre: qualquer nome, sem conferência, gravado na trilha
    como responsável clínico pela aprovação. Numa trilha de auditoria o
    responsável precisa ser alguém que existe e que se possa localizar depois —
    daí o CRM junto do nome.
    """
    medicos = medicos_store.listar(apenas_ativos=True)
    if not medicos:
        st.warning("Nenhum médico ativo no cadastro. Cadastre um em Cadastro › Médicos.")
        return None

    escolha = st.selectbox(
        rotulo,
        ["Selecione…", *(f"{m['nome']} · {m['crm']}" for m in medicos)],
        key=chave,
    )
    return None if escolha == "Selecione…" else escolha


# ---------------------------------------------------------------------------
# Módulo: Médicos
# ---------------------------------------------------------------------------


def modulo_medicos() -> None:
    st.markdown("### Médicos")
    st.caption(
        "Quem pode validar respostas do assistente. O nome escolhido aqui é o que fica "
        "gravado na trilha de auditoria como responsável pela aprovação."
    )

    with st.expander("Cadastrar médico"):
        with st.form("cadastro_medico", clear_on_submit=True):
            campos = st.columns([2, 1, 2])
            nome = campos[0].text_input("Nome")
            crm = campos[1].text_input("CRM", placeholder="CRM-SP 000000")
            especialidade = campos[2].selectbox("Especialidade", medicos_store.ESPECIALIDADES)

            if st.form_submit_button("Cadastrar", type="primary"):
                try:
                    criado = medicos_store.criar(nome, crm, especialidade)
                except ValueError as erro:
                    st.error(str(erro))
                else:
                    st.success(f"{criado['nome']} cadastrado.")
                    st.rerun()

    medicos = medicos_store.listar()
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Nome": m["nome"],
                    "CRM": m["crm"],
                    "Especialidade": m["especialidade"],
                    "Situação": "Ativo" if m.get("ativo", True) else "Inativo",
                }
                for m in medicos
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Inativar em vez de excluir: o nome do validador fica na trilha de
    # auditoria, e apagar o cadastro deixaria registros apontando para ninguém.
    alvo = st.selectbox(
        "Ativar ou inativar",
        ["Selecione…", *(f"{m['nome']} · {m['crm']}" for m in medicos)],
        key="alvo_medico",
    )
    if alvo != "Selecione…" and st.button("Alternar situação"):
        indice = [f"{m['nome']} · {m['crm']}" for m in medicos].index(alvo)
        medicos_store.alternar_ativo(medicos[indice]["id"])
        st.rerun()


# ---------------------------------------------------------------------------
# Módulo: Laudos
# ---------------------------------------------------------------------------


# Qual laudo está sendo editado. `None` mostra a listagem — a tela abre pelo que
# já existe, e emitir é uma ação a partir dali, não o estado inicial.
EDITANDO = "laudo_em_edicao"
LAUDOS_POR_PAGINA = 3


def modulo_laudos() -> None:
    st.markdown("### Prontuário eletrônico")

    em_edicao = st.session_state.get(EDITANDO)
    if em_edicao is None:
        _lista_laudos()
    else:
        _formulario_laudo(em_edicao)


def _situacao(linha: AuditRow) -> str:
    """Em que pé está o laudo de uma análise validada."""
    rascunho = laudo.obter_rascunho(linha["id"])
    paciente_id = linha["paciente_id"] or rascunho["paciente_id"]
    if laudo.esta_completo(linha["id"], paciente_id):
        return "laudo_concluido"
    if rascunho["anamnese"] or rascunho["prescricao"]:
        return "laudo_pendente"
    return "sem_laudo"


def _lista_laudos() -> None:
    """Laudos já iniciados, com a emissão de um novo como ação do topo."""
    validadas = [linha for linha in carregar_registros() if linha["status"] == "aprovado"]
    situacoes = {linha["id"]: _situacao(linha) for linha in validadas}
    com_laudo = [linha for linha in validadas if situacoes[linha["id"]] != "sem_laudo"]
    disponiveis = [linha for linha in validadas if situacoes[linha["id"]] == "sem_laudo"]

    st.caption(
        "Registro clínico dos atendimentos: anamnese, classificação de risco, conduta "
        "validada e prescrição do médico responsável."
    )

    acao = st.columns([1, 3])
    if acao[0].button(
        "Emitir laudo", type="primary", use_container_width=True, disabled=not disponiveis
    ):
        st.session_state[EDITANDO] = disponiveis[0]["id"]
        st.rerun()
    if not disponiveis:
        acao[1].caption("Todas as análises validadas já têm laudo. Aprove outra na fila.")

    if not com_laudo:
        st.info("Nenhum registro ainda. Use **Emitir laudo** para abrir o primeiro.")
        return

    pacientes = {p["id"]: p for p in list_patients()}

    # Cada laudo ocupa uma linha alta, com botões e expander. Passando de três a
    # tela vira rolagem longa e o botão de emitir sai de vista.
    pagina = st.session_state.get("pagina_laudos", 1)
    itens, total_paginas = ui.paginar(com_laudo, pagina, LAUDOS_POR_PAGINA)

    for linha in itens:
        _linha_da_grid(linha, pacientes, situacoes[linha["id"]])

    if total_paginas > 1:
        navegacao = st.columns([1, 2, 1])
        if navegacao[0].button("← Anteriores", disabled=pagina <= 1):
            st.session_state["pagina_laudos"] = pagina - 1
            st.rerun()
        navegacao[1].markdown(
            f"<div style='text-align:center;color:{tema.TEXTO_TENUE};font-size:.82rem'>"
            f"Página {min(pagina, total_paginas)} de {total_paginas}</div>",
            unsafe_allow_html=True,
        )
        if navegacao[2].button("Próximos →", disabled=pagina >= total_paginas):
            st.session_state["pagina_laudos"] = pagina + 1
            st.rerun()


def _linha_da_grid(linha: AuditRow, pacientes: dict[str, Any], situacao: str) -> None:
    """Uma linha da listagem, com as ações de abrir, visualizar e baixar.

    Cada linha é um container e não uma linha de `st.dataframe` porque a tabela
    do Streamlit não comporta botão por registro — e sem botão não haveria como
    ver ou baixar o documento sem sair da listagem.
    """
    registro_id = linha["id"]
    rascunho = laudo.obter_rascunho(registro_id)
    paciente = pacientes.get(linha["paciente_id"] or rascunho["paciente_id"] or "")
    concluido = situacao == "laudo_concluido"

    with st.container(border=True):
        colunas = st.columns([3, 2, 2, 1, 1])
        colunas[0].markdown(f"**{paciente['nome'] if paciente else 'Paciente não definido'}**")
        colunas[0].caption(ui.resumir_texto(linha["pergunta"], limite=70))
        colunas[1].caption("Validado por")
        colunas[1].markdown(linha["aprovador"] or "—")
        colunas[2].markdown(ui.badge_status(situacao), unsafe_allow_html=True)
        colunas[2].caption(ui.formatar_data_hora(str(linha["timestamp_aprovacao"] or "")))

        if colunas[3].button("Abrir", key=f"abrir-{registro_id}", use_container_width=True):
            st.session_state[EDITANDO] = registro_id
            st.rerun()

        if not concluido:
            colunas[4].caption("incompleto")
            return

        argumentos = (
            dict(linha),
            paciente,
            rascunho["anamnese"],
            rascunho["prescricao"],
            rascunho["risco"],
        )
        colunas[4].download_button(
            "Baixar",
            data=laudo.gerar_pdf(*argumentos),
            file_name=f"laudo-{registro_id}.pdf",
            mime="application/pdf",
            key=f"baixar-{registro_id}",
            use_container_width=True,
        )
        with st.expander("Visualizar"):
            st.markdown(laudo.gerar(*argumentos))


def _formulario_laudo(registro_id: int) -> None:
    """Emissão e edição de um laudo."""
    validadas = [linha for linha in carregar_registros() if linha["status"] == "aprovado"]
    por_id = {linha["id"]: linha for linha in validadas}

    if st.button("← Voltar para o prontuário"):
        st.session_state[EDITANDO] = None
        st.rerun()

    if registro_id not in por_id:
        st.warning("Essa análise não está mais disponível para laudo.")
        return

    # Trocar de análise sem voltar à listagem. As opções são as que ainda não
    # viraram laudo, mais a que está aberta.
    disponiveis = [
        linha
        for linha in validadas
        if _situacao(linha) == "sem_laudo" or linha["id"] == registro_id
    ]
    rotulos_laudo = {
        f"nº {linha['id']} · {ui.resumir_texto(linha['pergunta'], limite=60)}": linha["id"]
        for linha in disponiveis
    }
    atual = next(rotulo for rotulo, i in rotulos_laudo.items() if i == registro_id)
    escolha = st.selectbox(
        "Análise validada", list(rotulos_laudo), index=list(rotulos_laudo).index(atual)
    )
    if rotulos_laudo[escolha] != registro_id:
        st.session_state[EDITANDO] = rotulos_laudo[escolha]
        st.rerun()

    linha = por_id[registro_id]
    rascunho = laudo.obter_rascunho(registro_id)
    consulta = conhecimento_store.obter(registro_id)

    # O paciente do laudo é o da consulta; quando ela foi feita sem prontuário
    # vinculado, o médico escolhe aqui. Um laudo é um documento sobre alguém.
    paciente_id = linha["paciente_id"] or rascunho["paciente_id"]
    if not linha["paciente_id"]:
        por_rotulo = {f"{p['nome']} ({p['prontuario']})": p["id"] for p in list_patients()}
        anterior = next(
            (rotulo for rotulo, i in por_rotulo.items() if i == rascunho["paciente_id"]), None
        )
        opcoes = ["Selecione…", *por_rotulo]
        selecionado = st.selectbox(
            "Paciente do laudo",
            opcoes,
            index=opcoes.index(anterior) if anterior else 0,
            help="A consulta foi feita sem prontuário vinculado.",
        )
        paciente_id = por_rotulo.get(selecionado)

    paciente = {p["id"]: p for p in list_patients()}.get(paciente_id or "")
    completo = bool(paciente) and laudo.esta_completo(registro_id, paciente_id)

    st.markdown(
        ui.badge_status("laudo_concluido" if completo else "laudo_pendente"),
        unsafe_allow_html=True,
    )
    st.caption(
        "Pronto para emissão."
        if completo
        else "Faltam dados que só o médico preenche: paciente, anamnese e prescrição."
    )

    # Cabeçalho com quem é quem. A trilha guarda quem aprovou; quem solicitou
    # vem do registro do atendimento.
    with st.container(border=True):
        identificacao = st.columns(3)
        identificacao[0].markdown("**Paciente**")
        identificacao[0].markdown(paciente["nome"] if paciente else "— a selecionar")
        identificacao[1].markdown("**Solicitado por**")
        identificacao[1].markdown((consulta or {}).get("medico") or "— não informado")
        identificacao[2].markdown("**Validado por**")
        identificacao[2].markdown(linha["aprovador"] or "—")

    # Resumo em destaque: quem emite o laudo precisa reler o que foi analisado
    # sem sair da tela para conferir na fila.
    st.markdown("#### Resumo da análise")
    st.info(f"**Questão avaliada:** {linha['pergunta']}", icon="🔎")
    with st.container(border=True):
        st.markdown(ui.resumir_texto(linha["resposta_llm"], limite=600))
        st.caption(f"Fundamentação: {ui.formatar_fontes(linha['fontes_rag'])}")

    # Anamnese e prescrição são digitadas, nunca sugeridas. A avaliação
    # comparativa mostrou o modelo ajustado devolvendo dose e posologia onde o
    # base recusava — num documento assinado, essa parte precisa ter saído de
    # quem assina.
    with st.form(f"laudo-{registro_id}"):
        risco = st.selectbox(
            "Classificação de risco",
            ["Não classificado", *laudo.RISCOS],
            index=(
                list(laudo.RISCOS).index(rascunho["risco"]) + 1 if rascunho["risco"] else 0
            ),
            format_func=lambda chave: laudo.RISCOS.get(chave, "Não classificado"),
        )
        anamnese = st.text_area(
            "Anamnese",
            value=rascunho["anamnese"],
            height=140,
            placeholder="Quadro clínico, história e exame físico.",
        )
        prescricao = st.text_area(
            "Prescrição",
            value=rascunho["prescricao"],
            height=140,
            placeholder="Medicação, dose, via e posologia, sob responsabilidade do prescritor.",
        )
        alerta = st.text_input(
            "Alerta para a equipe (opcional)",
            value=rascunho["alerta"] or "",
            placeholder="Ex.: reavaliar em 6 h; risco de deterioração.",
        )

        # O prontuário guarda medicação em campos separados, e a prescrição é
        # texto livre. Quebrar o texto por heurística produziria registro
        # clínico errado, então o que vai para o histórico é digitado à parte —
        # e só se o médico quiser.
        st.caption("Registrar no prontuário do paciente (opcional)")
        registro = st.columns(3)
        med_nome = registro[0].text_input("Medicamento", placeholder="Ceftriaxona")
        med_dose = registro[1].text_input("Dose", placeholder="1 g EV")
        med_freq = registro[2].text_input("Frequência", placeholder="12/12h")

        if st.form_submit_button("Salvar laudo", type="primary"):
            # Salva mesmo incompleto: o médico pode escrever a anamnese, sair
            # para conferir um exame e voltar. Voltar para a listagem depois de
            # salvar é o que fecha o ciclo — de lá ele vê o que acabou de fazer.
            laudo.salvar_rascunho(
                registro_id,
                anamnese,
                prescricao,
                paciente_id,
                risco if risco in laudo.RISCOS else None,
                alerta.strip() or None,
            )
            _registrar_no_prontuario(paciente_id, med_nome, med_dose, med_freq, alerta, risco)
            st.session_state[EDITANDO] = None
            st.rerun()


def _registrar_no_prontuario(
    paciente_id: str | None,
    medicamento: str,
    dose: str,
    frequencia: str,
    alerta: str,
    risco: str,
) -> None:
    """Leva prescrição e alerta do laudo para o histórico do paciente.

    Sem isto o laudo diria uma coisa e o prontuário outra — e é o prontuário que
    a próxima consulta ao assistente vai ler.

    A severidade do alerta vem da classificação de risco do atendimento: são o
    mesmo julgamento clínico, e pedir os dois separadamente abriria espaço para
    se contradizerem no mesmo documento.
    """
    if not paciente_id:
        return

    hoje = datetime.now(UTC).date().isoformat()

    if medicamento.strip():
        registrar_medicacao(
            paciente_id, medicamento.strip(), dose.strip() or "—", frequencia.strip() or "—", hoje
        )
        st.toast(f"{medicamento.strip()} registrado no prontuário.")

    if alerta.strip():
        severidade = {"vermelho": "alta", "amarelo": "media"}.get(risco, "baixa")
        registrar_alerta(paciente_id, alerta.strip(), severidade, hoje)
        st.toast("Alerta registrado no prontuário.")


PAGINAS = {
    "assistente": modulo_assistente,
    "validacao": modulo_validacao,
    "conhecimento": modulo_conhecimento,
    "laudos": modulo_laudos,
    "pacientes": modulo_pacientes,
    "medicos": modulo_medicos,
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

    # O modelo é resolvido antes de qualquer tela. Se o ambiente não puder
    # carregá-lo, a aplicação inteira recusa em vez de abrir parcialmente: o
    # assistente responderia com o stand-in e as telas de fila e auditoria
    # arquivariam essas respostas como se fossem do modelo treinado.
    try:
        # O spinner não é enfeite: a primeira carga baixa os pesos e monta o
        # modelo na GPU, e sem ele a tela fica em branco por minutos sem dizer
        # se está trabalhando ou travada.
        # Aquecer aqui, e não deixar para a primeira pergunta: `get_llm` só
        # escolhe o backend, e os pesos subiam dentro da consulta — o médico
        # esperava minutos achando que a pergunta dele é que era lenta.
        backend = get_llm()
        if hasattr(backend, "aquecer"):
            painel = st.empty()

            def mostrar(fracao: float, rotulo: str) -> None:
                painel.markdown(ui.anel_progresso(fracao, rotulo), unsafe_allow_html=True)

            mostrar(0.0, "Preparando o modelo clínico")
            backend.aquecer(mostrar)
            painel.empty()
    except AmbienteSemModelo as erro:
        st.error(str(erro), icon="⛔")
        st.caption(
            "Nenhuma tela é aberta neste estado para que nada seja registrado na auditoria "
            "como se viesse do modelo treinado."
        )
        return

    pendencias = len(pendentes(carregar_registros()))
    with st.sidebar:
        barra_lateral(pendencias)

    PAGINAS[pagina_atual()]()


if __name__ == "__main__":
    main()
