"""Portal Clínico — interface modular do Assistente Virtual Médico.

Entry point **paralelo** ao `app.py`. As três telas originais e suas quatro
suítes de teste continuam intactas: nada aqui as importa, e nada aqui altera
módulo compartilhado. Rodar um não interfere no outro.

    streamlit run portal.py

O que muda em relação às telas originais:

- **Navegação por módulos** em vez de três rádios soltos: Atendimento,
  Conhecimento e Registros.
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

MODULOS: dict[str, list[str]] = {
    "Atendimento": ["Assistente", "Fila de validação"],
    "Conhecimento": ["Base de conhecimento"],
    "Registros": ["Pacientes", "Auditoria"],
}


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


def modulo_assistente() -> None:
    st.markdown("### Assistente")
    st.caption(
        "As respostas são sugestões de apoio à decisão e seguem para revisão de um médico "
        "antes de valer como conduta."
    )

    historico: list[dict[str, Any]] = st.session_state.setdefault("conversa", [])

    with st.container(border=True):
        pacientes = list_patients()
        opcoes = {SEM_PACIENTE: None} | {
            f"{p['nome']} ({p['prontuario']})": p["id"] for p in pacientes
        }
        escolha = st.selectbox("Paciente em atendimento", list(opcoes), key="paciente_conversa")
        paciente_id = opcoes[escolha]

        if paciente_id:
            historico_paciente = get_patient_history(paciente_id)
            colunas = st.columns(3)
            colunas[0].markdown(
                ui.metrica(len(historico_paciente["exames"]), "Exames"), unsafe_allow_html=True
            )
            colunas[1].markdown(
                ui.metrica(len(historico_paciente["medicacoes"]), "Medicações"),
                unsafe_allow_html=True,
            )
            colunas[2].markdown(
                ui.metrica(len(historico_paciente["alertas"]), "Alertas"), unsafe_allow_html=True
            )

    for turno in historico:
        with st.chat_message(turno["papel"]):
            st.markdown(turno["texto"])
            if turno.get("fontes"):
                with st.expander(f"Fontes consultadas ({len(turno['fontes'])})"):
                    for posicao, fonte in enumerate(turno["fontes"], start=1):
                        st.markdown(ui.cartao_fonte(fonte, posicao), unsafe_allow_html=True)

    pergunta = st.chat_input("Descreva o caso ou faça uma pergunta clínica")
    if not pergunta:
        return

    historico.append({"papel": "user", "texto": pergunta, "fontes": []})

    with st.spinner("Consultando protocolos e prontuário…"):
        resultado = run(pergunta, paciente_id)

    partes = [resultado["sugestao_llm"]]
    if resultado.get("alerta"):
        partes.append(f"\n\n**Alerta emitido para a equipe:** {resultado['alerta']}")

    historico.append(
        {
            "papel": "assistant",
            "texto": "\n\n".join(partes),
            "fontes": resultado.get("contexto_rag", []),
        }
    )
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

    for linha in fila:
        with st.expander(ui.resumir_texto(linha["pergunta"], limite=110)):
            cabecalho = st.columns([3, 1])
            cabecalho[0].markdown(
                f"**Paciente:** {linha['paciente_id'] or '—'} &nbsp;·&nbsp; "
                f"**Registrado em:** {ui.formatar_data_hora(linha['timestamp'])}",
                unsafe_allow_html=True,
            )
            cabecalho[1].markdown(ui.badge_status(linha["status"]), unsafe_allow_html=True)

            st.markdown("**Resposta sugerida**")
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
            acoes = st.columns(3)

            if acoes[0].button("Aprovar", key=f"portal-aprovar-{linha['id']}", type="primary"):
                registrar_decisao(linha, "aprovado", aprovador or None)
                st.rerun()
            if acoes[1].button("Rejeitar", key=f"portal-rejeitar-{linha['id']}"):
                registrar_decisao(linha, "rejeitado", aprovador or None)
                st.rerun()
            if acoes[2].button("Editar", key=f"portal-editar-{linha['id']}"):
                st.session_state[f"editar_portal_{linha['id']}"] = True
                st.rerun()

            if editando:
                texto = st.text_area(
                    "Resposta corrigida",
                    value=linha["resposta_llm"],
                    key=f"portal-edicao-{linha['id']}",
                    height=200,
                )
                if st.button("Salvar e aprovar", key=f"portal-salvar-{linha['id']}", type="primary"):
                    st.session_state[f"editar_portal_{linha['id']}"] = False
                    registrar_decisao(linha, "aprovado", aprovador or None, resposta_editada=texto)
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
    "Assistente": modulo_assistente,
    "Fila de validação": modulo_validacao,
    "Base de conhecimento": modulo_conhecimento,
    "Pacientes": modulo_pacientes,
    "Auditoria": modulo_auditoria,
}


def main() -> None:
    st.markdown(tema.css(), unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(tema.cabecalho_marca(), unsafe_allow_html=True)

        atual = st.session_state.get("modulo_atual", "Assistente")
        for grupo, paginas in MODULOS.items():
            st.markdown(f'<div class="grupo-menu">{grupo}</div>', unsafe_allow_html=True)
            for nome in paginas:
                marcador = "primary" if nome == atual else "secondary"
                if st.button(nome, key=f"nav-{nome}", use_container_width=True, type=marcador):
                    st.session_state["modulo_atual"] = nome
                    st.rerun()

        st.markdown("---")
        pendencias = len(pendentes(carregar_registros()))
        if pendencias:
            st.markdown(
                ui.badge_status("pendente").replace(
                    "Pendente de validação", f"{pendencias} aguardando validação"
                ),
                unsafe_allow_html=True,
            )
        st.caption(f"Modelo: {descrever_backend(get_llm())}")

    PAGINAS[st.session_state.get("modulo_atual", "Assistente")]()


if __name__ == "__main__":
    main()
