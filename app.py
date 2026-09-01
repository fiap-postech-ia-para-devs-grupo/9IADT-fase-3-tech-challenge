"""Streamlit entrypoint — 3 telas, per ESTRATEGIA.md §7.

All three screens run against real data (issue #17, Bloco 4): Tela 1 calls
the real graph (`hospital_assistant.graph.flow.run`) and real patient list;
Telas 2 and 3 read the real audit trail (`audit_log.real_audit_rows`), with
Tela 2's Aprovar/Rejeitar/Editar decisions held in `st.session_state` so
Tela 3 reflects them within the same session.
"""

from __future__ import annotations

from typing import Literal

import streamlit as st

from hospital_assistant.db.patient_tools import list_patients
from hospital_assistant.graph.flow import run
from hospital_assistant.safety.audit_log import AuditRow, apply_decision, filter_audit_rows, real_audit_rows

st.set_page_config(page_title="Assistente Virtual Médico", layout="wide")

_SEM_PACIENTE = "Nenhum paciente selecionado"


def _audit_rows() -> list[AuditRow]:
    """Session-scoped audit rows, shared by Telas 2 and 3.

    Seeded once from `real_audit_rows()`; Tela 2's decisions mutate this
    directly so Tela 3 reflects them without a real persisted `auditoria`
    table.
    """
    if "audit_rows" not in st.session_state:
        st.session_state.audit_rows = real_audit_rows()
    return st.session_state.audit_rows


def _decidir(
    row_id: int,
    decisao: Literal["aprovado", "rejeitado"],
    aprovador: str | None,
    resposta_editada: str | None = None,
) -> None:
    """Apply a Tela 2 decision and rerun so the row leaves the pending list."""
    st.session_state.audit_rows = apply_decision(
        _audit_rows(), row_id, decisao, aprovador, resposta_editada=resposta_editada
    )
    st.rerun()


def tela_consulta() -> None:
    st.header("Tela 1 · Consulta ao Assistente")
    pergunta = st.text_area("Pergunta do médico")

    pacientes = list_patients()
    opcoes = {_SEM_PACIENTE: None} | {
        f"{p['nome']} ({p['prontuario']})": p["id"] for p in pacientes
    }
    escolha = st.selectbox("Paciente (opcional)", list(opcoes))
    paciente_id = opcoes[escolha]

    if st.button("Consultar", type="primary") and pergunta:
        resultado = run(pergunta, paciente_id)
        st.warning("Pendente de validação humana")
        st.json(resultado)


def tela_validacao() -> None:
    st.header("Tela 2 · Fila de Validação Humana")

    aprovador = st.text_input("Aprovador", key="aprovador_nome")

    pendentes = [row for row in _audit_rows() if row["status"] == "pendente"]
    if not pendentes:
        st.info("Nenhuma resposta pendente de validação.")
        return

    for row in pendentes:
        editando_key = f"editando-{row['id']}"
        with st.expander(row["pergunta"]):
            st.write(row["resposta_llm"])
            st.caption("Fontes RAG")
            st.json(row["fontes_rag"])

            c1, c2, c3 = st.columns(3)
            if c1.button("Aprovar", key=f"aprovar-{row['id']}"):
                _decidir(row["id"], "aprovado", aprovador or None)
            if c2.button("Rejeitar", key=f"rejeitar-{row['id']}"):
                _decidir(row["id"], "rejeitado", aprovador or None)
            if c3.button("Editar", key=f"editar-{row['id']}"):
                st.session_state[editando_key] = True

            if st.session_state.get(editando_key):
                resposta_editada = st.text_area("Editar resposta", value=row["resposta_llm"], key=f"edicao-{row['id']}")
                if st.button("Salvar edição e aprovar", key=f"salvar-{row['id']}"):
                    st.session_state[editando_key] = False
                    _decidir(row["id"], "aprovado", aprovador or None, resposta_editada=resposta_editada)


def tela_auditoria() -> None:
    st.header("Tela 3 · Auditoria e Histórico")
    rows = _audit_rows()

    paciente_ids = {r["paciente_id"] for r in rows if r["paciente_id"]}
    paciente_options = ["todos"] + sorted(paciente_ids, key=lambda p: (0, int(p)) if p.isdigit() else (1, p))
    data_options = ["todas"] + sorted({r["timestamp"][:10] for r in rows})

    col1, col2, col3 = st.columns(3)
    status_filter = col1.selectbox("Status", ["todos", "pendente", "aprovado", "rejeitado", "nao_necessaria"])
    paciente_filter = col2.selectbox("Paciente", paciente_options)
    data_filter = col3.selectbox("Data", data_options)

    filtered = filter_audit_rows(rows, status_filter, paciente_filter, data_filter)
    st.dataframe(filtered, use_container_width=True)


def main() -> None:
    st.sidebar.title("Assistente Virtual Médico")
    tela = st.sidebar.radio(
        "Navegação",
        ["Tela 1 · Consulta", "Tela 2 · Validação", "Tela 3 · Auditoria"],
    )
    if tela.startswith("Tela 1"):
        tela_consulta()
    elif tela.startswith("Tela 2"):
        tela_validacao()
    else:
        tela_auditoria()


if __name__ == "__main__":
    main()
