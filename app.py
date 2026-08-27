"""Streamlit entrypoint — 3 telas, per ESTRATEGIA.md §7.

Scaffolding only: navigation works and every screen renders against mock
data so the app is demoable from day one. Pessoa D replaces each screen's
body with the real thing block by block (see the wayfinder tickets Bloco 1-4).
"""

from __future__ import annotations

import streamlit as st

from hospital_assistant.graph.flow import run
from hospital_assistant.safety.audit_log import filter_audit_rows, mock_audit_rows

st.set_page_config(page_title="Assistente Virtual Médico", layout="wide")


def tela_consulta() -> None:
    st.header("Tela 1 · Consulta ao Assistente")
    pergunta = st.text_area("Pergunta do médico")
    paciente_id = st.text_input("ID do paciente (opcional)")
    if st.button("Consultar", type="primary") and pergunta:
        resultado = run(pergunta, paciente_id or None)
        st.warning("Pendente de validação humana")
        st.json(resultado)


def tela_validacao() -> None:
    st.header("Tela 2 · Fila de Validação Humana")
    for row in mock_audit_rows():
        if row["status"] != "pendente":
            continue
        with st.expander(row["pergunta"]):
            st.write(row["resposta_llm"])
            st.caption("Fontes RAG")
            st.json(row["fontes_rag"])
            c1, c2, c3 = st.columns(3)
            c1.button("Aprovar", key=f"aprovar-{row['id']}")
            c2.button("Rejeitar", key=f"rejeitar-{row['id']}")
            c3.button("Editar", key=f"editar-{row['id']}")


def tela_auditoria() -> None:
    st.header("Tela 3 · Auditoria e Histórico")
    rows = mock_audit_rows()

    paciente_ids = {r["paciente_id"] for r in rows if r["paciente_id"]}
    paciente_options = ["todos"] + sorted(paciente_ids, key=lambda p: (0, int(p)) if p.isdigit() else (1, p))
    data_options = ["todas"] + sorted({r["timestamp"][:10] for r in rows})

    col1, col2, col3 = st.columns(3)
    status_filter = col1.selectbox("Status", ["todos", "pendente", "aprovado", "rejeitado"])
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
