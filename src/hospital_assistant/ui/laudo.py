"""Emissão do laudo a partir de uma resposta validada.

Só respostas **aprovadas** viram laudo. Antes da aprovação não há o que assinar:
o texto é sugestão de um modelo, e o laudo é o documento que atribui aquilo a um
médico com nome e CRM. Emitir a partir de uma resposta pendente inverteria a
ordem que o projeto inteiro existe para garantir — por isso `gerar` levanta em
vez de produzir um documento com ressalva.

Fica separado da tela para poder ser testado sem subir o Streamlit, e para que o
formato do documento não dependa de detalhes de renderização.
"""

from __future__ import annotations

from typing import Any

from hospital_assistant.ui import componentes as ui


class RespostaNaoAprovada(RuntimeError):
    """Tentativa de emitir laudo de uma resposta que ninguém validou."""


def gerar(linha: dict[str, Any], paciente: dict[str, Any] | None = None) -> str:
    """Documento em markdown de uma resposta aprovada.

    `paciente` é opcional porque a consulta pode ter sido feita sem prontuário
    vinculado — nesse caso o laudo diz isso, em vez de omitir o campo e sugerir
    que houve um paciente identificado.
    """
    if linha.get("status") != "aprovado":
        raise RespostaNaoAprovada(
            "Só respostas aprovadas geram laudo. Esta está como "
            f"'{ui.nome_do_status(str(linha.get('status')))}'."
        )

    identificacao = (
        f"{paciente['nome']} — prontuário {paciente['prontuario']}"
        if paciente
        else "Consulta sem paciente vinculado"
    )

    fontes = ui.formatar_fontes(linha.get("fontes_rag"))

    return "\n".join(
        [
            "# Laudo de apoio à decisão clínica",
            "",
            f"**Paciente:** {identificacao}  ",
            f"**Emitido em:** {ui.formatar_data_hora(str(linha.get('timestamp_aprovacao') or ''))}  ",
            f"**Registro de auditoria:** nº {linha.get('id')}",
            "",
            "## Questão clínica avaliada",
            "",
            str(linha.get("pergunta", "")).strip(),
            "",
            "## Análise e conduta",
            "",
            str(linha.get("resposta_llm", "")).strip(),
            "",
            "## Fundamentação",
            "",
            f"Protocolos consultados: {fontes}",
            "",
            "## Responsável",
            "",
            f"{linha.get('aprovador') or 'não informado'}",
            "",
            "---",
            "",
            "Documento gerado por assistente de apoio à decisão clínica e validado por "
            "médico responsável. A conduta final é do profissional que assina este laudo.",
        ]
    )
