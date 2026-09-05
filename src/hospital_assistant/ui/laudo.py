"""Emissão do laudo a partir de uma resposta validada.

O laudo tem três partes de origens diferentes, e a distinção é o ponto do
módulo:

- **Anamnese** — escrita pelo médico, descreve o quadro do paciente.
- **Análise e conduta** — o que o assistente sugeriu, já aprovado na fila.
- **Prescrição** — escrita pelo médico. **Não vem do modelo.** A avaliação
  comparativa deste projeto documentou que o modelo ajustado passou a devolver
  dose e posologia onde o modelo base recusava, incluindo um esquema
  clinicamente errado. Prescrição é ato médico, e num documento assinado ela
  precisa ter sido digitada por quem assina.

Só respostas **aprovadas** viram laudo, e só com anamnese e prescrição
preenchidas. Antes disso o laudo fica pendente de conclusão: um documento
incompleto que já pode ser baixado circula como se estivesse pronto.

A geração fica separada da tela para ser testável sem subir o Streamlit. A
persistência do rascunho vive aqui também porque anamnese e prescrição são
conteúdo do laudo, não estado de interface — perdê-las ao trocar de página
significaria redigitar o texto clínico inteiro.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any, TypedDict

from hospital_assistant.paths import DATA_DIR
from hospital_assistant.ui import componentes as ui

logger = logging.getLogger(__name__)

ARQUIVO = DATA_DIR / "laudos.json"


class Rascunho(TypedDict):
    anamnese: str
    prescricao: str


class RespostaNaoAprovada(RuntimeError):
    """Tentativa de emitir laudo de uma resposta que ninguém validou."""


class LaudoIncompleto(RuntimeError):
    """Faltam a anamnese ou a prescrição, que são do médico."""


# --- rascunho ---------------------------------------------------------------


def _todos() -> dict[str, Any]:
    try:
        return json.loads(ARQUIVO.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError):
        logger.exception("Laudos ilegíveis em %s; seguindo sem eles.", ARQUIVO)
        return {}


def obter_rascunho(audit_id: int) -> Rascunho:
    """Anamnese e prescrição já digitadas, ou vazias."""
    guardado = _todos().get(str(audit_id)) or {}
    return {
        "anamnese": guardado.get("anamnese", ""),
        "prescricao": guardado.get("prescricao", ""),
    }


def salvar_rascunho(audit_id: int, anamnese: str, prescricao: str) -> None:
    """Guarda o texto do médico, mesmo incompleto.

    Salvar incompleto é intencional: o médico pode escrever a anamnese, sair
    para conferir um exame e voltar. Quem decide se está pronto é `gerar`.
    """
    laudos = _todos()
    laudos[str(audit_id)] = {"anamnese": anamnese, "prescricao": prescricao}

    ARQUIVO.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=ARQUIVO.parent, delete=False, suffix=".tmp"
    ) as temporario:
        temporario.write(json.dumps(laudos, ensure_ascii=False, indent=2))
        caminho_temporario = temporario.name

    os.replace(caminho_temporario, ARQUIVO)


def esta_completo(audit_id: int) -> bool:
    """Se o laudo já tem o que só o médico pode escrever."""
    rascunho = obter_rascunho(audit_id)
    return bool(rascunho["anamnese"].strip() and rascunho["prescricao"].strip())


def limpar() -> None:
    """Remove todos os laudos. Existe para os testes começarem do zero."""
    ARQUIVO.unlink(missing_ok=True)


# --- documento --------------------------------------------------------------


def gerar(
    linha: dict[str, Any],
    paciente: dict[str, Any] | None = None,
    anamnese: str = "",
    prescricao: str = "",
) -> str:
    """Documento em markdown de uma resposta aprovada e de um laudo completo.

    `paciente` é opcional porque a consulta pode ter sido feita sem prontuário
    vinculado — nesse caso o laudo diz isso, em vez de omitir o campo e sugerir
    que houve um paciente identificado.
    """
    if linha.get("status") != "aprovado":
        raise RespostaNaoAprovada(
            "Só respostas aprovadas geram laudo. Esta está como "
            f"'{ui.nome_do_status(str(linha.get('status')))}'."
        )

    if not anamnese.strip():
        raise LaudoIncompleto("A anamnese é obrigatória e deve ser escrita pelo médico.")
    if not prescricao.strip():
        raise LaudoIncompleto("A prescrição é obrigatória e deve ser escrita pelo médico.")

    identificacao = (
        f"{paciente['nome']} — prontuário {paciente['prontuario']}"
        if paciente
        else "Consulta sem paciente vinculado"
    )
    emitido_em = ui.formatar_data_hora(str(linha.get("timestamp_aprovacao") or ""))

    return "\n".join(
        [
            "# Laudo de apoio à decisão clínica",
            "",
            f"**Paciente:** {identificacao}  ",
            f"**Emitido em:** {emitido_em}  ",
            f"**Registro de auditoria:** nº {linha.get('id')}",
            "",
            "## Anamnese",
            "",
            anamnese.strip(),
            "",
            "## Questão clínica avaliada",
            "",
            str(linha.get("pergunta", "")).strip(),
            "",
            "## Análise e conduta sugeridas pelo assistente",
            "",
            str(linha.get("resposta_llm", "")).strip(),
            "",
            "## Fundamentação",
            "",
            f"Protocolos consultados: {ui.formatar_fontes(linha.get('fontes_rag'))}",
            "",
            "## Prescrição",
            "",
            prescricao.strip(),
            "",
            "## Responsável",
            "",
            str(linha.get("aprovador") or "não informado"),
            "",
            "---",
            "",
            "A análise acima foi produzida por assistente de apoio à decisão clínica e "
            "validada por médico responsável. A anamnese e a prescrição foram redigidas "
            "pelo profissional que assina este laudo, e a conduta final é dele.",
        ]
    )
