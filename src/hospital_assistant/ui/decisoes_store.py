"""Persistência das decisões de validação humana.

A trilha de auditoria registra a pergunta, a resposta e as sinalizações do
guardrail — mas não a decisão do médico. `apply_decision` devolve a linha
atualizada e deixa a cargo de quem chama guardá-la, e as telas guardavam em
`st.session_state`. Um F5 apagava a revisão: o sistema cujo argumento central é
"nenhuma resposta chega ao médico sem revisão rastreável" perdia justamente a
rastreabilidade da revisão.

**Arquivo próprio, e não a trilha de auditoria.** Aquele arquivo é append-only e
pertence à trilha de segurança, com formato consumido por outros módulos;
acrescentar um tipo de evento ali mudaria um contrato de outra trilha. Um
arquivo separado, de responsabilidade da camada de apresentação, resolve a
persistência sem reescrever nada de ninguém.

A escrita é atômica (grava num temporário e renomeia) porque a fila é usada por
mais de uma aba ao mesmo tempo: uma escrita interrompida no meio deixaria um
JSON truncado, e a fila inteira sumiria na próxima leitura.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any, TypedDict

from hospital_assistant.paths import DATA_DIR

logger = logging.getLogger(__name__)

ARQUIVO = DATA_DIR / "decisoes_validacao.json"


class Decisao(TypedDict):
    status: str
    aprovador: str | None
    timestamp_aprovacao: str | None
    resposta_llm: str | None


def carregar() -> dict[int, Decisao]:
    """Decisões já tomadas, indexadas pelo id do registro de auditoria.

    Um arquivo ausente ou corrompido devolve vazio em vez de estourar: perder o
    histórico de decisões é ruim, mas derrubar a tela de validação inteira é
    pior — sem ela ninguém consegue nem revisar as pendentes.
    """
    try:
        bruto = json.loads(ARQUIVO.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError):
        logger.exception("Decisões de validação ilegíveis em %s; seguindo sem elas.", ARQUIVO)
        return {}

    # As chaves voltam do JSON como texto; o id da auditoria é inteiro.
    return {int(chave): valor for chave, valor in bruto.items()}


def registrar(registro_id: int, decisao: Decisao) -> None:
    """Grava uma decisão, preservando as demais."""
    decisoes: dict[int, Any] = dict(carregar())
    decisoes[registro_id] = decisao

    ARQUIVO.parent.mkdir(parents=True, exist_ok=True)
    conteudo = json.dumps(
        {str(chave): valor for chave, valor in decisoes.items()},
        ensure_ascii=False,
        indent=2,
    )

    # `delete=False` + `os.replace`: o rename é atômico no mesmo sistema de
    # arquivos, então nenhum leitor jamais vê o arquivo pela metade.
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=ARQUIVO.parent, delete=False, suffix=".tmp"
    ) as temporario:
        temporario.write(conteudo)
        caminho_temporario = temporario.name

    os.replace(caminho_temporario, ARQUIVO)


def limpar() -> None:
    """Remove todas as decisões. Existe para os testes começarem do zero."""
    ARQUIVO.unlink(missing_ok=True)
