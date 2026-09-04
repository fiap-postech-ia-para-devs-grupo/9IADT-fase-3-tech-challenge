"""Formato canônico de um exemplo de instruction tuning, per ESTRATEGIA.md §3.

Vive num módulo próprio (e não em `data_prep.py`) porque `anonymize.py`,
`sources.py` e `synthetic.py` também produzem/consomem esse tipo — deixá-lo em
`data_prep.py` criaria import circular assim que o orquestrador passasse a
importar os três.
"""

from __future__ import annotations

from typing import TypedDict


class InstructionExample(TypedDict):
    """Um par instrução/resposta no formato consumido pelo SFTTrainer.

    `input` é o contexto opcional (ex.: o abstract do PubMedQA); fica string
    vazia quando a instrução já é autocontida.
    """

    instruction: str
    input: str
    output: str
