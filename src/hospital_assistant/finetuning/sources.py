"""Coleta e normalização das fontes públicas de conhecimento clínico, per
ESTRATEGIA.md §3 e a tabela "Sugestão para Datasets" do PDF do Tech Challenge.

Duas fontes, dois papéis distintos:

- **PubMedQA** (`pqa_labeled`, 1000 exemplos anotados por especialistas):
  pergunta clínica *com* o abstract como contexto. Treina exatamente o
  comportamento que o grafo exige do modelo no nó `gerar_sugestao_llm` —
  responder a partir de um contexto recuperado, não do que memorizou.
- **MedQuAD** (`lavita/MedQuAD`): pergunta/resposta autocontida de saúde,
  a partir de fontes do NIH. É o mesmo dataset de onde saiu a amostra do RAG
  em `data/raw/medquad_sample/`, então o conhecimento do fine-tuning e o do
  vector store ficam coerentes em vez de competirem.

As funções `normalize_*_row` são puras de propósito: `tests/` verifica a forma
de cada dataset sem baixar nada, e o download fica isolado em `load_*`.
"""

from __future__ import annotations

import logging
from typing import Any

from hospital_assistant.finetuning.schema import InstructionExample

logger = logging.getLogger(__name__)

PUBMEDQA_DATASET = "qiaojin/PubMedQA"
PUBMEDQA_CONFIG = "pqa_labeled"
MEDQUAD_DATASET = "lavita/MedQuAD"

# Teto de caracteres do campo `input`. O treino roda num T4 (16GB) com
# `max_seq_length` modesto; abstract inteiro do PubMedQA passa de 3000
# caracteres em boa parte das linhas e faria o batch estourar memória.
MAX_CONTEXT_CHARS = 2000

# Volumes-alvo de ESTRATEGIA.md §3.
PUBMEDQA_LIMIT = 500
MEDQUAD_LIMIT = 300


def normalize_pubmedqa_row(row: dict[str, Any]) -> InstructionExample | None:
    """Converte uma linha do PubMedQA em exemplo de instrução, ou `None` se inútil."""
    pergunta = (row.get("question") or "").strip()
    resposta = (row.get("long_answer") or "").strip()
    if not pergunta or not resposta:
        return None

    contexto_bruto = (row.get("context") or {}).get("contexts") or []
    contexto = " ".join(trecho.strip() for trecho in contexto_bruto if trecho).strip()

    return {
        "instruction": pergunta,
        "input": contexto[:MAX_CONTEXT_CHARS],
        "output": resposta,
    }


def normalize_medquad_row(row: dict[str, Any]) -> InstructionExample | None:
    """Converte uma linha do MedQuAD em exemplo de instrução, ou `None` se inútil.

    Parte do MedQuAD teve as respostas removidas por questão de direito autoral
    (as fontes CancerGov/GARD etc.) e chega com `answer` vazio ou nulo — essas
    linhas existem no dataset mas não têm nada a ensinar.
    """
    pergunta = (row.get("question") or "").strip()
    resposta = (row.get("answer") or "").strip()
    if not pergunta or not resposta:
        return None

    return {"instruction": pergunta, "input": "", "output": resposta}


def load_pubmedqa(limit: int = PUBMEDQA_LIMIT) -> list[InstructionExample]:
    """Baixa e normaliza até `limit` exemplos do PubMedQA."""
    from datasets import load_dataset

    logger.info("Baixando %s (%s)...", PUBMEDQA_DATASET, PUBMEDQA_CONFIG)
    dataset = load_dataset(PUBMEDQA_DATASET, PUBMEDQA_CONFIG, split="train")

    exemplos: list[InstructionExample] = []
    for row in dataset:
        exemplo = normalize_pubmedqa_row(dict(row))
        if exemplo is not None:
            exemplos.append(exemplo)
        if len(exemplos) >= limit:
            break

    logger.info("PubMedQA: %d exemplos aproveitados", len(exemplos))
    return exemplos


def load_medquad(limit: int = MEDQUAD_LIMIT) -> list[InstructionExample]:
    """Baixa e normaliza até `limit` exemplos do MedQuAD.

    O dataset é ordenado por documento, então varrer do começo traria centenas
    de perguntas sobre os mesmos poucos temas. O embaralhamento com semente
    fixa espalha a amostra pelas fontes (CDC, GHR, NIDDK...) mantendo a
    execução reprodutível.
    """
    from datasets import load_dataset

    logger.info("Baixando %s...", MEDQUAD_DATASET)
    dataset = load_dataset(MEDQUAD_DATASET, split="train").shuffle(seed=42)

    exemplos: list[InstructionExample] = []
    for row in dataset:
        exemplo = normalize_medquad_row(dict(row))
        if exemplo is not None:
            exemplos.append(exemplo)
        if len(exemplos) >= limit:
            break

    logger.info("MedQuAD: %d exemplos aproveitados", len(exemplos))
    return exemplos
