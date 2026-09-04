"""Pipeline de preparação do dataset de fine-tuning, per ESTRATEGIA.md §3.

Orquestra as quatro etapas exigidas pelo PDF ("preprocessing, anonimização e
curadoria") sobre as três fontes definidas na ESTRATEGIA:

    PubMedQA (~500)  ┐
    MedQuAD  (~300)  ├─► normalizar ─► anonimizar ─► curar ─► dedup ─► split 90/10
    sintéticos (~180)┘                                                    │
                                                                          ▼
                                              data/processed/{train,val}.jsonl

O download e a chamada ao LLM vivem em `sources.py`/`synthetic.py`; aqui ficam
só transformações puras sobre listas, que é o que os testes exercitam.

Uso:
    uv run python -m hospital_assistant.finetuning.data_prep
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from pathlib import Path

from hospital_assistant.finetuning.anonymize import anonymize_example
from hospital_assistant.finetuning.schema import InstructionExample
from hospital_assistant.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)

MIN_INSTRUCTION_CHARS = 10
MIN_OUTPUT_CHARS = 40
# Teto de caracteres da resposta. Acima disso o exemplo domina o batch e
# desperdiça o orçamento de memória do T4 sem ensinar mais nada.
MAX_OUTPUT_CHARS = 3000

VAL_RATIO = 0.1
SEED = 42

TRAIN_PATH = PROCESSED_DATA_DIR / "train.jsonl"
VAL_PATH = PROCESSED_DATA_DIR / "val.jsonl"
# O corpus sintético é versionado (data/raw/ não está no .gitignore): é ele que
# atende o entregável "dataset anonimizado ou exemplo de dados sintéticos" do
# PDF, já que data/processed/ é derivado e fica fora do Git.
SYNTHETIC_PATH = RAW_DATA_DIR / "sinteticos_finetuning.jsonl"
STATS_PATH = RESULTS_DIR / "dataset_stats.json"
# Amostra versionada do dataset *já anonimizado*. O PDF pede "dataset
# anonimizado ou exemplo de dados sintéticos" entre os entregáveis, e
# `data/processed/` inteiro é derivado e fica fora do Git — esta amostra é o
# que permite conferir a anonimização sem regenerar nada.
SAMPLE_PATH = RESULTS_DIR / "dataset_sample.jsonl"
SAMPLE_SIZE = 30

# Restos de geração que não podem entrar no treino: placeholders do
# scaffolding e recusas do provider (rate limit, filtro de conteúdo).
_LIXO = ("[mock]", "desculpe, não posso", "desculpe, nao posso", "i cannot help", "as an ai")


def dedupe(examples: list[InstructionExample]) -> list[InstructionExample]:
    """Remove exemplos com a mesma pergunta *e* o mesmo contexto.

    A chave inclui `input` porque o PubMedQA repete perguntas quase idênticas
    sobre abstracts diferentes — são exemplos distintos, não duplicatas.
    """
    vistos: set[tuple[str, str]] = set()
    resultado: list[InstructionExample] = []
    for exemplo in examples:
        chave = (
            " ".join(exemplo["instruction"].lower().split()),
            " ".join(exemplo["input"].lower().split()),
        )
        if chave in vistos:
            continue
        vistos.add(chave)
        resultado.append(exemplo)
    return resultado


def curate(examples: list[InstructionExample]) -> list[InstructionExample]:
    """Descarta exemplos curtos, longos demais ou com resíduo de mock/recusa."""
    resultado: list[InstructionExample] = []
    for exemplo in examples:
        instrucao = exemplo["instruction"].strip()
        saida = exemplo["output"].strip()

        if len(instrucao) < MIN_INSTRUCTION_CHARS:
            continue
        if not MIN_OUTPUT_CHARS <= len(saida) <= MAX_OUTPUT_CHARS:
            continue
        if any(marcador in saida.lower() for marcador in _LIXO):
            continue

        resultado.append(exemplo)
    return resultado


def split_train_val(
    examples: list[InstructionExample],
    val_ratio: float = VAL_RATIO,
    seed: int = SEED,
) -> tuple[list[InstructionExample], list[InstructionExample]]:
    """Divide em treino/validação embaralhando com semente fixa.

    Embaralha antes de cortar porque as fontes entram concatenadas: cortar a
    cauda direto deixaria a validação inteira composta de exemplos sintéticos,
    e a loss de validação do #3 mediria só uma das três fontes. A semente fixa
    mantém a comparação base vs. fine-tuned do #4 reproduzível.
    """
    embaralhados = list(examples)
    random.Random(seed).shuffle(embaralhados)

    n_val = max(1, round(len(embaralhados) * val_ratio)) if embaralhados else 0
    n_val = min(n_val, max(0, len(embaralhados) - 1))

    return embaralhados[n_val:], embaralhados[:n_val]


def write_jsonl(path: Path, examples: list[InstructionExample]) -> None:
    """Grava um exemplo por linha, em UTF-8 legível (sem escapar acentos)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as arquivo:
        for exemplo in examples:
            arquivo.write(json.dumps(exemplo, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[InstructionExample]:
    """Lê um arquivo gravado por `write_jsonl`."""
    with path.open(encoding="utf-8") as arquivo:
        return [json.loads(linha) for linha in arquivo if linha.strip()]


def _coletar_sinteticos(total: int, reaproveitar: bool) -> list[InstructionExample]:
    """Carrega o corpus sintético do disco, ou gera via LLM se ainda não existe.

    Reaproveitar por padrão evita queimar cota do free tier — e mantém o
    dataset estável entre execuções, que é o que torna o treino reproduzível.
    """
    if reaproveitar and SYNTHETIC_PATH.exists():
        existentes = read_jsonl(SYNTHETIC_PATH)
        if len(existentes) >= total:
            logger.info("Reaproveitando %d sintéticos de %s", len(existentes), SYNTHETIC_PATH)
            return existentes[:total]

    from hospital_assistant.finetuning.synthetic import generate_synthetic

    gerados = generate_synthetic(total=total)
    write_jsonl(SYNTHETIC_PATH, gerados)
    logger.info("Gravados %d sintéticos em %s", len(gerados), SYNTHETIC_PATH)
    return gerados


def prepare_dataset(
    pubmedqa_limit: int | None = None,
    medquad_limit: int | None = None,
    sinteticos: int | None = None,
    reaproveitar_sinteticos: bool = True,
) -> tuple[list[InstructionExample], list[InstructionExample]]:
    """Executa o pipeline completo e grava `data/processed/{train,val}.jsonl`.

    Devolve `(train, val)`. As estatísticas por fonte vão para
    `results/dataset_stats.json`, que alimenta a seção 3.1 do relatório técnico.
    """
    from hospital_assistant.finetuning.sources import (
        MEDQUAD_LIMIT,
        PUBMEDQA_LIMIT,
        load_medquad,
        load_pubmedqa,
    )
    from hospital_assistant.finetuning.synthetic import TOTAL_PADRAO

    por_fonte: dict[str, list[InstructionExample]] = {
        "pubmedqa": load_pubmedqa(pubmedqa_limit or PUBMEDQA_LIMIT),
        "medquad": load_medquad(medquad_limit or MEDQUAD_LIMIT),
        "sintetico": _coletar_sinteticos(sinteticos or TOTAL_PADRAO, reaproveitar_sinteticos),
    }

    brutos = Counter({fonte: len(itens) for fonte, itens in por_fonte.items()})

    # Anonimizar antes de curar: o scrubber pode encurtar uma resposta que
    # era só PII, e nesse caso ela deve cair no filtro de tamanho.
    anonimizados: list[InstructionExample] = []
    origem: list[str] = []
    for fonte, itens in por_fonte.items():
        for item in itens:
            anonimizados.append(anonymize_example(item))
            origem.append(fonte)

    curados = dedupe(curate(anonimizados))
    train, val = split_train_val(curados)

    write_jsonl(TRAIN_PATH, train)
    write_jsonl(VAL_PATH, val)
    write_jsonl(SAMPLE_PATH, train[:SAMPLE_SIZE])

    stats = {
        "brutos_por_fonte": dict(brutos),
        "total_bruto": sum(brutos.values()),
        "total_apos_curadoria_e_dedup": len(curados),
        "descartados": sum(brutos.values()) - len(curados),
        "train": len(train),
        "val": len(val),
        "parametros": {
            "min_instruction_chars": MIN_INSTRUCTION_CHARS,
            "min_output_chars": MIN_OUTPUT_CHARS,
            "max_output_chars": MAX_OUTPUT_CHARS,
            "val_ratio": VAL_RATIO,
            "seed": SEED,
        },
    }
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Dataset pronto: %d treino / %d validação", len(train), len(val))
    return train, val


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from dotenv import load_dotenv

    from hospital_assistant.paths import PROJECT_ROOT

    # `override=True` porque um `GOOGLE_API_KEY` inválido já setado no ambiente
    # do sistema tem precedência sobre o `.env` no comportamento padrão do
    # python-dotenv — falha silenciosa difícil de diagnosticar.
    load_dotenv(PROJECT_ROOT / ".env", override=True)

    prepare_dataset()
