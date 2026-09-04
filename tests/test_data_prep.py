"""Curadoria, deduplicação e split do dataset de fine-tuning (ESTRATEGIA.md §3).

Estas são as etapas que decidem o que o modelo vê. Erram em silêncio: um split
não determinístico invalida a comparação base vs. fine-tuned do #4, e um filtro
frouxo deixa passar resposta vazia do MedQuAD que o modelo aprende a imitar.
Por isso a lógica é pura e testada aqui, com o download isolado em `sources.py`.
"""

from __future__ import annotations

import json

from hospital_assistant.finetuning.data_prep import (
    MAX_OUTPUT_CHARS,
    curate,
    dedupe,
    read_jsonl,
    split_train_val,
    write_jsonl,
)
from hospital_assistant.finetuning.schema import InstructionExample


def exemplo(instruction: str = "Qual a conduta na sepse?", output: str | None = None) -> InstructionExample:
    return {
        "instruction": instruction,
        "input": "",
        "output": output if output is not None else "Sugiro considerar coleta de lactato e hemoculturas antes do antimicrobiano.",
    }


# --------------------------------------------------------------------------
# dedupe
# --------------------------------------------------------------------------


def test_dedupe_remove_instrucao_repetida() -> None:
    assert len(dedupe([exemplo(), exemplo()])) == 1


def test_dedupe_ignora_caixa_e_espaco() -> None:
    """O gerador sintético repete o mesmo tema com pontuação diferente."""
    duplicatas = [exemplo("Qual a conduta na sepse?"), exemplo("  qual a conduta na SEPSE?  ")]

    assert len(dedupe(duplicatas)) == 1


def test_dedupe_preserva_ordem_da_primeira_ocorrencia() -> None:
    itens = [exemplo("A pergunta sobre sepse"), exemplo("B pergunta sobre pneumonia"), exemplo("A pergunta sobre sepse")]

    resultado = dedupe(itens)

    assert [e["instruction"] for e in resultado] == ["A pergunta sobre sepse", "B pergunta sobre pneumonia"]


def test_dedupe_mantem_mesma_pergunta_com_contexto_diferente() -> None:
    """PubMedQA repete perguntas parecidas sobre abstracts distintos — não é duplicata."""
    a: InstructionExample = {**exemplo(), "input": "contexto A"}
    b: InstructionExample = {**exemplo(), "input": "contexto B"}

    assert len(dedupe([a, b])) == 2


# --------------------------------------------------------------------------
# curate
# --------------------------------------------------------------------------


def test_curate_descarta_resposta_curta_demais() -> None:
    assert curate([exemplo(output="Sim.")]) == []


def test_curate_descarta_instrucao_curta_demais() -> None:
    assert curate([exemplo(instruction="Oi?")]) == []


def test_curate_descarta_resposta_longa_demais() -> None:
    """Sequência gigante estoura a memória do T4 e domina o batch."""
    assert curate([exemplo(output="palavra " * (MAX_OUTPUT_CHARS // 4))]) == []


def test_curate_descarta_residuo_de_mock() -> None:
    assert curate([exemplo(output="[MOCK] Resposta de exemplo para substituir depois.")]) == []


def test_curate_descarta_recusa_do_gerador() -> None:
    recusa = "Desculpe, não posso ajudar com esse pedido específico neste momento."
    assert curate([exemplo(output=recusa)]) == []


def test_curate_mantem_exemplo_valido() -> None:
    assert len(curate([exemplo()])) == 1


# --------------------------------------------------------------------------
# split
# --------------------------------------------------------------------------


def test_split_respeita_proporcao_90_10() -> None:
    itens = [exemplo(f"Pergunta clínica número {i}") for i in range(100)]

    train, val = split_train_val(itens)

    assert len(train) == 90
    assert len(val) == 10


def test_split_e_deterministico() -> None:
    """Sem semente fixa, a validação muda a cada execução e a métrica do #3 vira ruído."""
    itens = [exemplo(f"Pergunta clínica número {i}") for i in range(50)]

    assert split_train_val(itens) == split_train_val(itens)


def test_split_nao_vaza_exemplo_entre_os_lados() -> None:
    itens = [exemplo(f"Pergunta clínica número {i}") for i in range(50)]

    train, val = split_train_val(itens)

    instrucoes_train = {e["instruction"] for e in train}
    assert all(e["instruction"] not in instrucoes_train for e in val)
    assert len(train) + len(val) == 50


def test_split_embaralha_antes_de_cortar() -> None:
    """Cortar os últimos 10% sem embaralhar deixa a validação com uma fonte só."""
    itens = [exemplo(f"Pergunta clínica número {i}") for i in range(100)]

    _, val = split_train_val(itens)

    ultimos_dez = {f"Pergunta clínica número {i}" for i in range(90, 100)}
    assert {e["instruction"] for e in val} != ultimos_dez


def test_split_de_dataset_minusculo_nao_esvazia_validacao() -> None:
    train, val = split_train_val([exemplo(f"Pergunta clínica número {i}") for i in range(3)])

    assert len(val) >= 1
    assert len(train) >= 1


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------


def test_write_jsonl_e_read_jsonl_fazem_roundtrip(tmp_path) -> None:
    destino = tmp_path / "train.jsonl"
    itens = [exemplo("Pergunta com acento: há sepse?"), exemplo("Outra pergunta clínica válida")]

    write_jsonl(destino, itens)

    assert read_jsonl(destino) == itens


def test_write_jsonl_grava_uma_linha_por_exemplo(tmp_path) -> None:
    destino = tmp_path / "train.jsonl"

    write_jsonl(destino, [exemplo("Primeira pergunta clínica"), exemplo("Segunda pergunta clínica")])

    linhas = destino.read_text(encoding="utf-8").strip().split("\n")
    assert len(linhas) == 2
    assert json.loads(linhas[0])["instruction"] == "Primeira pergunta clínica"


def test_write_jsonl_preserva_acento_sem_escapar(tmp_path) -> None:
    """`ensure_ascii=True` deixaria o arquivo ilegível para revisão manual da curadoria."""
    destino = tmp_path / "train.jsonl"

    write_jsonl(destino, [exemplo("Há indicação de antibiótico?")])

    assert "Há indicação" in destino.read_text(encoding="utf-8")


def test_write_jsonl_cria_diretorio_ausente(tmp_path) -> None:
    destino = tmp_path / "processed" / "train.jsonl"

    write_jsonl(destino, [exemplo()])

    assert destino.exists()
