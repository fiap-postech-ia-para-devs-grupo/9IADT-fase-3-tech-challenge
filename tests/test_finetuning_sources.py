"""Normalização das fontes públicas (PubMedQA/MedQuAD) para o formato de instrução.

Os testes batem só nas funções puras de normalização — a linha divisória do
módulo existe justamente para que a forma de cada dataset seja verificável sem
baixar centenas de MB do Hugging Face a cada `pytest`. As fixtures abaixo são
cópias fiéis de uma linha real de cada dataset (schema conferido via
datasets-server em 2026-09-04).
"""

from __future__ import annotations

from hospital_assistant.finetuning.sources import normalize_medquad_row, normalize_pubmedqa_row

PUBMEDQA_ROW = {
    "pubid": 21645374,
    "question": "Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?",
    "context": {
        "contexts": [
            "Programmed cell death (PCD) is the regulated death of cells within an organism.",
            "The lace plant produces perforations in its leaves through PCD.",
        ],
        "labels": ["BACKGROUND", "RESULTS"],
    },
    "long_answer": "Results depicted mitochondrial dynamics in vivo as PCD progresses within the lace plant.",
    "final_decision": "yes",
}

MEDQUAD_ROW = {
    "document_id": "0000105",
    "document_source": "CDC",
    "question": "What is (are) pneumonia ?",
    "answer": "Pneumonia is an infection of the lungs that can cause mild to severe illness in people of all ages.",
}


def test_pubmedqa_vira_instrucao_com_contexto() -> None:
    exemplo = normalize_pubmedqa_row(PUBMEDQA_ROW)

    assert exemplo is not None
    assert exemplo["instruction"] == PUBMEDQA_ROW["question"]
    assert "Programmed cell death" in exemplo["input"]
    assert "perforations" in exemplo["input"]
    assert exemplo["output"] == PUBMEDQA_ROW["long_answer"]


def test_pubmedqa_trunca_contexto_muito_longo() -> None:
    """Contexto sem limite estoura o seq_len do treino no T4 — corta em fronteira de caractere."""
    row = {**PUBMEDQA_ROW, "context": {"contexts": ["palavra " * 2000]}}

    exemplo = normalize_pubmedqa_row(row)

    assert exemplo is not None
    assert len(exemplo["input"]) <= 2000


def test_pubmedqa_sem_resposta_longa_e_descartado() -> None:
    assert normalize_pubmedqa_row({**PUBMEDQA_ROW, "long_answer": ""}) is None


def test_medquad_vira_instrucao_sem_contexto() -> None:
    exemplo = normalize_medquad_row(MEDQUAD_ROW)

    assert exemplo is not None
    assert exemplo["instruction"] == "What is (are) pneumonia ?"
    assert exemplo["input"] == ""
    assert exemplo["output"] == MEDQUAD_ROW["answer"]


def test_medquad_com_resposta_vazia_e_descartado() -> None:
    """Parte do MedQuAD teve as respostas removidas por direito autoral — vêm vazias."""
    assert normalize_medquad_row({**MEDQUAD_ROW, "answer": ""}) is None
    assert normalize_medquad_row({**MEDQUAD_ROW, "answer": None}) is None


def test_medquad_sem_pergunta_e_descartado() -> None:
    assert normalize_medquad_row({**MEDQUAD_ROW, "question": ""}) is None
