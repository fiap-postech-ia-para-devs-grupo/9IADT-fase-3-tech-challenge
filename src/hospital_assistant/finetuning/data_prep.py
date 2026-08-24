"""Dataset preparation for QLoRA fine-tuning. Placeholder for Bloco 1 (Pessoa A).

Replace with the real pipeline described in ESTRATEGIA.md §3: collect/filter
PubMedQA + MedQuAD subsets, generate synthetic protocols via Groq/Gemini,
normalize to {"instruction", "input", "output"}, anonymize (regex scrubber
for names/CPF/dates/record numbers), dedupe, curate, and split 90/10 into
data/processed/{train,val}.jsonl.
"""

from __future__ import annotations

from typing import TypedDict


class InstructionExample(TypedDict):
    instruction: str
    input: str
    output: str


def mock_training_examples() -> list[InstructionExample]:
    """Fake instruction-tuning examples, standing in for the real dataset."""
    return [
        {
            "instruction": "Qual o protocolo para dor torácica aguda?",
            "input": "",
            "output": "[MOCK] Encaminhar para avaliação de emergência e ECG imediato.",
        },
        {
            "instruction": "Explique o resultado de hemograma completo.",
            "input": "",
            "output": "[MOCK] Resposta de exemplo — substituir pelo dataset real.",
        },
    ]


def prepare_dataset() -> tuple[list[InstructionExample], list[InstructionExample]]:
    """Returns (train, val) splits. TODO(Bloco 1 — Pessoa A): implement per ESTRATEGIA.md §3."""
    examples = mock_training_examples()
    return examples, examples[:1]
