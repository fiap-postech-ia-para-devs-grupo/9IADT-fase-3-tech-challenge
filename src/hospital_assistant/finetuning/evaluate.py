"""Base-vs-fine-tuned comparison. Placeholder for Bloco 3 (Pessoa A).

Replace with the real comparison described in ESTRATEGIA.md §3: run 8-10 test
questions through both the base and fine-tuned model, save loss/perplexity
curves to results/finetuning_metrics.json and the side-by-side comparison to
results/eval_comparativo.json.
"""

from __future__ import annotations

from typing import TypedDict


class ComparisonRow(TypedDict):
    question: str
    base_answer: str
    finetuned_answer: str


def mock_comparison() -> list[ComparisonRow]:
    """Fake base-vs-fine-tuned rows, standing in for the real evaluation."""
    return [
        {
            "question": "Qual o protocolo para dor torácica aguda?",
            "base_answer": "[MOCK base model]",
            "finetuned_answer": "[MOCK fine-tuned model]",
        }
    ]


def evaluate() -> list[ComparisonRow]:
    """TODO(Bloco 3 — Pessoa A): implement per ESTRATEGIA.md §3."""
    return mock_comparison()
