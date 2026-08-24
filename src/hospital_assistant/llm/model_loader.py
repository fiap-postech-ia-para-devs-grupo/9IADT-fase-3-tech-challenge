"""Loads base_model + LoRA adapter at runtime. Placeholder for Bloco 3 (Pessoa A).

Replace `load_llm` with the real loader described in ESTRATEGIA.md §3.3:
load meta-llama/Llama-3.2-3B-Instruct plus the adapter published on the
Hugging Face Hub (via `peft`). Until an adapter is published, everything in
this repo runs against `MockLLM` so the rest of the pipeline is demoable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class LLM(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass
class MockLLM:
    """Deterministic stand-in for the fine-tuned model."""

    def generate(self, prompt: str) -> str:
        return f"[MOCK LLM] Sugestão gerada para: {prompt[:80]!r}"


def load_llm() -> LLM:
    """TODO(Bloco 3 — Pessoa A): load base model + HF Hub adapter per ESTRATEGIA.md §3.3."""
    return MockLLM()
