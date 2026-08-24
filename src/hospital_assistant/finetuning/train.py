"""QLoRA training loop. Placeholder for Bloco 2 (Pessoa A).

This module documents the intended entry point; the actual training runs in
notebooks/finetuning_colab.ipynb on a Colab T4 GPU (4-bit NF4 quantization,
LoRA r=16, base model meta-llama/Llama-3.2-3B-Instruct — see ESTRATEGIA.md §3).
Not meant to run inside the devcontainer (CPU-only, no bitsandbytes installed
by default — see the `finetuning` extra in pyproject.toml).
"""

from __future__ import annotations


def train() -> None:
    """TODO(Bloco 2 — Pessoa A): implement per ESTRATEGIA.md §3 (LoraConfig, SFTTrainer)."""
    raise NotImplementedError(
        "Fine-tuning runs in notebooks/finetuning_colab.ipynb on Colab GPU, not here."
    )
