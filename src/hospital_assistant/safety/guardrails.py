"""Prescriptive-language guardrail. Placeholder for Bloco 2 (Pessoa C).

Ships a minimal, obviously-incomplete keyword check so the graph has
something to call end-to-end. Replace with the real regex/keyword policy
described in ESTRATEGIA.md §6: block direct prescriptive language (dosages,
"tome", "prescrevo"), force reformulation + disclaimer, and record flags.
"""

from __future__ import annotations

_PRESCRIPTIVE_KEYWORDS = ("prescrevo", "tome ")


def check_guardrails(text: str) -> list[str]:
    """TODO(Bloco 2 — Pessoa C): replace with the real policy per ESTRATEGIA.md §6."""
    lowered = text.lower()
    return [kw.strip() for kw in _PRESCRIPTIVE_KEYWORDS if kw in lowered]
