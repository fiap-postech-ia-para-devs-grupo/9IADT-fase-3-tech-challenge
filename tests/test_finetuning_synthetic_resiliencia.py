"""Resiliência do laço de geração sintética.

Escrito depois de um caso real: a cota do Gemini free tier estourou (429) e o
laço ficou ~20 minutos falhando em todos os lotes, dormindo entre eles e
seguindo até o teto de tentativas — sem produzir nada e sem sinalizar nada.
Erro transitório (um lote) tem que ser tolerado; provider indisponível
(todos os lotes) tem que abortar rápido e alto.
"""

from __future__ import annotations

import pytest

from hospital_assistant.finetuning import synthetic


class ChatFalso:
    """Substitui a chamada ao provider; falha as `n_falhas` primeiras vezes."""

    def __init__(self, n_falhas: int = 0, por_lote: int = 10) -> None:
        self.n_falhas = n_falhas
        self.por_lote = por_lote
        self.chamadas = 0

    def __call__(self, prompt: str) -> str:
        self.chamadas += 1
        if self.chamadas <= self.n_falhas:
            raise RuntimeError("429 ResourceExhausted: quota")
        itens = ", ".join(
            f'{{"instruction": "Pergunta clínica {self.chamadas}-{i}", "input": "", '
            f'"output": "Sugiro considerar avaliação conforme protocolo institucional."}}'
            for i in range(self.por_lote)
        )
        return f"[{itens}]"


@pytest.fixture
def sem_pausa(monkeypatch):
    monkeypatch.setattr(synthetic.time, "sleep", lambda _s: None)


def test_gera_o_total_pedido(sem_pausa, monkeypatch) -> None:
    monkeypatch.setitem(synthetic._PROVIDERS, "groq", ChatFalso())

    exemplos = synthetic.generate_synthetic(total=30, provider="groq")

    assert len(exemplos) == 30


def test_tolera_falha_isolada_de_lote(sem_pausa, monkeypatch) -> None:
    """Um 429 pontual no meio da geração não pode zerar o que já foi feito."""
    monkeypatch.setitem(synthetic._PROVIDERS, "groq", ChatFalso(n_falhas=1))

    exemplos = synthetic.generate_synthetic(total=20, provider="groq")

    assert len(exemplos) == 20


def test_aborta_quando_o_provider_falha_seguidamente(sem_pausa, monkeypatch) -> None:
    """Cota estourada = falha em todos os lotes. Tem que parar, não insistir 60 vezes."""
    chat = ChatFalso(n_falhas=10_000)
    monkeypatch.setitem(synthetic._PROVIDERS, "groq", chat)

    with pytest.raises(RuntimeError, match="falharam em sequência"):
        synthetic.generate_synthetic(total=180, provider="groq")

    assert chat.chamadas <= synthetic.MAX_FALHAS_SEGUIDAS


def test_contador_de_falhas_reseta_apos_sucesso(sem_pausa, monkeypatch) -> None:
    """Falhas alternadas com sucesso são ruído de rede, não indisponibilidade."""
    class Alternado:
        def __init__(self) -> None:
            self.chamadas = 0

        def __call__(self, prompt: str) -> str:
            self.chamadas += 1
            if self.chamadas % 2 == 1:
                raise RuntimeError("timeout")
            return (
                f'[{{"instruction": "Pergunta clínica {self.chamadas}", "input": "", '
                '"output": "Sugiro considerar avaliação conforme protocolo institucional."}]'
            )

    monkeypatch.setitem(synthetic._PROVIDERS, "groq", Alternado())

    exemplos = synthetic.generate_synthetic(total=3, provider="groq")

    assert len(exemplos) == 3


def test_provider_disponivel_prefere_groq(monkeypatch) -> None:
    """Groq é o primeiro citado em ESTRATEGIA.md §1 e o free tier aguenta o volume."""
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setenv("GOOGLE_API_KEY", "y")

    assert synthetic.provider_disponivel() == "groq"


def test_provider_disponivel_cai_para_gemini(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "y")

    assert synthetic.provider_disponivel() == "gemini"


def test_provider_disponivel_sem_chave_explica_o_que_falta(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        synthetic.provider_disponivel()
