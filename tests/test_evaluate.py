"""Comparativo base vs. fine-tuned (#4).

O comparativo é o que sustenta a seção 3.3 do relatório e o critério de
aprovação "avaliação real do modelo (não mock)". Os testes cobrem a mecânica
— mesma pergunta para os dois modelos, ordem preservada, falha de um lado não
derruba a rodada — e as métricas derivadas, que são o que transforma dez
respostas soltas em análise.
"""

from __future__ import annotations

from hospital_assistant.finetuning.evaluate import PERGUNTAS_AVALIACAO, comparar, resumir


class LLMFalso:
    def __init__(self, resposta: str, falhar: bool = False) -> None:
        self.resposta = resposta
        self.falhar = falhar
        self.perguntas_recebidas: list[str] = []

    def generate(self, pergunta, contexto_rag=None, exames_pendentes=None) -> str:
        self.perguntas_recebidas.append(pergunta)
        if self.falhar:
            raise RuntimeError("CUDA out of memory")
        return self.resposta


def test_perguntas_de_avaliacao_seguem_a_estrategia() -> None:
    """ESTRATEGIA.md §3 pede 8-10 perguntas de teste."""
    assert 8 <= len(PERGUNTAS_AVALIACAO) <= 10


def test_perguntas_sao_unicas() -> None:
    assert len(set(PERGUNTAS_AVALIACAO)) == len(PERGUNTAS_AVALIACAO)


def test_comparar_faz_a_mesma_pergunta_aos_dois_modelos() -> None:
    """Perguntas diferentes para cada lado invalidariam a comparação."""
    base, tuned = LLMFalso("resposta base"), LLMFalso("resposta tuned")

    comparar(base, tuned, perguntas=["Qual a conduta na sepse?"])

    assert base.perguntas_recebidas == tuned.perguntas_recebidas == ["Qual a conduta na sepse?"]


def test_comparar_preserva_a_ordem_das_perguntas() -> None:
    perguntas = ["primeira", "segunda", "terceira"]

    linhas = comparar(LLMFalso("b"), LLMFalso("t"), perguntas=perguntas)

    assert [linha["question"] for linha in linhas] == perguntas


def test_comparar_registra_as_duas_respostas() -> None:
    linha = comparar(LLMFalso("resposta base"), LLMFalso("resposta tuned"), perguntas=["P"])[0]

    assert linha["base_answer"] == "resposta base"
    assert linha["finetuned_answer"] == "resposta tuned"


def test_comparar_nao_aborta_quando_um_modelo_falha() -> None:
    """OOM na pergunta 7 de 10 não pode jogar fora as 6 anteriores."""
    linhas = comparar(LLMFalso("ok"), LLMFalso("", falhar=True), perguntas=["P1", "P2"])

    assert len(linhas) == 2
    assert linhas[0]["base_answer"] == "ok"
    assert "ERRO" in linhas[0]["finetuned_answer"]


def test_resumir_conta_as_perguntas() -> None:
    linhas = comparar(LLMFalso("b" * 60), LLMFalso("t" * 90), perguntas=["P1", "P2"])

    assert resumir(linhas)["perguntas"] == 2


def test_resumir_calcula_tamanho_medio_de_cada_lado() -> None:
    """Resposta que encurta demais depois do fine-tuning é sinal de overfitting."""
    resumo = resumir(comparar(LLMFalso("b" * 100), LLMFalso("t" * 50), perguntas=["P"]))

    assert resumo["tamanho_medio_base"] == 100
    assert resumo["tamanho_medio_finetuned"] == 50


def test_resumir_mede_quantas_respostas_disparam_o_guardrail() -> None:
    """A métrica que importa clinicamente: o fine-tuning ensinou a não prescrever?"""
    prescritivo = "Tome o remédio de 8 em 8 horas, dose de 500 mg."
    seguro = "Sugiro considerar avaliação médica presencial antes de qualquer conduta."

    resumo = resumir(comparar(LLMFalso(prescritivo), LLMFalso(seguro), perguntas=["P"]))

    assert resumo["respostas_que_exigem_validacao_base"] == 1
    assert resumo["respostas_que_exigem_validacao_finetuned"] == 0


def test_resumir_de_lista_vazia_nao_divide_por_zero() -> None:
    resumo = resumir([])

    assert resumo["perguntas"] == 0
    assert resumo["tamanho_medio_base"] == 0
