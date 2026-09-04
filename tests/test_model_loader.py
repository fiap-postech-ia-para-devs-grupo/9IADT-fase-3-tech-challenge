"""Carregamento do modelo em runtime (#4) e a degradação para mock.

O ponto sensível: enquanto o adapter LoRA não está publicado, o app inteiro
precisa continuar rodando — mas a diferença entre "estou usando o modelo
fine-tunado" e "estou usando um stand-in" não pode ser silenciosa, senão o
time grava o vídeo demonstrando o mock sem perceber.
"""

from __future__ import annotations

from hospital_assistant.llm.model_loader import MockLLM, descrever_backend, load_llm


def test_mock_llm_e_deterministico() -> None:
    """Determinismo é o que torna os testes do grafo estáveis."""
    mock = MockLLM()

    assert mock.generate("Qual a conduta na sepse?") == mock.generate("Qual a conduta na sepse?")


def test_mock_llm_marca_a_resposta_como_mock() -> None:
    """A marca `[MOCK LLM]` é o sinal visível na Tela 1 de que não há adapter."""
    assert "[MOCK LLM]" in MockLLM().generate("Qual a conduta na sepse?")


def test_mock_llm_aceita_contexto_e_exames() -> None:
    """Mesma assinatura do modelo real — o nó do grafo não muda conforme o backend."""
    resposta = MockLLM().generate(
        "Qual a conduta?",
        contexto_rag=[{"text": "t", "source": "sepse.md", "score": 0.8}],
        exames_pendentes=[{"tipo": "Hemograma"}],
    )

    assert isinstance(resposta, str)


def test_mock_llm_cita_a_fonte_recuperada() -> None:
    """Sem isso a Tela 1 fica idêntica com e sem RAG, e a demo não mostra nada."""
    resposta = MockLLM().generate("P", contexto_rag=[{"text": "t", "source": "sepse.md", "score": 0.8}])

    assert "sepse.md" in resposta


def test_load_llm_sem_adapter_cai_no_mock(monkeypatch) -> None:
    monkeypatch.delenv("HF_ADAPTER_REPO", raising=False)

    assert isinstance(load_llm(local_adapter_dir=None), MockLLM)


def test_descrever_backend_diz_qual_esta_ativo(monkeypatch) -> None:
    """String usada no log e na Tela 1 para o operador saber o que está rodando."""
    monkeypatch.delenv("HF_ADAPTER_REPO", raising=False)

    descricao = descrever_backend(load_llm(local_adapter_dir=None))

    assert "mock" in descricao.lower()
