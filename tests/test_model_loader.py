"""Carregamento do modelo em runtime (#4) e a degradação para mock.

O ponto sensível: enquanto o adapter LoRA não está publicado, o app inteiro
precisa continuar rodando — mas a diferença entre "estou usando o modelo
fine-tunado" e "estou usando um stand-in" não pode ser silenciosa, senão o
time grava o vídeo demonstrando o mock sem perceber.
"""

from __future__ import annotations

import pytest

from hospital_assistant.llm import model_loader
from hospital_assistant.llm.model_loader import MockLLM, descrever_backend, load_llm


def test_mock_llm_e_deterministico() -> None:
    """Determinismo é o que torna os testes do grafo estáveis."""
    mock = MockLLM()

    assert mock.generate("Qual a conduta na sepse?") == mock.generate("Qual a conduta na sepse?")


def test_mock_llm_avisa_que_nao_e_o_modelo_fine_tunado() -> None:
    """O revisor precisa saber que a resposta não veio do modelo treinado."""
    resposta = MockLLM().generate("Qual a conduta na sepse?")

    assert "modo de demonstração" in resposta


def test_mock_llm_nao_expoe_jargao_de_infraestrutura() -> None:
    """A resposta é lida por médico: nada de nome de biblioteca ou variável."""
    resposta = MockLLM().generate("Qual a conduta na sepse?")

    for termo in ("bitsandbytes", "HF_ADAPTER_REPO", "quantização", "stand-in", "LoRA"):
        assert termo not in resposta


def test_mock_llm_responde_em_formato_de_artigo() -> None:
    """A fila de validação exibe este texto; bloco corrido é ilegível para revisão."""
    resposta = MockLLM().generate("Qual a conduta na sepse?")

    assert resposta.startswith("### ")
    assert "#### Base consultada" in resposta
    assert "#### Encaminhamento" in resposta


def test_mock_llm_aceita_contexto_e_exames() -> None:
    """Mesma assinatura do modelo real — o nó do grafo não muda conforme o backend."""
    resposta = MockLLM().generate(
        "Qual a conduta?",
        contexto_rag=[{"text": "t", "source": "sepse.md", "score": 0.8}],
        exames_pendentes=[{"tipo": "Hemograma"}],
    )

    assert isinstance(resposta, str)


def test_mock_llm_registra_quantas_fontes_chegaram() -> None:
    """Torna observável se o RAG alcançou o modelo — sem poluir o texto com caminhos."""
    resposta = MockLLM().generate(
        "P", contexto_rag=[{"text": "t", "source": "sepse.md", "score": 0.8}]
    )

    assert "1 trecho(s)" in resposta


def test_mock_llm_nao_repete_nome_de_arquivo_na_resposta() -> None:
    """A procedência pertence ao painel de fontes, não ao corpo da resposta."""
    resposta = MockLLM().generate(
        "P", contexto_rag=[{"text": "t", "source": "protocolos/sepse.md", "score": 0.8}]
    )

    assert "sepse.md" not in resposta
    assert "protocolos" not in resposta


def test_mock_llm_sinaliza_ausencia_de_fundamentacao() -> None:
    """Resposta sem fonte recuperada precisa dizer isso, não passar batido."""
    assert "não tem fundamentação documental" in MockLLM().generate("P")


def _sem_adapter_configurado(monkeypatch) -> None:
    """Ambiente realmente sem adapter.

    Apagar a variável não basta: `_carregar_env` relê o `.env` do projeto, que
    na máquina de quem treinou tem `HF_ADAPTER_REPO` preenchido, e a repõe. Sem
    neutralizar essa leitura os testes abaixo passavam sem testar o que dizem.
    """
    monkeypatch.setattr(model_loader, "_carregar_env", lambda: None)
    monkeypatch.delenv("HF_ADAPTER_REPO", raising=False)


def test_load_llm_sem_adapter_cai_no_mock(monkeypatch) -> None:
    _sem_adapter_configurado(monkeypatch)

    assert isinstance(load_llm(local_adapter_dir=None), MockLLM)


def test_descrever_backend_diz_qual_esta_ativo(monkeypatch) -> None:
    """String usada no log e na barra lateral para o operador saber o que roda."""
    _sem_adapter_configurado(monkeypatch)

    descricao = descrever_backend(load_llm(local_adapter_dir=None))

    assert "mock" in descricao.lower()


def test_adapter_configurado_sem_gpu_levanta(monkeypatch) -> None:
    """Produção é o padrão: degradar em silêncio já pôs o stand-in numa demo."""
    monkeypatch.setattr(model_loader, "_carregar_env", lambda: None)
    monkeypatch.setenv("HF_ADAPTER_REPO", "usuario/adapter")
    monkeypatch.delenv("MODO_DEMONSTRACAO", raising=False)
    monkeypatch.setattr(
        model_loader, "_dependencias_faltando", lambda exigir_gpu=True: ["GPU CUDA"]
    )

    with pytest.raises(model_loader.AmbienteSemModelo, match="GPU CUDA"):
        load_llm(local_adapter_dir=None)


def test_modo_demonstracao_explicito_libera_o_mock(monkeypatch) -> None:
    """Escape hatch para quem sabe que não tem GPU e quer a interface assim mesmo."""
    monkeypatch.setattr(model_loader, "_carregar_env", lambda: None)
    monkeypatch.setenv("HF_ADAPTER_REPO", "usuario/adapter")
    monkeypatch.setenv("MODO_DEMONSTRACAO", "true")
    monkeypatch.setattr(
        model_loader, "_dependencias_faltando", lambda exigir_gpu=True: ["GPU CUDA"]
    )

    assert isinstance(load_llm(local_adapter_dir=None), MockLLM)


# --- caminho de CPU ---------------------------------------------------------


def _sem_gpu_mas_com_o_resto(monkeypatch) -> None:
    """Ambiente típico de máquina sem placa: falta CUDA, o resto está instalado."""
    monkeypatch.setattr(model_loader, "_carregar_env", lambda: None)
    monkeypatch.setenv("HF_ADAPTER_REPO", "usuario/adapter")
    monkeypatch.setattr(
        model_loader,
        "_dependencias_faltando",
        lambda exigir_gpu=True: ["bitsandbytes", "GPU CUDA"] if exigir_gpu else [],
    )


def test_permitir_cpu_usa_o_modelo_real_sem_gpu(monkeypatch) -> None:
    """Sem placa, resposta real ainda é possível — só lenta."""
    _sem_gpu_mas_com_o_resto(monkeypatch)
    monkeypatch.setenv("PERMITIR_CPU", "true")

    llm = load_llm(local_adapter_dir=None)

    assert isinstance(llm, model_loader.FineTunedLLM)
    assert llm.em_cpu


def test_cpu_tem_precedencia_sobre_demonstracao(monkeypatch) -> None:
    """Com as duas flags, ganha a que dá resposta de verdade.

    O mock é último recurso: quem ligou as duas aceitou esperar, e devolver o
    stand-in nesse caso entregaria menos do que o ambiente permite.
    """
    _sem_gpu_mas_com_o_resto(monkeypatch)
    monkeypatch.setenv("PERMITIR_CPU", "true")
    monkeypatch.setenv("MODO_DEMONSTRACAO", "true")

    assert isinstance(load_llm(local_adapter_dir=None), model_loader.FineTunedLLM)


def test_sem_flag_de_cpu_continua_levantando(monkeypatch) -> None:
    """CPU é opt-in: minutos por resposta não podem ser uma surpresa."""
    _sem_gpu_mas_com_o_resto(monkeypatch)
    monkeypatch.delenv("PERMITIR_CPU", raising=False)
    monkeypatch.delenv("MODO_DEMONSTRACAO", raising=False)

    with pytest.raises(model_loader.AmbienteSemModelo, match="PERMITIR_CPU"):
        load_llm(local_adapter_dir=None)


def test_descrever_backend_distingue_cpu_de_gpu() -> None:
    """A barra lateral precisa dizer em que modo está — muda a expectativa de tempo."""
    em_cpu = descrever_backend(model_loader.FineTunedLLM(adapter="u/a", em_cpu=True))
    em_gpu = descrever_backend(model_loader.FineTunedLLM(adapter="u/a"))

    assert "CPU" in em_cpu
    assert "4-bit" in em_gpu
