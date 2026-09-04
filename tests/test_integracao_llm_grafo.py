"""Integração do modelo ao nó `gerar_sugestao_llm` (#5).

Antes desta integração o nó chamava `llm.generate(state["pergunta"])`: o
`contexto_rag` era recuperado pelo nó anterior, exibido na Tela 2 como
explicabilidade — e nunca chegava ao modelo. O RAG existia como enfeite. O PDF
do Tech Challenge exige "contextualizar as respostas da LLM com informações
atualizadas do paciente", então os testes abaixo travam exatamente isso: o que
o grafo recuperou tem que entrar no prompt.
"""

from __future__ import annotations

from hospital_assistant.graph import nodes
from hospital_assistant.graph.state import AssistantState
from hospital_assistant.llm.model_loader import MockLLM, get_llm


class LLMEspiao:
    """Captura os argumentos com que o nó chamou o modelo."""

    def __init__(self) -> None:
        self.chamadas: list[dict] = []

    def generate(self, pergunta, contexto_rag=None, exames_pendentes=None) -> str:
        self.chamadas.append(
            {"pergunta": pergunta, "contexto_rag": contexto_rag, "exames_pendentes": exames_pendentes}
        )
        return "sugestão"


def estado(**overrides) -> AssistantState:
    base: AssistantState = {
        "paciente_id": "1",
        "pergunta": "Qual a conduta na sepse?",
        "exames_pendentes": [],
        "contexto_rag": [],
        "sugestao_llm": "",
        "flags_seguranca": [],
        "alerta": None,
        "status": "pendente",
    }
    return {**base, **overrides}  # type: ignore[return-value]


def test_no_repassa_o_contexto_rag_ao_modelo(monkeypatch) -> None:
    espiao = LLMEspiao()
    monkeypatch.setattr(nodes, "get_llm", lambda: espiao)
    chunks = [{"text": "Coletar lactato em 3h.", "source": "sepse.md", "score": 0.8}]

    nodes.gerar_sugestao_llm(estado(contexto_rag=chunks))

    assert espiao.chamadas[0]["contexto_rag"] == chunks


def test_no_repassa_os_exames_pendentes_ao_modelo(monkeypatch) -> None:
    espiao = LLMEspiao()
    monkeypatch.setattr(nodes, "get_llm", lambda: espiao)
    exames = [{"tipo": "Hemograma", "status": "pendente"}]

    nodes.gerar_sugestao_llm(estado(exames_pendentes=exames))

    assert espiao.chamadas[0]["exames_pendentes"] == exames


def test_no_preserva_a_pergunta_original(monkeypatch) -> None:
    espiao = LLMEspiao()
    monkeypatch.setattr(nodes, "get_llm", lambda: espiao)

    nodes.gerar_sugestao_llm(estado(pergunta="Posso dar alta?"))

    assert espiao.chamadas[0]["pergunta"] == "Posso dar alta?"


def test_no_grava_a_sugestao_no_estado(monkeypatch) -> None:
    monkeypatch.setattr(nodes, "get_llm", lambda: LLMEspiao())

    novo = nodes.gerar_sugestao_llm(estado())

    assert novo["sugestao_llm"] == "sugestão"


def test_get_llm_reaproveita_a_mesma_instancia() -> None:
    """Sem cache, cada consulta da Tela 1 recarregaria o modelo do zero."""
    assert get_llm() is get_llm()


def test_fluxo_real_leva_as_fontes_do_rag_ate_a_resposta(limpar_auditoria) -> None:
    """Ponta a ponta com o MockLLM: o chunk recuperado aparece na sugestão final."""
    from hospital_assistant.graph.flow import run

    resultado = run("Qual a conduta inicial na sepse?")

    assert resultado["contexto_rag"], "o retriever não devolveu nada — Chroma não indexado?"
    fontes = {c["source"] for c in resultado["contexto_rag"]}
    assert any(fonte in resultado["sugestao_llm"] for fonte in fontes)


def test_mock_llm_continua_sendo_o_padrao_sem_adapter(monkeypatch) -> None:
    """Enquanto o adapter não é publicado, o app tem que seguir demonstrável."""
    monkeypatch.delenv("HF_ADAPTER_REPO", raising=False)
    get_llm.cache_clear()

    assert isinstance(get_llm(), MockLLM)

    get_llm.cache_clear()
