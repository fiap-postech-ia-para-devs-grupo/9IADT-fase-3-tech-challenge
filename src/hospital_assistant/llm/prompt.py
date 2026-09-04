"""Template de prompt único, usado no treino (`finetuning/train.py`) e na
inferência (`llm/model_loader.py`), per ESTRATEGIA.md §3 e §6.

Existe um só motivo para este módulo: **treino e inferência precisam montar o
prompt do mesmo jeito**. Um adapter LoRA aprende a responder ao formato exato
que viu no SFT; se o app monta o prompt de outra forma, o adapter degrada
silenciosamente — sem exceção, sem log, só respostas piores. Manter as duas
pontas atrás de `build_messages` torna a divergência impossível por construção,
e `tests/test_prompt.py` fixa o formato.

O bloco de usuário tem três seções opcionais, sempre nesta ordem:

    [protocolos recuperados pelo RAG]   ← src/rag/retriever.py
    [exames pendentes do paciente]      ← src/db/patient_tools.py
    <pergunta do médico>                ← sempre por último

São exatamente os dois campos que `AssistantState` já carrega (`contexto_rag`
e `exames_pendentes`) e que o nó `gerar_sugestao_llm` precisa repassar ao
modelo para atender o requisito do PDF de "contextualizar as respostas da LLM
com informações atualizadas do paciente".
"""

from __future__ import annotations

from typing import Any

from hospital_assistant.finetuning.schema import InstructionExample

Message = dict[str, str]

SYSTEM_PROMPT = (
    "Você é um assistente clínico de apoio à decisão médica de um hospital. "
    "Seu papel é SUGERIR condutas e organizar informação para o médico "
    "responsável, sempre fundamentado nos protocolos institucionais fornecidos "
    "no contexto. Você não prescreve medicamentos nem define dosagens por conta "
    "própria: toda conduta que você propõe é uma sugestão sujeita a validação e "
    "assinatura do médico responsável. Quando o contexto fornecido não sustentar "
    "uma resposta, diga isso explicitamente em vez de completar com suposição."
)


def _bloco_contexto(contexto_rag: list[dict[str, Any]]) -> str:
    linhas = ["## Protocolos e referências recuperados"]
    for i, chunk in enumerate(contexto_rag, start=1):
        fonte = chunk.get("source", "desconhecida")
        score = chunk.get("score")
        cabecalho = f"[{i}] Fonte: {fonte}"
        if isinstance(score, int | float):
            cabecalho += f" (similaridade {score:.2f})"
        linhas.append(cabecalho)
        linhas.append(str(chunk.get("text", "")).strip())
    return "\n".join(linhas)


def _bloco_exames(exames_pendentes: list[dict[str, Any]]) -> str:
    linhas = ["## Exames pendentes deste paciente"]
    for exame in exames_pendentes:
        tipo = exame.get("tipo") or exame.get("nome") or "exame"
        solicitado = exame.get("data_solicitacao")
        linhas.append(f"- {tipo}" + (f" (solicitado em {solicitado})" if solicitado else ""))
    return "\n".join(linhas)


def build_messages(
    pergunta: str,
    contexto_rag: list[dict[str, Any]] | None = None,
    exames_pendentes: list[dict[str, Any]] | None = None,
) -> list[Message]:
    """Monta as mensagens de inferência: system + user com o contexto disponível.

    Seções sem conteúdo são omitidas por completo — cabeçalho vazio ensinaria
    o modelo que "sem contexto" é um estado normal em que ele deve responder
    de memória, que é justamente o oposto do comportamento desejado.
    """
    partes: list[str] = []

    if contexto_rag:
        partes.append(_bloco_contexto(contexto_rag))
    if exames_pendentes:
        partes.append(_bloco_exames(exames_pendentes))

    partes.append(pergunta.strip())

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(partes)},
    ]


def build_training_messages(example: InstructionExample) -> list[Message]:
    """Monta as mensagens de um exemplo de treino (inclui a resposta esperada).

    O `input` do exemplo (abstract do PubMedQA, contexto do exemplo sintético)
    entra pelo mesmo caminho que os chunks do RAG na inferência — assim o
    modelo aprende uma única convenção de "contexto vem antes da pergunta".
    """
    contexto = example.get("input", "").strip()
    contexto_rag = [{"text": contexto, "source": "contexto fornecido"}] if contexto else None

    mensagens = build_messages(example["instruction"], contexto_rag=contexto_rag)
    return [*mensagens, {"role": "assistant", "content": example["output"]}]
