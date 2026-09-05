"""Template de prompt compartilhado entre treino e inferência.

Este é o contrato mais frágil do fine-tuning: se `train.py` formata o exemplo
de um jeito e `model_loader.py` monta o prompt de outro, o adapter treinado
não transfere e o sintoma é "o fine-tuning não melhorou nada" — sem erro
nenhum aparecendo. Por isso as duas pontas saem da mesma função e os testes
fixam o formato.
"""

from __future__ import annotations

from hospital_assistant.finetuning.schema import InstructionExample
from hospital_assistant.llm.prompt import SYSTEM_PROMPT, build_messages, build_training_messages, filtrar_relevantes

CHUNKS = [
    {"text": "Na sepse, coletar lactato nas primeiras 3 horas.", "source": "sepse.md", "score": 0.82},
    {"text": "Hemocultura antes do antimicrobiano.", "source": "protocolos/sepse.md", "score": 0.61},
]

EXAMES = [
    {"tipo": "Hemograma completo", "status": "pendente", "data_solicitacao": "2026-08-30"},
]


def test_system_prompt_define_papel_nao_prescritivo() -> None:
    """A persona do system prompt é o guardrail de primeira linha (ESTRATEGIA.md §6)."""
    texto = SYSTEM_PROMPT.lower()

    assert "não prescreve" in texto or "nunca prescreve" in texto
    assert "validação" in texto


def test_mensagens_comecam_pelo_system() -> None:
    mensagens = build_messages("Qual a conduta na sepse?")

    assert mensagens[0]["role"] == "system"
    assert mensagens[0]["content"] == SYSTEM_PROMPT
    assert mensagens[-1]["role"] == "user"


def test_contexto_rag_entra_no_prompt_com_a_fonte() -> None:
    """O nó `consultar_protocolo` recupera os chunks — eles precisam chegar ao modelo."""
    conteudo = build_messages("Qual a conduta na sepse?", contexto_rag=CHUNKS)[-1]["content"]

    assert "coletar lactato" in conteudo
    assert "sepse.md" in conteudo


def test_exames_pendentes_entram_no_prompt() -> None:
    conteudo = build_messages("Posso dar alta?", exames_pendentes=EXAMES)[-1]["content"]

    assert "Hemograma completo" in conteudo


def test_pergunta_fica_por_ultimo() -> None:
    """Instrução no fim reduz a chance de o modelo responder sobre o contexto e ignorar a pergunta."""
    conteudo = build_messages("Qual a conduta na sepse?", contexto_rag=CHUNKS, exames_pendentes=EXAMES)[-1]["content"]

    assert conteudo.rstrip().endswith("Qual a conduta na sepse?")


def test_sem_contexto_nao_deixa_secao_vazia() -> None:
    conteudo = build_messages("Qual a conduta na sepse?")[-1]["content"]

    assert "Protocolos" not in conteudo
    assert "Exames" not in conteudo
    assert conteudo.strip() == "Qual a conduta na sepse?"


def test_contexto_vazio_equivale_a_ausente() -> None:
    assert build_messages("P", contexto_rag=[], exames_pendentes=[]) == build_messages("P")


def test_chunk_sem_score_nao_quebra() -> None:
    conteudo = build_messages("P", contexto_rag=[{"text": "t", "source": "s.md"}])[-1]["content"]

    assert "s.md" in conteudo


def test_training_messages_incluem_a_resposta_do_assistente() -> None:
    exemplo: InstructionExample = {"instruction": "Qual a conduta na sepse?", "input": "", "output": "Sugiro coletar lactato."}

    mensagens = build_training_messages(exemplo)

    assert mensagens[-1] == {"role": "assistant", "content": "Sugiro coletar lactato."}


def test_training_messages_usam_o_mesmo_bloco_de_usuario_da_inferencia() -> None:
    """Se estes dois divergirem, o adapter é treinado num formato que nunca vê em produção."""
    exemplo: InstructionExample = {"instruction": "Qual a conduta na sepse?", "input": "", "output": "resposta"}

    treino = build_training_messages(exemplo)
    inferencia = build_messages("Qual a conduta na sepse?")

    assert treino[:-1] == inferencia


def test_training_messages_colocam_o_input_como_contexto() -> None:
    """O `input` do PubMedQA é o abstract — entra como contexto, igual ao chunk do RAG."""
    exemplo: InstructionExample = {"instruction": "A droga X funciona?", "input": "Abstract: ensaio clínico...", "output": "resposta"}

    conteudo = build_training_messages(exemplo)[-2]["content"]

    assert "Abstract: ensaio clínico" in conteudo
    assert conteudo.rstrip().endswith("A droga X funciona?")


# --- relevância do contexto -------------------------------------------------
#
# Motivados por uma resposta clinicamente errada em produção: à pergunta
# "vômito, diarreia e tosse seca", o assistente sugeriu ceftriaxona e uma
# ressonância de joelho. O corpus não cobre o quadro, mas o retriever devolve
# sempre os três mais próximos — e o modelo tratou 0.38 de similaridade como
# protocolo do hospital.


def test_trecho_irrelevante_nao_entra_no_prompt() -> None:
    mensagens = build_messages(
        "Paciente com vômito, diarreia e tosse seca",
        contexto_rag=[
            {"text": "Emitir alerta quando qSOFA >= 2", "source": "sepse.md", "score": 0.384},
            {"text": "Condutas por HEART", "source": "dor_toracica.md", "score": 0.419},
        ],
    )

    conteudo = mensagens[1]["content"]
    assert "qSOFA" not in conteudo
    assert "HEART" not in conteudo


def test_sem_trecho_relevante_o_prompt_separa_geral_de_institucional() -> None:
    """Recusar seco tornaria o assistente inútil fora dos poucos protocolos.

    O problema nunca foi usar conhecimento clínico geral — foi apresentá-lo
    como protocolo do hospital. A instrução permite orientar e exige declarar
    a origem. Omitir a seção, por outro lado, deixaria o modelo achar que
    ninguém consultou nada.
    """
    mensagens = build_messages(
        "Pergunta fora do corpus",
        contexto_rag=[{"text": "x", "source": "y.md", "score": 0.2}],
    )
    conteudo = mensagens[1]["content"]

    assert "Nenhum protocolo institucional" in conteudo
    assert "conhecimento clínico geral" in conteudo
    assert "não conduta padronizada" in conteudo


def test_trecho_relevante_continua_entrando() -> None:
    mensagens = build_messages(
        "Qual a conduta na sepse?",
        contexto_rag=[{"text": "Coletar lactato", "source": "sepse.md", "score": 0.81}],
    )

    assert "Coletar lactato" in mensagens[1]["content"]


def test_contexto_de_treino_passa_sem_score() -> None:
    """No treino o contexto é o abstract do próprio exemplo: relevância dada."""
    relevantes = filtrar_relevantes([{"text": "abstract", "source": "contexto fornecido"}])

    assert len(relevantes) == 1


def test_exames_pendentes_vem_com_ressalva() -> None:
    """Sem ela o modelo sugeriu ressonância de joelho para vômito e diarreia."""
    mensagens = build_messages(
        "Paciente com vômito e diarreia",
        exames_pendentes=[{"tipo": "Ressonância magnética - joelho"}],
    )

    assert "Não presuma relação com a queixa atual" in mensagens[1]["content"]
