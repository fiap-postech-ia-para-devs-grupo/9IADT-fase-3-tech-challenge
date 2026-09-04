"""Template de prompt compartilhado entre treino e inferência.

Este é o contrato mais frágil do fine-tuning: se `train.py` formata o exemplo
de um jeito e `model_loader.py` monta o prompt de outro, o adapter treinado
não transfere e o sintoma é "o fine-tuning não melhorou nada" — sem erro
nenhum aparecendo. Por isso as duas pontas saem da mesma função e os testes
fixam o formato.
"""

from __future__ import annotations

from hospital_assistant.finetuning.schema import InstructionExample
from hospital_assistant.llm.prompt import SYSTEM_PROMPT, build_messages, build_training_messages

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
