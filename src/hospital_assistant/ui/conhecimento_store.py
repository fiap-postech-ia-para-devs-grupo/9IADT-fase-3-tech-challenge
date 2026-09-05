"""Base de conhecimento que cresce com os atendimentos.

Cada resposta do assistente é registrada aqui, e uma pergunta semelhante feita
depois reaproveita a resposta em vez de acionar o modelo de novo. Numa GPU isso
troca dezenas de segundos por milissegundos; em CPU, minutos por milissegundos.

**O reaproveitamento exige aprovação médica.** Registrar tudo é útil — dá para
ver o que já foi perguntado —, mas devolver uma resposta que ninguém revisou
como se fosse conhecimento consolidado transformaria um erro do modelo em
resposta oficial para todos os atendimentos seguintes. Por isso `registrar`
aceita qualquer resposta e `buscar_similar` só devolve as aprovadas, consultando
a decisão do médico em `decisoes_store`.

A semelhança é textual, por `difflib`, e não semântica por embeddings. Duas
razões: o limiar alto (0.88) torna o reuso conservador, o que é o que se quer
quando o custo de errar é clínico; e uma busca semântica traria "conduta na
sepse" e "conduta na sepse neonatal" como equivalentes, que é exatamente o tipo
de aproximação que não se pode fazer aqui.

O paciente faz parte da chave: a mesma pergunta muda de resposta conforme o
prontuário, e reaproveitar entre pacientes diferentes devolveria a conduta de
um para outro.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import unicodedata
from datetime import UTC, datetime
from difflib import SequenceMatcher
from typing import Any, TypedDict

from hospital_assistant.paths import DATA_DIR
from hospital_assistant.ui import decisoes_store

logger = logging.getLogger(__name__)

ARQUIVO = DATA_DIR / "base_conhecimento.json"

# Alto de propósito: abaixo disso as perguntas já são clinicamente distintas o
# bastante para merecerem análise nova.
LIMIAR_SEMELHANCA = 0.88


class Entrada(TypedDict):
    audit_id: int
    pergunta: str
    resposta: str
    paciente_id: str | None
    medico: str | None
    timestamp: str


def _normalizar(texto: str) -> str:
    """Reduz a pergunta ao que importa para comparar.

    Sem isto, "Qual a conduta na sepse?" e "qual a conduta na sepse" contariam
    como perguntas diferentes e o cache nunca acertaria.
    """
    sem_acento = "".join(
        caractere
        for caractere in unicodedata.normalize("NFD", texto.lower())
        if unicodedata.category(caractere) != "Mn"
    )
    return re.sub(r"[^a-z0-9 ]+", " ", sem_acento).strip()


def semelhanca(uma: str, outra: str) -> float:
    """Grau de semelhança entre duas perguntas, de 0 a 1."""
    return SequenceMatcher(None, _normalizar(uma), _normalizar(outra)).ratio()


def listar() -> list[Entrada]:
    """Tudo o que já foi perguntado, do mais recente para o mais antigo."""
    try:
        entradas = json.loads(ARQUIVO.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError):
        logger.exception("Base de conhecimento ilegível em %s; seguindo sem ela.", ARQUIVO)
        return []

    return list(reversed(entradas))


def obter(audit_id: int) -> Entrada | None:
    """A consulta correspondente a um registro de auditoria.

    É por aqui que o laudo descobre quem **solicitou** a análise: a trilha de
    auditoria guarda quem aprovou, não quem perguntou.
    """
    for entrada in listar():
        if entrada["audit_id"] == audit_id:
            return entrada
    return None


def registrar(
    audit_id: int,
    pergunta: str,
    resposta: str,
    paciente_id: str | None = None,
    medico: str | None = None,
) -> None:
    """Acrescenta uma resposta à base, sem julgar se ela é boa.

    A triagem de qualidade é a fila de validação, e ela acontece depois. Filtrar
    aqui esconderia do médico o que o assistente respondeu.
    """
    entradas: list[Any] = list(reversed(listar()))
    entradas.append(
        {
            "audit_id": audit_id,
            "pergunta": pergunta.strip(),
            "resposta": resposta,
            "paciente_id": paciente_id,
            "medico": medico,
            "timestamp": datetime.now(UTC).isoformat(),
        }
    )

    ARQUIVO.parent.mkdir(parents=True, exist_ok=True)
    conteudo = json.dumps(entradas, ensure_ascii=False, indent=2)

    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=ARQUIVO.parent, delete=False, suffix=".tmp"
    ) as temporario:
        temporario.write(conteudo)
        caminho_temporario = temporario.name

    os.replace(caminho_temporario, ARQUIVO)


def buscar_similar(pergunta: str, paciente_id: str | None = None) -> Entrada | None:
    """Resposta **aprovada** para uma pergunta semelhante, no mesmo paciente.

    Devolve `None` quando não há nada suficientemente próximo, quando a resposta
    mais próxima ainda não foi validada, ou quando ela pertence a outro
    paciente. Em qualquer desses casos a pergunta segue para o modelo.
    """
    aprovadas = {
        registro_id
        for registro_id, decisao in decisoes_store.carregar().items()
        if decisao.get("status") == "aprovado"
    }
    if not aprovadas:
        return None

    candidatas = [
        entrada
        for entrada in listar()
        if entrada["audit_id"] in aprovadas and entrada["paciente_id"] == paciente_id
    ]
    if not candidatas:
        return None

    melhor = max(candidatas, key=lambda entrada: semelhanca(pergunta, entrada["pergunta"]))
    if semelhanca(pergunta, melhor["pergunta"]) < LIMIAR_SEMELHANCA:
        return None

    # A resposta aprovada pode ter sido editada pelo médico antes de aprovar; é
    # o texto revisado que vale, não o que o modelo produziu.
    editada = decisoes_store.carregar()[melhor["audit_id"]].get("resposta_llm")
    if editada:
        return {**melhor, "resposta": editada}
    return melhor


def limpar() -> None:
    """Remove a base inteira. Existe para os testes começarem do zero."""
    ARQUIVO.unlink(missing_ok=True)
