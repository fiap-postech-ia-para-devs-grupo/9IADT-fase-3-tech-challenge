"""Rótulos legíveis e conteúdo estático da base de conhecimento.

As tabelas do portal nunca mostram o nome do campo. `paciente_id` vira
"Paciente", `timestamp` vira "Data e hora", `nao_necessaria` vira "Sem
validação exigida". O mapa vive aqui, num lugar só, para que a mesma coluna
tenha o mesmo nome em todos os módulos — e para que renomear seja uma linha,
não uma caçada.

A base de conhecimento é conteúdo versionado em código, e não linhas no banco:
as respostas espelham os protocolos indexados no vector store e precisam
evoluir junto com eles, no mesmo commit. Guardá-las no SQLite as separaria da
fonte que elas resumem.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TypedDict

from hospital_assistant.paths import RAW_DATA_DIR

# --- rótulos de coluna ------------------------------------------------------

COLUNAS: dict[str, str] = {
    "id": "Nº",
    "timestamp": "Data e hora",
    "pergunta": "Pergunta do médico",
    "paciente_id": "Paciente",
    "paciente": "Paciente",
    "medico_solicitante": "Solicitado por",
    "tipo_operacao": "Operação",
    "fontes_rag": "Fontes consultadas",
    "resposta_llm": "Resposta do assistente",
    "flags_seguranca": "Sinalizações",
    "status": "Situação",
    "aprovador": "Validado por",
    "timestamp_aprovacao": "Validado em",
    "nome": "Nome",
    "prontuario": "Prontuário",
    "tipo": "Exame",
    "data_solicitacao": "Solicitado em",
    "data_resultado": "Resultado em",
    "resultado": "Resultado",
    "dosagem": "Dosagem",
    "frequencia": "Frequência",
    "data_inicio": "Início",
    "descricao": "Descrição",
    "severidade": "Severidade",
    "data": "Data",
}

# Nome legível de cada flag interna do guardrail.
FLAGS: dict[str, str] = {
    "emergencia_clinica": "Sinal de emergência",
    "suspeita_violencia_domestica": "Suspeita de violência",
    "requer_validacao_humana": "Exige validação",
}


def rotular(campo: str) -> str:
    """Nome de exibição de um campo. Sem entrada no mapa, formata o próprio nome."""
    if campo in COLUNAS:
        return COLUNAS[campo]
    return campo.replace("_", " ").capitalize()


def rotular_flag(flag: str) -> str:
    """Nome de exibição de uma sinalização do guardrail."""
    return FLAGS.get(flag, flag.replace("_", " ").capitalize())


# --- base de conhecimento ---------------------------------------------------


class PerguntaFrequente(TypedDict):
    pergunta: str
    resposta: str
    categoria: str
    fonte: str  # referência legível, nunca caminho de código


CATEGORIAS: dict[str, str] = {
    "protocolo": "Protocolo clínico",
    "exames": "Exames",
    "medicacao": "Medicação",
}

FAQ: list[PerguntaFrequente] = [
    {
        "pergunta": "Qual a conduta inicial na suspeita de sepse?",
        "resposta": (
            "Coletar lactato e hemoculturas antes do antimicrobiano, iniciar antibioticoterapia "
            "empírica de amplo espectro dentro da primeira hora e iniciar ressuscitação volêmica "
            "conforme a resposta hemodinâmica. Toda prescrição depende de validação e assinatura "
            "do médico responsável."
        ),
        "categoria": "protocolo",
        "fonte": "Protocolo interno — Suspeita de sepse",
    },
    {
        "pergunta": "Quais critérios do qSOFA indicam gravidade?",
        "resposta": (
            "O qSOFA pontua alteração do nível de consciência, frequência respiratória ≥ 22 irpm "
            "e pressão arterial sistólica ≤ 100 mmHg. Escore ≥ 2 indica risco elevado e aciona o "
            "protocolo institucional de sepse."
        ),
        "categoria": "protocolo",
        "fonte": "Protocolo interno — Suspeita de sepse",
    },
    {
        "pergunta": "Como conduzir dor torácica aguda pela classificação de risco?",
        "resposta": (
            "HEART 0-3 (baixo risco): considerar alta com reavaliação ambulatorial em 72 h se as "
            "troponinas seriadas forem negativas e o ECG não mostrar alterações isquêmicas. "
            "HEART 4-6 (moderado): observação hospitalar com troponina seriada e avaliação da "
            "cardiologia antes da alta. HEART 7-10 (alto): internação com cardiologia acionada "
            "imediatamente."
        ),
        "categoria": "protocolo",
        "fonte": "Protocolo interno — Dor torácica aguda",
    },
    {
        "pergunta": "Qual o tempo máximo para liberação de exame urgente?",
        "resposta": (
            "Meta de coleta em até 30 minutos e liberação do resultado em até 2 horas para os "
            "exames laboratoriais básicos (hemograma, eletrólitos, função renal, troponina, "
            "lactato). O sistema emite alerta automático quando um exame urgente passa de 2 horas "
            "sem resultado."
        ),
        "categoria": "exames",
        "fonte": "FAQ interno — Solicitação de exames urgentes",
    },
    {
        "pergunta": "Posso reclassificar um exame de rotina para urgente?",
        "resposta": (
            "Sim, desde que haja justificativa clínica registrada. A reclassificação reinicia a "
            "contagem da meta de tempo de liberação do resultado."
        ),
        "categoria": "exames",
        "fonte": "FAQ interno — Solicitação de exames urgentes",
    },
]


def filtrar_faq(categoria: str = "todas", busca: str = "") -> list[PerguntaFrequente]:
    """Filtra a base de conhecimento por categoria e por termo livre."""
    itens = FAQ
    if categoria != "todas":
        itens = [item for item in itens if item["categoria"] == categoria]

    termo = busca.strip().lower()
    if termo:
        itens = [
            item
            for item in itens
            if termo in item["pergunta"].lower() or termo in item["resposta"].lower()
        ]
    return itens


# --- procedencia das respostas ----------------------------------------------

PROTOCOLOS_DIR = RAW_DATA_DIR / "protocolos_sinteticos"


@lru_cache(maxsize=64)
def nome_da_fonte(origem: str) -> str:
    """Titulo legivel de um documento da base, lido do proprio arquivo.

    O RAG grava o caminho (`protocolos_sinteticos/sepse.md`), que nao diz nada a
    quem revisa a resposta. Cada protocolo ja abre com um H1 escrito para leitura
    humana, entao o titulo sai dali: manter um mapa de arquivo para titulo aqui
    sairia de sincronia assim que alguem acrescentasse um protocolo novo.

    Sem o arquivo em disco, formata o proprio nome — a tela degrada para um
    rotulo feio, nunca para um erro.
    """
    caminho = PROTOCOLOS_DIR / Path(origem.replace("\\", "/")).name
    try:
        linhas = caminho.read_text(encoding="utf-8").splitlines()
    except OSError:
        linhas = []

    for linha in linhas:
        texto = linha.strip()
        if texto.startswith("#"):
            titulo = texto.lstrip("#").strip()
            if titulo:
                return titulo
        if texto:
            break

    return Path(origem).stem.replace("_", " ").capitalize()
