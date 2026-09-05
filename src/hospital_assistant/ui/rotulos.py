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
    "fluxo": "Fluxo do sistema",
    "seguranca": "Segurança",
}

# Categorias que valem como atalho de pergunta no assistente. As demais
# (`fluxo`, `seguranca`, `medicacao`) documentam o comportamento do próprio
# sistema — "por que a resposta não veio do modelo treinado" é informação útil
# na base de conhecimento, mas oferecê-la ao médico como sugestão de pergunta
# clínica desvia o composer da função dele.
CATEGORIAS_CLINICAS: tuple[str, ...] = ("protocolo", "exames")

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
    {
        "pergunta": "O assistente pode prescrever medicamento ou definir dosagem?",
        "resposta": (
            "Não. Ele sugere e apoia a decisão clínica, mas nunca prescreve de forma autônoma. "
            "Toda resposta que mencione medicamento ou dosagem recebe sinalização automática de "
            "segurança e fica retida na fila de validação até que um médico aprove, rejeite ou "
            "edite o texto."
        ),
        "categoria": "seguranca",
        "fonte": "Política de segurança do assistente",
    },
    {
        "pergunta": "O que acontece quando o assistente detecta sinal de emergência?",
        "resposta": (
            "O caso é sinalizado como emergência clínica, um alerta é emitido para a equipe "
            "médica e a resposta orienta atendimento presencial imediato. O evento fica "
            "registrado na auditoria com a sinalização correspondente."
        ),
        "categoria": "seguranca",
        "fonte": "Fluxo de atendimento automatizado",
    },
    {
        "pergunta": "Como o assistente indica de onde veio a informação?",
        "resposta": (
            "Cada resposta carrega os trechos de protocolo que a fundamentaram, com a origem de "
            "cada um e o grau de similaridade com a pergunta. Esses dados aparecem na fila de "
            "validação e ficam gravados na auditoria, de modo que sempre é possível reconstruir "
            "em que o assistente se baseou."
        ),
        "categoria": "fluxo",
        "fonte": "Política de explicabilidade das respostas",
    },
    {
        "pergunta": "Por que a resposta às vezes indica que não veio do modelo treinado?",
        "resposta": (
            "O modelo ajustado exige placa de vídeo dedicada. Quando o ambiente não a possui, o "
            "sistema opera com um gerador de demonstração e informa isso na barra lateral e na "
            "própria resposta. O fluxo, as fontes consultadas e o encaminhamento para validação "
            "continuam reais — apenas o texto da sugestão não vem do modelo treinado."
        ),
        "categoria": "fluxo",
        "fonte": "Nota técnica — Modelo em uso",
    },
    {
        "pergunta": "O modelo fine-tunado já pode ser o padrão do assistente?",
        "resposta": (
            "Ainda não. O comparativo base vs. fine-tuned mostrou regressão de segurança: o "
            "modelo ajustado passou a responder com dose e posologia a perguntas que o modelo "
            "original recusava. A base de treinamento precisa ser revisada antes. Os detalhes "
            "estão no relatório técnico do projeto."
        ),
        "categoria": "medicacao",
        "fonte": "Relatório técnico — Avaliação do modelo",
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
