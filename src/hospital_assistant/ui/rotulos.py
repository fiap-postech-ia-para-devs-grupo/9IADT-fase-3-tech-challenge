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

from typing import TypedDict

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
    fonte: str


CATEGORIAS: dict[str, str] = {
    "protocolo": "Protocolo clínico",
    "exames": "Exames",
    "medicacao": "Medicação",
    "fluxo": "Fluxo do sistema",
    "seguranca": "Segurança",
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
        "fonte": "data/raw/protocolos_sinteticos/sepse.md",
    },
    {
        "pergunta": "Quais critérios do qSOFA indicam gravidade?",
        "resposta": (
            "O qSOFA pontua alteração do nível de consciência, frequência respiratória ≥ 22 irpm "
            "e pressão arterial sistólica ≤ 100 mmHg. Escore ≥ 2 indica risco elevado e aciona o "
            "protocolo institucional de sepse."
        ),
        "categoria": "protocolo",
        "fonte": "data/raw/protocolos_sinteticos/sepse.md",
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
        "fonte": "data/raw/protocolos_sinteticos/dor_toracica_aguda.md",
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
        "fonte": "data/raw/protocolos_sinteticos/faq_medicos_exames_urgentes.md",
    },
    {
        "pergunta": "Posso reclassificar um exame de rotina para urgente?",
        "resposta": (
            "Sim, desde que haja justificativa clínica registrada. A reclassificação reinicia a "
            "contagem da meta de tempo de liberação do resultado."
        ),
        "categoria": "exames",
        "fonte": "data/raw/protocolos_sinteticos/faq_medicos_exames_urgentes.md",
    },
    {
        "pergunta": "O assistente pode prescrever medicamento ou definir dosagem?",
        "resposta": (
            "Não. Ele sugere e apoia a decisão clínica, mas nunca prescreve de forma autônoma. "
            "Toda resposta que mencione medicamento ou dosagem é sinalizada pelo guardrail e fica "
            "retida na fila de validação até que um médico aprove, rejeite ou edite."
        ),
        "categoria": "seguranca",
        "fonte": "src/hospital_assistant/safety/guardrails.py",
    },
    {
        "pergunta": "O que acontece quando o assistente detecta sinal de emergência?",
        "resposta": (
            "O nó de entrada do grafo sinaliza `emergencia_clinica`, o nó de alerta emite aviso "
            "para a equipe médica e a resposta orienta atendimento presencial imediato. O evento "
            "fica registrado na auditoria com a flag correspondente."
        ),
        "categoria": "seguranca",
        "fonte": "src/hospital_assistant/graph/nodes.py",
    },
    {
        "pergunta": "Como o assistente indica de onde veio a informação?",
        "resposta": (
            "Cada resposta carrega os trechos recuperados do vector store com o arquivo de origem "
            "e o score de similaridade de cosseno. Esses dados aparecem na fila de validação e "
            "ficam gravados na auditoria — é o requisito de explicabilidade do desafio."
        ),
        "categoria": "fluxo",
        "fonte": "src/hospital_assistant/rag/retriever.py",
    },
    {
        "pergunta": "Por que a resposta às vezes aparece marcada como [MOCK LLM]?",
        "resposta": (
            "Quando não há adapter LoRA configurado (`HF_ADAPTER_REPO`) ou o ambiente não tem GPU "
            "com bitsandbytes, o carregador degrada para um stand-in determinístico e registra "
            "isso no log. O pipeline continua demonstrável, mas a sugestão não vem do modelo "
            "fine-tunado."
        ),
        "categoria": "fluxo",
        "fonte": "src/hospital_assistant/llm/model_loader.py",
    },
    {
        "pergunta": "O modelo fine-tunado já pode ser o padrão do assistente?",
        "resposta": (
            "Ainda não. O comparativo base vs. fine-tuned mostrou regressão de segurança: o "
            "modelo ajustado passou a responder com dose e posologia a perguntas que o modelo "
            "base recusava. O corpus sintético precisa ser revisado antes. Detalhes na seção 3.3 "
            "do relatório técnico."
        ),
        "categoria": "medicacao",
        "fonte": "docs/relatorio_tecnico.md",
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
