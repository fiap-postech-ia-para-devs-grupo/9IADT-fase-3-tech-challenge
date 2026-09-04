"""Geração dos "dados próprios do hospital" via Groq/Gemini, per ESTRATEGIA.md §3.

O PDF pede fine-tuning com *protocolos médicos do hospital*, *perguntas
frequentes de médicos* e *modelos de laudos, receitas e procedimentos
internos*. Como o hospital é fictício, esse material é gerado sinteticamente —
é a metade em português do dataset (PubMedQA e MedQuAD são em inglês) e a
única parte que ensina o modelo o vocabulário institucional que o RAG indexa.

**Decisão de persona**: os exemplos gerados nunca prescrevem diretamente; toda
resposta é formulada como sugestão ao médico, sujeita a validação humana. Sem
isso o fine-tuning ensinaria exatamente o comportamento que
`safety/guardrails.py` bloqueia depois, e o sistema passaria a brigar consigo
mesmo — o guardrail reescreveria toda saída do modelo que ele mesmo treinou.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

from hospital_assistant.finetuning.schema import InstructionExample

logger = logging.getLogger(__name__)

# `gemini-2.5-flash` ainda aparece em `list_models()` mas o endpoint recusa
# contas novas ("no longer available to new users"), então o nome do modelo
# precisa ser o atual, não o citado na ESTRATEGIA.
GEMINI_MODEL = "gemini-3.6-flash"
GROQ_MODEL = "openai/gpt-oss-120b"

# ESTRATEGIA.md §3: ~150-200 exemplos sintéticos.
TOTAL_PADRAO = 180
LOTE = 10

# Falhas seguidas antes de desistir. Cota estourada faz *todos* os lotes
# falharem; insistir até o teto de tentativas gasta ~20 minutos para produzir
# zero exemplo e sem sinal claro do motivo (observado com o free tier do
# Gemini retornando 429 em série).
MAX_FALHAS_SEGUIDAS = 3

CATEGORIAS: dict[str, str] = {
    "protocolos_internos": (
        "protocolos clínicos internos do hospital (conduta inicial, critérios de "
        "internação, fluxo de encaminhamento, escalas de gravidade)"
    ),
    "faq_medicos": (
        "perguntas frequentes feitas por médicos da casa sobre rotinas assistenciais "
        "(como solicitar exame urgente, quando acionar a equipe de resposta rápida, "
        "prazos de resultado, fluxo de interconsulta)"
    ),
    "modelos_de_laudo": (
        "modelos de laudo e de evolução clínica usados internamente (estrutura do "
        "texto, campos obrigatórios, exemplo preenchido com dados fictícios)"
    ),
    "modelos_de_receita": (
        "modelos de prescrição e de orientação de alta usados internamente, sempre "
        "como rascunho a ser revisado e assinado pelo médico responsável"
    ),
    "interpretacao_de_exames": (
        "apoio à interpretação de exames de rotina (hemograma, gasometria, marcadores "
        "inflamatórios, culturas), relacionando achado e conduta sugerida"
    ),
}

_PERSONA = (
    "Você gera dados de treino para um assistente clínico de apoio à decisão "
    "de um hospital fictício brasileiro. O assistente SUGERE e apoia a decisão "
    "do médico; ele NUNCA prescreve diretamente nem define dosagem por conta "
    "própria, e toda conduta que ele propõe fica explicitamente sujeita a "
    "validação humana pelo médico responsável."
)


def build_prompt(categoria: str, quantidade: int, evitar: list[str] | None = None) -> str:
    """Monta o prompt de um lote. Puro — o teste fixa persona, idioma e formato."""
    descricao = CATEGORIAS[categoria]

    partes = [
        _PERSONA,
        "",
        f"Gere {quantidade} exemplos de treino sobre: {descricao}.",
        "",
        "Regras obrigatórias:",
        "- Escreva tudo em português do Brasil.",
        "- Use terminologia clínica correta e realista.",
        "- Dados de paciente devem ser fictícios e genéricos.",
        "- A resposta ('output') deve ter entre 3 e 8 frases, no tom de sugestão "
        "ao médico ('sugiro considerar', 'recomenda-se avaliar'), nunca em tom "
        "de ordem ao paciente.",
        "- Quando a resposta envolver medicamento ou dosagem, deixe explícito que "
        "a prescrição depende de validação e assinatura do médico responsável.",
        "- Varie o assunto entre os exemplos; não repita o mesmo tema.",
    ]

    if evitar:
        partes += [
            "",
            "Os temas abaixo JÁ foram gerados — escolha outros:",
            *[f"- {tema}" for tema in evitar],
        ]

    partes += [
        "",
        "Responda APENAS com um array JSON válido, sem texto em volta e sem "
        "cercas de markdown. Cada item deve ter exatamente as chaves "
        '"instruction" (a pergunta do médico), "input" (contexto adicional, '
        'pode ser string vazia) e "output" (a resposta do assistente).',
    ]

    return "\n".join(partes)


def parse_llm_batch(raw: str) -> list[InstructionExample]:
    """Extrai os exemplos válidos da resposta do modelo.

    Tolerante de propósito: um lote malformado no meio de ~18 chamadas não pode
    derrubar a geração inteira. Resposta sem JSON reconhecível devolve lista
    vazia e o chamador segue para o próximo lote.
    """
    if not raw:
        return []

    texto = re.sub(r"^\s*```(?:json)?|```\s*$", "", raw.strip(), flags=re.MULTILINE).strip()

    dados: Any = None
    try:
        dados = json.loads(texto)
    except json.JSONDecodeError:
        # Modelo mandou prosa em volta do array — pega do primeiro "[" ao último "]".
        inicio, fim = texto.find("["), texto.rfind("]")
        if inicio != -1 and fim > inicio:
            try:
                dados = json.loads(texto[inicio : fim + 1])
            except json.JSONDecodeError:
                logger.warning("Lote descartado: JSON inválido mesmo após recorte.")
                return []
        else:
            logger.warning("Lote descartado: nenhum array JSON na resposta.")
            return []

    if not isinstance(dados, list):
        return []

    exemplos: list[InstructionExample] = []
    for item in dados:
        if not isinstance(item, dict):
            continue
        instruction = str(item.get("instruction") or "").strip()
        output = str(item.get("output") or "").strip()
        if not instruction or not output:
            continue
        exemplos.append(
            {
                "instruction": instruction,
                "input": str(item.get("input") or "").strip(),
                "output": output,
            }
        )
    return exemplos


def _chat_gemini(prompt: str) -> str:
    import google.generativeai as _genai

    # `google.generativeai` não declara `configure`/`GenerativeModel` no
    # `__all__`, então o type checker acusa uso de símbolo privado apesar de a
    # API pública ser exatamente essa. O alias tipado como Any encerra o ruído
    # sem espalhar `# type: ignore` por cada linha.
    genai: Any = _genai

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    resposta = genai.GenerativeModel(GEMINI_MODEL).generate_content(prompt)
    return resposta.text or ""


def _chat_groq(prompt: str) -> str:
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resposta = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
    )
    return resposta.choices[0].message.content or ""


_PROVIDERS = {"gemini": _chat_gemini, "groq": _chat_groq}


def provider_disponivel() -> str:
    """Escolhe o provider pela chave presente no ambiente.

    Groq tem precedência: é o primeiro citado em ESTRATEGIA.md §1 e o free
    tier dele aguenta as ~20 chamadas desta geração, enquanto o do Gemini
    estoura antes (429) para contas novas.
    """
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    raise RuntimeError(
        "Nenhuma chave encontrada. Defina GOOGLE_API_KEY ou GROQ_API_KEY no .env "
        "(ver .env.example) antes de gerar os dados sintéticos."
    )


def generate_synthetic(
    total: int = TOTAL_PADRAO,
    provider: str | None = None,
    # O free tier de ambos os providers limita requisições por minuto; com
    # lotes de 10 exemplos são ~20 chamadas, e 4s entre elas mantém a geração
    # inteira abaixo do teto sem depender do retry por exceção.
    pausa_s: float = 4.0,
) -> list[InstructionExample]:
    """Gera `total` exemplos sintéticos, distribuídos entre as categorias.

    Os temas já gerados voltam no prompt do lote seguinte como lista de
    exclusão: sem isso o modelo converge para meia dúzia de assuntos e a
    deduplicação em `data_prep` descarta metade do que foi gerado.
    """
    escolhido = provider or provider_disponivel()
    chat = _PROVIDERS[escolhido]
    logger.info("Gerando %d exemplos sintéticos via %s...", total, escolhido)

    exemplos: list[InstructionExample] = []
    categorias = list(CATEGORIAS)
    lote_idx = 0
    falhas_seguidas = 0
    ultimo_erro: Exception | None = None
    max_lotes = len(categorias) * 12

    while len(exemplos) < total and lote_idx < max_lotes:
        categoria = categorias[lote_idx % len(categorias)]
        ja_usados = [e["instruction"] for e in exemplos][-12:]
        prompt = build_prompt(categoria, quantidade=LOTE, evitar=ja_usados)
        lote_idx += 1

        try:
            bruto = chat(prompt)
        except Exception as erro:  # noqa: BLE001 — um lote com erro transitório é tolerável
            falhas_seguidas += 1
            ultimo_erro = erro
            logger.warning(
                "Lote %d (%s) falhou (%d/%d): %s",
                lote_idx,
                categoria,
                falhas_seguidas,
                MAX_FALHAS_SEGUIDAS,
                erro,
            )
            if falhas_seguidas >= MAX_FALHAS_SEGUIDAS:
                raise RuntimeError(
                    f"{falhas_seguidas} lotes falharam em sequência via '{escolhido}' — "
                    f"provider indisponível ou cota esgotada. Último erro: {ultimo_erro}. "
                    "Tente o outro provider (GROQ_API_KEY / GOOGLE_API_KEY)."
                ) from ultimo_erro
            time.sleep(pausa_s * 3)
            continue

        falhas_seguidas = 0
        novos = parse_llm_batch(bruto)
        logger.info("Lote %d (%s): +%d exemplos (total %d)", lote_idx, categoria, len(novos), len(exemplos) + len(novos))
        exemplos.extend(novos)
        time.sleep(pausa_s)

    if lote_idx >= max_lotes and len(exemplos) < total:
        logger.warning("Teto de %d lotes atingido com %d exemplos.", max_lotes, len(exemplos))

    return exemplos[:total]
