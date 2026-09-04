"""Scrubber de PII aplicado a todo exemplo antes do treino, per ESTRATEGIA.md §3.

O PDF do Tech Challenge exige "preparar os dados com técnicas de preprocessing,
anonimização e curadoria". Nenhuma das fontes usadas aqui (PubMedQA, MedQuAD,
protocolos sintéticos) contém dado real de paciente — a anonimização é aplicada
mesmo assim, por dois motivos: documenta a técnica exigida e garante que o
pipeline continue seguro se algum dia receber prontuário real como entrada.

**Critério de desenho**: falso positivo custa mais que falso negativo aqui. Um
scrubber agressivo demais transforma "amoxicilina 500 mg" ou "Streptococcus
pneumoniae" em marcadores e destrói silenciosamente o valor clínico do dataset.
Por isso todo padrão exige uma âncora explícita — formato rígido (CPF, CEP,
e-mail), rótulo textual ("prontuário nº"), ou marcador de papel ("Dr.",
"paciente") — em vez de tentar reconhecer nome próprio ou número solto pelo
formato. `tests/test_anonymize.py` fixa os dois lados: o que some e o que fica.
"""

from __future__ import annotations

import re

from hospital_assistant.finetuning.schema import InstructionExample

# Sequência de nome próprio: uma ou mais palavras capitalizadas, aceitando
# partículas em minúscula no meio ("Ana de Souza", "João dos Santos").
_NOME = r"[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][\w'’-]+(?:\s+(?:d[aeo]s?\s+)?[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][\w'’-]+)*"

# Títulos que marcam o que vem a seguir como nome de pessoa. O título em si é
# preservado: "Dr." não identifica ninguém e dá contexto ao modelo.
_TITULOS = r"Dr|Dra|Sr|Sra|Srta|Srª|Enf|Prof|Profa|Mr|Mrs|Ms|Dr\.ª"

_MESES = (
    r"janeiro|fevereiro|março|marco|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro"
)

# Ordem importa: padrões mais longos/específicos primeiro, senão um padrão
# curto consome parte do texto que o seguinte precisaria ver inteiro (ex.: o
# CPF de 11 dígitos soltos morderia o CNS de 15).
_SCRUBBERS: tuple[tuple[re.Pattern[str], str], ...] = (
    # --- identificadores de formato rígido ---
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL]"),
    (re.compile(r"https?://\S+|\bwww\.[\w.-]+\.\w{2,}\S*"), "[URL]"),
    (re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"), "[CPF]"),
    (re.compile(r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b"), "[CNPJ]"),
    (re.compile(r"\b\d{3}\s\d{4}\s\d{4}\s\d{4}\b|\b\d{15}\b"), "[CNS]"),
    (re.compile(r"\b\d{11}\b"), "[CPF]"),
    (re.compile(r"\(?\b\d{2}\)?\s?9?\d{4}-\d{4}\b"), "[TELEFONE]"),
    (re.compile(r"\b\d{5}-\d{3}\b"), "[CEP]"),
    # --- datas ---
    (re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"), "[DATA]"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "[DATA]"),
    (re.compile(rf"\b\d{{1,2}}\s+de\s+(?:{_MESES})\s+de\s+\d{{4}}\b", re.IGNORECASE), "[DATA]"),
    # --- identificadores ancorados por rótulo textual ---
    (
        re.compile(
            r"\b(prontu[áa]rio|registro|matr[íi]cula|guia|atendimento|medical\s+record)\b"
            r"[\s:#]*(?:n[º°o]\.?|no\.?|number)?[\s:#]*[\dA-Z][\dA-Z.-]{2,}",
            re.IGNORECASE,
        ),
        r"\1 [PRONTUARIO]",
    ),
    # --- nomes de pessoa, sempre ancorados por título ou papel ---
    (re.compile(rf"\b({_TITULOS})\.?\s+{_NOME}"), r"\1. [NOME]"),
    # "paciente"/"patient" exige nome composto (2+ palavras capitalizadas) —
    # uma palavra só depois de "paciente" é quase sempre outra coisa
    # ("paciente idoso", "paciente Covid-19"), não um nome.
    #
    # **Sem `re.IGNORECASE` de propósito.** A flag valeria para o padrão
    # inteiro e anularia a exigência de inicial maiúscula, que é o único sinal
    # que separa nome próprio de texto corrido: com ela, "o paciente
    # apresentou dor torácica" virava "o paciente [NOME] torácica" e
    # "patient satisfaction was assessed" virava "patient [NOME]" — os dois
    # observados rodando o scrubber sobre PubMedQA/MedQuAD reais. A variação
    # de caixa da âncora entra explicitamente na alternância.
    (
        re.compile(
            r"\b([Pp]aciente|[Pp]atient|Sr\.|Sra\.)\s+"
            r"([A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][\w'’-]+(?:\s+(?:d[aeo]s?\s+)?[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][\w'’-]+)+)"
        ),
        r"\1 [NOME]",
    ),
)


def anonymize(text: str) -> str:
    """Remove identificadores diretos de `text`, preservando o conteúdo clínico.

    Cada categoria vira um marcador estável (`[CPF]`, `[NOME]`, `[DATA]`...) em
    vez de ser apagada: o modelo aprende que ali existia um dado identificável,
    o que é o comportamento desejado num assistente que nunca deve repetir PII.
    """
    if not text:
        return text

    for padrao, substituto in _SCRUBBERS:
        text = padrao.sub(substituto, text)
    return text


def anonymize_example(example: InstructionExample) -> InstructionExample:
    """Aplica `anonymize` aos três campos, devolvendo um novo dict."""
    return {
        "instruction": anonymize(example["instruction"]),
        "input": anonymize(example["input"]),
        "output": anonymize(example["output"]),
    }
