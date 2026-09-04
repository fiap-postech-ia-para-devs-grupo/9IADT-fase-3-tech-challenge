"""Scrubber de PII, per ESTRATEGIA.md §3 ("anonimização") e o requisito do PDF
("preparar os dados com técnicas de preprocessing, anonimização e curadoria").

Os casos aqui cobrem os dois lados que importam: o que *deve* ser removido
(identificadores diretos) e o que *não pode* ser removido (o conteúdo clínico
que é a razão de o exemplo existir). Um scrubber agressivo demais destrói o
dataset silenciosamente, então os testes de preservação são tão importantes
quanto os de remoção.
"""

from __future__ import annotations

import pytest

from hospital_assistant.finetuning.anonymize import anonymize, anonymize_example
from hospital_assistant.finetuning.schema import InstructionExample


@pytest.mark.parametrize(
    ("texto", "esperado_ausente", "marcador"),
    [
        ("Paciente CPF 123.456.789-00 admitido ontem.", "123.456.789-00", "[CPF]"),
        ("CPF 12345678900 na ficha.", "12345678900", "[CPF]"),
        ("Contato: maria.silva@hospital.org.br", "maria.silva@hospital.org.br", "[EMAIL]"),
        ("Telefone (11) 98765-4321 para retorno.", "98765-4321", "[TELEFONE]"),
        ("Internado em 12/03/2024 pela manhã.", "12/03/2024", "[DATA]"),
        ("Coleta em 2024-03-12 às 8h.", "2024-03-12", "[DATA]"),
        ("Admitido em 12 de março de 2024.", "12 de março de 2024", "[DATA]"),
        ("Ver prontuário nº 998877 no sistema.", "998877", "[PRONTUARIO]"),
        ("Residente no CEP 01310-100.", "01310-100", "[CEP]"),
        ("CNS 123456789012345 validado.", "123456789012345", "[CNS]"),
        ("Fonte: https://www.nlm.nih.gov/artigo/123", "https://www.nlm.nih.gov", "[URL]"),
    ],
)
def test_remove_identificador_direto(texto: str, esperado_ausente: str, marcador: str) -> None:
    resultado = anonymize(texto)
    assert esperado_ausente not in resultado
    assert marcador in resultado


@pytest.mark.parametrize(
    ("texto", "nome"),
    [
        ("Avaliado pelo Dr. Carlos Andrade na admissão.", "Carlos Andrade"),
        ("Conduta discutida com a Dra. Ana Paula Souza.", "Ana Paula Souza"),
        ("O paciente João Silva relatou dor torácica.", "João Silva"),
        ("The patient Mary Johnson reported chest pain.", "Mary Johnson"),
    ],
)
def test_remove_nome_proprio_identificavel(texto: str, nome: str) -> None:
    resultado = anonymize(texto)
    assert nome not in resultado
    assert "[NOME]" in resultado


def test_preserva_titulo_ao_remover_nome() -> None:
    """`Dr.` sozinho não identifica ninguém e dá contexto ao modelo — some só o nome."""
    assert anonymize("Avaliado pelo Dr. Carlos Andrade.") == "Avaliado pelo Dr. [NOME]."


@pytest.mark.parametrize(
    "termo_clinico",
    [
        "dor torácica aguda",
        "qSOFA maior ou igual a 2",
        "amoxicilina 500 mg de 8 em 8 horas",
        "pressão arterial 180x120 mmHg",
        "saturação de 88% em ar ambiente",
        "hemoglobina 9,4 g/dL",
        "escala de Glasgow 15",
        "Streptococcus pneumoniae",
        "doença de Alzheimer",
        "síndrome de Guillain-Barré",
    ],
)
def test_preserva_conteudo_clinico(termo_clinico: str) -> None:
    """Números e nomes próprios clínicos são o valor do dataset — não podem virar marcador."""
    texto = f"Conduta: considerar {termo_clinico} conforme protocolo."
    assert anonymize(texto) == texto


@pytest.mark.parametrize(
    "frase",
    [
        # Encontrados rodando o scrubber sobre PubMedQA/MedQuAD reais: com
        # re.IGNORECASE na regra de "paciente/patient", a exigência de inicial
        # maiúscula era anulada e qualquer par de palavras seguintes virava
        # [NOME] — em português isso apagaria metade do texto clínico.
        "O paciente apresentou dor torácica ao esforço.",
        "A paciente relatou febre persistente há três dias.",
        "Patient satisfaction was assessed 48 hours after discharge.",
        "International Patient Organisation for Primary Immunodeficiencies",
        "O paciente evoluiu com melhora clínica progressiva.",
    ],
)
def test_nao_confunde_texto_comum_depois_de_paciente_com_nome(frase: str) -> None:
    assert anonymize(frase) == frase


def test_texto_sem_pii_fica_intacto() -> None:
    texto = "Pneumonia adquirida na comunidade exige avaliação de gravidade antes da conduta."
    assert anonymize(texto) == texto


def test_string_vazia_nao_quebra() -> None:
    assert anonymize("") == ""


def test_anonymize_example_cobre_os_tres_campos() -> None:
    exemplo: InstructionExample = {
        "instruction": "O que fazer com o paciente João Silva?",
        "input": "CPF 123.456.789-00, internado em 12/03/2024.",
        "output": "Contatar o Dr. Carlos Andrade para reavaliação.",
    }

    resultado = anonymize_example(exemplo)

    assert "João Silva" not in resultado["instruction"]
    assert "123.456.789-00" not in resultado["input"]
    assert "12/03/2024" not in resultado["input"]
    assert "Carlos Andrade" not in resultado["output"]


def test_anonymize_example_nao_muta_a_entrada() -> None:
    exemplo: InstructionExample = {"instruction": "Paciente João Silva", "input": "", "output": "ok"}

    anonymize_example(exemplo)

    assert exemplo["instruction"] == "Paciente João Silva"
