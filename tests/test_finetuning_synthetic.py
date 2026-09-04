"""Geração dos dados sintéticos do hospital fictício (ESTRATEGIA.md §3).

A chamada ao Groq/Gemini não é testável offline, mas as duas partes que de
fato quebram são: (1) o parsing da resposta do modelo, que quase nunca vem
como JSON limpo, e (2) o prompt carregar a persona correta — o modelo
fine-tunado não pode aprender a prescrever direto, senão o treino trabalha
contra os guardrails de `safety/guardrails.py`. Ambas são puras e estão
cobertas aqui.
"""

from __future__ import annotations

from hospital_assistant.finetuning.synthetic import CATEGORIAS, build_prompt, parse_llm_batch


def test_parse_json_limpo() -> None:
    bruto = """
    [
      {"instruction": "Qual a conduta inicial na sepse?", "input": "", "output": "Sugiro considerar coleta de lactato."}
    ]
    """

    exemplos = parse_llm_batch(bruto)

    assert len(exemplos) == 1
    assert exemplos[0]["instruction"] == "Qual a conduta inicial na sepse?"
    assert exemplos[0]["output"] == "Sugiro considerar coleta de lactato."


def test_parse_com_cerca_markdown() -> None:
    """Groq e Gemini devolvem ```json ... ``` mesmo quando o prompt pede só JSON."""
    bruto = '```json\n[{"instruction": "P", "input": "", "output": "R suficientemente longa"}]\n```'

    assert len(parse_llm_batch(bruto)) == 1


def test_parse_com_texto_em_volta() -> None:
    bruto = 'Claro! Seguem os exemplos:\n[{"instruction": "P", "input": "", "output": "R"}]\nEspero ter ajudado.'

    assert len(parse_llm_batch(bruto)) == 1


def test_parse_preenche_input_ausente() -> None:
    exemplos = parse_llm_batch('[{"instruction": "P", "output": "R"}]')

    assert exemplos[0]["input"] == ""


def test_parse_descarta_entrada_incompleta() -> None:
    bruto = """
    [
      {"instruction": "boa", "input": "", "output": "resposta"},
      {"instruction": "sem resposta"},
      {"output": "sem pergunta"},
      "isso nem e um objeto",
      {"instruction": "", "output": "vazia"}
    ]
    """

    exemplos = parse_llm_batch(bruto)

    assert len(exemplos) == 1
    assert exemplos[0]["instruction"] == "boa"


def test_parse_resposta_sem_json_devolve_vazio() -> None:
    """Recusa do modelo / rate limit não pode virar exceção no meio de um lote longo."""
    assert parse_llm_batch("Desculpe, não posso ajudar com isso.") == []
    assert parse_llm_batch("") == []


def test_parse_converte_valores_nao_string() -> None:
    exemplos = parse_llm_batch('[{"instruction": "P", "input": null, "output": "R"}]')

    assert exemplos[0]["input"] == ""


def test_prompt_cobre_a_categoria_pedida() -> None:
    prompt = build_prompt("protocolos_internos", quantidade=5)

    assert CATEGORIAS["protocolos_internos"] in prompt
    assert "5" in prompt


def test_prompt_impoe_persona_nao_prescritiva() -> None:
    """O treino tem que reforçar o guardrail, não competir com ele."""
    prompt = build_prompt("modelos_de_receita", quantidade=3)

    assert "nunca prescreve" in prompt.lower() or "não prescreve" in prompt.lower()
    assert "validação" in prompt.lower()


def test_prompt_pede_portugues() -> None:
    """PubMedQA e MedQuAD são em inglês; o corpus sintético é o lado pt-BR do dataset."""
    assert "português" in build_prompt("faq_medicos", quantidade=3).lower()


def test_prompt_evita_repeticao_com_temas_ja_usados() -> None:
    prompt = build_prompt("faq_medicos", quantidade=3, evitar=["Como solicitar hemocultura?"])

    assert "Como solicitar hemocultura?" in prompt
