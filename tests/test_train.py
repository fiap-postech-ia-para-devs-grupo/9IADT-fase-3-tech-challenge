"""Configuração do treino QLoRA (#3).

O treino em si roda no Colab e não é testável aqui — mas três coisas que
decidem se ele funciona são puras e estão cobertas: os hiperparâmetros
fechados em ESTRATEGIA.md §3 (teste de regressão contra alguém "ajustar" um
valor sem reabrir a decisão), a formatação do exemplo via chat template do
tokenizer, e a extração das curvas de loss que alimentam
`results/finetuning_metrics.json`.
"""

from __future__ import annotations

import logging

from hospital_assistant.finetuning.schema import InstructionExample
from hospital_assistant.finetuning.train import (
    BASE_MODEL_ESPELHO,
    BASE_MODEL_OFICIAL,
    LORA_KWARGS,
    MAX_SEQ_LENGTH,
    TRAINING_KWARGS,
    _sft_kwargs,
    extract_loss_curves,
    format_for_sft,
)


class TokenizerFalso:
    """Stub do tokenizer: registra o que recebeu em `apply_chat_template`."""

    def __init__(self) -> None:
        self.recebido: list[dict[str, str]] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False) -> str:
        self.recebido = messages
        return " || ".join(f"{m['role']}:{m['content']}" for m in messages)


def test_hiperparametros_lora_batem_com_a_estrategia() -> None:
    """ESTRATEGIA.md §1 marca isso como decisão fechada — o teste é o cadeado."""
    assert LORA_KWARGS["r"] == 16
    assert LORA_KWARGS["lora_alpha"] == 32
    assert LORA_KWARGS["lora_dropout"] == 0.05
    assert LORA_KWARGS["target_modules"] == ["q_proj", "v_proj"]
    assert LORA_KWARGS["task_type"] == "CAUSAL_LM"


def test_hiperparametros_de_treino_batem_com_a_estrategia() -> None:
    assert TRAINING_KWARGS["per_device_train_batch_size"] == 4
    assert TRAINING_KWARGS["gradient_accumulation_steps"] == 4
    assert TRAINING_KWARGS["num_train_epochs"] == 3
    assert TRAINING_KWARGS["learning_rate"] == 2e-4


def test_modelo_base_e_o_llama_3_2_3b_nas_duas_origens() -> None:
    """O espelho existe porque o repo da Meta é gated; os pesos são os mesmos."""
    assert BASE_MODEL_OFICIAL == "meta-llama/Llama-3.2-3B-Instruct"
    assert BASE_MODEL_ESPELHO.endswith("Llama-3.2-3B-Instruct")


def test_format_for_sft_usa_o_chat_template_do_tokenizer() -> None:
    tokenizer = TokenizerFalso()
    exemplo: InstructionExample = {"instruction": "Qual a conduta na sepse?", "input": "", "output": "Sugiro coletar lactato."}

    texto = format_for_sft(exemplo, tokenizer)

    assert "system:" in texto
    assert "Qual a conduta na sepse?" in texto
    assert "Sugiro coletar lactato." in texto


def test_format_for_sft_inclui_a_resposta_como_turno_do_assistente() -> None:
    """Sem o turno do assistente no texto, o SFTTrainer não tem alvo para aprender."""
    tokenizer = TokenizerFalso()

    exemplo: InstructionExample = {"instruction": "P", "input": "", "output": "R"}
    format_for_sft(exemplo, tokenizer)

    assert tokenizer.recebido[-1] == {"role": "assistant", "content": "R"}


def test_extract_loss_curves_separa_treino_e_validacao() -> None:
    log_history = [
        {"loss": 1.8, "epoch": 0.5, "step": 10},
        {"eval_loss": 1.7, "epoch": 1.0, "step": 20},
        {"loss": 1.2, "epoch": 1.5, "step": 30},
        {"eval_loss": 1.1, "epoch": 2.0, "step": 40},
    ]

    curvas = extract_loss_curves(log_history)

    assert [p["loss"] for p in curvas["train"]] == [1.8, 1.2]
    assert [p["loss"] for p in curvas["eval"]] == [1.7, 1.1]


def test_extract_loss_curves_calcula_perplexidade_final() -> None:
    """ESTRATEGIA.md §1 pede loss *e* perplexity como métrica de avaliação."""
    curvas = extract_loss_curves([{"eval_loss": 0.0, "epoch": 1.0, "step": 1}])

    assert curvas["final_eval_perplexity"] == 1.0


def test_extract_loss_curves_sem_validacao_nao_quebra() -> None:
    curvas = extract_loss_curves([{"loss": 1.5, "epoch": 1.0, "step": 1}])

    assert curvas["eval"] == []
    assert curvas["final_eval_perplexity"] is None


def test_extract_loss_curves_ignora_registro_sem_loss() -> None:
    """O `log_history` termina com um resumo (`train_runtime`) que não é ponto de curva."""
    curvas = extract_loss_curves([{"train_runtime": 123.4, "epoch": 3.0}])

    assert curvas["train"] == []


# --------------------------------------------------------------------------
# _sft_kwargs — tolerância à deriva de API do trl
# --------------------------------------------------------------------------


class SFTConfigAntigo:
    """Assinatura do `trl` anterior à 1.12: `max_seq_length` e `warmup_ratio`."""

    def __init__(
        self,
        output_dir=None,
        max_seq_length=None,
        warmup_ratio=None,
        per_device_train_batch_size=None,
        gradient_accumulation_steps=None,
        num_train_epochs=None,
        learning_rate=None,
        lr_scheduler_type=None,
        logging_steps=None,
        optim=None,
        fp16=None,
        eval_strategy=None,
        save_strategy=None,
        save_total_limit=None,
        report_to=None,
    ) -> None: ...


class SFTConfigNovo:
    """`trl` 1.12: renomeou para `max_length` e removeu `warmup_ratio`."""

    def __init__(
        self,
        output_dir=None,
        max_length=None,
        per_device_train_batch_size=None,
        gradient_accumulation_steps=None,
        num_train_epochs=None,
        learning_rate=None,
        lr_scheduler_type=None,
        logging_steps=None,
        optim=None,
        fp16=None,
        eval_strategy=None,
        save_strategy=None,
        save_total_limit=None,
        report_to=None,
    ) -> None: ...


def test_sft_kwargs_usa_max_seq_length_na_versao_antiga() -> None:
    kwargs = _sft_kwargs(SFTConfigAntigo)

    assert kwargs["max_seq_length"] == MAX_SEQ_LENGTH
    assert "max_length" not in kwargs


def test_sft_kwargs_usa_max_length_na_versao_nova() -> None:
    """O `trl` renomeou o parâmetro; fixar o nome antigo mata o treino no Colab."""
    kwargs = _sft_kwargs(SFTConfigNovo)

    assert kwargs["max_length"] == MAX_SEQ_LENGTH
    assert "max_seq_length" not in kwargs


def test_sft_kwargs_descarta_parametro_inexistente() -> None:
    """`warmup_ratio` sumiu do SFTConfig na 1.12 — passá-lo levanta TypeError."""
    assert "warmup_ratio" not in _sft_kwargs(SFTConfigNovo)


def test_sft_kwargs_preserva_os_hiperparametros_fechados() -> None:
    """Descartar o incompatível não pode levar junto o que a ESTRATEGIA fixou."""
    kwargs = _sft_kwargs(SFTConfigNovo)

    assert kwargs["per_device_train_batch_size"] == 4
    assert kwargs["gradient_accumulation_steps"] == 4
    assert kwargs["num_train_epochs"] == 3
    assert kwargs["learning_rate"] == 2e-4


def test_sft_kwargs_avisa_no_log_ao_descartar(caplog) -> None:
    """Perder um parâmetro em silêncio mudaria o treino sem rastro no relatório."""
    with caplog.at_level(logging.WARNING):
        _sft_kwargs(SFTConfigNovo)

    assert "warmup_ratio" in caplog.text
