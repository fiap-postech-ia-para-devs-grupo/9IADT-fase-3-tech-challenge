"""Fine-tuning QLoRA do Llama-3.2-3B-Instruct, per ESTRATEGIA.md §3.

Este módulo é o código real do treino; `notebooks/finetuning_colab.ipynb` só o
importa e executa, porque a GPU está no Colab (T4) e não no devcontainer. Isso
atende literalmente o requisito do PDF de "projeto modularizado em Python" —
o notebook não é o lugar onde a lógica vive.

Configuração (decisões fechadas em ESTRATEGIA.md §1 e §3):

    quantização   4-bit NF4 + double quant, compute em float16 (T4 não tem bf16)
    LoRA          r=16, alpha=32, dropout=0.05, alvo q_proj/v_proj
    treino        batch 4, grad_accum 4 (batch efetivo 16), 3 epochs, lr 2e-4

As dependências pesadas (`torch`, `transformers`, `peft`, `trl`,
`bitsandbytes`) são importadas dentro das funções: o extra `finetuning` do
pyproject não é instalado no devcontainer nem na imagem Docker, e o import no
topo quebraria `pytest` para o time inteiro.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

from hospital_assistant.finetuning.schema import InstructionExample
from hospital_assistant.llm.prompt import build_training_messages
from hospital_assistant.paths import FINETUNING_METRICS, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

# Repositório oficial da Meta: `gated: manual`, exige licença aprovada.
BASE_MODEL_OFICIAL = "meta-llama/Llama-3.2-3B-Instruct"
# Re-upload dos MESMOS pesos, sem gate. Fallback quando a licença ainda não
# saiu — mantém a decisão de modelo base da ESTRATEGIA §1 intacta, muda só a
# origem do download.
BASE_MODEL_ESPELHO = "unsloth/Llama-3.2-3B-Instruct"

ADAPTER_DIR = Path("outputs/adapter")
MAX_SEQ_LENGTH = 1024

LORA_KWARGS: dict[str, Any] = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],
    "task_type": "CAUSAL_LM",
    "bias": "none",
}

TRAINING_KWARGS: dict[str, Any] = {
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 3,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "logging_steps": 5,
    "optim": "paged_adamw_8bit",
    "fp16": True,
    # Avalia e salva a cada época: a sessão do Colab cai (ESTRATEGIA.md §13) e
    # sem checkpoint por época o treino recomeça do zero.
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "save_total_limit": 2,
    "report_to": "none",
}


def resolve_base_model(token: str | None = None) -> str:
    """Devolve o repositório do modelo base acessível com o token atual.

    Tenta o repo oficial da Meta e cai para o espelho se o acesso ainda não
    tiver sido aprovado — em vez de estourar `GatedRepoError` no meio de uma
    sessão de Colab que custou minutos para subir.
    """
    from huggingface_hub import hf_hub_download

    token = token or os.environ.get("HF_TOKEN")
    try:
        hf_hub_download(BASE_MODEL_OFICIAL, filename="config.json", token=token)
        logger.info("Usando o repositório oficial: %s", BASE_MODEL_OFICIAL)
        return BASE_MODEL_OFICIAL
    except Exception as erro:  # noqa: BLE001 — qualquer falha de acesso cai no espelho
        logger.warning(
            "Sem acesso a %s (%s). Usando o espelho não-gated %s — mesmos pesos.",
            BASE_MODEL_OFICIAL,
            type(erro).__name__,
            BASE_MODEL_ESPELHO,
        )
        return BASE_MODEL_ESPELHO


def format_for_sft(example: InstructionExample, tokenizer: Any) -> str:
    """Renderiza um exemplo no chat template do modelo, com a resposta inclusa.

    Passa por `build_training_messages` (o mesmo módulo que a inferência usa)
    justamente para que treino e `model_loader` não divirjam — ver
    `llm/prompt.py`.
    """
    return tokenizer.apply_chat_template(
        build_training_messages(example),
        tokenize=False,
    )


def extract_loss_curves(log_history: list[dict[str, Any]]) -> dict[str, Any]:
    """Separa as curvas de treino/validação do `log_history` do Trainer.

    Entrega o material de `results/finetuning_metrics.json` exigido pelo #3:
    loss por passo dos dois lados e a perplexidade final (exp da eval loss),
    que é a métrica que a ESTRATEGIA §1 cita junto da loss.
    """
    train = [
        {"step": r.get("step"), "epoch": r.get("epoch"), "loss": r["loss"]}
        for r in log_history
        if "loss" in r
    ]
    eval_ = [
        {"step": r.get("step"), "epoch": r.get("epoch"), "loss": r["eval_loss"]}
        for r in log_history
        if "eval_loss" in r
    ]

    perplexidade = math.exp(eval_[-1]["loss"]) if eval_ else None

    return {"train": train, "eval": eval_, "final_eval_perplexity": perplexidade}


def _carregar_splits() -> tuple[Any, Any]:
    from datasets import load_dataset

    train_path = PROCESSED_DATA_DIR / "train.jsonl"
    val_path = PROCESSED_DATA_DIR / "val.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} não existe. Rode antes: "
            "uv run python -m hospital_assistant.finetuning.data_prep"
        )

    dados = load_dataset(
        "json",
        data_files={"train": str(train_path), "validation": str(val_path)},
    )
    return dados["train"], dados["validation"]


def train(output_dir: Path = ADAPTER_DIR, resume_from_checkpoint: bool = False) -> dict[str, Any]:
    """Roda o fine-tuning QLoRA e salva o adapter + as métricas.

    Devolve as curvas de loss (também gravadas em
    `results/finetuning_metrics.json`). Pensado para ser chamado do notebook:

        from hospital_assistant.finetuning.train import train
        train()
    """
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # `trl` e `bitsandbytes` só existem no extra `finetuning` (Colab), então o
    # type checker não os encontra no ambiente padrão — é esperado, não um bug.
    from trl import SFTConfig, SFTTrainer  # pyright: ignore[reportMissingImports]

    modelo_base = resolve_base_model()
    train_ds, val_ds = _carregar_splits()
    logger.info("Dataset: %d treino / %d validação", len(train_ds), len(val_ds))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(modelo_base)
    if tokenizer.pad_token is None:
        # Llama não define pad token; sem isso o SFTTrainer quebra ao montar batch.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        modelo_base,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False

    def formatting_func(batch: dict[str, list[Any]]) -> list[str]:
        return [
            format_for_sft(
                {"instruction": i, "input": c or "", "output": o},
                tokenizer,
            )
            for i, c, o in zip(batch["instruction"], batch["input"], batch["output"], strict=True)
        ]

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=LoraConfig(**LORA_KWARGS),
        formatting_func=formatting_func,
        args=SFTConfig(
            output_dir=str(output_dir),
            max_seq_length=MAX_SEQ_LENGTH,
            **TRAINING_KWARGS,
        ),
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint or None)

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Adapter salvo em %s", output_dir)

    metricas = extract_loss_curves(trainer.state.log_history)
    metricas["modelo_base"] = modelo_base
    metricas["hiperparametros"] = {**LORA_KWARGS, **TRAINING_KWARGS}
    metricas["exemplos"] = {"train": len(train_ds), "val": len(val_ds)}

    FINETUNING_METRICS.parent.mkdir(parents=True, exist_ok=True)
    FINETUNING_METRICS.write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Métricas salvas em %s", FINETUNING_METRICS)

    return metricas


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    train()
