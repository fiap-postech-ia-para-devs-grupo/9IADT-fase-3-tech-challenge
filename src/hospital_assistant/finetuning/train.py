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
from pathlib import Path
from typing import Any

from hospital_assistant.finetuning.schema import InstructionExample
from hospital_assistant.llm.model_loader import (
    BASE_MODEL as BASE_MODEL_OFICIAL,
)
from hospital_assistant.llm.model_loader import (
    BASE_MODEL_ESPELHO,
    resolve_base_model,
)
from hospital_assistant.llm.prompt import build_training_messages
from hospital_assistant.paths import FINETUNING_METRICS, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

# `resolve_base_model` e os dois repositórios vêm de `llm/model_loader.py`, não
# são redefinidos aqui: treino e inferência **precisam** resolver para o mesmo
# repositório. Duas cópias da regra permitiriam treinar sobre o espelho e
# carregar contra o repo gated, que estoura `GatedRepoError` na primeira
# pergunta da Tela 1.
__all__ = ["BASE_MODEL_ESPELHO", "BASE_MODEL_OFICIAL", "resolve_base_model", "train"]

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


def _kwarg_de_sequencia(sft_config: Any) -> dict[str, int]:
    """Nome do parâmetro de comprimento máximo aceito por esta versão do `trl`.

    O `trl` renomeou `SFTConfig.max_seq_length` para `max_length`. Como o
    notebook instala sempre a versão mais recente (`pip install -U trl`), fixar
    um dos dois nomes faria o `SFTTrainer` morrer com `TypeError` — depois de
    ~3GB de instalação, do preparo do dataset e do download do modelo, numa
    sessão de T4 que é cara em tempo. Descobrir pela assinatura evita apostar
    numa versão.
    """
    import inspect

    parametros = inspect.signature(sft_config.__init__).parameters
    chave = "max_seq_length" if "max_seq_length" in parametros else "max_length"
    return {chave: MAX_SEQ_LENGTH}


def _carregar_splits() -> tuple[Any, Any]:
    from datasets import load_dataset

    train_path = PROCESSED_DATA_DIR / "train.jsonl"
    val_path = PROCESSED_DATA_DIR / "val.jsonl"
    # Os dois são checados: `eval_strategy="epoch"` exige o split de validação,
    # e sem esta checagem a ausência dele apareceria como erro cru do
    # `datasets` em vez da instrução de como resolver.
    for caminho in (train_path, val_path):
        if not caminho.exists() or caminho.stat().st_size == 0:
            raise FileNotFoundError(
                f"{caminho} não existe ou está vazio. Rode antes: "
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

    def formatting_func(exemplos: dict[str, Any]) -> str | list[str]:
        """Formata um lote ou um exemplo isolado.

        Versões diferentes do `trl` chamam `formatting_func` de formas
        diferentes: umas passam o batch inteiro (valores são listas), outras um
        exemplo por vez (valores são strings). Assumir só o batch faz o treino
        morrer com `TypeError` bem depois do download do modelo, dentro de uma
        sessão de Colab — detectar pela forma do dado custa três linhas.
        """
        if isinstance(exemplos["instruction"], str):
            return format_for_sft(
                {
                    "instruction": exemplos["instruction"],
                    "input": exemplos.get("input") or "",
                    "output": exemplos["output"],
                },
                tokenizer,
            )

        return [
            format_for_sft({"instruction": i, "input": c or "", "output": o}, tokenizer)
            for i, c, o in zip(
                exemplos["instruction"], exemplos["input"], exemplos["output"], strict=True
            )
        ]

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=LoraConfig(**LORA_KWARGS),
        formatting_func=formatting_func,
        args=SFTConfig(
            output_dir=str(output_dir),
            **_kwarg_de_sequencia(SFTConfig),
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
