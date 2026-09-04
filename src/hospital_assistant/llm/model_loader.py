"""Carrega o modelo base + adapter LoRA em runtime, per ESTRATEGIA.md §3.3.

Dois backends implementam a mesma interface `LLM`:

- `FineTunedLLM`: Llama-3.2-3B-Instruct quantizado em 4-bit + o adapter LoRA
  publicado no Hugging Face Hub (ou um diretório local, logo após o treino).
- `MockLLM`: stand-in determinístico, usado quando não há adapter disponível
  ou quando as dependências de GPU não estão instaladas — o caso do
  devcontainer, do Docker e do `pytest`.

`load_llm()` escolhe sozinho e **registra qual escolheu**: a diferença entre
demonstrar o modelo fine-tunado e demonstrar o stand-in não pode ser
silenciosa. `descrever_backend()` existe para a Tela 1 poder mostrar isso.

A montagem do prompt não vive aqui — vem de `llm/prompt.py`, o mesmo módulo
que `finetuning/train.py` usa, para que o formato de treino e o de inferência
não possam divergir.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from hospital_assistant.llm.prompt import build_messages

logger = logging.getLogger(__name__)

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
BASE_MODEL_ESPELHO = "unsloth/Llama-3.2-3B-Instruct"

# Diretório onde `finetuning/train.py` grava o adapter no Colab.
LOCAL_ADAPTER_DIR = Path("outputs/adapter")

MAX_NEW_TOKENS = 512


@runtime_checkable
class LLM(Protocol):
    def generate(
        self,
        pergunta: str,
        contexto_rag: list[dict[str, Any]] | None = None,
        exames_pendentes: list[dict[str, Any]] | None = None,
    ) -> str: ...


@dataclass
class MockLLM:
    """Stand-in determinístico para quando não há adapter carregado.

    Ecoa as fontes recuperadas de propósito: assim a Tela 1 mostra visivelmente
    que o RAG chegou até o modelo, mesmo sem modelo de verdade — e um chunk
    que não chega ao prompt vira um sintoma observável em vez de um silêncio.
    """

    def generate(
        self,
        pergunta: str,
        contexto_rag: list[dict[str, Any]] | None = None,
        exames_pendentes: list[dict[str, Any]] | None = None,
    ) -> str:
        partes = [f"[MOCK LLM] Sugestão gerada para: {pergunta[:80]!r}"]

        if contexto_rag:
            fontes = ", ".join(str(c.get("source", "desconhecida")) for c in contexto_rag)
            partes.append(f"Fontes consultadas: {fontes}.")
        if exames_pendentes:
            partes.append(f"Exames pendentes considerados: {len(exames_pendentes)}.")

        return " ".join(partes)


@dataclass
class FineTunedLLM:
    """Modelo base 4-bit + adapter LoRA, carregados uma única vez (lazy)."""

    adapter: str
    modelo_base: str = BASE_MODEL
    _pipeline: Any = field(default=None, init=False, repr=False)

    def _carregar(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info("Carregando %s + adapter %s...", self.modelo_base, self.adapter)

        tokenizer = AutoTokenizer.from_pretrained(self.modelo_base)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            self.modelo_base,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            ),
            device_map="auto",
        )
        modelo = PeftModel.from_pretrained(base, self.adapter)
        modelo.eval()

        self._pipeline = (tokenizer, modelo)
        return self._pipeline

    def generate(
        self,
        pergunta: str,
        contexto_rag: list[dict[str, Any]] | None = None,
        exames_pendentes: list[dict[str, Any]] | None = None,
    ) -> str:
        tokenizer, modelo = self._carregar()

        mensagens = build_messages(pergunta, contexto_rag, exames_pendentes)
        entrada = tokenizer.apply_chat_template(
            mensagens, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(modelo.device)

        saida = modelo.generate(
            entrada,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        # Só os tokens novos: o prompt inteiro volta concatenado na saída.
        return tokenizer.decode(saida[0][entrada.shape[-1] :], skip_special_tokens=True).strip()


def _adapter_disponivel(local_adapter_dir: Path | None) -> str | None:
    """Descobre onde está o adapter: variável de ambiente ou diretório local."""
    repo = os.environ.get("HF_ADAPTER_REPO")
    if repo:
        return repo
    if local_adapter_dir is not None and (local_adapter_dir / "adapter_config.json").exists():
        return str(local_adapter_dir)
    return None


def load_llm(local_adapter_dir: Path | None = LOCAL_ADAPTER_DIR) -> LLM:
    """Devolve o backend disponível: fine-tunado se houver adapter, senão mock.

    Nunca levanta exceção por falta de adapter ou de GPU — o resto do pipeline
    (grafo, guardrails, auditoria, telas) precisa continuar demonstrável mesmo
    antes de o adapter existir, que é a situação prevista no plano B da
    ESTRATEGIA §13.
    """
    adapter = _adapter_disponivel(local_adapter_dir)
    if adapter is None:
        logger.warning("Nenhum adapter LoRA encontrado (defina HF_ADAPTER_REPO). Usando MockLLM.")
        return MockLLM()

    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        logger.warning("peft/torch indisponíveis neste ambiente. Usando MockLLM.")
        return MockLLM()

    return FineTunedLLM(adapter=adapter)


@lru_cache(maxsize=1)
def get_llm() -> LLM:
    """`load_llm()` memoizado — o backend usado pelo grafo em runtime.

    `load_llm()` sozinho reconstrói o backend a cada chamada, e o nó
    `gerar_sugestao_llm` roda uma vez por consulta: com o adapter real isso
    significaria recarregar 3B de parâmetros a cada pergunta da Tela 1. Os
    testes que precisam trocar o backend chamam `get_llm.cache_clear()`.
    """
    llm = load_llm()
    logger.info("Backend do assistente: %s", descrever_backend(llm))
    return llm


def descrever_backend(llm: LLM) -> str:
    """Texto curto para log e para a Tela 1 — qual modelo está de fato ativo."""
    if isinstance(llm, FineTunedLLM):
        return f"Llama-3.2-3B-Instruct + adapter LoRA ({llm.adapter})"
    return "MockLLM (stand-in determinístico — nenhum adapter LoRA carregado)"
