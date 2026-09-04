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
from hospital_assistant.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

# Repositório oficial da Meta: `gated: manual`, exige licença aprovada.
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
# Re-upload dos MESMOS pesos, sem gate. Mantém a decisão de modelo base da
# ESTRATEGIA §1 intacta quando a licença ainda não saiu; muda só a origem.
BASE_MODEL_ESPELHO = "unsloth/Llama-3.2-3B-Instruct"

# Diretório onde `finetuning/train.py` grava o adapter no Colab.
LOCAL_ADAPTER_DIR = Path("outputs/adapter")

MAX_NEW_TOKENS = 512


def resolve_base_model(token: str | None = None) -> str:
    """Devolve o repositório do modelo base acessível com o token atual.

    Vive aqui, e não em `finetuning/train.py`, porque **treino e inferência
    precisam resolver para o mesmo repositório**. Com a resolução só no lado do
    treino, um adapter treinado sobre o espelho seria carregado em runtime
    contra o repo gated da Meta e a primeira pergunta da Tela 1 estouraria
    `GatedRepoError` — justamente o caminho que o README documenta como
    "funciona sem mudança de código".
    """
    from huggingface_hub import hf_hub_download

    token = token or os.environ.get("HF_TOKEN")
    try:
        hf_hub_download(BASE_MODEL, filename="config.json", token=token)
        return BASE_MODEL
    except Exception as erro:  # noqa: BLE001 — qualquer falha de acesso cai no espelho
        logger.warning(
            "Sem acesso a %s (%s). Usando o espelho não-gated %s — mesmos pesos.",
            BASE_MODEL,
            type(erro).__name__,
            BASE_MODEL_ESPELHO,
        )
        return BASE_MODEL_ESPELHO


def _carregar_env() -> None:
    """Lê o `.env` do projeto sem sobrescrever variáveis já definidas.

    Sem isto, `HF_ADAPTER_REPO=...` no `.env` (que é o que o README manda
    fazer) seria ignorado ao rodar `streamlit run app.py`: nada no caminho do
    app chama `load_dotenv`, só os `__main__` de data_prep/evaluate. O
    operador acharia estar demonstrando o modelo fine-tunado enquanto o app
    seguiria no MockLLM — exatamente a falha silenciosa que este módulo existe
    para evitar. `override=False` mantém a precedência do ambiente real
    (Docker passa as variáveis por `env_file`).
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


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
    # `None` = resolver na hora de carregar (oficial se acessível, senão
    # espelho). Fixar `BASE_MODEL` aqui quebraria quem treinou sobre o espelho.
    modelo_base: str | None = None
    _pipeline: Any = field(default=None, init=False, repr=False)

    def _carregar(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        base = self.modelo_base or resolve_base_model()
        self.modelo_base = base
        logger.info("Carregando %s + adapter %s...", base, self.adapter)

        tokenizer = AutoTokenizer.from_pretrained(base)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        modelo_base = AutoModelForCausalLM.from_pretrained(
            base,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            ),
            device_map="auto",
        )
        modelo = PeftModel.from_pretrained(modelo_base, self.adapter)
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
    _carregar_env()
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

    faltando = _dependencias_faltando()
    if faltando:
        logger.warning(
            "Adapter %s configurado, mas o ambiente não suporta carregá-lo (%s). Usando MockLLM.",
            adapter,
            ", ".join(faltando),
        )
        return MockLLM()

    return FineTunedLLM(adapter=adapter)


def _dependencias_faltando() -> list[str]:
    """Lista o que impede carregar o modelo quantizado neste ambiente.

    Checar só `peft`/`torch` não bastava: os dois são dependências principais
    e estão na imagem Docker, enquanto `bitsandbytes` vive apenas no extra
    `finetuning`, que o Dockerfile não instala. Com `HF_ADAPTER_REPO` definido
    no compose, a guarda antiga passava e o erro só aparecia como traceback na
    primeira pergunta da Tela 1 — em vez da degradação para MockLLM que o
    docstring deste módulo promete. A carga em 4-bit também exige GPU.
    """
    faltando: list[str] = []

    for modulo in ("torch", "peft", "bitsandbytes"):
        try:
            __import__(modulo)
        except Exception:  # noqa: BLE001 — bitsandbytes falha em import por falta de CUDA, não só ImportError
            faltando.append(modulo)

    if "torch" not in faltando:
        import torch

        if not torch.cuda.is_available():
            faltando.append("GPU CUDA")

    return faltando


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
