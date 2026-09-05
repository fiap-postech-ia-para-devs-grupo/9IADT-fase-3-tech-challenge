"""Carrega o modelo base + adapter LoRA em runtime, per ESTRATEGIA.md §3.3.

Dois backends implementam a mesma interface `LLM`:

- `FineTunedLLM`: Llama-3.2-3B-Instruct + o adapter LoRA publicado no Hugging
  Face Hub (ou um diretório local, logo após o treino). Carrega em fp16 quando
  a GPU comporta e cai para 4-bit só quando a memória aperta.
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


# Pesos em fp16 (~6,4 GB) mais folga para o cache de atenção. Numa T4 de 15 GB
# sobra espaço; placas menores caem na quantização, que é o que a torna viável.
VRAM_MINIMA_FP16_GB = 9.0


def _cabe_em_fp16() -> bool:
    """Se a GPU comporta o modelo sem quantizar.

    Consulta a memória **livre**, não a total: o processo divide a placa com o
    modelo de embeddings do RAG e com o que mais estiver carregado na sessão.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        livre, _total = torch.cuda.mem_get_info()
    except Exception:  # noqa: BLE001 — sem GPU utilizável, a resposta é a mesma
        return False

    return livre / 2**30 >= VRAM_MINIMA_FP16_GB


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

    Devolve texto em formato de artigo — título, resumo do que foi consultado e
    encaminhamento — em vez de uma linha única. O formato importa porque é o
    que a fila de validação exibe: o médico revisor lê a resposta, e um bloco
    corrido com caminhos de arquivo no meio é ilegível.

    **Não repete o nome dos arquivos recuperados.** A procedência pertence ao
    painel de fontes, que mostra arquivo, trecho e score lado a lado; repeti-la
    dentro da resposta polui o texto sem acrescentar informação. O que fica é a
    contagem — suficiente para tornar observável se o RAG chegou ou não ao
    modelo, que é a razão de este stand-in existir.
    """

    def generate(
        self,
        pergunta: str,
        contexto_rag: list[dict[str, Any]] | None = None,
        exames_pendentes: list[dict[str, Any]] | None = None,
    ) -> str:
        n_fontes = len(contexto_rag or [])
        n_exames = len(exames_pendentes or [])

        linhas = [
            "### Sugestão preliminar",
            "",
            f"**Pergunta analisada:** {pergunta.strip()}",
            "",
            "#### Base consultada",
        ]

        if n_fontes:
            linhas.append(
                f"Foram recuperados {n_fontes} trecho(s) de protocolo relevantes para o caso. "
                "O detalhamento de cada um, com origem e grau de similaridade, está no painel "
                "de fontes desta resposta."
            )
        else:
            linhas.append(
                "Nenhum trecho de protocolo foi recuperado para esta pergunta. A sugestão "
                "abaixo não tem fundamentação documental e exige conferência redobrada."
            )

        if n_exames:
            linhas += [
                "",
                "#### Situação do paciente",
                f"Há {n_exames} exame(s) pendente(s) no prontuário, considerados nesta análise.",
            ]

        linhas += [
            "",
            "#### Encaminhamento",
            "Esta sugestão segue para a fila de validação e só vale como conduta após revisão "
            "de um médico responsável.",
            "",
            "> Resposta gerada em modo de demonstração: este ambiente não possui placa de vídeo "
            "dedicada, necessária para executar o modelo treinado. O fluxo de atendimento, as "
            "fontes consultadas e a validação humana são reais.",
        ]

        return "\n".join(linhas)


@dataclass
class FineTunedLLM:
    """Modelo base + adapter LoRA, carregados uma única vez (lazy).

    Três modos, mesmo adapter e mesma saída:

    - **GPU com folga** (padrão numa T4): fp16 sem quantização. É o mais rápido.
    - **GPU apertada**: 4-bit NF4 pelo bitsandbytes, que troca velocidade por
      memória. Foi o modo do treino, onde a memória era o gargalo.
    - **CPU**: bfloat16, sem quantização, porque o bitsandbytes exige CUDA. Os
      pesos entram inteiros na RAM (~6,4 GB) e a geração cai para minutos por
      resposta. Serve para ter resposta real onde não há placa, não para uso
      interativo.

    A escolha entre os dois primeiros é por memória livre, não por preferência:
    quantizar sem precisar só deixa a resposta lenta.
    """

    adapter: str
    # `None` = resolver na hora de carregar (oficial se acessível, senão
    # espelho). Fixar `BASE_MODEL` aqui quebraria quem treinou sobre o espelho.
    modelo_base: str | None = None
    em_cpu: bool = False
    _pipeline: Any = field(default=None, init=False, repr=False)

    def _carregar(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base = self.modelo_base or resolve_base_model()
        self.modelo_base = base
        logger.info(
            "Carregando %s + adapter %s em %s...",
            base,
            self.adapter,
            "CPU (sem quantização)" if self.em_cpu else "GPU (4-bit NF4)",
        )

        tokenizer = AutoTokenizer.from_pretrained(base)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if self.em_cpu:
            # Sem `quantization_config`: o bitsandbytes precisa de CUDA para
            # quantizar. bfloat16 em vez de float32 corta o consumo de RAM pela
            # metade (~6,4 GB no lugar de ~12,8 GB), que costuma ser a diferença
            # entre carregar e estourar a memória da máquina.
            modelo_base = AutoModelForCausalLM.from_pretrained(
                base, dtype=torch.bfloat16, device_map="cpu"
            )
        elif _cabe_em_fp16():
            # fp16 sem quantização **é mais rápido** que 4-bit, ao contrário do
            # que a intuição sugere: a NF4 existe para o treino caber na
            # memória (o "Q" de QLoRA), e paga desquantização a cada passo de
            # geração. Na inferência isso é custo puro. O modelo em fp16 ocupa
            # ~6,4 GB e sobra espaço numa T4 de 15 GB, então aqui a quantização
            # só tornava a resposta lenta sem necessidade.
            logger.info("VRAM suficiente: carregando em fp16 (mais rápido que 4-bit).")
            modelo_base = AutoModelForCausalLM.from_pretrained(
                base, dtype=torch.float16, device_map="auto"
            )
        else:
            from transformers import BitsAndBytesConfig

            logger.info("VRAM apertada: carregando em 4-bit NF4.")
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

        # `return_dict=True` é obrigatório: no transformers 5 o
        # `apply_chat_template` com `tokenize=True` devolve um `BatchEncoding`,
        # não mais um tensor de ids. Sem isso, `.shape` estoura AttributeError
        # e a máscara de atenção se perde.
        entrada = tokenizer.apply_chat_template(
            mensagens,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(modelo.device)

        saida = modelo.generate(
            **entrada,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        # Só os tokens novos: o prompt inteiro volta concatenado na saída.
        n_prompt = entrada["input_ids"].shape[-1]
        return tokenizer.decode(saida[0][n_prompt:], skip_special_tokens=True).strip()


def _adapter_disponivel(local_adapter_dir: Path | None) -> str | None:
    """Descobre onde está o adapter: variável de ambiente ou diretório local."""
    _carregar_env()
    repo = os.environ.get("HF_ADAPTER_REPO")
    if repo:
        return repo
    if local_adapter_dir is not None and (local_adapter_dir / "adapter_config.json").exists():
        return str(local_adapter_dir)
    return None


class AmbienteSemModelo(RuntimeError):
    """O adapter foi configurado, mas o ambiente não consegue carregá-lo."""


def _flag(nome: str) -> bool:
    _carregar_env()
    return os.environ.get(nome, "").strip().lower() in ("1", "true", "sim")


def _demonstracao_autorizada() -> bool:
    """Se o operador aceitou explicitamente rodar sem o modelo treinado."""
    return _flag("MODO_DEMONSTRACAO")


def _cpu_autorizada() -> bool:
    """Se o operador aceitou rodar o modelo real em CPU, ciente da lentidão."""
    return _flag("PERMITIR_CPU")


def load_llm(local_adapter_dir: Path | None = LOCAL_ADAPTER_DIR) -> LLM:
    """Devolve o backend a usar. Produção é o padrão.

    Configurar `HF_ADAPTER_REPO` é declarar que se quer o modelo treinado. Se o
    ambiente não puder carregá-lo, isto levanta `AmbienteSemModelo` em vez de
    devolver o stand-in: a degradação silenciosa já colocou uma demonstração no
    ar com o gerador de mentira e o único sinal foi um aviso perdido no log.
    Falhar alto custa um erro visível; falhar baixo custa a credibilidade do
    que está sendo demonstrado.

    Duas saídas continuam abertas para o mock, ambas explícitas:

    - `MODO_DEMONSTRACAO=true`, para quem sabe que não tem GPU e quer a
      interface mesmo assim;
    - nenhum adapter configurado, que é o caso dos testes e do plano B da
      ESTRATEGIA §13 — aí não há intenção de produção para frustrar.
    """
    adapter = _adapter_disponivel(local_adapter_dir)
    if adapter is None:
        logger.warning("Nenhum adapter LoRA encontrado (defina HF_ADAPTER_REPO). Usando MockLLM.")
        return MockLLM()

    faltando_gpu = _dependencias_faltando(exigir_gpu=True)
    if not faltando_gpu:
        return FineTunedLLM(adapter=adapter)

    # Sem GPU, o modelo real ainda é possível em CPU — devagar, mas real. Fica
    # atrás de uma flag porque a diferença é grande demais para ser automática:
    # minutos por resposta e ~6,4 GB de RAM não podem surpreender quem só subiu
    # a aplicação esperando a experiência normal.
    if _cpu_autorizada():
        faltando_cpu = _dependencias_faltando(exigir_gpu=False)
        if not faltando_cpu:
            logger.warning(
                "Sem GPU (falta %s). PERMITIR_CPU ativo: carregando o adapter %s em CPU. "
                "As respostas são do modelo treinado, mas levam minutos.",
                ", ".join(faltando_gpu),
                adapter,
            )
            return FineTunedLLM(adapter=adapter, em_cpu=True)
        faltando_gpu = faltando_cpu

    if _demonstracao_autorizada():
        logger.warning(
            "Adapter %s configurado, mas falta %s. MODO_DEMONSTRACAO ativo: usando MockLLM.",
            adapter,
            ", ".join(faltando_gpu),
        )
        return MockLLM()

    raise AmbienteSemModelo(
        "\n".join(
            [
                f"O adapter {adapter} está configurado, mas este ambiente não consegue "
                f"carregá-lo: falta {', '.join(faltando_gpu)}. A carga em 4-bit exige placa "
                "de vídeo com CUDA.",
                "",
                "Três saídas:",
                "· rode num ambiente com GPU — é o caso do Google Colab com T4;",
                "· defina PERMITIR_CPU=true para usar o modelo treinado em CPU: respostas "
                "reais, porém em minutos, e exige ~6,4 GB de RAM livre e ~7 GB de disco "
                "para o download;",
                "· defina MODO_DEMONSTRACAO=true para abrir a interface com respostas de "
                "demonstração.",
            ]
        )
    )


def _dependencias_faltando(exigir_gpu: bool = True) -> list[str]:
    """Lista o que impede carregar o modelo quantizado neste ambiente.

    Checar só `peft`/`torch` não bastava: os dois são dependências principais
    e estão na imagem Docker, enquanto `bitsandbytes` vive apenas no extra
    `finetuning`, que o Dockerfile não instala. Com `HF_ADAPTER_REPO` definido
    no compose, a guarda antiga passava e o erro só aparecia como traceback na
    primeira pergunta da Tela 1 — em vez da degradação para MockLLM que o
    docstring deste módulo promete. A carga em 4-bit também exige GPU.
    """
    modulos = ("torch", "peft", "bitsandbytes") if exigir_gpu else ("torch", "peft")

    faltando: list[str] = []
    for modulo in modulos:
        try:
            __import__(modulo)
        except Exception:  # noqa: BLE001 — bitsandbytes falha em import por falta de CUDA, não só ImportError
            faltando.append(modulo)

    if exigir_gpu and "torch" not in faltando:
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
        onde = "CPU, sem quantização" if llm.em_cpu else "GPU, 4-bit"
        return f"Llama-3.2-3B-Instruct + adapter LoRA ({llm.adapter}) — {onde}"
    return "MockLLM (stand-in determinístico — nenhum adapter LoRA carregado)"
