# 9IADT — Fase 3 — Tech Challenge

**Assistente Virtual Médico**: fine-tuning de LLM (QLoRA) + RAG (LangChain/Chroma) +
orquestração (LangGraph) + interface (Streamlit), com guardrails de segurança,
validação humana obrigatória e auditoria completa.

O plano de entrega completo — arquitetura, decisões fechadas, especificação técnica de
cada módulo e a divisão de trabalho entre o time — vive em
[ESTRATEGIA.md](https://github.com/fiap-postech-ia-para-devs-grupo/exercicios/blob/main/src/fase-03/99-tech-challenge/ESTRATEGIA.md)
(repo `exercicios`). O trabalho está quebrado em tickets rastreáveis a partir do
[issue de mapa](https://github.com/fiap-postech-ia-para-devs-grupo/9IADT-fase-3-tech-challenge/issues/1)
neste repositório — comece por lá.

## Estado atual

Pipeline real de ponta a ponta: `db/`, `rag/` e `graph/`/`safety/` (Pessoas B e C) e as 3
telas do Streamlit (Pessoa D) já rodam contra dados reais — SQLite, Chroma, o grafo
LangGraph com guardrails, e a auditoria real em `clinical_audit.jsonl`.

O bloco de fine-tuning (Pessoa A) está com **o código completo e o dataset gerado**:
`data/processed/{train,val}.jsonl` tem 869 exemplos de treino e 97 de validação, vindos
de PubMedQA + MedQuAD + 180 protocolos sintéticos, anonimizados e curados
(`results/dataset_stats.json`). Falta **executar o treino**, que roda no Google Colab
com GPU T4 — veja [notebooks/finetuning_colab.ipynb](notebooks/finetuning_colab.ipynb).

Até o adapter LoRA ser publicado, `llm/model_loader.py` cai no `MockLLM` (stand-in
determinístico) e o registra no log — a Tela 1 continua funcionando ponta a ponta, mas
a sugestão não vem do modelo fine-tunado. Depois de publicar o adapter, basta definir
`HF_ADAPTER_REPO` no `.env` e o app passa a usá-lo sem mais nenhuma mudança de código.

Veja os tickets no GitHub Issues para o estado de cada bloco.

## Rodando localmente

Requer [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env  # preencher HF_TOKEN / GROQ_API_KEY / GOOGLE_API_KEY quando necessário
uv run python -m hospital_assistant.db.seed_mock_data  # popula data/patients_mock.db
uv run python -m hospital_assistant.rag.ingest          # popula data/chroma/
uv run streamlit run app.py
```

`uv run pytest` faz esse seed/ingest sozinho na primeira vez que rodar (via
`tests/conftest.py`), mas `app.py` não — sem os dois comandos acima, a Tela 1
consulta um banco de pacientes e um índice RAG vazios.

```bash
uv run pytest
```

## Rodando via Docker

`compose.yaml` monta `./data` no container (`volumes:`), então isso sobrepõe o que o
Dockerfile copiou no build — sem popular `./data` no host, o container sobe com banco de
pacientes e índice RAG vazios, do mesmo jeito que `app.py` local sem seed/ingest. Rode os
dois comandos da seção anterior antes (host) ou depois (dentro do container):

```bash
docker compose up --build -d
docker compose exec streamlit python -m hospital_assistant.db.seed_mock_data
docker compose exec streamlit python -m hospital_assistant.rag.ingest
```

App em http://localhost:8501.

## Dev Container

Abra o repositório no VS Code com a extensão "Dev Containers" e escolha
"Reopen in Container". O `postCreateCommand` roda `uv sync` e configura o kernel Jupyter
automaticamente. Copie `.devcontainer/.env.example` para `.devcontainer/.env` antes de
abrir, preenchendo `GIT_USER_NAME`/`GIT_USER_EMAIL`/`GITHUB_TOKEN`.

## Estrutura

```
src/hospital_assistant/
├── finetuning/   # schema, anonymize, sources, synthetic, data_prep, train, evaluate — Pessoa A
├── llm/          # prompt (template treino+inferência), model_loader (base + adapter) — Pessoa A
├── rag/          # ingest, retriever (Chroma) — Pessoa B
├── db/           # schema, seed, patient_tools (SQLite mock) — Pessoa B
├── graph/        # state, nodes, flow (LangGraph) — Pessoa C
└── safety/       # guardrails, audit_log — Pessoa C
app.py            # Streamlit, 3 telas — Pessoa D
notebooks/finetuning_colab.ipynb  # roda no Colab (GPU T4) — Pessoa A
docs/relatorio_tecnico.md         # relatório técnico — Pessoa E
tests/                            # Pessoa E (+ smoke tests da base)
```

Fine-tuning real (QLoRA) roda no Google Colab, não no devcontainer — o extra
`finetuning` do `pyproject.toml` (`bitsandbytes`, `trl`, `accelerate`) não é instalado
por padrão.

## Fine-tuning

O dataset já está gerado e é reprodutível. Para regenerá-lo do zero:

```bash
uv run python -m hospital_assistant.finetuning.data_prep
```

Baixa PubMedQA e MedQuAD, reaproveita o corpus sintético versionado em
`data/raw/sinteticos_finetuning.jsonl` (só chama o Groq/Gemini se ele não existir — para
isso, preencha `GROQ_API_KEY` ou `GOOGLE_API_KEY` no `.env`), anonimiza, cura, deduplica
e escreve `data/processed/{train,val}.jsonl` + `results/dataset_stats.json`.

O treino roda no Colab (GPU T4): abra `notebooks/finetuning_colab.ipynb`, cadastre
`HF_TOKEN` nos *Secrets* do notebook e execute as células na ordem. O notebook clona o
repositório, chama `train()`, plota as curvas de loss, publica o adapter LoRA no
Hugging Face Hub e roda o comparativo base vs. fine-tuned.

> O modelo base `meta-llama/Llama-3.2-3B-Instruct` é *gated* (revisão manual da Meta).
> Peça acesso com antecedência na [página do modelo](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).
> Sem a aprovação, `resolve_base_model()` cai automaticamente para
> `unsloth/Llama-3.2-3B-Instruct` — os mesmos pesos, sem gate.

Depois de publicar, aponte o app para o adapter:

```bash
echo "HF_ADAPTER_REPO=seu-usuario/hospital-assistant-llama32-3b-lora" >> .env
```

## Vídeo

_(link a adicionar por Pessoa E ao final da Fase 3)_
