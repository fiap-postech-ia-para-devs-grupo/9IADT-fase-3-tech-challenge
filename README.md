# 9IADT — Fase 3 — Tech Challenge

**Assistente Virtual Médico**: fine-tuning de LLM (QLoRA) + RAG (LangChain/Chroma) +
orquestração (LangGraph) + interface (Streamlit), com guardrails de segurança,
validação humana obrigatória e auditoria completa.

O plano de entrega completo — arquitetura, decisões fechadas, especificação técnica de
cada módulo e a divisão de trabalho entre o time — vive em
[docs/ESTRATEGIA.md](docs/ESTRATEGIA.md). O trabalho está quebrado em tickets
rastreáveis a partir do
[issue de mapa](https://github.com/fiap-postech-ia-para-devs-grupo/9IADT-fase-3-tech-challenge/issues/1)
neste repositório — comece por lá.

## Time

| Integrante | Responsabilidade | Entregável |
| --- | --- | --- |
| **Marcelo Costa** | Fine-tuning da LLM | `src/hospital_assistant/finetuning/`, `src/hospital_assistant/llm/`, `notebooks/finetuning_colab.ipynb`, adapter LoRA no HF Hub |
| **Vinicius Geizler** | LangChain / RAG / Dados | `src/hospital_assistant/rag/`, `src/hospital_assistant/db/`, vector store em `data/chroma/` |
| **Antonio Bazo** | LangGraph e Segurança | `src/hospital_assistant/graph/`, `src/hospital_assistant/safety/` |
| **Renato Mattos** | Interface Streamlit | `app.py`, `src/hospital_assistant/ui/` |
| **Vinicius Blasque** | Relatório, testes e vídeo | `docs/relatorio_tecnico.md`, `tests/`, vídeo de demonstração |

## Estado atual

Pipeline real de ponta a ponta: `db/`, `rag/` e `graph/`/`safety/` (Geizler e Antonio) e as 3
telas do Streamlit (Renato) já rodam contra dados reais — SQLite, Chroma, o grafo
LangGraph com guardrails, e a auditoria real em `clinical_audit.jsonl`.

O bloco de fine-tuning (Marcelo) está **executado de ponta a ponta**. O dataset tem 869
exemplos de treino e 97 de validação (PubMedQA + MedQuAD + 180 protocolos sintéticos,
anonimizados e curados — `results/dataset_stats.json`). O treino QLoRA rodou numa GPU T4
por 51 minutos, com loss de validação caindo 1.2504 → 1.2008 e perplexidade final 3.32
(`results/finetuning_metrics.json`). O adapter LoRA está publicado em
[`agendesse/hospital-assistant-llama32-3b-lora`](https://huggingface.co/agendesse/hospital-assistant-llama32-3b-lora).

> ⚠️ **O comparativo base vs. fine-tuned revelou uma regressão de segurança.** O modelo
> fine-tunado ficou 60% mais conciso, mas passou a responder com dose e posologia a
> perguntas que o modelo base corretamente recusava. A causa está no corpus sintético
> (categorias de receita e interpretação de exames), não no treino. O guardrail e a fila
> de validação humana retêm essas respostas — mas o adapter **não deve ser promovido a
> padrão do app sem revisar o corpus**. Detalhes e mitigações na seção 3.3 do
> [relatório técnico](docs/relatorio_tecnico.md).

Sem `HF_ADAPTER_REPO` no `.env`, `llm/model_loader.py` cai no `MockLLM` (stand-in
determinístico) e registra isso no log — a Tela 1 continua funcionando ponta a ponta.
Definir a variável faz o app usar o adapter, sem nenhuma mudança de código.

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
`tests/conftest.py`), mas `app.py` não — sem os dois comandos acima, o assistente
consulta um banco de pacientes e um índice RAG vazios.

Sem placa de vídeo há duas saídas, ambas explícitas no `.env` — com
`HF_ADAPTER_REPO` preenchido o app recusa abrir em vez de responder com o
stand-in em silêncio:

| Variável | O que faz | Custo |
| --- | --- | --- |
| `PERMITIR_CPU=true` | roda o modelo **treinado** em CPU, sem quantização | minutos por resposta; ~6,4 GB de RAM e ~7 GB de disco |
| `MODO_DEMONSTRACAO=true` | abre a interface com respostas de demonstração | instantâneo, mas o texto não vem do modelo |

`PERMITIR_CPU` tem precedência: resposta real ganha do stand-in.

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
├── finetuning/   # schema, anonymize, sources, synthetic, data_prep, train, evaluate — Marcelo
├── llm/          # prompt (template treino+inferência), model_loader (base + adapter) — Marcelo
├── rag/          # ingest, retriever (Chroma) — Geizler
├── db/           # schema, seed, patient_tools (SQLite mock) — Geizler
├── graph/        # state, nodes, flow (LangGraph) — Antonio
└── safety/       # guardrails, audit_log — Antonio
app.py            # Streamlit, portal clínico — Renato
notebooks/finetuning_colab.ipynb  # roda no Colab (GPU T4) — Marcelo
docs/relatorio_tecnico.md         # relatório técnico — Blasque
tests/                            # Blasque (+ testes de cada módulo por seu autor)
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

_(link a adicionar por Vinicius Blasque ao final da Fase 3)_
