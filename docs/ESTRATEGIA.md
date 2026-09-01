# Tech Challenge Fase 3 — Estratégia de Entrega

> Projeto: Assistente Virtual Médico (fine-tuning de LLM + LangChain + LangGraph)
> Time: 5 integrantes · 35 dias · ~1h/dia por pessoa · **~175h efetivas totais**

---

## 1. Decisões Fechadas (não reabrir)

| Decisão                    | Escolha                                                              | Motivo                                                          |
| --------------------------- | --------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Repositório                 | Novo repositório, do zero                                             | Domínio e stack completamente diferentes da Fase 2               |
| Time                        | Mesmas 5 pessoas, especialidade análoga à Fase 2                     | Preserva curva de aprendizado individual                         |
| Compute de fine-tuning      | Google Colab free (GPU T4)                                            | Sem custo, suficiente com QLoRA                                  |
| Técnica de fine-tuning      | QLoRA (4-bit) via `peft` + `bitsandbytes` + `trl`                     | Viável em 16GB de VRAM do T4 dentro do limite de sessão          |
| Modelo base                 | `meta-llama/Llama-3.2-3B-Instruct`                                     | Equilíbrio entre qualidade e velocidade de treino no T4          |
| Dataset de fine-tuning      | Híbrido: subset PubMedQA/MedQuAD + protocolos sintéticos gerados por LLM | Cobre "conhecimento clínico geral" + "dados próprios do hospital" |
| Conhecimento (RAG)          | Vector store (Chroma) com embeddings locais                           | Protocolos/FAQs/laudos — sem custo de API de embeddings          |
| Dados do paciente           | SQLite mockado + *tools* estruturadas (não SQL agent livre)           | Evita text-to-SQL alucinado sobre dados clínicos                 |
| Orquestração de fluxo       | LangGraph (grafo linear com checkpoints de segurança)                 | Requisito explícito do entregável; modela decisão automatizada   |
| Guardrail de prescrição     | System prompt + nó de validação no grafo + fila de aprovação humana   | Nenhuma resposta chega ao usuário sem revisão humana              |
| Explainability              | Exibir chunks do RAG + score de similaridade                          | Rastreável, não depende da LLM "dizer a verdade" sobre a fonte    |
| Logging/auditoria           | Tabela SQLite dedicada                                                | Consultável direto na Tela 3 do Streamlit                        |
| Avaliação do modelo         | Loss/perplexity do treino + comparação qualitativa base vs. fine-tuned | Viável no tempo disponível, gera material rico pro relatório      |
| Serving do modelo           | Adapter LoRA salvo e publicado no Hugging Face Hub                    | Leve de versionar, reprodutível por qualquer um do time          |
| Interface                   | Streamlit (mesmo padrão da Fase 2), 3 telas                           | Familiaridade do time + separação de responsabilidades           |
| Organização do código       | `src/` modularizado é o código real; notebook só para treino no Colab | Atende literalmente o requisito "projeto modularizado em Python" |
| LLM provider (dados sintéticos) | Groq / Gemini free tier                                          | Mesma filosofia "sem custo, portável" da Fase 2                  |

---

## 2. Arquitetura da Solução

```
┌───────────────────────────────────────────────────────────────────┐
│                  notebooks/finetuning_colab.ipynb (Colab T4)        │
│                                                                       │
│  [1] Preparação de dados          [2] Fine-tuning QLoRA             │
│      • Curadoria + anonimização       • Llama-3.2-3B-Instruct       │
│        (src/finetuning/data_prep.py)  • LoRA r=16, 4-bit NF4         │
│      • PubMedQA/MedQuAD + sintéticos  • src/finetuning/train.py     │
│                                                                       │
│  [3] Avaliação                    [4] Publicação                    │
│      • Loss/perplexity                • Push adapter → HF Hub       │
│      • Comparativo base x tuned       • results/finetuning_*.json   │
└───────────────────────────────────────────────────────────────────┘
                         │ adapter LoRA (HF Hub)
                         ▼
┌───────────────────────────────────────────────────────────────────┐
│                         src/graph/flow.py (LangGraph)                │
│                                                                       │
│  receber_paciente                                                    │
│        │                                                             │
│        ▼                                                             │
│  verificar_exames_pendentes ──► src/db (SQLite mock prontuários)     │
│        │                                                             │
│        ▼                                                             │
│  consultar_protocolo (RAG) ────► src/rag (Chroma + embeddings)       │
│        │                                                             │
│        ▼                                                             │
│  gerar_sugestao_llm ────────────► modelo base + adapter LoRA         │
│        │                                                             │
│        ▼                                                             │
│  validar_seguranca ─────────────► src/safety/guardrails.py           │
│        │                                                             │
│        ▼                                                             │
│  emitir_alerta_se_necessario                                         │
│        │                                                             │
│        ▼                                                             │
│  log_auditoria ─────────────────► src/safety/audit_log.py (SQLite)   │
│        │                                                             │
│        ▼                                                             │
│  [status = "pendente de validação humana"]                           │
└───────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌───────────────────────────────────────────────────────────────────┐
│                        app.py (Streamlit)                            │
│                                                                       │
│  Tela 1: Consulta ao Assistente                                      │
│    • Médico digita pergunta / seleciona paciente                     │
│    • Roda o grafo LangGraph                                          │
│    • Mostra resposta com status "pendente de validação"              │
│                                                                       │
│  Tela 2: Fila de Validação Humana                                    │
│    • Lista respostas pendentes                                       │
│    • Exibe fontes RAG (chunks + score) para explainability           │
│    • Botões: Aprovar / Rejeitar / Editar                             │
│                                                                       │
│  Tela 3: Auditoria e Histórico                                       │
│    • Consulta a tabela de auditoria (SQLite)                         │
│    • Filtros por status, paciente, data                              │
└───────────────────────────────────────────────────────────────────┘
```

---

## 3. Fine-tuning — Especificação Técnica

### Dataset

| Fonte                              | Papel                                              | Volume aproximado |
| ----------------------------------- | --------------------------------------------------- | ------------------ |
| PubMedQA (subset)                   | Conhecimento clínico geral, formato pergunta/resposta | ~500 exemplos      |
| MedQuAD (subset)                    | Perguntas frequentes de saúde                        | ~300 exemplos      |
| Protocolos sintéticos (gerados via Groq/Gemini) | Simula "protocolos internos", modelos de laudo/receita, FAQs de médicos do hospital fictício | ~150-200 exemplos |

**Pipeline (`src/finetuning/data_prep.py`):**

1. Normalizar todas as fontes para o formato de instrução:
   ```json
   {"instruction": "...", "input": "...", "output": "..."}
   ```
2. **Anonimização:** aplicar scrubber de regex (nomes, CPF, datas, números de prontuário) mesmo nos dados sintéticos — documenta a técnica exigida pelo PDF e serve de exemplo caso dados reais sejam usados no futuro.
3. **Curadoria:** deduplicação, filtro de exemplos curtos/incoerentes, revisão manual de uma amostra.
4. Split treino/validação (90/10).
5. Salvar em `data/processed/` (versão final usada no treino).

### Treino (`src/finetuning/train.py`, executado via `notebooks/finetuning_colab.ipynb`)

```python
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
# Quantização 4-bit (NF4) via BitsAndBytesConfig
# SFTTrainer (trl): batch_size=4, grad_accum=4, epochs=3, lr=2e-4
```

### Avaliação (`src/finetuning/evaluate.py`)

- Curva de loss treino/validação → `results/finetuning_metrics.json`
- Tabela comparativa: 8-10 perguntas de teste, resposta do modelo base vs. fine-tuned lado a lado → `results/eval_comparativo.json`

### Publicação

- Push do adapter LoRA (não o modelo completo) para um repositório no Hugging Face Hub.
- `src/llm/model_loader.py` carrega `base_model + adapter` em runtime (app e testes).

---

## 4. LangChain / RAG — Especificação Técnica

### Indexação (`src/rag/ingest.py`)

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local, sem custo de API)
- Vector store: Chroma, persistido em disco (`data/chroma/`)
- Documentos indexados: protocolos sintéticos + amostra de PubMedQA/MedQuAD usada como base de conhecimento

### Retriever (`src/rag/retriever.py`)

- Top-k (k=3) por similaridade, retorna texto + metadado de origem + score — usado tanto para gerar contexto quanto para explainability na Tela 2.

### Dados estruturados do paciente (`src/db/`)

- `schema.sql`: tabelas `pacientes`, `exames`, `medicacoes`, `alertas`
- `seed_mock_data.py`: popula com dados sintéticos de pacientes fictícios
- `patient_tools.py`: funções Python parametrizadas expostas como *tools* do LangChain (`get_pending_exams(paciente_id)`, `get_patient_history(paciente_id)`) — **não** um agente de SQL livre, para eliminar risco de query alucinada sobre dados clínicos.

---

## 5. LangGraph — Especificação Técnica

### Estado (`src/graph/state.py`)

```python
class AssistantState(TypedDict):
    paciente_id: str | None
    pergunta: str
    exames_pendentes: list[dict]
    contexto_rag: list[dict]       # chunks + fonte + score
    sugestao_llm: str
    flags_seguranca: list[str]
    alerta: str | None
    status: str                     # "pendente" | "aprovado" | "rejeitado"
```

### Nós (`src/graph/nodes.py`)

| Nó                              | Responsabilidade                                                        |
| --------------------------------- | --------------------------------------------------------------------- |
| `receber_paciente`                | Valida input, carrega dados básicos do paciente (se houver)             |
| `verificar_exames_pendentes`      | Consulta `patient_tools.get_pending_exams`                              |
| `consultar_protocolo`             | Chama o retriever RAG                                                   |
| `gerar_sugestao_llm`              | Monta prompt com contexto + chama modelo fine-tuned                     |
| `validar_seguranca`               | Roda `guardrails.py`; bloqueia/reformula linguagem prescritiva direta   |
| `emitir_alerta_se_necessario`     | Gera alerta para equipe médica conforme regras (ex: exame crítico pendente) |
| `log_auditoria`                   | Grava todo o rastro da execução na tabela de auditoria                  |

### Montagem (`src/graph/flow.py`)

`StateGraph` linear conectando os nós acima, compilado uma vez e reutilizado pelo `app.py`.

---

## 6. Segurança, Guardrails e Auditoria

### Guardrail de prescrição (`src/safety/guardrails.py`)

- System prompt do modelo: papel de "sugerir e apoiar decisão clínica", nunca prescrever diretamente.
- Pós-processamento: regex/keyword check por linguagem prescritiva direta (dosagens, "tome", "prescrevo") → se detectado, força reformulação + disclaimer e marca `flags_seguranca`.
- **Toda resposta gerada fica com `status = "pendente"` até aprovação humana explícita na Tela 2 do Streamlit.** Nenhuma resposta chega ao médico solicitante sem esse passo.

### Auditoria (`src/safety/audit_log.py`)

Tabela SQLite `auditoria`:

| Campo | Descrição |
| --- | --- |
| `id`, `timestamp` | identificação e momento da interação |
| `pergunta`, `paciente_id` | input original |
| `fontes_rag` (JSON) | chunks recuperados + score |
| `resposta_llm` | saída bruta do modelo |
| `flags_seguranca` (JSON) | sinalizações do guardrail |
| `status` | pendente / aprovado / rejeitado |
| `aprovador`, `timestamp_aprovacao` | quem validou e quando |

Consultável na **Tela 3** do Streamlit.

---

## 7. Interface Streamlit — 3 Telas

### Tela 1: Consulta ao Assistente
- Campo de pergunta + seletor opcional de paciente (mock)
- Executa o grafo LangGraph (`src/graph/flow.py`)
- Mostra resultado com badge "Pendente de validação humana"

### Tela 2: Fila de Validação Humana
- Lista de respostas com `status = pendente`
- Expander com fontes RAG (documento + score) para explainability
- Botões Aprovar / Rejeitar / Editar → atualiza `auditoria.status`

### Tela 3: Auditoria e Histórico
- Tabela da base de auditoria com filtros (status, paciente, data)
- Destaque de respostas sinalizadas por `flags_seguranca`

> **Importante:** o Streamlit não retreina nem reindexa nada — consome modelo, vector store e SQLite já preparados.

---

## 8. Divisão de Responsabilidades

Cronograma em blocos de 7 dias (35 dias totais, ~1h/dia/pessoa).

### Pessoa A (Marcelo Costa) — Fine-tuning (~35h)

**Dias 1-7:** Preparação de dados
- [ ] Coletar/filtrar subset PubMedQA e MedQuAD
- [ ] Gerar protocolos/FAQs/laudos sintéticos via Groq/Gemini
- [ ] Implementar `data_prep.py` (formatação, anonimização, curadoria)
- [ ] Salvar `data/processed/train.jsonl` e `val.jsonl`

**Dias 8-21:** Fine-tuning
- [ ] Montar `notebooks/finetuning_colab.ipynb` com QLoRA
- [ ] Rodar treino (Llama-3.2-3B, LoRA r=16, 3 epochs)
- [ ] Lidar com limites de sessão do Colab (checkpoints no Drive)
- [ ] Salvar `results/finetuning_metrics.json` (loss curves)

**Dias 22-28:** Avaliação e publicação
- [ ] Rodar `evaluate.py` (comparativo base vs. fine-tuned, 8-10 perguntas)
- [ ] Push do adapter LoRA para o Hugging Face Hub
- [ ] Implementar `src/llm/model_loader.py`

**Dias 29-35:** Suporte à integração
- [ ] Apoiar Pessoa C na integração do modelo no nó `gerar_sugestao_llm`
- [ ] Documentar processo de fine-tuning para o relatório

**Entregável:** `notebooks/finetuning_colab.ipynb` + `src/finetuning/` + adapter no HF Hub + `results/finetuning_*.json`

---

### Pessoa B (Vinicius Geizler) — LangChain / RAG / Dados (~35h)

**Dias 1-7:** Setup de dados estruturados
- [ ] Definir `schema.sql` (pacientes, exames, medicações, alertas)
- [ ] Implementar `seed_mock_data.py` com pacientes sintéticos
- [ ] Implementar `patient_tools.py` (funções parametrizadas)

**Dias 8-21:** RAG
- [ ] Implementar `ingest.py` (Chroma + sentence-transformers)
- [ ] Indexar protocolos sintéticos e amostra de PubMedQA/MedQuAD
- [ ] Implementar `retriever.py` com retorno de fonte + score

**Dias 22-28:** Integração
- [ ] Testar retriever + patient_tools em conjunto (dados coerentes)
- [ ] Apoiar Pessoa C na integração dos nós `verificar_exames_pendentes` e `consultar_protocolo`

**Dias 29-35:** Polimento
- [ ] Ajustar qualidade dos chunks/retrieval com base em feedback dos testes end-to-end
- [ ] Apoiar relatório na seção de arquitetura de dados

**Entregável:** `src/db/` + `src/rag/` + vector store persistido em `data/chroma/`

---

### Pessoa C (Antonio Bazo) — LangGraph e Segurança (~35h)

**Dias 1-7:** Setup
- [ ] Definir `AssistantState` (`src/graph/state.py`)
- [ ] Esqueleto dos nós (`nodes.py`) com stubs/mocks

**Dias 8-21:** Implementação do grafo
- [ ] Implementar cada nó de fato (exames, RAG, sugestão LLM, validação, alerta, log)
- [ ] Montar `flow.py` (StateGraph completo)
- [ ] Implementar `guardrails.py` (regex/keyword de linguagem prescritiva)

**Dias 22-28:** Auditoria
- [ ] Implementar `audit_log.py` (tabela SQLite + funções de consulta)
- [ ] Testar fluxo completo ponta a ponta (pergunta → grafo → log)

**Dias 29-35:** Integração e testes
- [ ] Integrar com modelo fine-tuned (Pessoa A) e RAG/DB (Pessoa B)
- [ ] Apoiar Pessoa D na chamada do grafo a partir do Streamlit

**Entregável:** `src/graph/` + `src/safety/` + fluxo LangGraph funcional ponta a ponta

---

### Pessoa D (Renato Mattos) — Interface Streamlit (~35h)

> **Dependência:** Tela 1 depende do grafo LangGraph minimamente funcional (~dia 21, Pessoa C). Tela 2 depende da estrutura de auditoria (~dia 22-28). Tela 3 pode começar com dados mock desde o dia 1.

**Dias 1-7:** Setup + Tela 3 com mock
- [ ] Criar `app.py` com navegação entre as 3 telas
- [ ] Implementar Tela 3 com dados mock de auditoria

**Dias 8-21:** Tela 1
- [ ] Formulário de pergunta / seleção de paciente
- [ ] Integração inicial com o grafo (usando stubs enquanto os nós reais não estão prontos)

**Dias 22-28:** Tela 2
- [ ] Fila de validação humana (aprovar/rejeitar/editar)
- [ ] Exibição de fontes RAG (explainability)

**Dias 29-35:** Integração real e polimento
- [ ] Substituir mocks por dados reais (grafo, auditoria)
- [ ] Testar fluxo completo: pergunta → grafo → fila → aprovação → auditoria
- [ ] Gravar tela para o vídeo

**Entregável:** `app.py` funcionando com as 3 telas

---

### Pessoa E (Vinicius Blasque) — Relatório, Testes e Vídeo (~35h)

**Dias 1-7:** Estrutura do relatório
- [ ] Criar `docs/relatorio_tecnico.md` com esqueleto (ver seção 11)
- [ ] Documentar arquitetura (diagrama do fluxo LangGraph)

**Dias 8-21:** Testes automatizados
- [ ] `tests/test_guardrails.py` — casos que devem/não devem ser bloqueados
- [ ] `tests/test_retriever.py` — retorno de chunks com fonte
- [ ] `tests/test_patient_tools.py` — consulta a paciente mockado
- [ ] Rodar `pytest` e garantir que passam

**Dias 22-28:** Preencher relatório com resultados reais
- [ ] Processo de fine-tuning e anonimização (Pessoa A)
- [ ] Descrição do assistente e diagrama LangChain/LangGraph
- [ ] Avaliação do modelo e análise dos resultados

**Dias 29-35:** Vídeo
- [ ] Roteiro (≤15 min):
  - 0–2min: contexto do desafio (hospital, assistente médico)
  - 2–5min: treino e funcionamento da LLM personalizada
  - 5–9min: execução do fluxo automatizado (LangGraph) com pergunta clínica real
  - 9–12min: fila de validação humana + explainability (fontes)
  - 12–15min: logs de auditoria e conclusão
- [ ] Gravar e subir no YouTube (não listado)

**Entregável:** `docs/relatorio_tecnico.md` + `tests/` + link do vídeo no README

---

## 9. Estrutura de Arquivos

```
hospital-assistant-fase3/
├── notebooks/
│   └── finetuning_colab.ipynb        ← roda no Colab (GPU T4)
├── src/
│   ├── finetuning/
│   │   ├── data_prep.py              ← anonimização + curadoria (Pessoa A)
│   │   ├── train.py                  ← QLoRA training loop (Pessoa A)
│   │   └── evaluate.py               ← comparativo base x tuned (Pessoa A)
│   ├── rag/
│   │   ├── ingest.py                 ← indexação Chroma (Pessoa B)
│   │   └── retriever.py              ← (Pessoa B)
│   ├── db/
│   │   ├── schema.sql                ← (Pessoa B)
│   │   ├── seed_mock_data.py         ← (Pessoa B)
│   │   └── patient_tools.py          ← (Pessoa B)
│   ├── graph/
│   │   ├── state.py                  ← (Pessoa C)
│   │   ├── nodes.py                  ← (Pessoa C)
│   │   └── flow.py                   ← (Pessoa C)
│   ├── safety/
│   │   ├── guardrails.py             ← (Pessoa C)
│   │   └── audit_log.py              ← (Pessoa C)
│   └── llm/
│       └── model_loader.py           ← carrega base + adapter LoRA (Pessoa A)
├── app.py                             ← Streamlit, 3 telas (Pessoa D)
├── data/
│   ├── raw/                          ← PubMedQA/MedQuAD subset + protocolos sintéticos brutos
│   ├── processed/                    ← dataset formatado para fine-tuning
│   └── chroma/                       ← vector store persistido
├── results/
│   ├── finetuning_metrics.json       ← (Pessoa A)
│   └── eval_comparativo.json         ← (Pessoa A)
├── tests/
│   ├── test_guardrails.py            ← (Pessoa E)
│   ├── test_retriever.py             ← (Pessoa E)
│   └── test_patient_tools.py         ← (Pessoa E)
├── docs/
│   └── relatorio_tecnico.md          ← (Pessoa E)
├── requirements.txt
├── .env.example
└── README.md
```

---

## 10. Dependências (`requirements.txt`)

```
transformers>=4.40.0
peft>=0.10.0
bitsandbytes>=0.43.0
trl>=0.8.0
accelerate>=0.29.0
datasets>=2.18.0
langchain>=0.1.16
langchain-community>=0.0.34
langgraph>=0.0.40
chromadb>=0.4.24
sentence-transformers>=2.6.0
streamlit>=1.32.0
huggingface_hub>=0.22.0
groq>=0.4.0              # ou google-generativeai — geração de dados sintéticos
python-dotenv>=1.0.0
pytest>=7.4.0
```

> **Segurança:** criar `.env` com `HF_TOKEN=...` e `GROQ_API_KEY=...` (ou `GOOGLE_API_KEY=...`), adicionar `.env` ao `.gitignore`. Nunca commitar chaves nem pesos de modelo — só o adapter LoRA vive no HF Hub.

---

## 11. Estrutura do Relatório Técnico

```markdown
# Relatório Técnico — Tech Challenge Fase 3

## 1. Contexto e Motivação

## 2. Arquitetura da Solução (diagrama)

## 3. Fine-tuning da LLM

### 3.1 Dataset (fontes, anonimização, curadoria)

### 3.2 Configuração do treino (QLoRA, hiperparâmetros)

### 3.3 Avaliação: base vs. fine-tuned

## 4. Assistente Médico com LangChain e LangGraph

### 4.1 Arquitetura de dados (RAG + base estruturada de pacientes)

### 4.2 Diagrama do fluxo LangGraph

### 4.3 Segurança e limites de atuação (guardrails)

### 4.4 Explainability (fontes das respostas)

### 4.5 Logging e auditoria

## 5. Interface Streamlit

## 6. Desafios e Soluções

## 7. Conclusão e Próximos Passos
```

---

## 12. Critérios Mínimos para Aprovação

Antes de entregar, verificar:

- [ ] Fine-tuning executado com dataset próprio (real + sintético), curadoria e anonimização documentadas
- [ ] Adapter LoRA publicado no Hugging Face Hub e carregado com sucesso no app
- [ ] RAG responde com fontes citadas (documento + score de similaridade)
- [ ] `patient_tools` consulta corretamente o SQLite mock de prontuários
- [ ] LangGraph implementado com todos os nós: exames pendentes, RAG, sugestão, validação de segurança, alerta, log de auditoria
- [ ] Nenhuma resposta chega ao usuário final sem passar pela fila de aprovação humana (Tela 2)
- [ ] Log de auditoria completo e consultável na Tela 3
- [ ] `pytest` passa (guardrails, retriever, patient_tools)
- [ ] Relatório técnico com diagrama do fluxo LangGraph + avaliação real do modelo (não mock)
- [ ] Vídeo no YouTube ≤15min mostrando treino, fluxo automatizado, pergunta clínica contextualizada e logs/validação
- [ ] `.env` no `.gitignore`, chaves e pesos de modelo não commitados
- [ ] README atualizado com instruções de instalação e execução

---

## 13. Riscos e Plano B

| Risco                                              | Probabilidade | Plano B                                                                                     |
| --------------------------------------------------- | -------------- | --------------------------------------------------------------------------------------------- |
| Colab desconecta/expira durante o treino             | Alta           | Salvar checkpoints intermediários no Google Drive; retomar treino de onde parou              |
| Push do adapter para o HF Hub falha ou API limitada  | Baixa          | Fallback: salvar adapter localmente e versionar via Git LFS ou zip anexado ao repo            |
| Modelo 3B produz respostas de baixa qualidade médica | Média          | Aumentar/curar mais exemplos sintéticos; se persistir, documentar como limitação conhecida e reforçar via RAG |
| Guardrail de regex gera falsos positivos/negativos   | Média          | Documentar limitação no relatório; complementar com verificação manual na fila de validação   |
| Atraso da Pessoa A no adapter fine-tuned              | Média          | Nós do grafo usam o modelo base (sem fine-tuning) como stub até o adapter estar pronto        |
| Tempo insuficiente para o vídeo                      | Média          | Screen recording simples com narração, sem edição                                            |

---

## 14. Comunicação do Time

- **Check-in semanal:** mensagem no grupo ao final de cada bloco de 7 dias com: ✅ o que foi feito / 🔴 bloqueios
- **Dependências críticas:** notificar imediatamente quando o adapter LoRA (Pessoa A), o vector store + patient_tools (Pessoa B) e o grafo LangGraph (Pessoa C) estiverem prontos
- **Branch strategy:** cada pessoa trabalha em branch próprio (`feature/finetuning`, `feature/rag`, `feature/graph`, `feature/streamlit`, `feature/docs-tests`). PR para `main` quando estável
- **Não reabrir escopo:** qualquer nova ideia vai para o README como "trabalhos futuros"

---

_Gerado a partir de sessão de grilling em 2026-08-07. Baseado na análise do PDF do Tech Challenge Fase 3 e na estrutura validada da estratégia da Fase 2._
