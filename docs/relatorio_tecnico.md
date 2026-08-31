# Relatório Técnico — Tech Challenge Fase 3

> Esqueleto per ESTRATEGIA.md §11. Seção 4.1 preenchida por Pessoa B (RAG/Dados,
> issue #9) — as demais seções são responsabilidade de quem executa o bloco
> correspondente e ficam como placeholder até lá.

## 1. Contexto e Motivação

_(pendente — Pessoa E, #18)_

## 2. Arquitetura da Solução (diagrama)

_(pendente — Pessoa E, #18)_

## 3. Fine-tuning da LLM

### 3.1 Dataset (fontes, anonimização, curadoria)

_(pendente — Pessoa A)_

### 3.2 Configuração do treino (QLoRA, hiperparâmetros)

_(pendente — Pessoa A)_

### 3.3 Avaliação: base vs. fine-tuned

_(pendente — Pessoa A)_

## 4. Assistente Médico com LangChain e LangGraph

### 4.1 Arquitetura de dados (RAG + base estruturada de pacientes)

O assistente combina duas fontes de dados, deliberadamente separadas por
responsabilidade: conhecimento clínico geral/protocolar (RAG, não estruturado)
e dados de um paciente específico (SQLite, estruturado).

#### Dados estruturados de pacientes (`src/hospital_assistant/db/`)

Schema com quatro tabelas (`pacientes`, `exames`, `medicacoes`, `alertas`),
todas referenciando `pacientes.id`. `exames.status` é restrito a
`pendente`/`concluido` via `CHECK`, e `alertas.severidade` a
`baixa`/`media`/`alta` — validação no schema, não só na aplicação.

Esses dados **não são expostos via um agente de SQL livre**. Essa é uma decisão
fechada em ESTRATEGIA.md §1: um LLM gerando SQL sobre prontuários teria risco
real de query alucinada retornando dado incorreto ou de outro paciente — um
tipo de erro silencioso, sem sinal de que algo deu errado, num domínio onde o
custo de um erro silencioso é alto. Em vez disso, `patient_tools.py` expõe só
duas funções parametrizadas e estreitas, com queries fixas
(`get_pending_exams(paciente_id)`, `get_patient_history(paciente_id)`),
usando placeholders (`?`) do `sqlite3` — o LLM nunca vê nem gera SQL, só chama
essas funções como *tools* do LangChain. Um `paciente_id` inexistente levanta
erro explícito (`ValueError`) em `get_patient_history` em vez de devolver dado
parcial ou inconsistente.

#### RAG — conhecimento clínico geral (`src/hospital_assistant/rag/`)

**Corpus** (`data/raw/`): dois grupos de documentos, ambos versionados no
repositório (não regenerados em runtime, ao contrário do índice vetorial):

- `protocolos_sinteticos/`: 4 protocolos fictícios escritos para este projeto
  (dor torácica aguda, crise hipertensiva, sepse, FAQ de exames urgentes),
  representando o conhecimento interno de um hospital fictício.
- `medquad_sample/`: 7 pares pergunta/resposta reais extraídos do
  [MedQuAD](https://github.com/abachaa/MedQuAD) (licença CC BY 4.0, atribuição
  mantida em cada arquivo), cobrindo leucemia e pneumonia — a amostra pública
  de conhecimento clínico geral prevista em ESTRATEGIA.md §4.

**Indexação** (`ingest.py`): os documentos são divididos em chunks de 800
caracteres (100 de sobreposição) via `RecursiveCharacterTextSplitter`, e
embedados com `sentence-transformers/all-MiniLM-L6-v2` — modelo fixado em
ESTRATEGIA.md §1 por não ter custo de API. Os embeddings são normalizados
(`normalize_embeddings=True`) e o índice Chroma é criado em espaço de
similaridade de cosseno (`hnsw:space: cosine`) — a combinação recomendada para
esse modelo; a primeira versão, usando a distância L2 padrão do Chroma sobre
vetores não normalizados, produzia scores fora de qualquer faixa estável
(inclusive negativos). Antes de embedar, o texto de proveniência (rodapé
"Fonte: ..." do MedQuAD, boilerplate institucional dos protocolos sintéticos)
é removido: são strings quase idênticas entre documentos que, num modelo
pequeno, competem com o conteúdo clínico real pela similaridade.

**Retrieval** (`retriever.py`): top-k (k=3 por padrão) por similaridade de
cosseno, retornando texto + fonte (nome do arquivo original) + score. O score
é a própria similaridade de cosseno (`1 - distância`), então "quanto maior,
mais relevante" — é esse valor que a Tela 2 exibe cru para explicabilidade, ao
lado do texto do chunk e do arquivo de origem.

**Limitação conhecida**: `all-MiniLM-L6-v2` é treinado majoritariamente em
inglês. Em teste manual, das 5 perguntas clínicas em português usadas para
validar o retriever, 4 rankeiam o documento correto em 1º lugar (pneumonia,
exame urgente, crise hipertensiva, sepse/qSOFA); a pergunta sobre "dor
torácica aguda" não traz o protocolo correspondente no top-3, mesmo após
normalizar embeddings, usar distância de cosseno e remover boilerplate
repetido do corpus — indicando que é uma limitação de cobertura do modelo
nesse idioma para esse par específico de documento/consulta, não um defeito de
implementação. Trocar o modelo de embeddings resolveria, mas está fora de
escopo (ESTRATEGIA.md §1, "Decisões Fechadas — não reabrir").

### 4.2 Diagrama do fluxo LangGraph

_(pendente — Pessoa C)_

### 4.3 Segurança e limites de atuação (guardrails)

_(pendente — Pessoa C)_

### 4.4 Explainability (fontes das respostas)

_(pendente — Pessoa C / Pessoa D — cobre a exibição na Tela 2 do que a seção
4.1 acima descreve como retorno do retriever)_

### 4.5 Logging e auditoria

_(pendente — Pessoa C)_

## 5. Interface Streamlit

_(pendente — Pessoa D)_

## 6. Desafios e Soluções

_(pendente — consolidação final, Pessoa E; a limitação de retrieval em
português descrita na seção 4.1 é candidata a entrar aqui)_

## 7. Conclusão e Próximos Passos

_(pendente — Pessoa E)_
