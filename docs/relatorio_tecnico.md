# Relatório Técnico — Tech Challenge Fase 3

**Assistente Virtual Médico** — pós-graduação 9IADT, Fase 3.

| Integrante | Responsabilidade | Seções deste relatório |
| --- | --- | --- |
| Marcelo Costa | Fine-tuning da LLM | 3.1, 3.2, 3.3 |
| Vinicius Geizler | LangChain / RAG / Dados | 4.1 |
| Antonio Bazo | LangGraph e Segurança | 4.2, 4.3, 4.5 |
| Renato Mattos | Interface Streamlit | 4.4, 5 |
| Vinicius Blasque | Relatório, testes e vídeo | 1, 2, 6, 7 |

> Estrutura per ESTRATEGIA.md §11. Cada seção é escrita por quem executou o bloco
> correspondente; as ainda não preenchidas aparecem marcadas como pendentes.

## 1. Contexto e Motivação

_(pendente — Vinicius Blasque, #18)_

## 2. Arquitetura da Solução (diagrama)

_(pendente — Vinicius Blasque, #18)_

## 3. Fine-tuning da LLM

### 3.1 Dataset (fontes, anonimização, curadoria)

O dataset é híbrido por decisão de ESTRATEGIA.md §1: duas fontes públicas de
conhecimento clínico geral e um corpus sintético que simula o material interno
do hospital fictício — que é o que o PDF chama de "dados próprios do hospital".

| Fonte | Papel | Volume bruto |
| --- | --- | --- |
| **PubMedQA** (`qiaojin/PubMedQA`, config `pqa_labeled`) | Pergunta clínica **com o abstract como contexto**. Treina exatamente o comportamento que o grafo exige no nó `gerar_sugestao_llm`: responder a partir de um contexto recuperado, não de memória. | 500 |
| **MedQuAD** (`lavita/MedQuAD`) | Pergunta/resposta autocontida de saúde, a partir de fontes do NIH. É o **mesmo dataset** de onde saiu a amostra do RAG em `data/raw/medquad_sample/`, então o conhecimento do fine-tuning e o do vector store se reforçam em vez de competirem. | 300 |
| **Protocolos sintéticos** (gerados via Groq `openai/gpt-oss-120b`) | Protocolos internos, FAQs de médicos, modelos de laudo e de prescrição, apoio à interpretação de exames. É a metade em **português** do dataset — as duas fontes públicas são em inglês. Versionado em `data/raw/sinteticos_finetuning.jsonl`. | 180 |

**Persona embutida na geração sintética.** Os exemplos sintéticos nunca
prescrevem diretamente: toda resposta é formulada como sugestão ao médico,
explicitamente sujeita a validação humana. Isso não é cosmético — sem essa
restrição o fine-tuning ensinaria justamente o comportamento que
`safety/guardrails.py` bloqueia depois, e o sistema passaria a brigar consigo
mesmo, com o guardrail reescrevendo por regex toda saída do modelo que o
próprio projeto treinou.

**Anonimização** (`src/hospital_assistant/finetuning/anonymize.py`). Nenhuma
das três fontes contém dado real de paciente; o scrubber é aplicado mesmo
assim, porque documenta a técnica exigida pelo PDF e mantém o pipeline seguro
caso um dia receba prontuário real. Cada categoria vira um marcador estável
(`[CPF]`, `[NOME]`, `[DATA]`, `[EMAIL]`, `[TELEFONE]`, `[CEP]`, `[CNS]`,
`[URL]`, `[PRONTUARIO]`, `[CNPJ]`) em vez de ser apagada — o modelo aprende
que ali existia um dado identificável, que é o comportamento desejado num
assistente que nunca deve repetir PII.

O critério de desenho foi que **falso positivo custa mais que falso negativo**:
um scrubber agressivo transformaria "amoxicilina 500 mg" ou "Streptococcus
pneumoniae" em marcadores e destruiria o valor clínico do dataset em silêncio.
Por isso todo padrão exige uma âncora explícita — formato rígido (CPF, CEP,
e-mail), rótulo textual ("prontuário nº") ou marcador de papel ("Dr.",
"paciente") — em vez de tentar reconhecer nome próprio pelo formato.

Essa escolha se provou necessária na prática. A primeira versão usava
`re.IGNORECASE` na regra de "paciente/patient", o que anulava a exigência de
inicial maiúscula e fazia **qualquer** par de palavras seguintes virar
`[NOME]`: em português, `"o paciente apresentou dor torácica"` viraria
`"o paciente [NOME] torácica"`. O defeito só apareceu ao medir o scrubber
contra texto real das duas fontes públicas — 24 campos corrompidos em 150
exemplos do PubMedQA, contra 2 depois da correção (e 1 → 0 no MedQuAD). Os
dois lados estão travados em `tests/test_anonymize.py`: o que deve sumir e o
que não pode sumir.

**Curadoria e deduplicação** (`data_prep.py`), nesta ordem — anonimizar antes
de curar, porque o scrubber pode encurtar uma resposta que era só PII, e nesse
caso ela deve cair no filtro de tamanho:

- descarte de instrução com menos de 10 caracteres e de resposta fora da faixa
  de 40–3000 caracteres (o teto existe porque sequência longa demais domina o
  batch e estoura a memória do T4);
- descarte de resíduo de mock e de recusa do gerador ("desculpe, não posso…"),
  que o modelo aprenderia a imitar;
- deduplicação por pergunta **+ contexto** — a chave inclui o `input` porque o
  PubMedQA repete perguntas quase idênticas sobre abstracts diferentes, que são
  exemplos distintos e não duplicatas.

**Split 90/10 com semente fixa** (`seed=42`), embaralhando antes de cortar: as
três fontes entram concatenadas, e cortar a cauda direto deixaria a validação
inteira composta de exemplos sintéticos — a loss de validação mediria uma
fonte só. A semente fixa também é o que torna a comparação base vs.
fine-tuned reproduzível.

**Resultado.** Das 980 linhas brutas somadas das três fontes, 966 sobreviveram
à anonimização, à curadoria e à deduplicação (14 descartadas, 1,4%), divididas
em **869 exemplos de treino** e **97 de validação**.

| Fonte | Bruto | Após curadoria | Descartado |
| --- | ---: | ---: | ---: |
| PubMedQA | 500 | 500 | 0 |
| MedQuAD | 300 | 286 | 14 |
| Sintético | 180 | 180 | 0 |
| **Total** | **980** | **966** | **14** |

O descarte se concentra inteiramente no MedQuAD, e isso é coerente com a
natureza da fonte: o dataset agrega perguntas de vários portais do NIH sobre os
mesmos temas, e a deduplicação por pergunta + contexto remove as repetições —
além das respostas curtas demais que sobram de documentos com conteúdo
removido. PubMedQA e o corpus sintético passam íntegros: o primeiro é curado na
origem por especialistas, o segundo já foi gerado sob as restrições de formato
e tamanho que a curadoria verifica.

Estatísticas completas em `results/dataset_stats.json`. Amostra de 30 exemplos
já anonimizados versionada em `results/dataset_sample.jsonl` — `data/processed/`
é artefato derivado e fica fora do Git, então é essa amostra que permite
conferir a anonimização sem regenerar nada.

**Limitação conhecida: desequilíbrio de idioma.** Apenas 180 dos 966 exemplos
curados são em português — **18,6%**, os do corpus sintético. PubMedQA e
MedQuAD, que respondem pelos 81,4% restantes, são em inglês, porque são as duas
fontes que o próprio PDF do Tech Challenge sugere e que a ESTRATEGIA §1 fixou.
O assistente, no entanto, é inteiramente pt-BR: as telas, os guardrails, os
protocolos indexados no RAG e as perguntas do vídeo de demonstração.

A consequência prática é que o ganho do fine-tuning em português vem sobretudo
do corpus sintético, enquanto as fontes públicas contribuem com estrutura de
raciocínio clínico e com o comportamento de responder a partir de um contexto
fornecido — que é transferível entre idiomas — mais do que com vocabulário
utilizável diretamente nas respostas. Duas mitigações ficam registradas como
trabalho futuro, ambas fora do escopo desta fase por dependerem de reabrir uma
decisão fechada ou de orçamento de API relevante: traduzir/adaptar uma parcela
do MedQuAD para português na etapa de curadoria, ou aumentar o corpus sintético
até equiparar o volume das fontes públicas.

Vale registrar também que 52% dos exemplos de treino têm o campo `input`
preenchido (o abstract do PubMedQA). Isso é deliberado e não acidental: é a
parcela do dataset que ensina o modelo a responder **a partir do contexto
fornecido**, que é exatamente o formato em que o nó `gerar_sugestao_llm` o
invoca em produção, com os chunks vindos do RAG naquele mesmo campo.

### 3.2 Configuração do treino (QLoRA, hiperparâmetros)

O treino roda no Google Colab (GPU T4), mas **a lógica não vive no notebook**:
está em `src/hospital_assistant/finetuning/train.py`, e
`notebooks/finetuning_colab.ipynb` apenas o importa e executa. É o que atende
literalmente o requisito de "projeto modularizado em Python" do PDF.

| Parâmetro | Valor | Motivo |
| --- | --- | --- |
| Modelo base | `meta-llama/Llama-3.2-3B-Instruct` | ESTRATEGIA.md §1 — equilíbrio qualidade/velocidade no T4 |
| Quantização | 4-bit NF4 + double quant, compute `float16` | QLoRA cabe nos 16GB do T4; o T4 não tem `bfloat16` |
| LoRA | `r=16`, `alpha=32`, `dropout=0.05`, alvo `q_proj`/`v_proj` | ESTRATEGIA.md §3 |
| Treino | batch 4 × `grad_accum` 4 (batch efetivo 16), 3 épocas, `lr=2e-4`, scheduler cosine | ESTRATEGIA.md §3 |
| `max_seq_length` | 1024 | Acomoda o contexto do PubMedQA truncado em 2000 caracteres |
| Checkpoints | por época, no Google Drive | A sessão do Colab cai no meio do treino (risco de probabilidade **alta** em ESTRATEGIA.md §13) |

Os hiperparâmetros vivem em `LORA_KWARGS`/`TRAINING_KWARGS` e têm teste de
regressão em `tests/test_train.py` — são decisões fechadas (§1), e o teste é o
cadeado que impede alterá-las sem reabrir a decisão.

**Acesso ao modelo base.** O repositório oficial da Meta é *gated* com revisão
manual, e o pedido de licença pode levar dias. `resolve_base_model()` tenta o
repositório oficial e cai automaticamente para `unsloth/Llama-3.2-3B-Instruct`
— um re-upload dos **mesmos pesos**, sem gate — em vez de estourar
`GatedRepoError` no meio de uma sessão de Colab. A decisão de modelo base da
ESTRATEGIA §1 fica intacta; muda só a origem do download.

**Formato do prompt.** Treino e inferência montam o prompt pela mesma função
(`src/hospital_assistant/llm/prompt.py`). Esse é o contrato mais frágil do
fine-tuning: um adapter LoRA aprende a responder ao formato exato que viu no
SFT, e se o app montasse o prompt de outro jeito o adapter degradaria **sem
erro nenhum** — o sintoma seria "o fine-tuning não melhorou nada".
`tests/test_prompt.py` trava a equivalência entre os dois lados.

### 3.3 Avaliação: base vs. fine-tuned

_(pendente — depende da execução do treino no Colab; `evaluate.py` e o notebook
já estão prontos e gravam `results/eval_comparativo.json`.)_

A avaliação combina duas métricas com papéis diferentes:

- **Loss e perplexidade** (`results/finetuning_metrics.json`, geradas por
  `extract_loss_curves`): dizem que o modelo aprendeu a distribuição do
  dataset. Não dizem que ele ficou clinicamente melhor.
- **Comportamento sob o guardrail** (`results/eval_comparativo.json`): as 9
  perguntas de `evaluate.PERGUNTAS_AVALIACAO` rodam nos dois modelos, e o
  próprio `ClinicalGuardrails` de produção atua como juiz — quantas respostas
  de cada lado ele precisaria marcar para validação humana por linguagem de
  prescrição. Se o modelo fine-tunado dispara o guardrail menos vezes, ele
  internalizou uma regra que antes era imposta por regex depois do fato. É
  essa a métrica que responde à pergunta do projeto, não a perplexidade.

As 9 perguntas cobrem os quatro protocolos sintéticos indexados no RAG (sepse,
dor torácica, crise hipertensiva, exames urgentes), conhecimento geral vindo do
MedQuAD (pneumonia) e três casos de pressão explícita por prescrição — que é
onde base e fine-tunado devem divergir mais.

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

_(pendente — Antonio Bazo)_

### 4.3 Segurança e limites de atuação (guardrails)

_(pendente — Antonio Bazo)_

### 4.4 Explainability (fontes das respostas)

_(pendente — Antonio Bazo / Renato Mattos — cobre a exibição na Tela 2 do que a
seção 4.1 acima descreve como retorno do retriever)_

### 4.5 Logging e auditoria

_(pendente — Antonio Bazo)_

## 5. Interface Streamlit

_(pendente — Renato Mattos)_

## 6. Desafios e Soluções

_(pendente — consolidação final, Vinicius Blasque; a limitação de retrieval em
português descrita na seção 4.1 é candidata a entrar aqui)_

## 7. Conclusão e Próximos Passos

_(pendente — Vinicius Blasque)_
