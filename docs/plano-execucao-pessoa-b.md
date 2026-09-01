# Plano de execução — Pessoa B (Vinicius Geizler)

Plano de trabalho para #6, #7, #8 e #9, lido a partir do corpo de cada issue no GitHub,
do scaffolding já presente em `src/hospital_assistant/{db,rag}`, das dependências
registradas em [dependency-map.md](./dependency-map.md) e da seção 8 (Divisão de
Responsabilidades) + seção 12 (Critérios Mínimos de Aprovação) de
`exercicios/src/fase-03/99-tech-challenge/ESTRATEGIA.md`. Gerado em 2026-08-31.

**Branch**: `feature/rag` (convenção de branch por pessoa, ESTRATEGIA.md §14) — PR
para `main` quando cada bloco estiver estável, não só no fim dos 35 dias.

## Visão geral

| Ticket | Bloco / dias | Depende de | Bloqueia |
|---|---|---|---|
| #6 · Setup de dados estruturados | Bloco 1 · 1–7 | nenhuma | #7 (calendário) |
| #7 · RAG (ingest + retriever) | Bloco 2 · 8–21 | #6 | #8 (calendário) |
| #8 · Integração RAG + patient_tools | Bloco 3 · 22–28 | #7 | **#13 (Antonio)** |
| #9 · Polimento | Bloco 4 · 29–35 | #8 | #22 (entrega final) |

**Ponto crítico do meu lado**: #8 é a única das minhas entregas que outra pessoa
espera diretamente — Antonio não fecha #13 (integração final do grafo) sem ela. Os
outros três tickets só me bloqueiam a mim mesmo entre blocos. Isso muda a estratégia:
#6 e #7 têm folga (posso terminar antes do fim do bloco e adiantar), #8 não tem — se
atrasar, atraso Antonio no Bloco 4 dele.

Nota de escopo: `patient_tools.py` é deliberadamente **não** um agente de SQL livre —
funções parametrizadas e estreitas, para eliminar risco de query alucinada sobre dados
clínicos (decisão fechada em ESTRATEGIA.md §1, reafirmada em §4). Mantenha essa
restrição em #6 e não a reabra em #8/#9.

### Critérios de aprovação que dependem de mim (ESTRATEGIA.md §12)

Três dos itens do checklist final de aprovação (§22 no mapa de issues) são
responsabilidade direta desta trilha — vale usá-los como definition of done real,
não só o texto de cada issue:

- [ ] RAG responde com fontes citadas (documento + score de similaridade) — #7/#9
- [ ] `patient_tools` consulta corretamente o SQLite mock de prontuários — #6/#8
- [ ] `pytest` passa incluindo os testes de retriever e patient_tools (que a Pessoa E
      escreve no Bloco 2 dela, contra a minha implementação) — #7/#8/#9

---

## #6 · Setup de dados estruturados (Bloco 1 · dias 1–7)

**Escopo da issue**: `src/db/schema.sql` (pacientes, exames, medicacoes, alertas),
`src/db/seed_mock_data.py` (popular com pacientes sintéticos), `src/db/patient_tools.py`
(`get_pending_exams(paciente_id)`, `get_patient_history(paciente_id)` parametrizadas).

**Estado atual**: as três peças já existem como placeholder — schema com só `id` +
1 campo obrigatório por tabela, seed com 1 linha, `patient_tools.py` retornando dados
fixos. Nenhuma tem TODO bloqueado por outra pessoa.

1. Fechar o desenho de colunas em `schema.sql` antes de tocar no resto (os TODOs no
   arquivo já listam os candidatos): `pacientes` (data_nascimento, prontuario),
   `exames` (tipo, data_solicitacao, data_resultado, resultado), `medicacoes` (nome,
   dosagem, frequencia, data_inicio), `alertas` (severidade, data, resolvido).
   Confirme que `exames.status` cobre pelo menos `pendente`/`concluido` — é o campo
   que `verificar_exames_pendentes` (nodes.py) já lê.
2. Expandir `seed_mock_data.py` para múltiplos pacientes (não só 1 linha por tabela) —
   inclua pelo menos um paciente com exame `pendente` e um sem, para exercitar os dois
   ramos de `emitir_alerta_se_necessario`.
3. Implementar `get_pending_exams` e `get_patient_history` contra o SQLite real
   (`sqlite3`, placeholders `?`, nunca f-string na query).
4. Rodar `uv run python -m hospital_assistant.db.seed_mock_data` e depois
   `uv run pytest` — o smoke test `test_patient_tools_returns_history` já existe e
   deve continuar passando com dados reais em vez de mock.
5. Apagar os comentários `TODO(Bloco 1 — Pessoa B)` conforme cada arquivo fecha.

**Entregável**: `src/db/` completo, populável via `seed_mock_data.py`.
**Quem consome isso depois**: `nodes.py` (Antonio, #11) já importa
`get_pending_exams` — manter a assinatura e o formato de retorno (`ExamRecord`,
`PatientHistory`) estáveis evita retrabalho para ele.

---

## #7 · RAG (ingest + retriever) (Bloco 2 · dias 8–21)

**Escopo da issue**: `src/rag/ingest.py` (embeddings
`sentence-transformers/all-MiniLM-L6-v2`, Chroma persistido em `data/chroma/`,
indexando protocolos sintéticos + amostra PubMedQA/MedQuAD); `src/rag/retriever.py`
(top-3, texto + origem + score).

1. Montar o corpus antes de escrever código: alguns `.md` de protocolos sintéticos
   escritos à mão em `data/raw/` + uma amostra pequena de PubMedQA ou MedQuAD (baixar
   um subconjunto, não o dataset inteiro). **`data/raw/` não está no `.gitignore`** —
   ao contrário de `data/chroma/` e `data/patients_mock.db`, que são regenerados via
   script, os arquivos brutos aqui *serão* commitados. Mantenha o volume pequeno.
2. Implementar `ingest.py`: carregar os arquivos de `data/raw/`, dividir em chunks
   (`RecursiveCharacterTextSplitter` do LangChain é o caminho mais direto dado que
   `langchain`/`langchain-community` já são dependências do projeto), gerar embeddings
   com `all-MiniLM-L6-v2` e persistir no Chroma em `CHROMA_DIR`
   (`hospital_assistant.paths`).
3. Implementar `retriever.py`: abrir a coleção persistida, `similarity_search_with_score`
   (ou equivalente), devolver os top-k (k=3 por padrão) no formato `RetrievedChunk`
   já definido (`text`, `source`, `score`). Decida a normalização do score uma vez
   (Chroma retorna distância, não similaridade) e documente no docstring — a Tela 2
   do Renato (#16) exibe esse score cru para explicabilidade, então score maior deve
   significar "mais relevante", não o contrário.
4. Rodar `uv run python -m hospital_assistant.rag.ingest` localmente, confirmar que
   `data/chroma/` populou, e então validar `retrieve()` manualmente com 2–3 perguntas
   de teste antes de considerar fechado.
5. Atualizar `test_retriever_returns_scored_chunks` (ou os que a Pessoa E adicionar em
   paralelo no Bloco 2 dela) contra o retriever real.

**Entregável**: `src/rag/` completo + `data/chroma/` persistido.
**Risco a observar**: download do modelo de embeddings pode ser lento na primeira
execução do devcontainer — vale rodar o `ingest()` uma vez e confirmar cache antes do
fim do bloco, não no último dia.

---

## #8 · Integração RAG + patient_tools (Bloco 3 · dias 22–28)

**Este é o ticket com prazo rígido** — Antonio não fecha #13 sem ele.

**Escopo da issue**: testar `retriever` + `patient_tools` juntos validando coerência
dos dados; apoiar Antonio na integração dos nós `verificar_exames_pendentes` e
`consultar_protocolo` ao grafo.

1. Escrever cenários que exercitam os dois módulos juntos: dado um `paciente_id`,
   buscar exames pendentes E rodar uma consulta RAG relacionada na mesma interação,
   conferindo que nada colide (ex.: paciente inexistente não deve quebrar o retriever,
   query vazia não deve quebrar `patient_tools`).
2. Conferir contra `graph/state.py` (`AssistantState`) que os tipos batem exatamente
   com o que `nodes.py` espera gravar em `exames_pendentes` e `contexto_rag` — hoje
   `consultar_protocolo` já faz `dict(c) for c in chunks`, então mudanças de forma no
   `RetrievedChunk` do #7 se propagam para lá.
3. Combinar com Antonio (issue #11, em andamento no Bloco 2 dele) assim que ele tocar
   os dois nós — não esperar o fim do bloco para o primeiro contato. Ele já tem o
   placeholder funcionando contra o mock; o trabalho aqui é confirmar que a troca pelo
   real não muda nada na interface.
4. Rodar a suíte completa (`uv run pytest`) com o grafo ponta a ponta usando dados
   reais de `db`/`rag`, não mocks.
5. Deixar registrado (comentário na issue #8, ou nota curta linkada) o que foi
   validado — é o artefato que Antonio referencia ao fechar #13.

**Entregável da issue**: "retriever + patient_tools validados e prontos para os nós
do LangGraph".
**Ação de coordenação**: avisar Antonio no primeiro sinal de atraso, não no dia 28 —
qualquer atraso aqui empurra o Bloco 4 dele (#13) diretamente.

---

## #9 · Polimento (Bloco 4 · dias 29–35)

**Escopo da issue**: refinar qualidade de chunk e retrieval a partir de feedback do
teste ponta a ponta; contribuir documentação de arquitetura de dados para o relatório
técnico (Pessoa E).

**Já resolvido durante o #9**: tentei a mitigação óbvia — remover o boilerplate
institucional repetido entre os protocolos sintéticos do texto embedado (`ingest.py`)
— e retestei a query "dor torácica aguda". **Não resolveu**: `dor_toracica_aguda.md`
continua fora do top-3. As outras 4 queries de teste (pneumonia, exame urgente, crise
hipertensiva, qSOFA) seguem rankeando corretamente, com ou sem a mitigação. Conclusão:
é limitação de cobertura do `all-MiniLM-L6-v2` (fixado em ESTRATEGIA.md §1) para esse
par específico de query/documento em português, não bug de corpus corrigível por
chunking. Documentado no docstring de `retriever.py`. Não vale insistir mais nessa
frente sem trocar o modelo (fora de escopo) — se sobrar tempo no Bloco 4 de verdade
(com Tela 1 real do Renato gerando mais casos), vale catalogar novos casos e decidir
se um threshold mínimo de score na resposta (não confiar em fontes abaixo de X) é
uma mitigação melhor que tentar afinar o corpus.

1. Rodar a Tela 1 (Renato, #17, já integrada de verdade nesse bloco) com perguntas
   reais e catalogar onde o retriever traz chunk irrelevante ou score enganoso —
   o caso de dor torácica acima já está catalogado e não some sozinho.
2. Ajustar `chunk_size`/overlap do splitter, o valor de k, ou aplicar um threshold de
   score mínimo em `retrieve()` — o que os casos catalogados no passo 1 indicarem.
3. Revisar a amostra de protocolos sintéticos/PubMedQA-MedQuAD por qualidade, não só
   por quantidade.
4. Escrever a seção de arquitetura de dados (schema, pipeline de ingest, decisão de
   scoring) para `docs/relatorio_tecnico.md` — Blasque (#20) depende de #12 (Antonio) e
   #4 (Marcelo) para os *resultados*, mas o relatório final também precisa dessa parte
   sua; não deixe para o último dia do bloco dele.
5. Passar `uv run pytest` completo uma última vez antes de considerar o bloco fechado.

**Entregável**: componentes de RAG/dados refinados + material entregue à Pessoa E.
Este é o último bloco meu — fechar #9 é uma das cinco condições de #22 (entrega
final), junto com #5 (Marcelo), #13 (Antonio) e #17 (Renato).

---

## Pontos de sincronização com o time (resumo)

- **Fim do Bloco 3 (dia 28)**: #8 precisa estar fechado — é input direto de #13
  (Antonio). Combinar revisão conjunta antes do prazo, não depois.
- **Durante o Bloco 4**: seção de dados do relatório para Blasque (#20/#18) — sem
  dependência formal no mapa, mas é trabalho real que ele espera de você.
- **#6 e #7 não bloqueiam ninguém fora de você** — se adiantar, adianta só a sua
  própria folga para #8.

---

## Status de execução e gaps conhecidos (2026-08-31)

As quatro tickets foram implementadas e commitadas em branches próprias
(`feature/rag-6-setup-dados-estruturados`, `feature/rag-7-ingest-retriever`,
`feature/rag-8-integracao-rag-patient-tools`, `feature/rag-9-polimento`),
validadas com `pytest` (16 testes, incluindo os de integração do #8),
`ruff check` e `pyright` — todos limpos, rodando contra dependências reais
(não só inspeção de código).

**O que está genuinamente completo:**
- `src/db/` (#6): schema, seed com 3 pacientes cobrindo os casos que o grafo
  precisa (exame pendente/concluído/alerta), `patient_tools` contra SQLite real.
- `src/rag/` (#7): ingest (MiniLM + Chroma, cosseno normalizado) + retriever,
  indexando 4 protocolos sintéticos e 7 QA reais do MedQuAD (CC BY 4.0).
- Testes de integração retriever+patient_tools (#8) e fixture de bootstrap
  (`tests/conftest.py`) que corrige um bug real: sem ela, `pytest` quebrava
  numa checkout limpa porque `patient_tools`/`retriever` pararam de usar mock.
- Seção 4.1 (arquitetura de dados) do `docs/relatorio_tecnico.md` (#9).

**Gaps que não fecham sozinhos, mesmo com o código pronto:**
1. **"Apoiar Pessoa C na integração dos nós" (#8) e "ajustar com base em
   feedback de testes end-to-end" (#9)** — ambos pressupõem o grafo real do
   Antonio e a Tela 1 real do Renato existindo. Nesta sessão só a trilha da
   Pessoa B foi implementada, então o que foi feito é o máximo possível
   sozinho: validei que `nodes.py` funciona com as implementações reais, e
   testei/documentei o caso de ranking da "dor torácica aguda" (ver seção do
   #9 acima). O fechamento de fato desses dois itens depende do restante do
   time entregar as respectivas tickets.
2. **Limitação de ranking não resolvida**: `all-MiniLM-L6-v2` não traz
   `dor_toracica_aguda.md` no top-3 para a query correspondente, mesmo após
   cosseno normalizado + remoção de boilerplate. Documentado em
   `retriever.py` e nesta seção do #9. Não é bloqueante para os critérios de
   aprovação (§12 pede fontes citadas + score, não ranking perfeito), mas é
   um ponto real a mencionar no relatório (seção 6, Desafios e Soluções).
3. **`docs/relatorio_tecnico.md` é só o esqueleto + seção 4.1** — as outras
   seções (contexto, fine-tuning, grafo, interface, conclusão) são
   responsabilidade de quem executar os respectivos blocos.
4. **As issues #6-#9 no GitHub não foram fechadas nem tiveram checkbox
   marcado** — o trabalho está pronto em código/PR, mas o estado da issue em
   si (aberta/fechada, checklist) não foi tocado.
