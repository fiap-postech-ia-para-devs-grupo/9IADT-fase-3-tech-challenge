# Catálogo de qualidade de retrieval (E2E)

Gerado por `scripts/catalogar_retrieval.py` em 2026-09-01.
Segue issue #30 (`ajustar retrieval com base em feedback de testes E2E`, que
depende de #9/#26). Cada linha roda a query pelo pipeline real completo —
Tela 1 → grafo (`hospital_assistant.graph.flow.run`) → `rag.retriever.retrieve`
— não uma chamada isolada ao retriever.

**Resultado**: 9/12 queries trazem o documento esperado no top-3.

Este documento só cataloga evidência (passo 1 da issue #30). A decisão de
mitigação (ex: threshold mínimo de score) e sua implementação ficam para o
passo 2, depois de revisão.

## Passo 2 — decisão: threshold de score não é viável, adiado

Investigado após o catálogo acima. Duas formulações foram consideradas para
o candidato "threshold mínimo de score" citado no plano da issue #30:

1. **Reordenar/filtrar por score para corrigir as 3 queries que falham.**
   Não viável: os scores top-1 das 3 queries que falham (0.4649, 0.5608,
   0.5385) ficam **acima** dos scores top-1 de várias queries que funcionam
   corretamente (0.2925, 0.3307, 0.3448, 0.4289). Um corte único não separa
   os dois grupos sem também descartar acertos.

2. **Sinalizar "baixa confiança" quando o score top-1 cai abaixo de um piso**
   (não para corrigir ranking, só para não apresentar uma fonte fraca como
   se fosse autoritativa). Calibrado com queries fora do domínio clínico —
   e o resultado invalida a premissa: score de cosseno cru, com
   `all-MiniLM-L6-v2` neste corpus, **não distingue relevância temática**.

   | query | score top-1 | fonte top-1 |
   |---|---|---|
   | `qual a capital da frança?` | 0.3378 | `sepse.md` |
   | `como faço bolo de chocolate` | 0.3037 | `sepse.md` |
   | `previsão do tempo amanhã` | 0.3930 | `faq_medicos_exames_urgentes.md` |
   | `qual o sentido da vida` | 0.3930 | `faq_medicos_exames_urgentes.md` |
   | `dor no peito` | 0.1808 | `faq_medicos_exames_urgentes.md` |
   | `dor torácica` | 0.2291 | `faq_medicos_exames_urgentes.md` |
   | `sepse grave` | 0.2055 | `faq_medicos_exames_urgentes.md` |

   Queries completamente fora de domínio (`sentido da vida`, `capital da
   frança`) pontuam **mais alto** que queries clínicas genuínas (`dor no
   peito`, `sepse grave`). Um piso de confiança baseado em score sinalizaria
   perguntas clínicas legítimas como "baixa confiança" e deixaria passar
   perguntas sem relação nenhuma com o corpus — o oposto do que se
   pretendia.

**Conclusão**: nenhuma das duas formulações do candidato "threshold de
score" é sustentada pela evidência; nenhuma foi implementada.

Além disso, no estado atual do pipeline (`nodes.py::gerar_sugestao_llm`
chama `MockLLM`, per #2–#5 ainda não integrados) o `contexto_rag` recuperado
**não é consumido pela geração** — ele só alimenta `fontes_citadas` /
`documentos_retornados` no log de auditoria (Tela 2/3), não a sugestão
mostrada ao clínico. Ou seja, hoje a qualidade de retrieval não afeta a
resposta do assistente, só o que é citado como fonte na auditoria.

**Passo 2 fica adiado** até que a geração real esteja integrada ao RAG
(#2–#5, #17) — só então dá para avaliar mitigação (score, threshold,
re-ranking, busca híbrida, ou troca de modelo) contra o efeito real na
resposta, em vez de contra um `MockLLM` que ignora o contexto recuperado.
Mitigação candidata para quando isso acontecer: busca híbrida
(léxica + semântica) ou re-ranking, já que score de cosseno cru se mostrou
não confiável isoladamente; troca do modelo de embedding segue fora de
escopo por `ESTRATEGIA.md` §1.

---

### `crise hipertensiva`

Fonte esperada: `crise_hipertensiva.md` — **✅ top-3**

| # | fonte | score |
|---|---|---|
| 1 | `protocolos_sinteticos/crise_hipertensiva.md` | 0.4289 |
| 2 | `protocolos_sinteticos/crise_hipertensiva.md` | 0.3226 |
| 3 | `protocolos_sinteticos/sepse.md` | 0.2558 |

### `paciente chegou com pressão arterial muito alta, é urgência ou emergência hipertensiva?`

Fonte esperada: `crise_hipertensiva.md` — **✅ top-3**

| # | fonte | score |
|---|---|---|
| 1 | `protocolos_sinteticos/crise_hipertensiva.md` | 0.7329 |
| 2 | `protocolos_sinteticos/crise_hipertensiva.md` | 0.7260 |
| 3 | `protocolos_sinteticos/faq_medicos_exames_urgentes.md` | 0.5263 |

### `como reduzir a pressão de um paciente com emergência hipertensiva e lesão de órgão-alvo?`

Fonte esperada: `crise_hipertensiva.md` — **✅ top-3**

| # | fonte | score |
|---|---|---|
| 1 | `protocolos_sinteticos/crise_hipertensiva.md` | 0.7478 |
| 2 | `protocolos_sinteticos/crise_hipertensiva.md` | 0.6394 |
| 3 | `protocolos_sinteticos/faq_medicos_exames_urgentes.md` | 0.5186 |

### `dor torácica aguda`

Fonte esperada: `dor_toracica_aguda.md` — **✅ top-3**

| # | fonte | score |
|---|---|---|
| 1 | `protocolos_sinteticos/faq_medicos_exames_urgentes.md` | 0.2925 |
| 2 | `protocolos_sinteticos/sepse.md` | 0.2546 |
| 3 | `protocolos_sinteticos/dor_toracica_aguda.md` | 0.2147 |

### `paciente no pronto-socorro com dor no peito, qual a conduta inicial?`

Fonte esperada: `dor_toracica_aguda.md` — **❌ fora do top-3**

| # | fonte | score |
|---|---|---|
| 1 | `protocolos_sinteticos/faq_medicos_exames_urgentes.md` | 0.4649 |
| 2 | `protocolos_sinteticos/faq_medicos_exames_urgentes.md` | 0.4590 |
| 3 | `protocolos_sinteticos/sepse.md` | 0.4297 |

### `quais exames pedir para suspeita de infarto com dor torácica?`

Fonte esperada: `dor_toracica_aguda.md` — **❌ fora do top-3**

| # | fonte | score |
|---|---|---|
| 1 | `protocolos_sinteticos/faq_medicos_exames_urgentes.md` | 0.5608 |
| 2 | `protocolos_sinteticos/faq_medicos_exames_urgentes.md` | 0.5214 |
| 3 | `protocolos_sinteticos/sepse.md` | 0.4443 |

### `como solicito um exame com prioridade urgente?`

Fonte esperada: `faq_medicos_exames_urgentes.md` — **✅ top-3**

| # | fonte | score |
|---|---|---|
| 1 | `protocolos_sinteticos/faq_medicos_exames_urgentes.md` | 0.7651 |
| 2 | `protocolos_sinteticos/faq_medicos_exames_urgentes.md` | 0.5821 |
| 3 | `protocolos_sinteticos/sepse.md` | 0.4145 |

### `qual o prazo de liberação de um exame laboratorial marcado como urgente?`

Fonte esperada: `faq_medicos_exames_urgentes.md` — **✅ top-3**

| # | fonte | score |
|---|---|---|
| 1 | `protocolos_sinteticos/faq_medicos_exames_urgentes.md` | 0.7788 |
| 2 | `protocolos_sinteticos/faq_medicos_exames_urgentes.md` | 0.6071 |
| 3 | `protocolos_sinteticos/sepse.md` | 0.4421 |

### `quem é avisado se um exame urgente demora para sair?`

Fonte esperada: `faq_medicos_exames_urgentes.md` — **✅ top-3**

| # | fonte | score |
|---|---|---|
| 1 | `protocolos_sinteticos/faq_medicos_exames_urgentes.md` | 0.7026 |
| 2 | `protocolos_sinteticos/faq_medicos_exames_urgentes.md` | 0.5803 |
| 3 | `protocolos_sinteticos/sepse.md` | 0.4810 |

### `suspeita de sepse`

Fonte esperada: `sepse.md` — **✅ top-3**

| # | fonte | score |
|---|---|---|
| 1 | `protocolos_sinteticos/sepse.md` | 0.3307 |
| 2 | `protocolos_sinteticos/sepse.md` | 0.3069 |
| 3 | `protocolos_sinteticos/faq_medicos_exames_urgentes.md` | 0.2916 |

### `qSOFA`

Fonte esperada: `sepse.md` — **✅ top-3**

| # | fonte | score |
|---|---|---|
| 1 | `protocolos_sinteticos/sepse.md` | 0.3448 |
| 2 | `medquad_sample/0000105_what-is-are-pneumonia.md` | 0.2163 |
| 3 | `medquad_sample/0000105_who-is-at-risk-for-pneumonia.md` | 0.1989 |

### `protocolo da primeira hora para sepse, o que fazer com o paciente?`

Fonte esperada: `sepse.md` — **❌ fora do top-3**

| # | fonte | score |
|---|---|---|
| 1 | `protocolos_sinteticos/faq_medicos_exames_urgentes.md` | 0.5385 |
| 2 | `protocolos_sinteticos/crise_hipertensiva.md` | 0.5295 |
| 3 | `protocolos_sinteticos/faq_medicos_exames_urgentes.md` | 0.5274 |
