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
