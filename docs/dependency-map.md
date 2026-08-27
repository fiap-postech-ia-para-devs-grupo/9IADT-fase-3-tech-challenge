# Quem depende de quem

Dependências reais dos 22 tickets, lidas do GitHub (issue #1 e seus filhos). "Depende de" = precisa estar
fechado antes de começar. Gerado em 2026-08-27 — não se atualiza sozinho, reflete o estado das issues nesse momento.

## Pessoa A · Marcelo (fine-tuning)

| Ticket | Bloco / dias | Status | Depende de |
|---|---|---|---|
| #2 · Preparação de dados | Bloco 1 · 1–7 | aberto | nenhuma |
| #3 · Fine-tuning QLoRA | Bloco 2 · 8–21 | aberto | #2 |
| #4 · Avaliação e publicação do adapter | Bloco 3 · 22–28 | aberto | #3 |
| #5 · Suporte à integração | Bloco 4 · 29–35 | aberto | #4 |

## Pessoa B · Vinicius Geizler (RAG / dados)

| Ticket | Bloco / dias | Status | Depende de |
|---|---|---|---|
| #6 · Setup de dados estruturados | Bloco 1 · 1–7 | aberto | nenhuma |
| #7 · RAG (ingest + retriever) | Bloco 2 · 8–21 | aberto | #6 |
| #8 · Integração RAG + patient_tools | Bloco 3 · 22–28 | aberto | #7 |
| #9 · Polimento | Bloco 4 · 29–35 | aberto | #8 |

## Pessoa C · Antonio (LangGraph / guardrails)

| Ticket | Bloco / dias | Status | Depende de |
|---|---|---|---|
| #10 · Setup do estado e stubs dos nós | Bloco 1 · 1–7 | aberto | nenhuma |
| #11 · Implementação do grafo e guardrails | Bloco 2 · 8–21 | aberto | #10 |
| #12 · Auditoria | Bloco 3 · 22–28 | aberto | #11 |
| #13 · Integração final do grafo | Bloco 4 · 29–35 | aberto | #12, **#4 (Marcelo)**, **#8 (Geizler)** |

## Pessoa D · Renato (Streamlit)

| Ticket | Bloco / dias | Status | Depende de |
|---|---|---|---|
| #14 · Setup do app + Tela 3 mock | Bloco 1 · 1–7 | ✅ concluído | nenhuma |
| #15 · Tela 1 — Consulta ao Assistente | Bloco 2 · 8–21 | aberto | #14, **#11 (Antonio)** |
| #16 · Tela 2 — Fila de Validação Humana | Bloco 3 · 22–28 | aberto | #15, **#12 (Antonio)** |
| #17 · Integração real, polimento e gravação | Bloco 4 · 29–35 | aberto | #16 |

## Pessoa E · Vinicius Blasque (relatório / testes / vídeo)

| Ticket | Bloco / dias | Status | Depende de |
|---|---|---|---|
| #18 · Estrutura do relatório | Bloco 1 · 1–7 | aberto | nenhuma |
| #19 · Testes automatizados | Bloco 2 · 8–21 | aberto | #18 |
| #20 · Preencher relatório com resultados reais | Bloco 3 · 22–28 | aberto | #19, **#4 (Marcelo)**, **#12 (Antonio)** |
| #21 · Vídeo de demonstração | Bloco 4 · 29–35 | aberto | #20, **#17 (Renato)** |

## Entrega final

| Ticket | Depende de |
|---|---|
| #22 · Checklist de aprovação (ESTRATEGIA.md §12) | #5 (Marcelo), #9 (Geizler), #13 (Antonio), #17 (Renato), #21 (Blasque) — **todo mundo precisa fechar o Bloco 4** |

## Quem bloqueia quem entre pessoas (o que realmente importa numa reunião de time)

- **Antonio bloqueia Renato duas vezes**: #11 → #15, #12 → #16. Se o grafo/guardrails atrasar, as telas 1 e 2 do Renato atrasam junto.
- **Antonio bloqueia Blasque também**: #12 → #20 (relatório com resultados reais).
- **Marcelo bloqueia Antonio e Blasque**: #4 → #13 (integração final do grafo) e #4 → #20 (relatório).
- **Geizler bloqueia Antonio**: #8 → #13.
- **Renato bloqueia Blasque**: #17 → #21 (o vídeo só pode ser gravado depois da integração real do Renato estar pronta).
- **Todo mundo bloqueia a entrega final** (#22): ela só abre depois que #5, #9, #13, #17 e #21 (o último bloco de cada pessoa) estiverem fechados.

Nota: os blocos são janelas de calendário fixas e iguais para todo mundo (Bloco 1 = Dias 1–7, Bloco 2 = Dias 8–21,
Bloco 3 = Dias 22–28, Bloco 4 = Dias 29–35). As dependências acima não alongam esse calendário — dizem o que
precisa existir antes que o bloco seguinte de outra pessoa faça sentido. Um atraso em um ticket ainda empurra
quem depende dele.
