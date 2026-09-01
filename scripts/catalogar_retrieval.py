"""Catálogo de qualidade de retrieval via o pipeline E2E real (issue #30, segue #9/#26).

Roda cada query parafraseada por `graph.flow.run()` — o mesmo caminho que a
Tela 1 usa (grafo → `consultar_protocolo` → `rag.retriever.retrieve`) — e
registra o top-3 + score retornado em `docs/retrieval-eval.md`. Cobre os 4
protocolos do corpus sintético atual; não decide nem aplica nenhum ajuste de
parâmetro, só produz a evidência para essa decisão (passo 2 da issue #30).

Uso: `python scripts/catalogar_retrieval.py` (a partir da raiz do repo, com
o venv do projeto ativo — precisa do Chroma e do SQLite de pacientes já
populados; `tests/conftest.py` faz esse seed sob demanda nos testes, aqui
fazemos o mesmo diretamente).
"""

from __future__ import annotations

import datetime
from pathlib import Path

from hospital_assistant.db.patient_tools import list_patients
from hospital_assistant.db.seed_mock_data import seed
from hospital_assistant.graph.flow import run
from hospital_assistant.paths import CHROMA_DIR, PATIENTS_DB
from hospital_assistant.rag.ingest import ingest

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "docs" / "retrieval-eval.md"

# query parafraseada -> arquivo-fonte esperado em data/raw/protocolos_sinteticos/
CASOS: list[tuple[str, str]] = [
    ("crise hipertensiva", "crise_hipertensiva.md"),
    ("paciente chegou com pressão arterial muito alta, é urgência ou emergência hipertensiva?", "crise_hipertensiva.md"),
    ("como reduzir a pressão de um paciente com emergência hipertensiva e lesão de órgão-alvo?", "crise_hipertensiva.md"),
    ("dor torácica aguda", "dor_toracica_aguda.md"),
    ("paciente no pronto-socorro com dor no peito, qual a conduta inicial?", "dor_toracica_aguda.md"),
    ("quais exames pedir para suspeita de infarto com dor torácica?", "dor_toracica_aguda.md"),
    ("como solicito um exame com prioridade urgente?", "faq_medicos_exames_urgentes.md"),
    ("qual o prazo de liberação de um exame laboratorial marcado como urgente?", "faq_medicos_exames_urgentes.md"),
    ("quem é avisado se um exame urgente demora para sair?", "faq_medicos_exames_urgentes.md"),
    ("suspeita de sepse", "sepse.md"),
    ("qSOFA", "sepse.md"),
    ("protocolo da primeira hora para sepse, o que fazer com o paciente?", "sepse.md"),
]


def _garantir_dados() -> None:
    if not PATIENTS_DB.exists():
        seed()
    if not (CHROMA_DIR / "chroma.sqlite3").exists():
        ingest()


def _linha_resultado(query: str, fonte_esperada: str, contexto_rag: list[dict]) -> str:
    fontes_retornadas = [c["source"] for c in contexto_rag]
    acertou = any(fonte_esperada in f for f in fontes_retornadas)
    status = "✅ top-3" if acertou else "❌ fora do top-3"

    linhas = [f"### `{query}`", "", f"Fonte esperada: `{fonte_esperada}` — **{status}**", ""]
    linhas.append("| # | fonte | score |")
    linhas.append("|---|---|---|")
    for i, chunk in enumerate(contexto_rag, start=1):
        linhas.append(f"| {i} | `{chunk['source']}` | {chunk['score']:.4f} |")
    linhas.append("")
    return "\n".join(linhas)


def main() -> None:
    _garantir_dados()
    paciente_id = list_patients()[0]["id"]

    secoes = []
    total_ok = 0
    for query, fonte_esperada in CASOS:
        estado = run(query, paciente_id=paciente_id)
        contexto_rag = estado["contexto_rag"]
        fontes_retornadas = [c["source"] for c in contexto_rag]
        if any(fonte_esperada in f for f in fontes_retornadas):
            total_ok += 1
        secoes.append(_linha_resultado(query, fonte_esperada, contexto_rag))

    cabecalho = f"""# Catálogo de qualidade de retrieval (E2E)

Gerado por `scripts/catalogar_retrieval.py` em {datetime.date.today().isoformat()}.
Segue issue #30 (`ajustar retrieval com base em feedback de testes E2E`, que
depende de #9/#26). Cada linha roda a query pelo pipeline real completo —
Tela 1 → grafo (`hospital_assistant.graph.flow.run`) → `rag.retriever.retrieve`
— não uma chamada isolada ao retriever.

**Resultado**: {total_ok}/{len(CASOS)} queries trazem o documento esperado no top-3.

Este documento só cataloga evidência (passo 1 da issue #30). A decisão de
mitigação (ex: threshold mínimo de score) e sua implementação ficam para o
passo 2, depois de revisão.

---

"""
    OUTPUT_PATH.write_text(cabecalho + "\n".join(secoes), encoding="utf-8")
    print(f"Catálogo escrito em {OUTPUT_PATH} ({total_ok}/{len(CASOS)} ok)")


if __name__ == "__main__":
    main()
