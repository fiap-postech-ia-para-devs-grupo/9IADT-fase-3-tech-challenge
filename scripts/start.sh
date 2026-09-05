#!/usr/bin/env bash
# Sobe o Portal Clínico NESTA máquina.
#
#     ./scripts/start.sh
#
# Para rodar no Google Colab, com GPU e o modelo treinado, use o outro script:
# `scripts/colab_portal.sh`. Os dois existem porque o ambiente é diferente em
# tudo o que importa — quem tem GPU, onde os dados vivem, e qual endereço
# alcança o servidor. Ver docs/EXECUCAO.md.
#
# Idempotente: o seed do banco e a indexação do RAG podem rodar quantas vezes
# for preciso, então reexecutar é sempre seguro.

set -euo pipefail

cd "$(dirname "$0")/.."

PORTA="${PORTA:-8501}"

# `uv run` garante o ambiente do projeto sem exigir venv ativado. Se o uv não
# estiver instalado, cai no python do ambiente atual em vez de falhar seco.
if command -v uv >/dev/null 2>&1; then
  EXEC="uv run"
else
  echo "AVISO: uv não encontrado; usando o python do ambiente atual." >&2
  EXEC=""
fi

echo "==> 1/3  Dados (SQLite de pacientes)"
${EXEC} python -m hospital_assistant.db.seed_mock_data

echo "==> 2/3  Índice vetorial (Chroma)"
${EXEC} python -m hospital_assistant.rag.ingest

echo "==> 3/3  Modelo"
# O diagnóstico vem antes de subir o servidor porque a falha por falta de GPU
# acontece na primeira renderização, dentro do Streamlit, onde vira um erro na
# tela sem contexto. Aqui dá para explicar o que fazer.
${EXEC} python - <<'PY'
import os
import sys

sys.path.insert(0, "src")
from hospital_assistant.llm import model_loader  # noqa: E402
from hospital_assistant.llm.model_loader import _cabe_em_fp16, _dependencias_faltando  # noqa: E402

# O mesmo `.env` que o app lê. Sem isto o diagnóstico diria "nenhum adapter"
# com um adapter configurado, e daria a resposta errada sobre o modo.
model_loader._carregar_env()

adapter = os.environ.get("HF_ADAPTER_REPO") or "(nenhum)"
demo = os.environ.get("MODO_DEMONSTRACAO", "").lower() in ("1", "true", "sim")
cpu = os.environ.get("PERMITIR_CPU", "").lower() in ("1", "true", "sim")
faltando = _dependencias_faltando(exigir_gpu=True)

print(f"    adapter: {adapter}")
if not faltando:
    print(f"    modo: GPU ({'fp16' if _cabe_em_fp16() else '4-bit NF4'})")
elif cpu:
    print("    modo: CPU — respostas reais, porém em minutos")
elif demo:
    print("    modo: demonstração — as respostas NÃO vêm do modelo treinado")
elif adapter != "(nenhum)":
    print(f"    ERRO: falta {', '.join(faltando)} e nenhuma alternativa foi autorizada.",
          file=sys.stderr)
    print("    Defina PERMITIR_CPU=true ou MODO_DEMONSTRACAO=true no .env,",
          file=sys.stderr)
    print("    ou rode no Colab com GPU (scripts/colab_portal.sh).", file=sys.stderr)
    sys.exit(1)
else:
    print("    modo: demonstração (nenhum adapter configurado)")
PY

echo
echo "Portal subindo em http://localhost:${PORTA}"
${EXEC} streamlit run app.py --server.port "${PORTA}"
