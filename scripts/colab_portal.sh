#!/usr/bin/env bash
# Sobe o Portal Clínico numa sessão do Google Colab, com o adapter LoRA real.
#
# Por que existe: a sessão do Colab cai por inatividade e leva a VM junto.
# Refazer o ambiente célula a célula é frágil, demorado e não fica versionado —
# duas reconexões produzem dois ambientes ligeiramente diferentes. Este script é
# a fonte única: reconectou, roda uma linha e o ambiente volta idêntico.
#
# No Colab, numa célula:
#
#     !curl -sL https://raw.githubusercontent.com/fiap-postech-ia-para-devs-grupo/9IADT-fase-3-tech-challenge/main/scripts/colab_portal.sh | bash
#
# Pré-requisito na sessão: ambiente de execução com GPU T4. O adapter é público
# e o modelo base vem de um espelho não-gated, então nenhum token é necessário.
#
# O script é idempotente: rodar de novo sobre um ambiente já montado apenas
# atualiza o código e reinicia o servidor.

set -euo pipefail

BRANCH="${BRANCH:-main}"
REPO="https://github.com/fiap-postech-ia-para-devs-grupo/9IADT-fase-3-tech-challenge.git"
DESTINO="/content/portal"
PORTA="${PORTA:-8501}"

# Adapter treinado publicado no Hugging Face. Fica versionado aqui, e não como
# variável que alguém precisa lembrar de exportar: esquecê-la fazia o portal
# subir em modo de demonstração sem nenhum erro — só um aviso no meio do log,
# fácil de passar batido, e a demo ia ao ar com o modelo errado.
# Continua sobrescrevível por ambiente para testar outro adapter.
export HF_ADAPTER_REPO="${HF_ADAPTER_REPO:-agendesse/hospital-assistant-llama32-3b-lora}"

echo "==> 1/5  Código (branch ${BRANCH})"
if [ -d "${DESTINO}/.git" ]; then
  git -C "${DESTINO}" fetch --quiet origin "${BRANCH}"
  git -C "${DESTINO}" reset --hard --quiet "origin/${BRANCH}"
else
  git clone --quiet --branch "${BRANCH}" "${REPO}" "${DESTINO}"
fi
cd "${DESTINO}"

echo "==> 2/5  Dependências"
# Runtime do app + extras de GPU. O Colab já traz torch e transformers; o
# `-q` mantém a saída legível, e o pip resolve o que já estiver presente.
pip install -q -U streamlit langgraph langchain langchain-community langchain-text-splitters \
  chromadb sentence-transformers pandas python-dotenv >/dev/null
pip install -q -U peft bitsandbytes accelerate >/dev/null

echo "==> 3/5  Dados (SQLite de pacientes e índice vetorial)"
export PYTHONPATH="${DESTINO}/src"
# Idempotentes: recriam do zero, então rodar de novo não duplica nada.
python -m hospital_assistant.db.seed_mock_data
python -m hospital_assistant.rag.ingest

echo "==> 4/5  Modelo"
echo "    adapter: ${HF_ADAPTER_REPO}"
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo
  echo "    AVISO: nenhuma GPU visível. O carregamento em 4-bit exige CUDA, então o" >&2
  echo "    portal vai subir em modo de demonstração — as respostas NÃO virão do" >&2
  echo "    modelo treinado. Troque o ambiente de execução para T4 e rode de novo." >&2
  echo
fi

echo "==> 5/5  Servidor na porta ${PORTA}"
pkill -f "streamlit run" 2>/dev/null || true
nohup streamlit run portal.py \
  --server.port "${PORTA}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  > /content/portal.log 2>&1 &

# Espera o healthcheck em vez de dormir um tempo fixo: o primeiro start baixa o
# modelo de embeddings e a duração varia bastante entre sessões.
for _ in $(seq 1 60); do
  if curl -s -o /dev/null "http://localhost:${PORTA}/_stcore/health"; then
    echo
    echo "Portal no ar em http://localhost:${PORTA}"
    echo "Log: /content/portal.log"
    echo
    echo "Para abrir de fora do Colab, exponha a porta numa célula:"
    echo "    from google.colab.output import eval_js"
    echo "    print(eval_js(f'google.colab.kernel.proxyPort(${PORTA})'))"
    exit 0
  fi
  sleep 2
done

echo "Servidor não respondeu ao healthcheck. Últimas linhas do log:" >&2
tail -n 30 /content/portal.log >&2
exit 1
