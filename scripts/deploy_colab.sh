#!/usr/bin/env bash
# Publica o estado atual e atualiza a sessão do Colab que serve a demonstração.
#
#     ./scripts/deploy_colab.sh
#
# Por que não faz tudo sozinho: a máquina virtual do Colab não é alcançável a
# partir daqui. Ela puxa o código do GitHub, então "atualizar o Colab" é sempre
# em dois tempos — publicar na `main` (o que este script faz) e a VM buscar (a
# célula que ele imprime no fim).
#
# Roda a suíte antes de publicar porque o destino é um ambiente de demonstração:
# subir algo quebrado lá custa vários minutos de remontagem. Use `SEM_TESTES=1`
# para pular quando souber o que está fazendo.

set -euo pipefail

cd "$(dirname "$0")/.."

PORTA="${PORTA:-8501}"
BRANCH="${BRANCH:-main}"

if command -v uv >/dev/null 2>&1; then
  EXEC="uv run"
else
  EXEC=""
fi

if [ "${SEM_TESTES:-0}" != "1" ]; then
  echo "==> 1/3  Verificação"
  ${EXEC} ruff check .
  ${EXEC} pytest -q
else
  echo "==> 1/3  Verificação (pulada por SEM_TESTES=1)"
fi

echo "==> 2/3  Publicando na ${BRANCH}"
if [ -n "$(git status --porcelain)" ]; then
  echo "    Há alterações não commitadas. Commite antes de publicar:" >&2
  git status --short >&2
  exit 1
fi
git push origin "${BRANCH}"

echo "==> 3/3  Atualizar a sessão do Colab"
cat <<INSTRUCOES

Rode esta célula no notebook do Colab (ambiente de execução em T4 GPU):

    !curl -sL https://raw.githubusercontent.com/fiap-postech-ia-para-devs-grupo/9IADT-fase-3-tech-challenge/${BRANCH}/scripts/colab_portal.sh | bash

    from google.colab.output import eval_js
    print(eval_js('google.colab.kernel.proxyPort(${PORTA})'))

A primeira linha refaz o ambiente com o código recém-publicado — ela é
idempotente, então rodar de novo sobre uma sessão viva só atualiza e reinicia.
A segunda imprime o endereço de acesso.

O endereço muda quando a máquina virtual é reciclada, que é o que acontece
sempre que a sessão fica ociosa. Por isso ele é impresso a cada atualização em
vez de anotado em algum lugar.
INSTRUCOES
