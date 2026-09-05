#!/usr/bin/env bash
# Publica o Portal Clínico do Colab numa URL acessível de qualquer lugar.
#
#     !curl -sL https://raw.githubusercontent.com/fiap-postech-ia-para-devs-grupo/9IADT-fase-3-tech-challenge/main/scripts/colab_tunel.sh | bash
#
# Por que existe: a URL do `proxyPort` é amarrada à sessão autenticada do Colab
# naquele navegador. De outro dispositivo — celular, ou a máquina de um colega —
# o proxy do Google responde 404 com corpo vazio, e alguns navegadores móveis
# oferecem download em vez de mostrar página em branco. Não é link
# compartilhável; este script cria um que é.
#
# ATENÇÃO: o endereço gerado alcança QUALQUER PESSOA da internet. A aplicação
# não tem autenticação — sem a senha abaixo, um desconhecido com o link poderia
# aprovar laudos e ler o prontuário. Por isso o script recusa publicar sem
# senha, e gera uma quando você não fornece.

set -euo pipefail

PORTA="${PORTA:-8501}"
DESTINO="/content/portal"

if [ -z "${SENHA_PORTAL:-}" ]; then
  SENHA_PORTAL="$(python -c 'import secrets; print(secrets.token_urlsafe(9))')"
  echo "==> Senha gerada para este acesso: ${SENHA_PORTAL}"
  echo "    Anote: ela é pedida ao abrir o portal pela URL pública."
  echo
fi
export SENHA_PORTAL

SETUP_LOG="${SETUP_LOG:-/content/setup.log}"

echo "==> 1/3  Montando o ambiente e subindo o portal (1 a 4 min)"
# A saída do bootstrap vai para um log em vez da tela: são dezenas de linhas de
# pip e de avisos de biblioteca, e quem roda esta célula quer a URL, não o
# relatório da instalação. Em caso de falha o log é despejado, porque aí o
# detalhe é justamente o que importa.
if ! curl -sL "https://raw.githubusercontent.com/fiap-postech-ia-para-devs-grupo/9IADT-fase-3-tech-challenge/main/scripts/colab_portal.sh"      | bash > "${SETUP_LOG}" 2>&1; then
  echo "    Falhou ao montar o ambiente. Últimas linhas:" >&2
  tail -n 25 "${SETUP_LOG}" >&2
  exit 1
fi
echo "    pronto (detalhes em ${SETUP_LOG})"

echo "==> 2/3  Instalando o cloudflared"
if ! command -v cloudflared >/dev/null 2>&1; then
  wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    -O /usr/local/bin/cloudflared
  chmod +x /usr/local/bin/cloudflared
fi

echo "==> 3/3  Abrindo o túnel"

# Log em arquivo, e não só na tela: o terminal do Colab congela sem avisar, e
# quando isso acontece com o cloudflared em primeiro plano a URL fica presa num
# display que parou de atualizar. Com o arquivo, ela é recuperável a qualquer
# momento com `cat ${TUNEL_LOG}`.
TUNEL_LOG="${TUNEL_LOG:-/content/tunel.log}"
: > "${TUNEL_LOG}"

cloudflared tunnel --url "http://localhost:${PORTA}" --no-autoupdate   --logfile "${TUNEL_LOG}" >> "${TUNEL_LOG}" 2>&1 &
TUNEL_PID=$!

# Espera o endereço aparecer no log em vez de dormir um tempo fixo: o registro
# do túnel varia de alguns segundos a meio minuto.
ENDERECO=""
for _ in $(seq 1 45); do
  ENDERECO="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "${TUNEL_LOG}" | head -1 || true)"
  [ -n "${ENDERECO}" ] && break
  sleep 2
done

echo
if [ -z "${ENDERECO}" ]; then
  echo "    O túnel não registrou endereço. Últimas linhas do log:" >&2
  tail -n 20 "${TUNEL_LOG}" >&2
  kill "${TUNEL_PID}" 2>/dev/null || true
  exit 1
fi

echo "    ================================================================"
echo "    URL:   ${ENDERECO}"
echo "    Senha: ${SENHA_PORTAL}"
echo "    ================================================================"
echo
echo "    Recuperar depois:  grep -o 'https://.*trycloudflare.com' ${TUNEL_LOG} | head -1"
echo "    Encerrar o túnel:  kill ${TUNEL_PID}"
echo
echo "    Enquanto o túnel estiver no ar, o endereço alcança qualquer pessoa"
echo "    que o tenha. A senha é a única barreira."
echo

# Segura o processo em primeiro plano DEPOIS de imprimir o endereço: parar a
# célula derruba o túnel, que é o desejado, mas a URL já foi registrada em disco
# e sobrevive a um terminal congelado.
wait "${TUNEL_PID}"
