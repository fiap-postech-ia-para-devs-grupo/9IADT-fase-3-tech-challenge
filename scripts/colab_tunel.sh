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

echo "==> 1/3  Reiniciando o portal com a senha ativa"
# O servidor precisa nascer com a variável no ambiente; reaproveita o script de
# sempre, que é idempotente.
curl -sL "https://raw.githubusercontent.com/fiap-postech-ia-para-devs-grupo/9IADT-fase-3-tech-challenge/main/scripts/colab_portal.sh" | bash

echo "==> 2/3  Instalando o cloudflared"
if ! command -v cloudflared >/dev/null 2>&1; then
  wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    -O /usr/local/bin/cloudflared
  chmod +x /usr/local/bin/cloudflared
fi

echo "==> 3/3  Abrindo o túnel"
echo
echo "    A URL aparece abaixo, em https://<algo>.trycloudflare.com"
echo "    Senha do portal: ${SENHA_PORTAL}"
echo
echo "    Encerre com o botão de parar desta célula. Enquanto ela roda, o"
echo "    endereço está no ar para qualquer pessoa que o tenha."
echo

# Em primeiro plano de propósito: fechar a célula derruba o túnel, e um túnel
# esquecido rodando em background é exatamente o que não se quer aqui.
cloudflared tunnel --url "http://localhost:${PORTA}" --no-autoupdate
