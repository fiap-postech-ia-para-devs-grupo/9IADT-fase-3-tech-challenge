"""Trava de acesso para quando o portal fica exposto na internet.

**Isto não é autenticação.** É uma senha única compartilhada, comparada em
memória, sem usuários, sem sessão persistente, sem bloqueio por tentativas. Não
identifica ninguém e não substitui login — a identificação de quem valida
continua sendo a seleção de médico do cadastro.

Existe por um motivo estreito: ao publicar o portal por um túnel, o endereço
passa a alcançar qualquer pessoa da internet, e sem nenhuma barreira um
desconhecido poderia aprovar laudos e ler o prontuário. A senha transforma
"aberto" em "aberto para quem tem o link e a senha", que para uma demonstração
é a proteção proporcional.

Fica **desligada** quando `SENHA_PORTAL` não está definida, que é o caso do uso
local e da sessão privada do Colab. Ligar uma trava onde ela não é necessária
só atrapalharia quem desenvolve.
"""

from __future__ import annotations

import hmac
import os

import streamlit as st

VARIAVEL = "SENHA_PORTAL"
CHAVE_SESSAO = "acesso_liberado"


def exigida() -> bool:
    """Se há senha configurada para este ambiente."""
    return bool(os.environ.get(VARIAVEL, "").strip())


def liberado() -> bool:
    """Deixa passar, ou desenha a tela de senha e barra.

    Devolve `True` quando o portal pode ser renderizado.
    """
    if not exigida():
        return True
    if st.session_state.get(CHAVE_SESSAO):
        return True

    st.markdown("### Portal Clínico")
    st.caption(
        "Este ambiente está publicado na internet e pede a senha combinada com a equipe."
    )

    with st.form("acesso"):
        tentativa = st.text_input("Senha de acesso", type="password")
        if st.form_submit_button("Entrar", type="primary"):
            # `compare_digest` em vez de `==`: a comparação byte a byte do
            # Python sai no primeiro caractere diferente, e o tempo de resposta
            # vaza quantos caracteres estavam certos.
            if hmac.compare_digest(tentativa, os.environ[VARIAVEL].strip()):
                st.session_state[CHAVE_SESSAO] = True
                st.rerun()
            else:
                st.error("Senha incorreta.")

    return False
