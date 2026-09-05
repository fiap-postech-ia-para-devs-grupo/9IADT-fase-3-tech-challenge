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

import hashlib
import hmac
import os

import streamlit as st

VARIAVEL = "SENHA_PORTAL"
CHAVE_SESSAO = "acesso_liberado"
PARAMETRO = "acesso"


def _marca() -> str:
    """Marca derivada da senha, guardada na URL para o acesso durar.

    `session_state` morre a cada recarregamento de página, e num celular isso
    acontece o tempo todo — a senha era pedida de novo a cada volta. A marca no
    endereço sobrevive ao reload e ao compartilhamento do link entre abas.

    **É a senha em outra forma, não uma proteção adicional.** Quem receber a URL
    com a marca entra sem digitar nada. Isso não muda o modelo de segurança
    daqui, que já era "quem tem o link e a senha entra": a senha é única e
    compartilhada, e quem a tem passa o acesso adiante de um jeito ou de outro.
    O que muda é que o link fica perigoso de colar em lugar público — mais que
    o endereço sozinho.

    A senha em si não vai para a URL: o que vai é um resumo dela, para não
    aparecer em histórico de navegador e log de servidor.
    """
    senha = os.environ.get(VARIAVEL, "").strip()
    return hashlib.sha256(f"portal-clinico:{senha}".encode()).hexdigest()[:32]


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
    if hmac.compare_digest(st.query_params.get(PARAMETRO, ""), _marca()):
        st.session_state[CHAVE_SESSAO] = True
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
                # Na URL também: sem isto, recarregar a página pede a senha de
                # novo, que é o que acontecia no celular a cada volta.
                st.query_params[PARAMETRO] = _marca()
                st.rerun()
            else:
                st.error("Senha incorreta.")

    return False


def sufixo_url() -> str:
    """Trecho a anexar em links internos para não perder a autorização.

    A navegação do portal é feita de âncoras que reescrevem a query string
    inteira. Como a marca de acesso vive ali, um link que não a carrega adiante
    desloga o usuário a cada troca de tela.

    Devolve string vazia quando não há senha configurada, para o endereço não
    ganhar lixo em ambiente sem trava.
    """
    if not exigida():
        return ""
    return f"&{PARAMETRO}={_marca()}"
