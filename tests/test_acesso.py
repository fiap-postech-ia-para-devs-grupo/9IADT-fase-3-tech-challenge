"""Trava de acesso do portal publicado.

Não é autenticação — é senha única compartilhada, e os testes existem para
garantir que ela fique **desligada** onde não é necessária e que a comparação
não vaze informação pelo tempo de resposta.
"""

from __future__ import annotations

from hospital_assistant.ui import acesso


def test_desligada_sem_variavel(monkeypatch) -> None:
    """Uso local e sessão privada do Colab não devem pedir senha."""
    monkeypatch.delenv(acesso.VARIAVEL, raising=False)

    assert acesso.exigida() is False


def test_variavel_vazia_conta_como_desligada(monkeypatch) -> None:
    """`SENHA_PORTAL=` no .env não pode virar uma senha vazia que barra todo mundo."""
    monkeypatch.setenv(acesso.VARIAVEL, "   ")

    assert acesso.exigida() is False


def test_ligada_com_senha(monkeypatch) -> None:
    monkeypatch.setenv(acesso.VARIAVEL, "segredo")

    assert acesso.exigida() is True


def test_comparacao_e_de_tempo_constante() -> None:
    """`==` sai no primeiro caractere diferente e vaza quantos estavam certos."""
    import inspect

    fonte = inspect.getsource(acesso.liberado)

    assert "compare_digest" in fonte


def test_marca_nao_expoe_a_senha(monkeypatch) -> None:
    """A marca vai para a URL, e URL vaza em histórico e log de servidor."""
    monkeypatch.setenv(acesso.VARIAVEL, "FIAP2026")

    assert "FIAP2026" not in acesso._marca()


def test_marca_muda_com_a_senha(monkeypatch) -> None:
    """Trocar a senha precisa invalidar os links já distribuídos."""
    monkeypatch.setenv(acesso.VARIAVEL, "uma")
    primeira = acesso._marca()
    monkeypatch.setenv(acesso.VARIAVEL, "outra")

    assert acesso._marca() != primeira


def test_marca_e_estavel_para_a_mesma_senha(monkeypatch) -> None:
    """Se mudasse a cada chamada, o link salvo no celular pararia de valer."""
    monkeypatch.setenv(acesso.VARIAVEL, "FIAP2026")

    assert acesso._marca() == acesso._marca()
