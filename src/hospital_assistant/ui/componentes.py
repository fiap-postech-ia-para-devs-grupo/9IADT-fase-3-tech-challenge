"""Componentes de apresentação: badges, formatação de saída e paginação.

O problema que este módulo resolve: as telas originais entregam estruturas
cruas ao Streamlit — uma lista de dicionários vira `[object Object]` na
tabela, um timestamp ISO com microssegundos e fuso ocupa meia coluna, e o
status aparece como `nao_necessaria`. Nada disso é legível para um médico.

As funções aqui são puras (recebem dados, devolvem texto ou DataFrame) para
que possam ser testadas sem subir o Streamlit — o que também mantém o
`portal.py` fino, cuidando só de layout e interação.
"""

from __future__ import annotations

import html
from datetime import datetime
from typing import Any

import pandas as pd

from hospital_assistant.llm.prompt import filtrar_relevantes
from hospital_assistant.ui import rotulos, tema

# ---------------------------------------------------------------------------
# Formatação de valores
# ---------------------------------------------------------------------------


def formatar_data_hora(valor: str) -> str:
    """Converte timestamp ISO em `dd/mm/aaaa HH:MM`.

    A trilha de auditoria grava com microssegundos e fuso
    (`2026-09-04T22:30:46.812498+00:00`), o que é correto para o registro e
    ilegível numa tabela. Valor não reconhecido volta intacto: é melhor mostrar
    o dado bruto do que esconder que ele veio fora do formato esperado.
    """
    if not valor:
        return "—"
    try:
        return datetime.fromisoformat(valor).strftime("%d/%m/%Y %H:%M")
    except ValueError:
        return valor


def fontes_relevantes(fontes: Any) -> list[dict[str, Any]]:
    """Só os trechos que de fato sustentaram a resposta.

    Aplica o **mesmo** limiar que o prompt: o que não entrou no contexto do
    modelo não pode aparecer como fundamentação da resposta dele. Sem isto o
    assistente dizia "nenhum protocolo institucional cobre esta pergunta" e a
    tela, logo abaixo, listava três protocolos — o que desmente a própria
    resposta e destrói a explicabilidade que o painel existe para dar.

    A trilha de auditoria continua guardando tudo o que foi recuperado; o filtro
    é de exibição. O que se recuperou e o que fundamentou são perguntas
    diferentes, e a auditoria precisa responder as duas.
    """
    if not isinstance(fontes, list):
        return []
    return filtrar_relevantes([f for f in fontes if isinstance(f, dict)])


def formatar_fontes(fontes: Any) -> str:
    """Resume os chunks do RAG como `protocolo (score)`, separados por vírgula.

    É o campo que aparecia como `[object Object]`: uma lista de dicionários
    entregue direto ao `st.dataframe`. O nome do arquivo dá lugar ao título do
    protocolo — a auditoria é lida por médico, não por quem mantém o índice.
    """
    if not isinstance(fontes, list) or not fontes:
        return "—"

    fontes = fontes_relevantes(fontes)
    if not fontes:
        return "nenhum protocolo institucional sustentou esta resposta"

    # Agrupa por protocolo, guardando o melhor score. O RAG devolve trechos, e
    # dois trechos do mesmo documento viravam duas entradas idênticas depois que
    # o rótulo passou a ser o título em vez do nome do arquivo — a lista parecia
    # repetida e não dizia de quantos documentos distintos a resposta saiu.
    melhores: dict[str, float | None] = {}
    for fonte in fontes:
        if not isinstance(fonte, dict):
            continue
        titulo = rotulos.nome_da_fonte(str(fonte.get("source", "")))
        score = fonte.get("score")
        score = float(score) if isinstance(score, int | float) else None

        if titulo not in melhores:
            melhores[titulo] = score
        elif score is not None and (melhores[titulo] is None or score > melhores[titulo]):
            melhores[titulo] = score

    partes = [
        f"{titulo} ({score:.2f})" if score is not None else titulo
        for titulo, score in melhores.items()
    ]
    return ", ".join(partes) if partes else "—"


def formatar_flags(flags: Any) -> str:
    """Traduz as flags internas do guardrail para nomes legíveis."""
    if not isinstance(flags, list) or not flags:
        return "—"
    return ", ".join(rotulos.rotular_flag(str(flag)) for flag in flags)


def resumir_texto(texto: Any, limite: int = 90) -> str:
    """Colapsa quebras de linha e corta o texto para caber numa célula."""
    if not texto:
        return "—"
    limpo = " ".join(str(texto).split())
    return limpo if len(limpo) <= limite else limpo[: limite - 1] + "…"


def nome_do_status(status: str) -> str:
    """Rótulo legível de um status de validação."""
    return tema.CORES_STATUS.get(status, (tema.NEUTRO, tema.NEUTRO_FUNDO, status))[2]


# ---------------------------------------------------------------------------
# Badges e cartões
# ---------------------------------------------------------------------------


def badge_status(status: str) -> str:
    """Badge colorido de um status de validação."""
    cor, fundo, rotulo = tema.CORES_STATUS.get(
        status, (tema.NEUTRO, tema.NEUTRO_FUNDO, status)
    )
    return f'<span class="badge" style="color:{cor};background:{fundo}">{html.escape(rotulo)}</span>'


def badge_categoria(categoria: str) -> str:
    """Badge colorido de uma categoria da base de conhecimento."""
    cor, fundo = tema.CORES_CATEGORIA.get(categoria, (tema.NEUTRO, tema.NEUTRO_FUNDO))
    rotulo = rotulos.CATEGORIAS.get(categoria, categoria)
    return f'<span class="badge" style="color:{cor};background:{fundo}">{html.escape(rotulo)}</span>'


def cartao_fonte(fonte: dict[str, Any], posicao: int) -> str:
    """Cartão de uma fonte do RAG: protocolo, score com barra e trecho recuperado.

    Substitui o dump de JSON cru. O score vira barra além de número porque a
    comparação entre as três fontes é o que interessa ao médico ao decidir se
    a resposta está bem fundamentada — e comparar barras é mais rápido que
    comparar decimais.
    """
    titulo = rotulos.nome_da_fonte(str(fonte.get("source", "")))
    trecho = resumir_texto(fonte.get("text", ""), limite=320)
    score = fonte.get("score")

    if isinstance(score, int | float):
        largura = max(0.0, min(1.0, float(score))) * 100
        score_html = f'<span class="fonte-score">similaridade {score:.3f}</span>'
        barra = f'<div class="barra-score"><div style="width:{largura:.1f}%"></div></div>'
    else:
        score_html = '<span class="fonte-score">sem score</span>'
        barra = ""

    return f"""
<div class="fonte-card">
  <div class="fonte-topo">
    <span class="fonte-arquivo">[{posicao}] {html.escape(titulo)}</span>
    {score_html}
  </div>
  <div class="fonte-trecho">{html.escape(trecho)}</div>
  {barra}
</div>
"""


def cartao_faq(item: dict[str, Any]) -> str:
    """Cartão de uma pergunta frequente, com badge de categoria e fonte."""
    return f"""
<div class="faq-item">
  <div class="faq-pergunta">{html.escape(item["pergunta"])}</div>
  <div>{badge_categoria(item["categoria"])}</div>
  <div class="faq-resposta" style="margin-top:.55rem">{html.escape(item["resposta"])}</div>
  <span class="faq-fonte">{html.escape(item["fonte"])}</span>
</div>
"""


def metrica(valor: Any, rotulo: str) -> str:
    """Cartão de indicador numérico para o painel."""
    return f"""
<div class="metrica">
  <div class="metrica-valor">{html.escape(str(valor))}</div>
  <div class="metrica-rotulo">{html.escape(rotulo)}</div>
</div>
"""


# ---------------------------------------------------------------------------
# Tabelas
# ---------------------------------------------------------------------------

# Formatador por campo. Quem não estiver aqui passa por `resumir_texto`.
_FORMATADORES = {
    "timestamp": formatar_data_hora,
    "timestamp_aprovacao": formatar_data_hora,
    "fontes_rag": formatar_fontes,
    "flags_seguranca": formatar_flags,
    "status": nome_do_status,
}


def tabela_auditoria(linhas: list[dict[str, Any]]) -> pd.DataFrame:
    """Monta o DataFrame da auditoria com colunas legíveis e valores formatados.

    A ordem das colunas é fixada aqui (e não deixada por conta da ordem das
    chaves do dicionário) porque a leitura de uma linha de auditoria tem uma
    sequência natural: quando, que tipo de operação, quem pediu, sobre quem, o
    que foi perguntado e respondido, com base em quê, e em que situação está.

    `paciente` e `medico_solicitante` não vêm da trilha de auditoria — ela grava
    o id do paciente e só o nome de quem aprovou. Quem os resolve é a tela, que
    tem acesso ao cadastro; aqui eles chegam prontos, para este módulo continuar
    sendo só formatação.
    """
    ordem = [
        "id",
        "timestamp",
        "tipo_operacao",
        "medico_solicitante",
        "paciente",
        "pergunta",
        "resposta_llm",
        "fontes_rag",
        "flags_seguranca",
        "status",
        "aprovador",
    ]

    if not linhas:
        return pd.DataFrame(columns=[rotulos.rotular(campo) for campo in ordem])

    registros: list[dict[str, Any]] = []
    for linha in linhas:
        registro: dict[str, Any] = {}
        for campo in ordem:
            valor = linha.get(campo)
            formatador = _FORMATADORES.get(campo)
            if formatador is not None:
                registro[rotulos.rotular(campo)] = formatador(valor)  # type: ignore[arg-type]
            elif campo in ("paciente", "medico_solicitante", "tipo_operacao"):
                registro[rotulos.rotular(campo)] = valor or "—"
            else:
                registro[rotulos.rotular(campo)] = resumir_texto(valor)
        registros.append(registro)

    return pd.DataFrame(registros)


def tabela_generica(linhas: list[dict[str, Any]], campos: list[str] | None = None) -> pd.DataFrame:
    """DataFrame de uma lista de dicionários, com nomes de coluna legíveis."""
    if not linhas:
        return pd.DataFrame()

    campos = campos or list(linhas[0])
    registros = [
        {rotulos.rotular(campo): resumir_texto(linha.get(campo), limite=140) for campo in campos}
        for linha in linhas
    ]
    return pd.DataFrame(registros)


def paginar(linhas: list[Any], pagina: int, por_pagina: int) -> tuple[list[Any], int]:
    """Recorta uma página e devolve `(itens, total_de_paginas)`.

    A página é normalizada para o intervalo válido em vez de levantar erro:
    o número vem de um widget que pode ficar defasado quando um filtro reduz
    o total, e nesse caso a resposta certa é mostrar a última página, não
    quebrar a tela.
    """
    total = max(1, -(-len(linhas) // por_pagina))  # divisão para cima
    pagina = max(1, min(pagina, total))
    inicio = (pagina - 1) * por_pagina
    return linhas[inicio : inicio + por_pagina], total


def cartao_risco(risco: str | None, avaliado_em: str | None = None) -> str:
    """Cartão da classificação de risco, no formato dos demais indicadores.

    Sem classificação, diz isso em vez de ficar em branco: um espaço vazio ao
    lado de "Exames" e "Alertas" parece dado que não carregou, não ausência de
    avaliação.

    A data acompanha porque uma classificação de meses atrás não afirma o mesmo
    que a de hoje, e sem ela quem lê não tem como distinguir.
    """
    from hospital_assistant.ui import laudo as _laudo

    if risco not in tema.CORES_RISCO:
        return (
            '<div class="metrica">'
            f'<div class="metrica-valor" style="font-size:1rem;color:{tema.TEXTO_TENUE}">'
            "Não avaliado</div>"
            '<div class="metrica-rotulo">Classificação de risco</div></div>'
        )

    cor, fundo = tema.CORES_RISCO[risco]
    rotulo = _laudo.RISCOS[risco].split("—")[0].strip()
    quando = f" · {formatar_data_hora(avaliado_em).split(' ')[0]}" if avaliado_em else ""
    return (
        f'<div class="metrica" style="background:{fundo}">'
        f'<div class="metrica-valor" style="font-size:1.15rem;color:{cor}">'
        f"{html.escape(rotulo)}</div>"
        f'<div class="metrica-rotulo">Classificação de risco{html.escape(quando)}</div></div>'
    )


def badge_risco(risco: str | None, avaliado_em: str | None = None) -> str:
    """Badge compacto do risco, para a grid de pacientes.

    Versão de linha do `cartao_risco`: numa listagem de dez pacientes, dez
    cartões ocupariam a tela inteira e a comparação entre eles — que é para o
    que a grid serve — ficaria impossível.
    """
    from hospital_assistant.ui import laudo as _laudo

    if risco not in tema.CORES_RISCO:
        return f'<span style="color:{tema.TEXTO_TENUE};font-size:.8rem">Sem classificação</span>'

    cor, fundo = tema.CORES_RISCO[risco]
    rotulo = _laudo.RISCOS[risco].split("—")[0].strip()
    quando = f" · {formatar_data_hora(avaliado_em).split(' ')[0]}" if avaliado_em else ""
    return (
        f'<span class="badge" style="color:{cor};background:{fundo}">{html.escape(rotulo)}</span>'
        f'<span style="color:{tema.TEXTO_TENUE};font-size:.72rem">{html.escape(quando)}</span>'
    )


def badge_alertas(alertas: list[dict[str, Any]]) -> str:
    """Alertas abertos de um paciente, coloridos pela severidade mais grave.

    A contagem sozinha não diferencia três alertas leves de um grave, e é o
    grave que decide a ordem de atendimento. Sem alerta nenhum diz isso por
    extenso: um zero ao lado de uma classificação de risco lê-se como dado
    faltando, não como ausência de alerta.
    """
    if not alertas:
        return f'<span style="color:{tema.TEXTO_TENUE};font-size:.8rem">Nenhum</span>'

    ordem = {"alta": 0, "media": 1, "baixa": 2}
    pior = min((a.get("severidade", "baixa") for a in alertas), key=lambda s: ordem.get(s, 3))
    cor, fundo = {
        "alta": (tema.ALERTA, tema.ALERTA_FUNDO),
        "media": (tema.PENDENTE, tema.PENDENTE_FUNDO),
    }.get(pior, (tema.NEUTRO, tema.NEUTRO_FUNDO))

    rotulo = "1 alerta" if len(alertas) == 1 else f"{len(alertas)} alertas"
    return (
        f'<span class="badge" style="color:{cor};background:{fundo}">'
        f"{html.escape(rotulo)} · {html.escape(pior)}</span>"
    )
