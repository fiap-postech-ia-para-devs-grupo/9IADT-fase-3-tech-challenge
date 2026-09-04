"""Identidade visual do portal: paleta, tipografia, logo e CSS.

**Paleta.** Verde-petróleo em vez do azul-hospital genérico. A escolha não é
estética apenas: o sistema usa cor como informação — âmbar para pendência,
carmim para alerta, verde para aprovado — e um primário azul competiria com o
azul que a maioria das interfaces clínicas já usa para links e ações neutras.
O petróleo deixa os três estados semânticos legíveis sem disputar atenção.

Os neutros têm viés levemente esverdeado (não cinza puro) para assentar sobre
o primário sem parecer que vieram de outro sistema.
"""

from __future__ import annotations

# --- paleta -----------------------------------------------------------------

PRIMARIA = "#0F6B62"
PRIMARIA_ESCURA = "#0A4F49"
PRIMARIA_CLARA = "#E3F0EE"

PENDENTE = "#B45309"
PENDENTE_FUNDO = "#FDF3E7"
ALERTA = "#9F1239"
ALERTA_FUNDO = "#FCEBEF"
APROVADO = "#15803D"
APROVADO_FUNDO = "#EAF6EE"
NEUTRO = "#5A6E6B"
NEUTRO_FUNDO = "#EFF3F2"

TEXTO = "#0F1F1D"
TEXTO_SUAVE = "#45605C"
TEXTO_TENUE = "#7C918D"
BORDA = "#D3E0DC"
FUNDO = "#F6F9F8"
SUPERFICIE = "#FFFFFF"

# Cor por status de validação e por categoria da base de conhecimento. Manter
# o mapa aqui, e não espalhado nas telas, é o que garante que o mesmo estado
# tenha a mesma cor em todos os módulos.
CORES_STATUS: dict[str, tuple[str, str, str]] = {
    "pendente": (PENDENTE, PENDENTE_FUNDO, "Pendente de validação"),
    "aprovado": (APROVADO, APROVADO_FUNDO, "Aprovado"),
    "rejeitado": (ALERTA, ALERTA_FUNDO, "Rejeitado"),
    "nao_necessaria": (NEUTRO, NEUTRO_FUNDO, "Sem validação exigida"),
}

CORES_CATEGORIA: dict[str, tuple[str, str]] = {
    "protocolo": (PRIMARIA, PRIMARIA_CLARA),
    "exames": ("#1D4ED8", "#E8EEFD"),
    "medicacao": (PENDENTE, PENDENTE_FUNDO),
    "fluxo": ("#6D28D9", "#F1EBFC"),
    "seguranca": (ALERTA, ALERTA_FUNDO),
}


def logo_svg(altura: int = 34) -> str:
    """Logo do sistema: cruz clínica cujo braço horizontal é um traçado de ECG.

    Inline em SVG e não como arquivo de imagem porque o Streamlit renderiza
    markdown com HTML, e um asset externo exigiria servir estático — atrito
    desnecessário para um símbolo de 40 linhas.
    """
    return f"""
<svg width="{altura}" height="{altura}" viewBox="0 0 40 40" fill="none"
     xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Logo do portal clínico">
  <rect x="1" y="1" width="38" height="38" rx="10" fill="{PRIMARIA}"/>
  <rect x="17" y="7" width="6" height="26" rx="2" fill="#FFFFFF" opacity="0.92"/>
  <path d="M6 20 L13 20 L15.5 14 L19 26 L22.5 17 L25 20 L34 20"
        stroke="#FFFFFF" stroke-width="2.4" stroke-linecap="round"
        stroke-linejoin="round" fill="none"/>
</svg>
"""


def cabecalho_marca() -> str:
    """Bloco de marca da barra lateral: logo, nome do sistema e subtítulo."""
    return f"""
<div class="marca">
  {logo_svg(36)}
  <div class="marca-texto">
    <strong>Portal Clínico</strong>
    <span>Assistente virtual médico</span>
  </div>
</div>
"""


def css() -> str:
    """Folha de estilo do portal.

    Escopo deliberadamente estreito: estiliza a marca, os badges, os cartões de
    fonte e a densidade das tabelas. Não sobrescreve componentes do Streamlit
    além do necessário — quanto mais seletores internos forem alvo, mais frágil
    o tema fica a cada atualização da biblioteca.
    """
    return f"""
<style>
  :root {{
    --primaria: {PRIMARIA};
    --primaria-escura: {PRIMARIA_ESCURA};
    --texto: {TEXTO};
    --texto-suave: {TEXTO_SUAVE};
    --texto-tenue: {TEXTO_TENUE};
    --borda: {BORDA};
    --superficie: {SUPERFICIE};
  }}

  .marca {{
    display: flex; align-items: center; gap: .7rem;
    padding: .2rem 0 1.1rem;
    border-bottom: 1px solid {BORDA};
    margin-bottom: 1rem;
  }}
  .marca-texto {{ display: flex; flex-direction: column; line-height: 1.25; }}
  .marca-texto strong {{ font-size: 1.02rem; color: {TEXTO}; letter-spacing: -.01em; }}
  .marca-texto span {{ font-size: .76rem; color: {TEXTO_TENUE}; }}

  .grupo-menu {{
    font-size: .68rem; font-weight: 700; letter-spacing: .13em;
    text-transform: uppercase; color: {TEXTO_TENUE};
    margin: 1.1rem 0 .3rem;
  }}

  .badge {{
    display: inline-flex; align-items: center; gap: .34rem;
    font-size: .7rem; font-weight: 600; letter-spacing: .02em;
    padding: .2rem .58rem; border-radius: 100px; white-space: nowrap;
  }}
  .badge::before {{
    content: ""; width: 6px; height: 6px; border-radius: 50%;
    background: currentColor;
  }}

  .fonte-card {{
    border: 1px solid {BORDA};
    border-left: 3px solid {PRIMARIA};
    border-radius: 6px;
    padding: .75rem .9rem;
    margin-bottom: .55rem;
    background: {SUPERFICIE};
  }}
  .fonte-topo {{
    display: flex; justify-content: space-between; align-items: baseline;
    gap: 1rem; margin-bottom: .35rem;
  }}
  .fonte-arquivo {{
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    font-size: .76rem; color: {PRIMARIA_ESCURA}; word-break: break-all;
  }}
  .fonte-score {{
    font-size: .72rem; color: {TEXTO_TENUE}; white-space: nowrap;
    font-variant-numeric: tabular-nums;
  }}
  .fonte-trecho {{ font-size: .86rem; color: {TEXTO_SUAVE}; line-height: 1.5; }}

  .barra-score {{
    height: 4px; border-radius: 2px; background: {NEUTRO_FUNDO};
    margin-top: .5rem; overflow: hidden;
  }}
  .barra-score > div {{ height: 100%; background: {PRIMARIA}; }}

  .aviso-seguranca {{
    border: 1px solid {ALERTA}33;
    background: {ALERTA_FUNDO};
    border-radius: 6px; padding: .8rem 1rem;
    font-size: .88rem; color: {TEXTO};
  }}

  .faq-item {{
    border: 1px solid {BORDA}; border-radius: 8px;
    padding: 1rem 1.1rem; margin-bottom: .7rem; background: {SUPERFICIE};
  }}
  .faq-pergunta {{ font-weight: 600; color: {TEXTO}; margin-bottom: .45rem; }}
  .faq-resposta {{ font-size: .9rem; color: {TEXTO_SUAVE}; line-height: 1.6; }}
  .faq-fonte {{
    font-family: ui-monospace, Menlo, Consolas, monospace;
    font-size: .72rem; color: {TEXTO_TENUE}; margin-top: .55rem; display: block;
  }}

  .metrica {{
    border: 1px solid {BORDA}; border-radius: 8px;
    padding: .85rem 1rem; background: {SUPERFICIE};
  }}
  .metrica-valor {{
    font-size: 1.6rem; font-weight: 600; color: {TEXTO};
    font-variant-numeric: tabular-nums; line-height: 1.1;
  }}
  .metrica-rotulo {{
    font-size: .74rem; color: {TEXTO_TENUE};
    text-transform: uppercase; letter-spacing: .08em; margin-top: .2rem;
  }}

  div[data-testid="stDataFrame"] {{ border-radius: 6px; }}
  .stButton > button {{ border-radius: 6px; font-weight: 500; }}
</style>
"""
