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

# Cada estado recebe uma matiz claramente distinta das demais — âmbar, carmim,
# verde e ardósia — com o texto em tom escuro o suficiente para passar em
# contraste sobre o próprio fundo do chip. Tons pastel de mesma família
# confundiriam "pendente" com "alerta" numa lista longa, que é exatamente onde
# a distinção importa.
PENDENTE = "#9A5B00"
PENDENTE_FUNDO = "#FDF0DC"
ALERTA = "#A81F43"
ALERTA_FUNDO = "#FCE8EE"
APROVADO = "#0F7040"
APROVADO_FUNDO = "#E3F5EA"
NEUTRO = "#4E6360"
NEUTRO_FUNDO = "#EBF1EF"

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
    # Vocabulário do laudo. Fica no mesmo mapa para reusar o badge, mas com
    # chaves próprias: na tela de laudos a resposta **já foi validada**, e
    # reaproveitar "Pendente de validação" dizia ao médico que faltava algo que
    # ele acabara de fazer.
    "sem_laudo": (NEUTRO, NEUTRO_FUNDO, "Sem laudo"),
    "laudo_pendente": (PENDENTE, PENDENTE_FUNDO, "Pendente de conclusão"),
    "laudo_concluido": (APROVADO, APROVADO_FUNDO, "Laudo concluído"),
}

# Cores da classificação de risco. Seguem a convenção de triagem que qualquer
# equipe de pronto-socorro já lê sem legenda — inventar uma paleta própria aqui
# custaria clareza numa informação que precisa ser entendida num relance.
CORES_RISCO: dict[str, tuple[str, str]] = {
    "verde": (APROVADO, APROVADO_FUNDO),
    "amarelo": (PENDENTE, PENDENTE_FUNDO),
    "vermelho": (ALERTA, ALERTA_FUNDO),
}

CORES_CATEGORIA: dict[str, tuple[str, str]] = {
    "protocolo": (PRIMARIA, PRIMARIA_CLARA),
    "exames": ("#1B4FA8", "#E6EDF9"),
    "medicacao": (PENDENTE, PENDENTE_FUNDO),
    "fluxo": ("#5B3AA8", "#EFEAFA"),
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


# Ícones da navegação. Inline e geométricos, no traço de 1.6px que combina com
# o peso do texto do menu — um pacote de ícones inteiro seria dependência nova
# para cinco símbolos.
ICONES: dict[str, str] = {
    "assistente": '<path d="M3 5h10v7H7l-3 3V5z"/>',
    "validacao": '<path d="M4 8.5 6.8 11 12 4.5"/><path d="M2.5 2.5h11v11h-11z" opacity=".45"/>',
    "conhecimento": '<path d="M3 3.5h4.2c.9 0 1.8.5 1.8 1.4V13c0-.7-.7-1.2-1.6-1.2H3z"/>'
    '<path d="M13 3.5H8.8c-.9 0-1.8.5-1.8 1.4V13c0-.7.7-1.2 1.6-1.2H13z"/>',
    "pacientes": '<circle cx="8" cy="5.5" r="2.4"/><path d="M3.4 13.2c0-2.3 2-4 4.6-4s4.6 1.7 4.6 4"/>',
    "auditoria": '<path d="M3 4h10M3 8h10M3 12h6"/>',
}


def icone(nome: str, tamanho: int = 15) -> str:
    """Ícone de navegação como SVG inline, herdando a cor do item do menu."""
    traco = ICONES.get(nome, ICONES["auditoria"])
    return (
        f'<svg width="{tamanho}" height="{tamanho}" viewBox="0 0 16 16" fill="none" '
        f'stroke="currentColor" stroke-width="1.6" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{traco}</svg>'
    )


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

  /* --- navegação -------------------------------------------------------- */

  .grupo-menu {{
    font-size: .66rem; font-weight: 700; letter-spacing: .14em;
    text-transform: uppercase; color: {TEXTO_TENUE};
    margin: 1.15rem 0 .35rem; padding-left: .7rem;
    user-select: none;   /* agrupador não é clicável */
  }}
  .grupo-menu:first-of-type {{ margin-top: .3rem; }}

  .nav-item {{
    display: flex; align-items: center; gap: .6rem;
    padding: .46rem .7rem; margin-bottom: .1rem;
    border-radius: 7px;
    border-left: 3px solid transparent;   /* reserva o indicador lateral */
    color: {PRIMARIA}; font-size: .89rem; font-weight: 500;
    text-decoration: none; background: transparent;
    transition: background .12s ease, color .12s ease;
  }}
  /* O Streamlit estiliza todo `a` dentro de markdown com sublinhado e azul de
     link, com especificidade maior que a nossa. Sem `!important` nas duas
     propriedades o item de menu continua com cara de hyperlink. */
  .nav-item, .nav-item:hover, .nav-item:visited, .nav-item:active {{
    text-decoration: none !important;
    color: {PRIMARIA} !important;
  }}
  .nav-item:hover {{ background: {PRIMARIA_CLARA}; color: {PRIMARIA_ESCURA} !important; }}
  .nav-item svg {{ flex: 0 0 auto; opacity: .65; }}

  .nav-item.ativo, .nav-item.ativo:visited {{
    background: {PRIMARIA_CLARA};
    border-left-color: {PRIMARIA};
    color: {PRIMARIA_ESCURA} !important;
    font-weight: 600;
  }}
  .nav-item.ativo svg {{ opacity: 1; }}

  .nav-contagem {{
    margin-left: auto;
    min-width: 20px; height: 20px; padding: 0 6px;
    display: inline-flex; align-items: center; justify-content: center;
    background: {ALERTA}; color: #FFFFFF;
    font-size: .7rem; font-weight: 700; line-height: 1;
    border-radius: 100px; font-variant-numeric: tabular-nums;
  }}

  .rodape-status {{
    margin-top: 1.3rem; padding-top: .9rem;
    border-top: 1px solid {BORDA};
    display: flex; flex-direction: column; gap: .4rem;
  }}
  .chip {{
    display: inline-flex; align-items: center; gap: .4rem;
    font-size: .72rem; font-weight: 600;
    padding: .22rem .55rem; border-radius: 100px; width: fit-content;
  }}
  .chip-modelo {{
    font-family: ui-monospace, Menlo, Consolas, monospace;
    font-size: .7rem; color: {TEXTO_TENUE}; padding-left: .2rem;
    word-break: break-word; font-weight: 400;
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

  /* Com paciente, solicitante e tipo de operação a auditoria passou a ter mais
     colunas do que cabe na tela. Sem a rolagem própria, a página inteira rola
     de lado e a barra lateral sai de vista. */
  /* Spinner como overlay central. Inline, ele nasce onde o código chamou —
     no fim da conversa, quase sempre abaixo da dobra — e a tela parecia
     travada durante os segundos de geração.

     O conteúdo padrão do Streamlit (uma barra com texto ao lado) é escondido e
     substituído por um anel desenhado no `::after`: numa tela bloqueada, um
     único elemento centralizado comunica "aguarde" melhor que uma faixa com
     rótulo, que compete com o resto da página por atenção. */
  div[data-testid="stSpinner"] {{
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    z-index: 9999;
    display: flex; align-items: center; justify-content: center;
    background: {SUPERFICIE}d9;
    backdrop-filter: blur(2px);
  }}
  div[data-testid="stSpinner"] > div {{ display: none; }}
  div[data-testid="stSpinner"]::after {{
    content: "";
    width: 84px; height: 84px;
    border-radius: 50%;
    border: 6px solid {PRIMARIA_CLARA};
    border-top-color: {PRIMARIA};
    animation: girar .8s linear infinite;
  }}
  @keyframes girar {{ to {{ transform: rotate(360deg); }} }}

  /* Sem animação para quem pediu movimento reduzido: o anel vira um disco
     estático, que ainda marca a tela como bloqueada. */
  @media (prefers-reduced-motion: reduce) {{
    div[data-testid="stSpinner"]::after {{
      animation: none;
      border-color: {PRIMARIA_CLARA};
      border-top-color: {PRIMARIA_CLARA};
    }}
  }}

  div[data-testid="stDataFrame"] {{ border-radius: 6px; overflow-x: auto; }}
  .stButton > button {{ border-radius: 6px; font-weight: 500; }}

  /* --- assistente ------------------------------------------------------- */

  /* Coluna central estreita. Texto clínico corrido em 1400px de largura é
     desconfortável de ler; ~62rem mantém a linha perto da faixa legível e dá
     à conversa o eixo central que se espera de uma interface de chat. */
  .st-key-bloco_assistente {{ max-width: 62rem; margin: 0 auto; }}

  .saudacao {{
    text-align: center; margin: 2.2rem 0 1.4rem;
  }}
  .saudacao h2 {{
    font-size: 1.65rem; font-weight: 600; color: {TEXTO};
    letter-spacing: -.02em; margin: 0 0 .4rem;
  }}
  .saudacao p {{ color: {TEXTO_TENUE}; font-size: .92rem; margin: 0; }}

  /* Composer arredondado. O seletor é global aos textareas de propósito: o
     campo de edição da fila de validação ganha o mesmo tratamento, e manter
     uma única aparência de entrada de texto é mais coerente que criar duas. */
  div[data-testid="stTextArea"] textarea {{
    border-radius: 14px !important;
    border: 1px solid {BORDA} !important;
    padding: .85rem 1rem !important;
    font-size: .95rem;
    background: {SUPERFICIE};
  }}
  div[data-testid="stTextArea"] textarea:focus {{
    border-color: {PRIMARIA} !important;
    box-shadow: 0 0 0 3px {PRIMARIA}1f !important;
  }}

  .contador {{
    display: flex; justify-content: flex-end;
    font-size: .72rem; color: {TEXTO_TENUE};
    font-variant-numeric: tabular-nums; margin-top: -.4rem;
  }}
  .contador.excedido {{ color: {ALERTA}; font-weight: 600; }}

  /* Chips de sugestão: discretos, uma linha, sem competir com o composer. */
  .sugestoes-titulo {{
    font-size: .72rem; text-transform: uppercase; letter-spacing: .1em;
    color: {TEXTO_TENUE}; margin: 1.1rem 0 .45rem;
  }}
  /* Sugestões usam a mesma pílula verde dos indicadores de status do rodapé:
     um atalho para a base de conhecimento é um selo, não mais um botão
     disputando atenção com o composer. */
  /* Chips ancorados na `key` do widget, e não em `.stButton > button`: quando o
     botão tem `help`, o Streamlit o embrulha num alvo de tooltip e o seletor de
     filho direto para de casar — era por isso que só o "Gerar outras", único
     sem `help`, ficava verde. */
  /* Chips colados: o respiro padrão entre colunas do Streamlit é grande demais
     e desmancha a leitura de conjunto. */
  .st-key-linha_sugestoes div[data-testid="stHorizontalBlock"] {{ gap: .3rem; }}
  .st-key-linha_sugestoes div[data-testid="stColumn"] {{ min-width: 0; }}

  div[class*="st-key-sugestao-"] button,
  .st-key-regerar_sugestoes button,
  .st-key-limpar_conversa button {{
    border-radius: 100px;
    border: 1px solid {PRIMARIA};
    background: {SUPERFICIE};
    color: {PRIMARIA_ESCURA};
    padding: .2rem .78rem; min-height: 0; line-height: 1.4;
    white-space: nowrap;   /* sem isto o rótulo quebra e a pílula vira bolha */
  }}
  /* O rótulo mora num `p` com tamanho próprio; sem esta regra o chip encolhe
     mas o texto continua no corpo de botão comum. */
  div[class*="st-key-sugestao-"] button p,
  .st-key-regerar_sugestoes button p,
  .st-key-limpar_conversa button p {{
    font-size: .72rem; font-weight: 600; margin: 0; white-space: nowrap;
  }}
  div[class*="st-key-sugestao-"] button:hover,
  .st-key-limpar_conversa button:hover {{
    background: {PRIMARIA_CLARA}; border-color: {PRIMARIA}; color: {PRIMARIA_ESCURA};
  }}
  /* "Gerar outras" é ação, não atalho: fica preenchido para se distinguir dos
     chips de sugestão, que agora são contornados. */
  .st-key-regerar_sugestoes button {{
    background: {PRIMARIA}; color: #FFFFFF; border-color: {PRIMARIA};
  }}
  .st-key-regerar_sugestoes button:hover {{
    background: {PRIMARIA_ESCURA}; border-color: {PRIMARIA_ESCURA}; color: #FFFFFF;
  }}
  /* O botão de envio é a ação primária e mantém a forma de botão. Dentro de um
     st.form o Streamlit troca o `kind` para `primaryFormSubmit`, então os dois
     precisam constar — só `primary` deixaria o envio com cara de chip. */
  .st-key-enviar_pergunta button {{
    border-radius: 8px; padding: .45rem 1.4rem;
    background: {PRIMARIA}; border-color: {PRIMARIA}; color: #FFFFFF;
  }}
  .st-key-enviar_pergunta button p {{ font-size: .9rem; font-weight: 600; }}
  .st-key-enviar_pergunta button:hover {{
    background: {PRIMARIA_ESCURA}; border-color: {PRIMARIA_ESCURA};
  }}
</style>
"""
