# Execução do Portal Clínico

Como rodar a aplicação, onde cada coisa vive, e por que o acesso no Colab passa
por um endereço estranho.

## Os dois ambientes

| | Máquina local | Google Colab |
| --- | --- | --- |
| Script | `scripts/start.sh` | `scripts/colab_portal.sh` |
| Atualizar depois de mexer no código | reiniciar o script | `scripts/deploy_colab.sh` + a célula que ele imprime |
| GPU | normalmente não | T4 de 15 GB |
| Resposta do assistente | demonstração ou CPU | modelo treinado, em segundos |
| Endereço | `http://localhost:8501` | `https://8501-….prod.colab.dev` |

Os dois existem porque o ambiente difere em tudo o que importa: quem tem placa
de vídeo, onde os dados são preparados e — o ponto que mais confunde — qual
endereço alcança o servidor.

## Rodando localmente

```bash
./scripts/start.sh
```

O script semeia o banco de pacientes, indexa o RAG, **diagnostica em que modo o
modelo vai rodar** e sobe o Streamlit. O diagnóstico vem antes do servidor de
propósito: sem ele, a falta de GPU só apareceria na primeira renderização, como
um erro na tela sem contexto.

## Rodando no Colab

A **seção 0 do notebook** é o setup da aplicação: uma célula que monta o
ambiente do zero e devolve o endereço. É também a célula de recuperação — a VM
é reciclada por inatividade e leva tudo junto, e nada dentro dela se reergue
sozinho, porque ela deixou de existir.

Para um endereço que funciona em qualquer dispositivo, inclusive celular, ela
usa `colab_tunel.sh`. Para uso restrito ao navegador que abriu o Colab, a
célula seguinte usa `colab_portal.sh` com o `proxyPort`.

Numa célula do notebook, com o ambiente de execução em **T4 GPU**:

```python
!curl -sL https://raw.githubusercontent.com/fiap-postech-ia-para-devs-grupo/9IADT-fase-3-tech-challenge/main/scripts/colab_portal.sh | bash

from google.colab.output import eval_js
print(eval_js('google.colab.kernel.proxyPort(8501)'))
```

A primeira linha monta o ambiente; a segunda imprime o endereço de acesso.
Nenhum token é necessário: o adapter é público e o modelo base vem de um espelho
não-gated.

A primeira pergunta leva alguns minutos — baixa 6,4 GB de pesos e monta o modelo
na GPU. As seguintes saem em segundos.

### Atualizando a sessão depois de mexer no código

```bash
./scripts/deploy_colab.sh
```

Ele roda a suíte, publica na `main` e imprime a célula a rodar no notebook. Não
faz mais que isso por um limite real: **a VM do Colab não é alcançável a partir
da sua máquina.** Ela puxa do GitHub, então atualizar é sempre em dois tempos —
publicar aqui, buscar lá.

O endereço muda a cada reciclagem da VM. É por isso que ele é impresso a cada
atualização, em vez de anotado num lugar fixo.

### Abrindo de fora do computador que roda o Colab

A URL do `proxyPort` **não é compartilhável**: ela é amarrada à sessão
autenticada do Colab naquele navegador. De outro dispositivo — um celular, ou a
máquina de um colega — o proxy do Google responde `404` com corpo vazio, e
alguns navegadores móveis oferecem download em vez de exibir a página em branco.

Para um endereço que funciona em qualquer lugar:

```python
!curl -sL https://raw.githubusercontent.com/fiap-postech-ia-para-devs-grupo/9IADT-fase-3-tech-challenge/main/scripts/colab_tunel.sh | bash
```

O script reinicia o portal com uma senha ativa, sobe um túnel do Cloudflare e
imprime a URL pública junto da senha.

**A URL alcança qualquer pessoa da internet.** A aplicação não tem autenticação
— a seleção de médico identifica quem valida, não controla quem entra. Sem a
senha, um desconhecido com o link poderia aprovar laudos e ler o prontuário; por
isso o script recusa publicar sem uma, e gera se você não fornecer.

A senha é única e compartilhada, comparada em memória: protege uma demonstração,
não um sistema em produção. O túnel cai quando a célula para, o que é
intencional — um túnel esquecido em background é o que não se quer aqui.

## Por que aquele endereço, e não `localhost`

`localhost` significa "esta máquina", e muda de sentido conforme quem fala.

Quando o servidor no Colab diz que está em `http://localhost:8501`, esse
"localhost" é o **da máquina virtual do Google**, um computador num datacenter
(no nosso caso, `europe-west4`, na Holanda). Digitar `localhost:8501` no seu
navegador procura um servidor **no seu PC** — outra máquina, outro conteúdo, ou
nenhum.

O endereço `https://8501-gpu-t4-…-c.europe-west4-1.prod.colab.dev` é a ponte: um
nome público do Google que roteia até a sua VM específica, autenticado pelo
cookie da sua sessão do Colab. Ele é longo porque carrega a porta, o
identificador da máquina e a região. Ninguém sem a sua sessão o alcança, mesmo
tendo o link.

É `google.colab.kernel.proxyPort(8501)` que gera esse endereço, e ela só existe
dentro do kernel do notebook — não no terminal do Colab, não no seu shell. Por
isso a segunda linha da célula é necessária, e por isso `scripts/colab_portal.sh`
imprime a instrução em vez de tentar resolver sozinho.

### Duas consequências que já custaram tempo

**A VM recicla por inatividade.** Ela leva junto o clone, as dependências, os
dados e o servidor. Não há aviso: você volta ao notebook e o endereço responde
com erro. A correção é rodar a célula de novo — é para isso que o script existe
como fonte única e versionada, em vez de um punhado de células soltas.

**O Streamlit recusa o WebSocket atrás do proxy.** O acesso chega de um domínio
`*.prod.colab.dev` diferente do host onde o servidor escuta, e as proteções de
CORS/XSRF derrubam a conexão. O sintoma engana: a página HTTP carrega e trava no
esqueleto de carregamento, sem erro visível, porque é o WebSocket que entrega a
interface. O log mostra `Rejecting WebSocket connection with disallowed Origin
or Host header`. Por isso o script passa `--server.enableCORS false` e
`--server.enableXsrfProtection false`.

Essas duas flags valem **só** para este cenário, em que a porta é alcançável
apenas por quem está autenticado na sessão do Colab. Num deploy exposto à
internet elas não devem existir.

## Como o modelo é escolhido

Preencher `HF_ADAPTER_REPO` é declarar que se quer o modelo treinado. Se o
ambiente não puder carregá-lo, a aplicação **recusa abrir** em vez de responder
com o stand-in — a degradação silenciosa já colocou uma demonstração no ar com o
gerador de mentira, e o único sinal foi um aviso perdido no log.

Em ordem de precedência:

| Condição | Modo | Velocidade |
| --- | --- | --- |
| GPU com ≥ 9 GB livres | fp16, sem quantização | segundos |
| GPU apertada | 4-bit NF4 | segundos, mais lento |
| `PERMITIR_CPU=true` | bfloat16 na RAM | minutos por resposta |
| `MODO_DEMONSTRACAO=true` | stand-in | instantâneo, mas não é o modelo |
| nada disso | `AmbienteSemModelo` | — |

Vale notar que **quantizar deixa a inferência mais lenta**. A NF4 existe para o
treino caber na memória — é o "Q" de QLoRA — e cobra desquantização a cada passo
de geração. Como o modelo em fp16 ocupa ~6,4 GB e a T4 tem 15 GB, quantizar aqui
só custaria tempo. Daí a escolha por memória livre, e não por preferência.

`PERMITIR_CPU` tem precedência sobre `MODO_DEMONSTRACAO`: quem ligou as duas
aceitou esperar, e o stand-in entregaria menos do que o ambiente permite.

### Variáveis

Todas em `.env` (veja `.env.example`; o `.env` real é ignorado pelo git):

- `HF_ADAPTER_REPO` — repositório do adapter. No Colab tem padrão versionado no
  script, porque esquecê-lo era uma falha silenciosa.
- `MODO_DEMONSTRACAO` — libera a interface sem o modelo.
- `PERMITIR_CPU` — roda o modelo real sem GPU.
- `HF_TOKEN` — só para publicar o adapter; não é preciso para consumi-lo.

## Onde cada coisa está versionada

| O quê | Onde |
| --- | --- |
| Aplicação (entry point único) | `app.py` |
| Camada de apresentação | `src/hospital_assistant/ui/` |
| Carregamento do modelo e modos | `src/hospital_assistant/llm/model_loader.py` |
| Execução local | `scripts/start.sh` |
| Execução no Colab | `scripts/colab_portal.sh` |
| Protocolos que alimentam o RAG | `data/raw/protocolos_sinteticos/` |
| Índice vetorial (gerado) | `data/chroma/` |
| Métricas do treino e comparativo | `results/` |
| Adapter LoRA treinado | Hugging Face Hub, [`agendesse/hospital-assistant-llama32-3b-lora`](https://huggingface.co/agendesse/hospital-assistant-llama32-3b-lora) |

O adapter mora no Hub, e não no repositório, porque são centenas de megabytes de
pesos binários — versioná-los no git incharia o clone para todo mundo. O que
fica versionado é o que o produziu (`src/hospital_assistant/finetuning/`) e o que
ele produziu (`results/`).
