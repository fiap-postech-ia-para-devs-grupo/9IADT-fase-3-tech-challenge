import datetime
import importlib
import time
from typing import Any

from hospital_assistant.safety.audit_log import ClinicalAuditLogger
from hospital_assistant.safety.guardrails import ClinicalGuardrails
from hospital_assistant.state import HospitalAssistantState

# ============================================================
# FALLBACK LLM
# ============================================================


class MockLLM:
    """
    LLM local de fallback.

    Permite executar o grafo durante o desenvolvimento
    sem depender do adapter do Marcelo.
    """

    def invoke(self, prompt: str):
        class MockResponse:
            content = (
                "Esta é uma resposta informativa de demonstração. "
                "Os sintomas relatados devem ser avaliados no contexto "
                "clínico individual. Recomenda-se avaliação presencial "
                "com profissional de saúde."
            )

        return MockResponse()


# ============================================================
# INTEGRAÇÃO DO MARCELO
# ============================================================


def obter_llm_clinica():
    """
    Fábrica da LLM clínica.

    Atualmente utiliza uma LLM genérica quando disponível.

    Futuramente, o adapter fine-tuned/LoRA do Marcelo poderá
    substituir esta implementação sem alterar o grafo.

    Interface esperada:

        llm.invoke(prompt)
    """

    try:
        modulo = importlib.import_module("langchain_openai")

        ChatOpenAI = getattr(
            modulo,
            "ChatOpenAI",
        )

        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
        )

    except (
        ImportError,
        AttributeError,
    ):
        return MockLLM()

    except Exception:
        return MockLLM()


# ============================================================
# INTEGRAÇÃO DO VINICIUS
# ============================================================


def obter_provedor_rag():
    """
    Fábrica do provedor RAG.

    Tenta utilizar a implementação real do Vinicius.

    Caso o módulo/função ainda não exista, retorna:

        None, False

    O restante do sistema utiliza o FallbackRAGProvider.

    Interface esperada do RAG real:

        buscar_protocolos_clinicos(
            pergunta,
            categoria=...
        )
    """

    try:
        modulo = importlib.import_module("hospital_assistant.rag")

        buscar_protocolos_clinicos = getattr(
            modulo,
            "buscar_protocolos_clinicos",
        )

        class GeizlerRAGProvider:
            """
            Adapter mínimo para o RAG do Vinicius.
            """

            def buscar(
                self,
                pergunta: str,
                categoria: str,
            ):

                return buscar_protocolos_clinicos(
                    pergunta,
                    categoria=categoria,
                )

        return (
            GeizlerRAGProvider(),
            True,
        )

    except (
        ImportError,
        AttributeError,
    ):
        return (
            None,
            False,
        )

    except Exception:
        return (
            None,
            False,
        )


# ============================================================
# FALLBACK RAG
# ============================================================


class FallbackRAGProvider:
    """
    Fallback interno.

    Não substitui o RAG do Vinicius.

    Serve apenas para permitir a execução do grafo enquanto
    a integração real não estiver disponível.
    """

    BASES = {
        "ginecologia": {
            "contexto": (
                "Informações clínicas gerais sobre saúde ginecológica, rastreamento, prevenção e acompanhamento."
            ),
            "fonte": ("Fallback interno - ginecologia"),
        },
        "obstetricia": {
            "contexto": (
                "Informações clínicas gerais sobre acompanhamento "
                "obstétrico, pré-natal e reconhecimento de sinais "
                "de alerta."
            ),
            "fonte": ("Fallback interno - obstetrícia"),
        },
        "geral": {
            "contexto": (
                "Informações gerais de saúde da mulher. A resposta deve ser informativa, cautelosa e não prescritiva."
            ),
            "fonte": ("Fallback interno - saúde da mulher"),
        },
    }

    def buscar(
        self,
        pergunta: str,
        categoria: str,
    ) -> list[dict[str, Any]]:

        base = self.BASES.get(
            categoria,
            self.BASES["geral"],
        )

        return [
            {
                "page_content": base["contexto"],
                "metadata": {
                    "fonte": base["fonte"],
                    "categoria": categoria,
                    "tipo": "fallback",
                },
            }
        ]


# ============================================================
# NÓS
# ============================================================


class ClinicalNodes:
    def __init__(self):

        self.guardrails = ClinicalGuardrails()

        # -----------------------------------------------------
        # Marcelo
        # -----------------------------------------------------

        self.llm = obter_llm_clinica()

        # -----------------------------------------------------
        # Vinicius
        # -----------------------------------------------------

        self.rag, self.rag_real_disponivel = obter_provedor_rag()

        if self.rag is None:
            self.rag = FallbackRAGProvider()

    # ========================================================
    # ROUTER
    # ========================================================

    def analisar_entrada_router(
        self,
        state: HospitalAssistantState,
    ) -> dict[str, Any]:

        pergunta = state.get(
            "pergunta",
            "",
        ).strip()

        passos = state.get(
            "passos_processamento",
            [],
        )[:]

        passos.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Router iniciado")

        # ----------------------------------------------------
        # Guardrails
        # ----------------------------------------------------

        valido, mensagem, sinais = self.guardrails.validar_input(state)

        pergunta_lower = pergunta.lower()

        # ----------------------------------------------------
        # Classificação
        # ----------------------------------------------------

        categoria = "geral"

        if "suspeita_violencia_domestica" in sinais:
            categoria = "violencia_domestica"

        elif any(
            palavra in pergunta_lower
            for palavra in [
                "grávida",
                "gravida",
                "gravidez",
                "parto",
                "gestante",
                "pré-natal",
                "pre-natal",
            ]
        ):
            categoria = "obstetricia"

        elif any(
            palavra in pergunta_lower
            for palavra in [
                "menstruação",
                "menstruacao",
                "cólica",
                "colica",
                "útero",
                "utero",
                "ginecologia",
                "preventivo",
                "corrimento",
                "ovário",
                "ovario",
            ]
        ):
            categoria = "ginecologia"

        # ----------------------------------------------------
        # Segurança
        # ----------------------------------------------------

        emergencia_detectada = "emergencia_clinica" in sinais

        violencia_detectada = "suspeita_violencia_domestica" in sinais

        if emergencia_detectada:
            bloqueado = True

            motivo_bloqueio = mensagem

        elif violencia_detectada:
            bloqueado = False

            motivo_bloqueio = None

            categoria = "violencia_domestica"

        else:
            bloqueado = not valido

            motivo_bloqueio = mensagem if not valido else None

        # ----------------------------------------------------
        # Tracing
        # ----------------------------------------------------

        passos.append(f"Direcionado para: {categoria.upper()}")

        if emergencia_detectada:
            passos.append("PRIORIDADE MÁXIMA: emergência detectada")

        if violencia_detectada:
            passos.append("Acolhimento especializado ativado")

        return {
            "categoria_triagem": categoria,
            "sinais_alarme_detectados": sinais,
            "bloqueado_por_seguranca": bloqueado,
            "motivo_bloqueio": motivo_bloqueio,
            "passos_processamento": passos,
        }

    # ========================================================
    # RAG
    # ========================================================

    def _buscar_contexto(
        self,
        pergunta: str,
        categoria: str,
    ) -> tuple[
        str,
        list[str],
        list[dict[str, Any]],
    ]:

        try:
            documentos = self.rag.buscar(
                pergunta,
                categoria,
            )

        except Exception:
            self.rag = FallbackRAGProvider()

            self.rag_real_disponivel = False

            documentos = self.rag.buscar(
                pergunta,
                categoria,
            )

        if documentos is None:
            documentos = []

        contexto_partes: list[str] = []

        fontes: list[str] = []

        documentos_serializados: list[dict[str, Any]] = []

        for documento in documentos:
            if isinstance(
                documento,
                dict,
            ):
                conteudo = documento.get(
                    "page_content",
                    "",
                )

                metadata = documento.get(
                    "metadata",
                    {},
                )

            else:
                conteudo = getattr(
                    documento,
                    "page_content",
                    str(documento),
                )

                metadata = getattr(
                    documento,
                    "metadata",
                    {},
                )

            if not isinstance(
                metadata,
                dict,
            ):
                metadata = {}

            if conteudo:
                contexto_partes.append(str(conteudo))

            fonte = metadata.get("pmid") or metadata.get("fonte") or metadata.get("source") or metadata.get("titulo")

            if fonte:
                fontes.append(str(fonte))

            documentos_serializados.append(
                {
                    "page_content": str(conteudo),
                    "metadata": metadata,
                }
            )

        if not contexto_partes:
            contexto_partes.append(
                "Nenhum documento clínico foi recuperado. "
                "A resposta deve permanecer cautelosa, "
                "informativa e não prescritiva."
            )

        if not fontes:
            fontes.append("Nenhuma fonte clínica recuperada")

        return (
            "\n\n".join(contexto_partes),
            fontes,
            documentos_serializados,
        )

    # ========================================================
    # CHAMADA LLM
    # ========================================================

    def _gerar_resposta(
        self,
        prompt: str,
        passos: list[str],
    ) -> str:

        try:
            resposta = self.llm.invoke(prompt)

            resposta_texto = (
                resposta.content
                if hasattr(
                    resposta,
                    "content",
                )
                else str(resposta)
            )

            return str(resposta_texto)

        except Exception:
            passos.append("LLM principal indisponível: fallback MockLLM utilizado")

            fallback = MockLLM()

            resposta = fallback.invoke(prompt)

            return str(resposta.content)

    # ========================================================
    # GINECOLOGIA
    # ========================================================

    def triar_ginecologia(
        self,
        state: HospitalAssistantState,
    ) -> dict[str, Any]:

        start_time = time.time()

        pergunta = state.get(
            "pergunta",
            "",
        )

        passos = state.get(
            "passos_processamento",
            [],
        )[:]

        passos.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Nó Ginecologia iniciado")

        (
            contexto,
            fontes,
            documentos,
        ) = self._buscar_contexto(
            pergunta,
            "ginecologia",
        )

        prompt = f"""
Você é um assistente informativo de saúde da mulher.

Use o contexto clínico recuperado abaixo como fonte
de informação.

REGRAS OBRIGATÓRIAS:
- Não invente fontes.
- Não faça diagnóstico definitivo.
- Não prescreva medicamentos.
- Não informe dosagens.
- Não substitua avaliação presencial.
- Se não houver informação suficiente, diga isso.
- Mantenha linguagem cautelosa.

CONTEXTO:
{contexto}

PERGUNTA:
{pergunta}

Forneça uma resposta clara, informativa e segura.
"""

        resposta_texto = self._gerar_resposta(
            prompt,
            passos,
        )

        latencia = (time.time() - start_time) * 1000

        origem_rag = "RAG real do Vinicius" if self.rag_real_disponivel else "Fallback interno de RAG"

        passos.append(f"{origem_rag} utilizado")

        passos.append(f"Processamento ginecológico concluído em {latencia:.1f}ms")

        return {
            "resposta_bruta": resposta_texto,
            "fontes_citadas": fontes,
            "documentos_retornados": documentos,
            "passos_processamento": passos,
        }

    # ========================================================
    # OBSTETRÍCIA
    # ========================================================

    def triar_obstetricia(
        self,
        state: HospitalAssistantState,
    ) -> dict[str, Any]:

        start_time = time.time()

        pergunta = state.get(
            "pergunta",
            "",
        )

        passos = state.get(
            "passos_processamento",
            [],
        )[:]

        passos.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Nó Obstetrícia iniciado")

        (
            contexto,
            fontes,
            documentos,
        ) = self._buscar_contexto(
            pergunta,
            "obstetricia",
        )

        prompt = f"""
Você é um assistente informativo de saúde da mulher.

Use o contexto clínico recuperado abaixo como fonte
de informação.

REGRAS OBRIGATÓRIAS:
- Não faça diagnóstico definitivo.
- Não prescreva medicamentos.
- Não informe dosagens.
- Não invente informações.
- Não substitua avaliação presencial.
- Se não houver informação suficiente, diga isso.

CONTEXTO:
{contexto}

PERGUNTA:
{pergunta}

Forneça uma resposta clara, informativa e segura.
"""

        resposta_texto = self._gerar_resposta(
            prompt,
            passos,
        )

        latencia = (time.time() - start_time) * 1000

        origem_rag = "RAG real do Vinicius" if self.rag_real_disponivel else "Fallback interno de RAG"

        passos.append(f"{origem_rag} utilizado")

        passos.append(f"Processamento obstétrico concluído em {latencia:.1f}ms")

        return {
            "resposta_bruta": resposta_texto,
            "fontes_citadas": fontes,
            "documentos_retornados": documentos,
            "passos_processamento": passos,
        }

    # ========================================================
    # VIOLÊNCIA
    # ========================================================

    def acolher_violencia(
        self,
        state: HospitalAssistantState,
    ) -> dict[str, Any]:

        passos = state.get(
            "passos_processamento",
            [],
        )[:]

        passos.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Nó Acolhimento Violência iniciado")

        resposta_acolhimento = (
            "Seu relato foi recebido com respeito e sem julgamento. "
            "Você não precisa enfrentar essa situação sozinha. "
            "Se houver risco imediato, procure um local seguro e "
            "acione o serviço de emergência da sua região. "
            "No Brasil, você também pode procurar apoio pela "
            "Central de Atendimento à Mulher, telefone 180, "
            "ou buscar atendimento em um serviço de saúde ou "
            "na rede de proteção. "
            "Se for seguro para você, podemos continuar conversando "
            "sobre formas de buscar ajuda."
        )

        return {
            "resposta_bruta": resposta_acolhimento,
            "fontes_citadas": ["Protocolo de Acolhimento a Mulheres em Situação de Violência - MS"],
            "passos_processamento": passos + ["Acolhimento seguro ativado"],
        }

    # ========================================================
    # SEGURANÇA
    # ========================================================

    def validar_seguranca(
        self,
        state: HospitalAssistantState,
    ) -> dict[str, Any]:

        passos = state.get(
            "passos_processamento",
            [],
        )[:]

        passos.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Validação de Saída iniciada")

        resposta_bruta = state.get(
            "resposta_bruta",
            "",
        )

        # -----------------------------------------------------
        # Bloqueio por emergência
        # -----------------------------------------------------

        if state.get(
            "bloqueado_por_seguranca",
            False,
        ):
            resposta_bruta = state.get(
                "motivo_bloqueio",
                ("Situação de possível urgência detectada. Procure atendimento presencial imediatamente."),
            )

        (
            resposta_higienizada,
            requer_validacao,
        ) = self.guardrails.validar_output(
            state,
            resposta_bruta,
        )

        return {
            "resposta_final": resposta_higienizada,
            "requer_validacao_humana": requer_validacao,
            "passos_processamento": passos + [(f"Validação concluída. Requer validação humana: {requer_validacao}")],
        }

    # ========================================================
    # REVISÃO HUMANA
    # ========================================================

    def revisao_humana(
        self,
        state: HospitalAssistantState,
    ) -> dict[str, Any]:
        """
        Mock de revisão humana.

        IMPORTANTE:
        Este nó NÃO deve ser chamado automaticamente pelo fluxo
        de produção antes da integração com a fila do Renato.

        Ele permanece disponível como ponto de integração/teste.
        """

        passos = state.get(
            "passos_processamento",
            [],
        )[:]

        passos.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Revisão Clínica Humana")

        resposta_atual = state.get(
            "resposta_final",
            "",
        )

        resposta_revisada = f"[✓ Resposta Chancelada por Médico Supervisor]\n{resposta_atual}"

        return {
            "validado_por_humano": True,
            "resposta_final": resposta_revisada,
            "passos_processamento": passos + ["Chancela médica simulada"],
        }

    # ========================================================
    # AUDITORIA
    # ========================================================

    def registrar_auditoria_final(
        self,
        state: HospitalAssistantState,
    ) -> dict[str, Any]:

        ClinicalAuditLogger.registrar_evento(state)

        return {}
