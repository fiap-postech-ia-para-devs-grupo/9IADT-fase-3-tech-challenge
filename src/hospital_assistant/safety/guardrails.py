import re

from hospital_assistant.state import HospitalAssistantState


class ClinicalGuardrails:
    """
    Guardrails clínicos de entrada e saída.

    Objetivos:
    - detectar sinais de possível emergência;
    - detectar violência doméstica;
    - impedir prescrição/dosagem autônoma;
    - sinalizar necessidade de validação humana;
    - reduzir linguagem diagnóstica definitiva;
    - manter a resposta dentro de caráter informativo.
    """

    # =========================================================
    # SINAIS DE POSSÍVEL EMERGÊNCIA
    # =========================================================

    TERMOS_EMERGENCIA = [
        "sangramento abundante",
        "dor insuportável",
        "desmaiou",
        "hemorragia",
        "contrações de 5 em 5 minutos",
        "perda de líquido amniótico",
        "perda de sangue gestante",
        "falta de ar severa",
        "convulsão",
        "febre muito alta",
    ]

    # =========================================================
    # MEDICAMENTOS / PRESCRIÇÃO
    # =========================================================

    TERMOS_MEDICAMENTOS = [
        "remedio",
        "remédio",
        "receita",
        "dosagem",
        "dose",
        "prescrever",
        "prescrição",
        "mg",
        "comprimido",
        "comprimir",
        "gotas",
        "anticoncepcional",
        "pílula",
        "pilula",
        "ibuprofeno",
        "paracetamol",
        "buscopan",
        "dipirona",
        "amoxicilina",
        "nistatina",
        "miconazol",
        "metronidazol",
    ]

    # =========================================================
    # DIAGNÓSTICO DEFINITIVO
    # =========================================================

    EXPRESSOES_DIAGNOSTICO_DEFINITIVO = [
        r"\bvocê tem\b",
        r"\bvoce tem\b",
        r"\bo seu diagnóstico é\b",
        r"\bo seu diagnostico e\b",
        r"\bcom certeza é\b",
        r"\bcom certeza e\b",
        r"\bestá com a doença\b",
        r"\besta com a doença\b",
        r"\bestá com\b",
        r"\besta com\b",
        r"\bisso é uma infecção\b",
        r"\bisso e uma infecção\b",
        r"\bdiagnóstico final:",
        r"\bdiagnostico final:",
    ]

    # =========================================================
    # PRESCRIÇÃO DIRETA
    # =========================================================

    EXPRESSOES_PRESCRICAO_DIRETA = [
        r"\btome o remédio\b",
        r"\btome o remedio\b",
        r"\brecomendo tomar\b",
        r"\bvocê deve usar o medicamento\b",
        r"\bvoce deve usar o medicamento\b",
        r"\breceito para você\b",
        r"\breceito para voce\b",
        r"\buse a dose de\b",
        r"\bcompre o remédio\b",
        r"\bcompre o remedio\b",
    ]

    # =========================================================
    # VIOLÊNCIA DOMÉSTICA
    # =========================================================

    TERMOS_VIOLENCIA = [
        "violência",
        "violencia",
        "agressão",
        "agressao",
        "agredida",
        "agredido",
        "me bateu",
        "me bate",
        "bater em mim",
        "me machucou",
        "abuso",
        "ameaça",
        "ameaca",
        "violência doméstica",
        "violencia domestica",
    ]

    def __init__(self):
        emergencia_pattern = "|".join(re.escape(termo) for termo in self.TERMOS_EMERGENCIA)

        medicamento_pattern = "|".join(re.escape(termo) for termo in self.TERMOS_MEDICAMENTOS)

        violencia_pattern = "|".join(re.escape(termo) for termo in self.TERMOS_VIOLENCIA)

        prescricao_pattern = "|".join(self.EXPRESSOES_PRESCRICAO_DIRETA)

        diagnostico_pattern = "|".join(self.EXPRESSOES_DIAGNOSTICO_DEFINITIVO)

        self.regex_emergencia = re.compile(
            rf"(?<!\w)({emergencia_pattern})(?!\w)",
            re.IGNORECASE,
        )

        self.regex_medicamentos = re.compile(
            rf"(?<!\w)({medicamento_pattern})(?!\w)",
            re.IGNORECASE,
        )

        self.regex_violencia = re.compile(
            rf"({violencia_pattern})",
            re.IGNORECASE,
        )

        self.regex_diag_def = re.compile(
            diagnostico_pattern,
            re.IGNORECASE,
        )

        self.regex_presc_dir = re.compile(
            prescricao_pattern,
            re.IGNORECASE,
        )

    # =========================================================
    # INPUT
    # =========================================================

    def validar_input(
        self,
        state: HospitalAssistantState,
    ) -> tuple[bool, str, list[str]]:
        """
        Valida a pergunta antes da chamada à LLM.
        """

        pergunta = state.get(
            "pergunta",
            "",
        ).strip()

        sinais_alarme: list[str] = []

        # -----------------------------------------------------
        # Pergunta vazia
        # -----------------------------------------------------

        if not pergunta:
            return (
                False,
                "Pergunta vazia.",
                [],
            )

        # -----------------------------------------------------
        # Emergência
        # -----------------------------------------------------

        matches_emergencia = self.regex_emergencia.findall(pergunta)

        if matches_emergencia:
            # Marcador interno utilizado pelo router.
            sinais_alarme.append("emergencia_clinica")

            sinais_alarme.extend(sorted(set(match.lower() for match in matches_emergencia)))

            alerta = (
                "SINAL DE ALARME DETECTADO: os sintomas relatados "
                "podem indicar uma situação de urgência clínica. "
                "Procure atendimento presencial imediatamente "
                "em um serviço de emergência ou pronto atendimento. "
                "Em caso de risco imediato à vida, acione o serviço "
                "de emergência da sua região."
            )

            # A detecção de emergência não significa que a entrada
            # seja "inválida"; ela é uma condição de segurança que
            # deve ser roteada pelo grafo.
            return (
                True,
                alerta,
                sinais_alarme,
            )

        # -----------------------------------------------------
        # Violência
        # -----------------------------------------------------

        match_violencia = self.regex_violencia.search(pergunta)

        if match_violencia:
            return (
                True,
                "ACOLHIMENTO_VIOLENCIA",
                ["suspeita_violencia_domestica"],
            )

        # -----------------------------------------------------
        # Entrada normal
        # -----------------------------------------------------

        return (
            True,
            "OK",
            [],
        )

    # =========================================================
    # OUTPUT
    # =========================================================

    def validar_output(
        self,
        state: HospitalAssistantState,
        resposta_bruta: str,
    ) -> tuple[str, bool]:
        """
        Aplica os guardrails na resposta gerada.

        Retorna:
            resposta_higienizada
            requer_validacao_humana
        """

        pergunta = state.get(
            "pergunta",
            "",
        ).lower()

        resposta_final = resposta_bruta or ""

        requer_validacao_humana = state.get(
            "requer_validacao_humana",
            False,
        )

        # -----------------------------------------------------
        # Medicamentos / prescrição
        # -----------------------------------------------------

        tem_prescricao_direta = bool(self.regex_presc_dir.search(resposta_final))

        menciona_medicamento = bool(self.regex_medicamentos.search(resposta_final))

        paciente_pediu_medicamento = bool(self.regex_medicamentos.search(pergunta))

        if tem_prescricao_direta or menciona_medicamento or paciente_pediu_medicamento:
            requer_validacao_humana = True

            aviso_prescricao = (
                "\n\nAVISO DE SEGURANÇA: este assistente não "
                "realiza prescrição ou definição autônoma de "
                "medicamentos e dosagens. A solicitação foi "
                "marcada para validação clínica humana."
            )

            if aviso_prescricao not in resposta_final:
                resposta_final += aviso_prescricao

        # -----------------------------------------------------
        # Diagnóstico definitivo
        # -----------------------------------------------------

        if self.regex_diag_def.search(resposta_final):
            resposta_final = self.regex_diag_def.sub(
                "os seus sintomas podem ser compatíveis com",
                resposta_final,
            )

        # -----------------------------------------------------
        # Garantia de linguagem informativa
        # -----------------------------------------------------

        termos_consulta = [
            "consulte seu médico",
            "consulte um médico",
            "procure atendimento",
            "agende uma consulta",
            "avaliação presencial",
            "avaliação médica",
            "atendimento presencial",
            "profissional de saúde",
        ]

        if not any(termo in resposta_final.lower() for termo in termos_consulta):
            resposta_final += (
                "\n\nEstas informações têm caráter "
                "informativo e de triagem. Elas não substituem "
                "uma avaliação presencial por um profissional "
                "de saúde."
            )

        return (
            resposta_final,
            requer_validacao_humana,
        )
