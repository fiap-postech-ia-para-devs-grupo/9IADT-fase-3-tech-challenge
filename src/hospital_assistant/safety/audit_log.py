import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("ClinicalAuditTrail")


class ClinicalAuditLogger:
    """
    Auditoria estruturada do fluxo clínico.

    Formato:
        JSONL

    Cada execução gera um registro independente.

    O registro permite ao futuro painel do Renato identificar:
    - respostas pendentes;
    - decisões de segurança;
    - sinais de alarme;
    - fontes utilizadas;
    - passos do grafo;
    - status da validação humana.
    """

    LOG_TEXTO_PATH = "clinical_audit.log"

    LOG_ESTRUTURADO_PATH = "clinical_audit.jsonl"

    @staticmethod
    def registrar_evento(
        state: dict[str, Any],
        latencia_ms: float = 0.0,
    ):
        """
        Persiste o estado relevante para auditoria.

        Observação:
        pergunta e resposta podem conter dados de saúde.
        Em produção, o armazenamento deve possuir controle
        de acesso, retenção e proteção adequados.
        """

        timestamp = datetime.now(UTC).isoformat()

        paciente_id = state.get(
            "paciente_id",
            "ANONIMO",
        )

        categoria = state.get(
            "categoria_triagem",
            "GERAL",
        ).upper()

        bloqueado = state.get(
            "bloqueado_por_seguranca",
            False,
        )

        requer_fila = state.get(
            "requer_validacao_humana",
            False,
        )

        validado = state.get(
            "validado_por_humano",
            False,
        )

        fontes = state.get(
            "fontes_citadas",
            [],
        )

        # =====================================================
        # REGISTRO ESTRUTURADO
        # =====================================================

        log_entry = {
            "timestamp_utc": timestamp,
            "paciente_id": paciente_id,
            "paciente_idade": state.get("paciente_idade"),
            "categoria_triagem": categoria,
            "pergunta": state.get(
                "pergunta",
                "",
            ),
            "resposta_final": state.get(
                "resposta_final",
                "",
            ),
            "sinais_alarme": state.get(
                "sinais_alarme_detectados",
                [],
            ),
            "bloqueado_por_seguranca": (bloqueado),
            "motivo_bloqueio": state.get("motivo_bloqueio"),
            "requer_validacao_humana": (requer_fila),
            "validado_por_humano": (validado),
            "status_validacao": ("VALIDADO" if validado else ("PENDENTE" if requer_fila else "NAO_NECESSARIA")),
            "fontes_citadas": fontes,
            "documentos_retornados": state.get(
                "documentos_retornados",
                [],
            ),
            "latencia_ms": latencia_ms,
            "passos_grafo": state.get(
                "passos_processamento",
                [],
            ),
        }

        try:
            with open(
                ClinicalAuditLogger.LOG_ESTRUTURADO_PATH,
                "a",
                encoding="utf-8",
            ) as arquivo:
                arquivo.write(
                    json.dumps(
                        log_entry,
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )

        except OSError as erro:
            logger.error(
                "Falha ao persistir log JSONL: %s",
                erro,
            )

        # =====================================================
        # LOG DE CONSOLE
        # =====================================================

        msg_base = f"ID: {paciente_id} | Categoria: {categoria} | Fontes: {len(fontes)} | Tempo: {latencia_ms:.1f}ms"

        if bloqueado:
            logger.warning(
                f"[ALERTA DE SEGURANÇA] {msg_base} | Status: BLOQUEADO | Motivo: {state.get('motivo_bloqueio')}"
            )

        elif requer_fila and not validado:
            logger.info(f"[FILA MÉDICA] {msg_base} | Status: PENDENTE")

        elif validado:
            logger.info(f"[FILA MÉDICA] {msg_base} | Status: VALIDADO")

        else:
            logger.info(f"[OK] {msg_base} | Status: AUTORIZADO")

    @staticmethod
    def ler_logs_auditoria() -> list[dict[str, Any]]:
        """
        Carrega todos os registros JSONL.
        """

        logs: list[dict[str, Any]] = []

        caminho = ClinicalAuditLogger.LOG_ESTRUTURADO_PATH

        if not os.path.exists(caminho):
            return logs

        try:
            with open(
                caminho,
                encoding="utf-8",
            ) as arquivo:
                for linha in arquivo:
                    linha = linha.strip()

                    if not linha:
                        continue

                    try:
                        logs.append(json.loads(linha))

                    except json.JSONDecodeError as erro:
                        logger.error(
                            "Linha JSONL inválida: %s",
                            erro,
                        )

        except OSError as erro:
            logger.error(
                "Erro ao ler auditoria: %s",
                erro,
            )

        return logs

    @staticmethod
    def ler_pendencias() -> list[dict[str, Any]]:
        """
        Retorna somente respostas que ainda precisam
        de validação humana.
        """

        logs = ClinicalAuditLogger.ler_logs_auditoria()

        return [
            log
            for log in logs
            if log.get(
                "requer_validacao_humana",
                False,
            )
            and not log.get(
                "validado_por_humano",
                False,
            )
        ]
