"""Audit trail for every graph run, per ESTRATEGIA.md §6.

Two independent concerns share this module:

- `AuditRow` / `mock_audit_rows` / `filter_audit_rows`: the read contract Tela 2
  (Fila de Validação Humana) and Tela 3 (Auditoria) already consume from
  `app.py`. Still backed by mock data — wiring Telas 2/3 to real persisted
  rows instead of `mock_audit_rows()` is a separate ticket, so this contract
  is kept stable here on purpose.
- `ClinicalAuditLogger`: the real write path, called by the graph's
  `log_auditoria` node on every run. Persists JSONL independently of the
  contract above.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any, TypedDict

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("ClinicalAuditTrail")


class AuditRow(TypedDict):
    id: int
    timestamp: str
    pergunta: str
    paciente_id: str | None
    fontes_rag: list[dict]
    resposta_llm: str
    flags_seguranca: list[str]
    status: str
    aprovador: str | None
    timestamp_aprovacao: str | None


def mock_audit_rows() -> list[AuditRow]:
    """Fake audit history, standing in for the real persisted audit trail.

    Spans multiple patients, dates and statuses so Tela 3's status/paciente/data
    filters (Bloco 1, Pessoa D) have something to actually filter.
    """
    return [
        {
            "id": 1,
            "timestamp": "2026-08-20T09:00:00",
            "pergunta": "[MOCK] Qual o protocolo para dor torácica aguda?",
            "paciente_id": "1",
            "fontes_rag": [{"source": "mock_doc_1.md", "score": 0.83}],
            "resposta_llm": "[MOCK LLM] Sugestão de exemplo.",
            "flags_seguranca": [],
            "status": "pendente",
            "aprovador": None,
            "timestamp_aprovacao": None,
        },
        {
            "id": 2,
            "timestamp": "2026-08-21T10:30:00",
            "pergunta": "[MOCK] Paciente com histórico de hipertensão, ajustar dosagem?",
            "paciente_id": "2",
            "fontes_rag": [{"source": "mock_doc_2.md", "score": 0.77}],
            "resposta_llm": "[MOCK LLM] Sugestão de exemplo.",
            "flags_seguranca": [],
            "status": "aprovado",
            "aprovador": "Dr. Souza",
            "timestamp_aprovacao": "2026-08-21T11:00:00",
        },
        {
            "id": 3,
            "timestamp": "2026-08-21T14:15:00",
            "pergunta": "[MOCK] Interação medicamentosa suspeita?",
            "paciente_id": "1",
            "fontes_rag": [{"source": "mock_doc_3.md", "score": 0.65}],
            "resposta_llm": "[MOCK LLM] Sugestão de exemplo.",
            "flags_seguranca": ["interacao_medicamentosa"],
            "status": "rejeitado",
            "aprovador": "Dra. Lima",
            "timestamp_aprovacao": "2026-08-21T14:45:00",
        },
        {
            "id": 4,
            "timestamp": "2026-08-22T08:00:00",
            "pergunta": "[MOCK] Protocolo para febre pós-operatória?",
            "paciente_id": "3",
            "fontes_rag": [{"source": "mock_doc_4.md", "score": 0.71}],
            "resposta_llm": "[MOCK LLM] Sugestão de exemplo.",
            "flags_seguranca": [],
            "status": "pendente",
            "aprovador": None,
            "timestamp_aprovacao": None,
        },
        {
            "id": 5,
            "timestamp": "2026-08-23T16:20:00",
            "pergunta": "[MOCK] Reavaliação de exames pendentes?",
            "paciente_id": "2",
            "fontes_rag": [{"source": "mock_doc_5.md", "score": 0.88}],
            "resposta_llm": "[MOCK LLM] Sugestão de exemplo.",
            "flags_seguranca": [],
            "status": "aprovado",
            "aprovador": "Dr. Souza",
            "timestamp_aprovacao": "2026-08-23T16:50:00",
        },
    ]


def filter_audit_rows(
    rows: list[AuditRow],
    status: str = "todos",
    paciente_id: str = "todos",
    data: str = "todas",
) -> list[AuditRow]:
    """Apply Tela 3's status/paciente/data filters to a list of audit rows.

    `data` matches against the date portion (YYYY-MM-DD) of `timestamp`.
    """
    if status != "todos":
        rows = [r for r in rows if r["status"] == status]
    if paciente_id != "todos":
        rows = [r for r in rows if r["paciente_id"] == paciente_id]
    if data != "todas":
        rows = [r for r in rows if r["timestamp"].startswith(data)]
    return rows


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
