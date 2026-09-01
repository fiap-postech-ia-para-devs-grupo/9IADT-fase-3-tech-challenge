"""Audit trail for every graph run, per ESTRATEGIA.md §6.

Two independent concerns share this module:

- `AuditRow` / `real_audit_rows` / `filter_audit_rows` / `apply_decision`: the
  read/write contract Tela 2 (Fila de Validação Humana) and Tela 3 (Auditoria)
  consume from `app.py`. `real_audit_rows` reads the rows from
  `ClinicalAuditLogger`'s real JSONL trail; decisions (Aprovar/Rejeitar/Editar)
  are held in `st.session_state` so Tela 2's decisions are visible on Tela 3
  within the same session — writing those decisions back into a persisted
  `auditoria` table instead is a separate ticket, so that half of the contract
  is kept stable here on purpose. `mock_audit_rows` remains as fixture data
  for this module's own unit tests.
- `ClinicalAuditLogger`: the real write path, called by the graph's
  `log_auditoria` node on every run. Persists JSONL independently of the
  contract above; `real_audit_rows` is the read side of that same file.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any, Literal, TypedDict

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


def _status_from_log_entry(entry: dict[str, Any]) -> str:
    """Map a JSONL entry's validation fields to Tela 2/3's `status` values.

    `validado_por_humano` has no writer on the real graph path — only Tela
    2's session-scoped decisions ever set "aprovado"/"rejeitado" — so this
    only distinguishes "pendente" (needs Tela 2's attention) from
    "nao_necessaria" (the graph never required a human to look at it).
    """
    if not entry.get("requer_validacao_humana"):
        return "nao_necessaria"
    if entry.get("validado_por_humano"):
        return "aprovado"
    return "pendente"


def real_audit_rows() -> list[AuditRow]:
    """Real audit history, read from `ClinicalAuditLogger`'s JSONL trail.

    Every logged run becomes a row — not just the ones requiring human
    validation — so Tela 3's "Auditoria e Histórico" reflects the full real
    trail; only rows with `status == "pendente"` enter Tela 2's queue.

    `id` is the row's 1-based position in the full JSONL file, so it stays
    stable across a read even though every row is included.
    """
    rows: list[AuditRow] = []
    for i, entry in enumerate(ClinicalAuditLogger.ler_logs_auditoria(), start=1):
        rows.append(
            {
                "id": i,
                "timestamp": entry.get("timestamp_utc", ""),
                "pergunta": entry.get("pergunta", ""),
                "paciente_id": entry.get("paciente_id"),
                "fontes_rag": entry.get("documentos_retornados", []),
                "resposta_llm": entry.get("resposta_final", ""),
                "flags_seguranca": entry.get("sinais_alarme", []),
                "status": _status_from_log_entry(entry),
                "aprovador": None,
                "timestamp_aprovacao": None,
            }
        )
    return rows


def apply_decision(
    rows: list[AuditRow],
    row_id: int,
    decisao: Literal["aprovado", "rejeitado"],
    aprovador: str | None = None,
    resposta_editada: str | None = None,
) -> list[AuditRow]:
    """Apply Tela 2's Aprovar/Rejeitar/Editar decision to one audit row.

    Returns a new list with the matching row's `status`, `aprovador` and
    `timestamp_aprovacao` updated; other rows are returned unchanged.
    `resposta_editada` covers the "Editar" action: it replaces `resposta_llm`
    before the decision (always "aprovado" for an edit) is applied — editing
    isn't a separate status, it's an edit-then-approve.
    """
    if decisao not in ("aprovado", "rejeitado"):
        raise ValueError(f"decisao inválida: {decisao!r}")

    timestamp_aprovacao = datetime.now(UTC).isoformat()

    updated: list[AuditRow] = []
    for row in rows:
        if row["id"] != row_id:
            updated.append(row)
            continue
        novo: AuditRow = dict(row)  # type: ignore[assignment]
        if resposta_editada is not None:
            novo["resposta_llm"] = resposta_editada
        novo["status"] = decisao
        novo["aprovador"] = aprovador
        novo["timestamp_aprovacao"] = timestamp_aprovacao
        updated.append(novo)
    return updated


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
