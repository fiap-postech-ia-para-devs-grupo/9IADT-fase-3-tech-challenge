"""Tela 3 · Auditoria e Histórico, per issue #17 (Bloco 4): histórico real,
não só a fila de validação.

`real_audit_rows()` inclui toda linha do `clinical_audit.jsonl`, não só as
que passaram por validação humana — do contrário Tela 3 ("Histórico") nunca
mostraria as execuções que o grafo liberou sozinho, mesmo com o filtro de
Status oferecendo "aprovado"/"rejeitado"/"todos" sugerindo um histórico
completo. Roda app.py via streamlit.testing.v1.AppTest.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from hospital_assistant.paths import PROJECT_ROOT
from hospital_assistant.safety.audit_log import ClinicalAuditLogger

_APP_PATH = str(PROJECT_ROOT / "app.py")


def _registrar(pergunta: str, paciente_id: str | None, requer_validacao_humana: bool) -> None:
    ClinicalAuditLogger.registrar_evento(
        {
            "paciente_id": paciente_id,
            "pergunta": pergunta,
            "resposta_final": "Resposta de teste.",
            "requer_validacao_humana": requer_validacao_humana,
            "validado_por_humano": False,
            "fontes_citadas": ["protocolo_teste.md"],
        }
    )


def _tela_3(at: AppTest) -> AppTest:
    at.sidebar.radio[0].set_value("Tela 3 · Auditoria").run()
    return at


def test_historico_inclui_execucoes_que_nao_precisaram_de_validacao(limpar_auditoria):
    _registrar("Qual remédio devo prescrever para a dor?", "1", requer_validacao_humana=True)
    _registrar("Quais os sinais de alerta na sepse?", "2", requer_validacao_humana=False)

    at = _tela_3(AppTest.from_file(_APP_PATH).run())

    linhas = at.dataframe[0].value
    assert len(linhas) == 2
    assert set(linhas["status"]) == {"pendente", "nao_necessaria"}


def test_filtro_de_status_nao_necessaria(limpar_auditoria):
    _registrar("Quais os sinais de alerta na sepse?", "2", requer_validacao_humana=False)

    at = _tela_3(AppTest.from_file(_APP_PATH).run())
    at.selectbox[0].set_value("nao_necessaria").run()

    linhas = at.dataframe[0].value
    assert len(linhas) == 1
    assert linhas.iloc[0]["status"] == "nao_necessaria"
