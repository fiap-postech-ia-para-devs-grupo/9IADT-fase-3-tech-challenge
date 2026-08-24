from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma"

PATIENTS_DB = DATA_DIR / "patients_mock.db"
AUDIT_DB = DATA_DIR / "audit.db"

FINETUNING_METRICS = RESULTS_DIR / "finetuning_metrics.json"
EVAL_COMPARATIVO = RESULTS_DIR / "eval_comparativo.json"
