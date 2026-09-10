"""Inferencia con XGBoost sobre el dataset completo de Open Payments.

Carga `models/xgboost_is_excluded.pkl` y `models/supervised_preprocessing.pkl` (ya
entrenados/ajustados por train_supervised.py) y no reentrena ni reajusta nada: aplica
el preprocesamiento vía `apply_preprocessing` (train_supervised.py), que ya está hecho
para transformar un DataFrame crudo usando artefactos guardados sin volver a ajustar
target encoding / bucketing / imputación - se reutiliza tal cual, no se duplica esa
lógica acá.

Lee data/processed/fraud_features.csv (15,498,687 filas, ~3.39GB) en chunks - mismo
patrón memory-conscious que clean_data.py/build_features.py/train_supervised.py, dado
que esta máquina llegó a tener ~0.6GB de RAM libre en este proyecto.

Solo XGBoost (no Random Forest): ganó en todas las métricas de ranking en la
comparación de train_supervised.py. Umbral fijo `THRESHOLD = 0.724`, elegido en
tune_threshold.py (recall=0.803, precision=0.312 para XGBoost sobre el test set) -
acá se aplica tal cual, no se vuelve a buscar.

Output liviano (Record_ID, predicted_probability, flagged_for_review), no el feature
set completo - mismo principio que fraud_labels.csv, mismas razones de RAM/disco.
"""

import ctypes
import sys
import time
from pathlib import Path

import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, TARGET_COLUMN
from src.features.build_features import DEFAULT_OUTPUT_FILENAME as FRAUD_FEATURES_FILENAME
from src.models.model_io import load_model
from src.models.train_supervised import apply_preprocessing

DEFAULT_CHUNKSIZE = 300_000
DEFAULT_OUTPUT_FILENAME = "xgboost_predictions.csv"

# Elegido en tune_threshold.py (recall=0.803, precision=0.312 para XGBoost) - se aplica
# tal cual acá, no se vuelve a ajustar.
THRESHOLD = 0.724

# Columnas crudas necesarias de fraud_features.csv: las que pide apply_preprocessing
# (ver train_supervised.py) + Record_ID (clave del output) + TARGET_COLUMN (solo para
# el chequeo de sanidad del recall al final - NUNCA se le pasa al modelo).
PREDICT_USECOLS = [
    "Record_ID",
    "payment_amount_log",
    "num_payments_included",
    "payment_month",
    "payment_day_of_week",
    "payment_frequency",
    "is_related_product",
    "is_third_party_payment",
    "recipient_specialty",
    "manufacturer_name",
    "payment_form",
    "payment_nature",
    TARGET_COLUMN,
]

PREDICT_DTYPES = {
    "Record_ID": "int64",
    "payment_amount_log": "float32",
    "num_payments_included": "float32",
    "payment_month": "float32",
    "payment_day_of_week": "float32",
    "payment_frequency": "float32",
    "is_related_product": "int8",
    "is_third_party_payment": "int8",
    TARGET_COLUMN: "int8",
}


def _free_ram_mb():
    """Best-effort de RAM libre - mismo enfoque que train_unsupervised.py/la EDA
    (ctypes, solo Windows; devuelve None en otros sistemas)."""
    if sys.platform != "win32":
        return None

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return stat.ullAvailPhys / (1024 ** 2)


def _log(msg: str, start_time: float) -> None:
    elapsed = time.time() - start_time
    free_ram = _free_ram_mb()
    ram_note = f", free RAM ~{free_ram:,.0f} MB" if free_ram is not None else ""
    print(f"[{elapsed:7.1f}s{ram_note}] {msg}")


def run_inference(
    features_path: Path | None = None,
    output_path: Path | None = None,
    chunksize: int = DEFAULT_CHUNKSIZE,
    threshold: float = THRESHOLD,
) -> dict:
    """Puntúa fraud_features.csv completo con el XGBoost ya entrenado, chunk por
    chunk, y escribe (Record_ID, predicted_probability, flagged_for_review) a
    data/processed/. Devuelve un resumen con conteos y el chequeo de sanidad del recall.
    """
    start = time.time()
    features_path = features_path or (PROCESSED_DATA_DIR / FRAUD_FEATURES_FILENAME)
    output_path = output_path or (PROCESSED_DATA_DIR / DEFAULT_OUTPUT_FILENAME)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _log("loading model and preprocessing artifacts", start)
    model = load_model(MODELS_DIR / "xgboost_is_excluded.pkl")
    artifacts = load_model(MODELS_DIR / "supervised_preprocessing.pkl")

    total_rows = 0
    total_flagged = 0
    total_true_positive = 0
    total_true_positive_flagged = 0
    first_chunk = True

    _log(f"scoring {features_path.name} in chunks of {chunksize:,} rows (threshold={threshold}) ...", start)
    reader = pd.read_csv(
        features_path, usecols=PREDICT_USECOLS, dtype=PREDICT_DTYPES, chunksize=chunksize, low_memory=False
    )
    for i, chunk in enumerate(reader):
        X = apply_preprocessing(chunk, artifacts)
        proba = model.predict_proba(X)[:, 1]
        flagged = proba >= threshold

        out_chunk = pd.DataFrame(
            {
                "Record_ID": chunk["Record_ID"].to_numpy(),
                "predicted_probability": proba.astype("float32"),
                "flagged_for_review": flagged.astype("int8"),
            }
        )
        out_chunk.to_csv(output_path, mode="w" if first_chunk else "a", header=first_chunk, index=False)
        first_chunk = False

        true_positive_mask = (chunk[TARGET_COLUMN] == 1).to_numpy()
        total_rows += len(chunk)
        total_flagged += int(flagged.sum())
        total_true_positive += int(true_positive_mask.sum())
        total_true_positive_flagged += int((flagged & true_positive_mask).sum())

        if (i + 1) % 10 == 0:
            _log(f"...processed {total_rows:,} rows, {total_flagged:,} flagged so far", start)

    _log(f"done: {total_rows:,} rows scored", start)

    recall_sanity_check = (
        total_true_positive_flagged / total_true_positive if total_true_positive else float("nan")
    )

    return {
        "rows_scored": total_rows,
        "rows_flagged": total_flagged,
        "flagged_rate": total_flagged / total_rows if total_rows else float("nan"),
        "true_positives_total": total_true_positive,
        "true_positives_flagged": total_true_positive_flagged,
        "recall_sanity_check": recall_sanity_check,
        "threshold": threshold,
        "output_path": str(output_path),
        "runtime_seconds": time.time() - start,
    }


if __name__ == "__main__":
    summary = run_inference()
    print(f"\nrows_scored: {summary['rows_scored']:,}")
    print(f"rows_flagged: {summary['rows_flagged']:,} ({summary['flagged_rate'] * 100:.3f}% of scored rows)")
    print(f"true_positives_total (is_excluded=1): {summary['true_positives_total']:,}")
    print(f"true_positives_flagged: {summary['true_positives_flagged']:,}")
    print(f"recall_sanity_check: {summary['recall_sanity_check']:.4f}  (tuned for ~0.80 on the test split)")
    print(f"threshold: {summary['threshold']}")
    print(f"runtime_seconds: {summary['runtime_seconds']:.1f}")
    print(f"output_path: {summary['output_path']}")
