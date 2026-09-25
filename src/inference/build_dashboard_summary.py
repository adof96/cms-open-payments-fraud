"""Precalcula un resumen liviano para streamlit_app/app.py.

La app de Streamlit NUNCA debe cargar fraud_features.csv (3.39GB) ni
xgboost_predictions.csv (414MB) en vivo - este script lee ambos EN CHUNKS, una sola
vez, y agrega todo lo que el dashboard necesita a un JSON de unos pocos MB:

- Tasa de flagged_for_review y volumen por categoría, para recipient_specialty y
  manufacturer_name (top 20 por volumen) y payment_nature (todas, son pocas).
- Estadísticas generales: filas totales, flagged totales/tasa, is_excluded
  totales/tasa.
- Histograma precomputado (bins, no valores crudos) de payment_amount_log para filas
  flagged vs no-flagged.
- El barrido completo de threshold (precision/recall/F1) ya calculado por
  tune_threshold.py para ambos modelos - se reutiliza esa función tal cual (no se
  recalcula esa lógica acá), solo se exporta su tabla a este resumen.

## Por qué no hace falta un merge por Record_ID

fraud_features.csv y xgboost_predictions.csv se generaron ambos iterando sobre el
mismo CSV crudo en el mismo orden, sin muestrear ni reordenar filas en ningún momento
(ver build_features.py / predictor.py) - así que sus filas corresponden 1 a 1 por
posición. Este módulo lee ambos con el MISMO chunksize y verifica que los Record_ID de
cada par de chunks coincidan exactamente como chequeo de seguridad, en vez de hacer un
join por Record_ID que exigiría tener uno de los dos completo en memoria (esta máquina
llegó a tener ~0.6GB de RAM libre en este proyecto).
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR, TARGET_COLUMN
from src.features.build_features import DEFAULT_OUTPUT_FILENAME as FRAUD_FEATURES_FILENAME
from src.inference.predictor import DEFAULT_OUTPUT_FILENAME as PREDICTIONS_FILENAME
from src.models.tune_threshold import tune_thresholds

DEFAULT_CHUNKSIZE = 300_000
DEFAULT_OUTPUT_FILENAME = "dashboard_summary.json"

TOP_N_CATEGORIES = 20

# payment_amount_log observado en ~[0, 18.3] (ver EDA/numeric_summary de
# train_supervised.py) - 40 bins de ancho 0.5 dan resolución suficiente para un
# histograma de dashboard sin guardar valores crudos.
HISTOGRAM_BIN_EDGES = np.linspace(0, 20, 41)

CATEGORY_COLUMNS = ["recipient_specialty", "manufacturer_name", "payment_nature"]
FEATURES_USECOLS = ["Record_ID", "payment_amount_log", TARGET_COLUMN] + CATEGORY_COLUMNS
PREDICTIONS_USECOLS = ["Record_ID", "flagged_for_review"]


def _aggregate(features_path: Path, predictions_path: Path, chunksize: int) -> dict:
    category_stats = {col: defaultdict(lambda: {"count": 0, "flagged": 0}) for col in CATEGORY_COLUMNS}
    hist_flagged = np.zeros(len(HISTOGRAM_BIN_EDGES) - 1, dtype="int64")
    hist_not_flagged = np.zeros(len(HISTOGRAM_BIN_EDGES) - 1, dtype="int64")

    total_rows = 0
    total_flagged = 0
    total_excluded = 0

    features_reader = pd.read_csv(features_path, usecols=FEATURES_USECOLS, chunksize=chunksize, low_memory=False)
    predictions_reader = pd.read_csv(predictions_path, usecols=PREDICTIONS_USECOLS, chunksize=chunksize)

    for features_chunk, predictions_chunk in zip(features_reader, predictions_reader):
        if len(features_chunk) != len(predictions_chunk) or not np.array_equal(
            features_chunk["Record_ID"].to_numpy(), predictions_chunk["Record_ID"].to_numpy()
        ):
            raise RuntimeError(
                "fraud_features.csv y xgboost_predictions.csv no están alineados fila a "
                "fila en este chunk - no se puede combinar por posición sin un merge "
                "explícito por Record_ID."
            )

        flagged = predictions_chunk["flagged_for_review"].to_numpy()
        total_rows += len(features_chunk)
        total_flagged += int(flagged.sum())
        total_excluded += int(features_chunk[TARGET_COLUMN].sum())

        for col in CATEGORY_COLUMNS:
            values = features_chunk[col].fillna("Missing")
            grouped = pd.DataFrame({"cat": values, "flagged": flagged}).groupby("cat")["flagged"].agg(
                ["count", "sum"]
            )
            stats = category_stats[col]
            for cat, row in grouped.iterrows():
                entry = stats[cat]
                entry["count"] += int(row["count"])
                entry["flagged"] += int(row["sum"])

        log_amount = features_chunk["payment_amount_log"].to_numpy()
        hist_flagged += np.histogram(log_amount[flagged == 1], bins=HISTOGRAM_BIN_EDGES)[0]
        hist_not_flagged += np.histogram(log_amount[flagged == 0], bins=HISTOGRAM_BIN_EDGES)[0]

    return {
        "total_rows": total_rows,
        "total_flagged": total_flagged,
        "total_excluded": total_excluded,
        "category_stats": {col: dict(stats) for col, stats in category_stats.items()},
        "histogram_flagged": hist_flagged.tolist(),
        "histogram_not_flagged": hist_not_flagged.tolist(),
    }


def _top_n(stats: dict, n: int) -> dict:
    return dict(sorted(stats.items(), key=lambda kv: kv[1]["count"], reverse=True)[:n])


def build_summary(
    features_path: Path | None = None,
    predictions_path: Path | None = None,
    output_path: Path | None = None,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> dict:
    features_path = features_path or (PROCESSED_DATA_DIR / FRAUD_FEATURES_FILENAME)
    predictions_path = predictions_path or (PROCESSED_DATA_DIR / PREDICTIONS_FILENAME)
    output_path = output_path or (PROCESSED_DATA_DIR / DEFAULT_OUTPUT_FILENAME)

    agg = _aggregate(features_path, predictions_path, chunksize=chunksize)

    category_flagged_rate = {
        "recipient_specialty": _top_n(agg["category_stats"]["recipient_specialty"], TOP_N_CATEGORIES),
        "manufacturer_name": _top_n(agg["category_stats"]["manufacturer_name"], TOP_N_CATEGORIES),
        "payment_nature": agg["category_stats"]["payment_nature"],  # todas - son pocas categorías
    }

    # Reutiliza tune_threshold.py tal cual - no se recalcula esa lógica acá, solo se
    # exporta su tabla. Vuelve a leer fraud_features.csv una vez más (para reconstruir
    # el test set, como hace tune_threshold.py normalmente) y regenera de paso
    # docs/threshold_tuning_pr_curve.png (idempotente, mismo contenido).
    threshold_result = tune_thresholds(features_path=features_path)
    threshold_sweep = {
        name: model_summary["table"][["threshold", "precision", "recall", "f1"]].to_dict(orient="list")
        for name, model_summary in threshold_result["models"].items()
    }

    summary = {
        "overall": {
            "total_rows": agg["total_rows"],
            "total_flagged": agg["total_flagged"],
            "flagged_rate": agg["total_flagged"] / agg["total_rows"],
            "total_excluded": agg["total_excluded"],
            "excluded_rate": agg["total_excluded"] / agg["total_rows"],
        },
        "category_flagged_rate": category_flagged_rate,
        "amount_log_histogram": {
            "bin_edges": HISTOGRAM_BIN_EDGES.tolist(),
            "flagged": agg["histogram_flagged"],
            "not_flagged": agg["histogram_not_flagged"],
        },
        "threshold_sweep": threshold_sweep,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f)

    return {
        "output_path": str(output_path),
        "output_size_mb": output_path.stat().st_size / (1024 ** 2),
        **summary["overall"],
    }


if __name__ == "__main__":
    result = build_summary()
    for key, value in result.items():
        print(f"{key}: {value}")
