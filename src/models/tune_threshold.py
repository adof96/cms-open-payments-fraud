"""Ajuste del umbral de decisión para Random Forest y XGBoost sobre is_excluded.

train_supervised.py evalúa ambos modelos al threshold=0.5 (el default de
`.predict()`), pero con un desbalance ~1:100 en el set de entrenamiento ese valor no
tiene nada de especial. Este módulo NO reentrena nada ni toca los .pkl existentes: el
umbral de decisión es una transformación posterior a la probabilidad predicha, no algo
que se entrena.

No se guardó el split train/test como artefacto aparte, así que para "reusarlo" sin
reentrenar, este módulo reconstruye el mismo split llamando a las mismas funciones con
los mismos parámetros y `random_state` que `train_and_evaluate()` - `_load_stratified_sample`
y `train_test_split` son ambos determinísticos (mismo archivo, mismo random_state),
así que esto da EXACTAMENTE las mismas filas, no un split nuevo. Las probabilidades
salen de los modelos ya entrenados (cargados de disco) aplicados con
`apply_preprocessing`, que usa el preprocesamiento YA AJUSTADO guardado en
supervised_preprocessing.pkl (target encoding / bucketing / imputación) sin volver a
ajustar nada - así que X_test es idéntico al usado originalmente. `verify_reproduction()`
confirma esto comparando contra las métricas ya reportadas a threshold=0.5.

Reporta, para cada modelo:
- la tabla completa precision/recall/F1 a lo largo de una grilla de thresholds,
- el threshold que maximiza F1,
- el threshold más bajo que alcanza recall >= 0.80, con su precision en ese punto,
- el threshold más alto que alcanza precision >= 0.50 (si es alcanzable), con su
  recall en ese punto,
sin descartar ninguno de estos - la elección final queda para quien lea el reporte.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.config import DOCS_DIR, MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE, TARGET_COLUMN
from src.features.build_features import DEFAULT_OUTPUT_FILENAME as FRAUD_FEATURES_FILENAME
from src.models.model_io import load_model
from src.models.train_supervised import (
    NEGATIVE_SAMPLE_FRAC,
    TEST_SIZE,
    apply_preprocessing,
)
from src.models.train_supervised import _load_stratified_sample  # reutilizado, no reimplementado
from src.visualization.plots import plot_precision_recall_curves

# Grilla de thresholds para la tabla precision/recall/F1. 0.001 de paso (1001 puntos)
# es barato de calcular (son solo comparaciones sobre un array de probabilidades ya
# calculado, no reentrena nada) y ya de por sí "fino" en todo el rango - no hace falta
# una segunda pasada más fina alrededor del pico de F1.
THRESHOLD_STEP = 0.001
THRESHOLDS = np.round(np.arange(0.0, 1.0 + THRESHOLD_STEP, THRESHOLD_STEP), 3)

RECALL_FLOOR = 0.80
PRECISION_FLOOR = 0.50

PR_CURVE_PATH = DOCS_DIR / "threshold_tuning_pr_curve.png"


def rebuild_test_set(
    features_path: Path | None = None,
    negative_sample_frac: float = NEGATIVE_SAMPLE_FRAC,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Reconstruye determinísticamente el mismo split de train_and_evaluate() (misma
    muestra, mismo random_state) - no un split nuevo, el mismo recalculado, porque
    nunca se guardó como artefacto aparte.
    """
    features_path = features_path or (PROCESSED_DATA_DIR / FRAUD_FEATURES_FILENAME)
    sample = _load_stratified_sample(
        features_path, negative_sample_frac=negative_sample_frac, random_state=random_state
    )
    train_raw, test_raw = train_test_split(
        sample, test_size=test_size, stratify=sample[TARGET_COLUMN], random_state=random_state
    )
    return train_raw, test_raw


def threshold_metrics_table(y_true: pd.Series, y_proba: np.ndarray, thresholds=THRESHOLDS) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        rows.append(
            {
                "threshold": t,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def best_f1(table: pd.DataFrame) -> pd.Series:
    return table.loc[table["f1"].idxmax()]


def best_precision_at_recall_floor(table: pd.DataFrame, recall_floor: float) -> pd.Series | None:
    """Entre los thresholds con recall >= recall_floor, el de mejor precision (el
    threshold MÁS ALTO que aún cumple el piso de recall, ya que subir el threshold
    generalmente sube precision y baja recall)."""
    candidates = table[table["recall"] >= recall_floor]
    if candidates.empty:
        return None
    return candidates.loc[candidates["precision"].idxmax()]


def best_recall_at_precision_floor(table: pd.DataFrame, precision_floor: float) -> pd.Series | None:
    """Entre los thresholds con precision >= precision_floor, el de mejor recall."""
    candidates = table[table["precision"] >= precision_floor]
    if candidates.empty:
        return None
    return candidates.loc[candidates["recall"].idxmax()]


def verify_reproduction(y_true: pd.Series, y_proba: np.ndarray, expected_at_0_5: dict, model_name: str) -> None:
    """Chequeo de sanidad: al threshold=0.5 esto debe reproducir EXACTAMENTE las
    métricas ya reportadas por train_and_evaluate() - si no coincide, X_test no es el
    mismo que se usó originalmente y no hay que confiar en el resto del análisis.
    """
    y_pred = (y_proba >= 0.5).astype(int)
    actual = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    mismatches = {
        k: (actual[k], expected_at_0_5[k])
        for k in expected_at_0_5
        if abs(actual[k] - expected_at_0_5[k]) > 1e-6
    }
    if mismatches:
        raise RuntimeError(
            f"[{model_name}] La reconstrucción del test set NO reproduce las métricas ya "
            f"reportadas a threshold=0.5 - mismatches (actual, esperado): {mismatches}"
        )
    print(f"[{model_name}] OK: threshold=0.5 reproduce exactamente las métricas ya reportadas {actual}")


# Métricas a threshold=0.5 ya reportadas por train_and_evaluate() sobre el dataset real
# (ver reporte de esa corrida) - usadas solo para el chequeo de sanidad de arriba.
EXPECTED_AT_0_5 = {
    "random_forest": {"precision": 0.10587419857910241, "recall": 0.8018372703412073, "f1": 0.18705035971223022},
    "xgboost": {"precision": 0.1702548874041079, "recall": 0.9028871391076115, "f1": 0.2864876119092234},
}


def tune_thresholds(features_path: Path | None = None) -> dict:
    train_raw, test_raw = rebuild_test_set(features_path=features_path)
    artifacts = load_model(MODELS_DIR / "supervised_preprocessing.pkl")
    X_test = apply_preprocessing(test_raw, artifacts)
    y_test = test_raw[TARGET_COLUMN].reset_index(drop=True)

    models = {
        "random_forest": load_model(MODELS_DIR / "random_forest_is_excluded.pkl"),
        "xgboost": load_model(MODELS_DIR / "xgboost_is_excluded.pkl"),
    }

    summary: dict = {"n_test": int(len(y_test)), "n_positive_test": int(y_test.sum()), "models": {}}
    curves = {}

    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        verify_reproduction(y_test, y_proba, EXPECTED_AT_0_5[name], name)

        table = threshold_metrics_table(y_test, y_proba)
        f1_row = best_f1(table)
        recall_row = best_precision_at_recall_floor(table, RECALL_FLOOR)
        precision_row = best_recall_at_precision_floor(table, PRECISION_FLOOR)

        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
        curves[name] = (recall_curve, precision_curve)

        summary["models"][name] = {
            "best_f1": f1_row.to_dict(),
            f"best_precision_at_recall_{RECALL_FLOOR:.2f}": (
                recall_row.to_dict() if recall_row is not None else None
            ),
            f"best_recall_at_precision_{PRECISION_FLOOR:.2f}": (
                precision_row.to_dict() if precision_row is not None else None
            ),
            "table": table,
        }

    plot_precision_recall_curves(curves, PR_CURVE_PATH)
    summary["pr_curve_path"] = str(PR_CURVE_PATH)
    return summary


if __name__ == "__main__":
    result = tune_thresholds()
    print(f"\nn_test: {result['n_test']:,}  n_positive_test: {result['n_positive_test']:,}")
    for name, model_summary in result["models"].items():
        print(f"\n=== {name} ===")
        print(f"  best F1:                              {model_summary['best_f1']}")
        print(f"  best precision @ recall>={RECALL_FLOOR:.2f}:      {model_summary[f'best_precision_at_recall_{RECALL_FLOOR:.2f}']}")
        print(f"  best recall @ precision>={PRECISION_FLOOR:.2f}:     {model_summary[f'best_recall_at_precision_{PRECISION_FLOOR:.2f}']}")
    print(f"\nPR curve saved to: {result['pr_curve_path']}")
