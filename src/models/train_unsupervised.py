"""Detección de anomalías no supervisada (Isolation Forest) sobre Open Payments.

`is_excluded` **no** entra a la matriz de entrenamiento en ningún momento - Isolation
Forest se ajusta sin ver la etiqueta. `is_excluded` (y, exploratoriamente,
`is_excluded_name_match`) se cargan aparte y se usan únicamente después, para evaluar
si los scores de anomalía separan a los proveedores realmente excluidos.

## Diferencias respecto al set de features de train_supervised.py

- **`recipient_specialty` / `manufacturer_name`: frequency encoding (proporción de
  filas con esa categoría), NO target encoding.** El target encoding de
  train_supervised.py calcula la tasa media de `is_excluded` por categoría - usar eso
  acá sería filtrar la etiqueta hacia un método que se supone no supervisado.
  Frequency encoding no toca `is_excluded` en absoluto: solo cuenta filas.
- Todo lo demás es igual a train_supervised.py (mismas decisiones de la EDA, ninguna
  involucra la etiqueta): `payment_amount_log` en vez de `payment_amount` crudo;
  se descartan `manufacturer_id` (redundante) y `Record_ID` (identificador); se
  descartan `is_disputed`/`is_ownership_interest`/`is_charity` (lift 0.00x, sin
  señal); se conservan `is_related_product`/`is_third_party_payment`; categorías
  raras de `payment_form`/`payment_nature` se agrupan en "Other" con el mismo
  `MIN_CATEGORY_COUNT` ya establecido (importado de train_supervised.py, no
  reharcodeado). `is_excluded_name_match` y `match_confidence` tampoco entran como
  feature - solo se cargan para la comparación exploratoria de la evaluación.

## Muestreo

Muestra aleatoria SIMPLE (no oversample de positivos como train_supervised.py):
Isolation Forest necesita ver la distribución natural de clases para que sus scores
de anomalía signifiquen algo - oversamplear positivos sesgaría qué pinta "normal".
`SAMPLE_FRAC` se aplica por igual a todas las filas (sin importar `is_excluded`), leído
en una sola pasada por chunks (mismo patrón que clean_data.py/build_features.py/la
EDA), con `random_state` fijo. Apunta a un volumen similar a los ~391k de
train_supervised.py.

## Recursos

Esta máquina corrió caliente en entrenamientos previos, así que acá los límites son
explícitos y fijos, no se suben sin confirmar antes: `n_estimators=100` (default de
sklearn) y `n_jobs=2` (no -1, para no acaparar todos los cores). El script imprime
tiempo transcurrido y memoria estimada en cada paso, para detectar un job descontrolado
temprano.

## Evaluación

Un score de anomalía continuo no tiene un corte natural tipo 0.5, así que acá NO se
reporta precision/recall/F1 (eso es de train_supervised.py). En cambio: score y rango
percentil promedio de `is_excluded=1` vs `is_excluded=0`, y qué porcentaje de los
`is_excluded=1` reales cae dentro del top 1%/5%/10% de scores más anómalos - "si
investigáramos solo el N% más anómalo, ¿cuántas exclusiones reales atraparíamos?". La
misma comparación se repite, exploratoriamente, contra `is_excluded_name_match`.
"""

import ctypes
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE, TARGET_COLUMN
from src.features.build_features import DEFAULT_OUTPUT_FILENAME as FRAUD_FEATURES_FILENAME
from src.models.model_io import save_model
from src.models.train_supervised import (
    MIN_CATEGORY_COUNT,
    _apply_rare_category_bucket,
    _fit_rare_category_bucket,
)

DEFAULT_CHUNKSIZE = 300_000

# Fracción de TODAS las filas a muestrear (sin importar is_excluded) - simple, no
# estratificada. 2.5% de ~15.49M filas da ~387k, un volumen similar a los 391,176 de
# train_supervised.py sin necesitar conocer de antemano el total exacto de filas.
SAMPLE_FRAC = 0.025

# Fijos por instrucción explícita del usuario - no aumentar sin confirmar primero.
N_ESTIMATORS = 100
N_JOBS = 2

BUCKET_COLUMNS = ["payment_form", "payment_nature"]
FREQUENCY_ENCODED_COLUMNS = ["recipient_specialty", "manufacturer_name"]

USECOLS = [
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
    "is_excluded_name_match",
]


def _free_ram_mb():
    """Best-effort de RAM libre - mismo enfoque que notebooks/eda_fraud_features.ipynb
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


def _estimated_memory_mb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / (1024 ** 2)


def _load_random_sample(
    features_path: Path,
    sample_frac: float = SAMPLE_FRAC,
    chunksize: int = DEFAULT_CHUNKSIZE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Lee fraud_features.csv en chunks (una sola pasada) y arma una muestra aleatoria
    SIMPLE (misma fracción para is_excluded=0 y 1, sin oversample). No carga el
    dataset completo en memoria en ningún momento.
    """
    dtype = {
        "payment_amount_log": "float32",
        "num_payments_included": "float32",
        "payment_month": "float32",
        "payment_day_of_week": "float32",
        "payment_frequency": "float32",
        "is_related_product": "int8",
        "is_third_party_payment": "int8",
        TARGET_COLUMN: "int8",
        "is_excluded_name_match": "int8",
    }
    reader = pd.read_csv(
        features_path, usecols=USECOLS, dtype=dtype, chunksize=chunksize, low_memory=False
    )
    parts = [chunk.sample(frac=sample_frac, random_state=random_state) for chunk in reader]
    sample = pd.concat(parts, ignore_index=True)
    return sample.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def preprocess_features(df: pd.DataFrame):
    """Arma la matriz de features SIN usar is_excluded en ningún paso (a diferencia de
    train_supervised.py, que sí lo usa para el target encoding). Devuelve (X,
    artifacts) donde artifacts guarda todo lo necesario para reproducir la
    transformación sobre datos nuevos.
    """
    out = pd.DataFrame(index=df.index)
    artifacts: dict = {"bucket_categories": {}, "frequency_encoding": {}}

    out["payment_amount_log"] = df["payment_amount_log"].astype("float32")
    out["num_payments_included"] = df["num_payments_included"].astype("float32")
    out["payment_month"] = df["payment_month"].astype("float32")
    out["payment_day_of_week"] = df["payment_day_of_week"].astype("float32")
    out["is_related_product"] = df["is_related_product"].astype("int8")
    out["is_third_party_payment"] = df["is_third_party_payment"].astype("int8")

    # payment_frequency: mismo tratamiento que train_supervised.py (nulo estructural
    # de hospitales docentes sin NPI -> flag + imputación con la mediana), pero
    # calculado acá sobre esta muestra (no hay train/test split en este script).
    freq_missing = df["payment_frequency"].isna()
    freq_median = float(df["payment_frequency"].median())
    out["payment_frequency_missing"] = freq_missing.astype("int8")
    out["payment_frequency"] = df["payment_frequency"].fillna(freq_median).astype("float32")
    artifacts["payment_frequency_median"] = freq_median

    # recipient_specialty / manufacturer_name: frequency encoding, NO target encoding
    # (ver docstring del módulo). El nulo estructural de recipient_specialty se trata
    # como su propia categoría "Missing" antes de codificar, no se descartan filas.
    for col in FREQUENCY_ENCODED_COLUMNS:
        filled = df[col].fillna("Missing")
        mapping = filled.value_counts(normalize=True).to_dict()
        out[f"{col}_freq"] = filled.map(mapping).astype("float32")
        artifacts["frequency_encoding"][col] = mapping

    # payment_form / payment_nature: mismo bucket de categorías raras que
    # train_supervised.py (mismo MIN_CATEGORY_COUNT, importado no reharcodeado) + one-hot.
    for col in BUCKET_COLUMNS:
        keep = _fit_rare_category_bucket(df[col], MIN_CATEGORY_COUNT)
        bucketed = _apply_rare_category_bucket(df[col], keep)
        dummies = pd.get_dummies(bucketed, prefix=col, dtype="int8")
        out = pd.concat([out, dummies], axis=1)
        artifacts["bucket_categories"][col] = sorted(keep)

    artifacts["feature_columns"] = list(out.columns)
    return out, artifacts


def _evaluate_anomaly_ranking(anomaly_score: np.ndarray, y: pd.Series) -> dict:
    """Compara los scores de anomalía entre y=1 e y=0: promedio de score y de rango
    percentil, más qué fracción de los y=1 reales cae en el top 1%/5%/10% de scores
    más anómalos. No hay threshold de clasificación - el score es continuo.
    """
    mask_pos = (y == 1).to_numpy()
    pct_rank = pd.Series(anomaly_score).rank(pct=True).to_numpy() * 100  # 0-100, mayor = más anómalo

    n_pos = int(mask_pos.sum())
    result = {
        "n_positive": n_pos,
        "n_total": int(len(anomaly_score)),
        "mean_anomaly_score_positive": float(anomaly_score[mask_pos].mean()) if n_pos else float("nan"),
        "mean_anomaly_score_negative": float(anomaly_score[~mask_pos].mean()) if (~mask_pos).any() else float("nan"),
        "mean_percentile_rank_positive": float(pct_rank[mask_pos].mean()) if n_pos else float("nan"),
        "mean_percentile_rank_negative": float(pct_rank[~mask_pos].mean()) if (~mask_pos).any() else float("nan"),
    }
    for top_pct in (1, 5, 10):
        threshold = np.quantile(anomaly_score, 1 - top_pct / 100)
        in_top = anomaly_score >= threshold
        n_captured = int((in_top & mask_pos).sum())
        result[f"top_{top_pct}pct_capture_rate"] = (n_captured / n_pos) if n_pos else float("nan")
        result[f"top_{top_pct}pct_n_captured"] = n_captured
    return result


def train_and_evaluate(
    features_path: Path | None = None,
    sample_frac: float = SAMPLE_FRAC,
    random_state: int = RANDOM_STATE,
) -> dict:
    start = time.time()
    features_path = features_path or (PROCESSED_DATA_DIR / FRAUD_FEATURES_FILENAME)

    _log(f"starting - reading random sample (frac={sample_frac}) from {features_path.name}", start)
    sample = _load_random_sample(features_path, sample_frac=sample_frac, random_state=random_state)
    _log(f"sample loaded: {len(sample):,} rows, ~{_estimated_memory_mb(sample):.1f} MB", start)

    X, artifacts = preprocess_features(sample)
    _log(f"features built: {X.shape[1]} columns, ~{_estimated_memory_mb(X):.1f} MB", start)

    model = IsolationForest(n_estimators=N_ESTIMATORS, n_jobs=N_JOBS, random_state=random_state)
    _log(f"fitting IsolationForest (n_estimators={N_ESTIMATORS}, n_jobs={N_JOBS}) ...", start)
    model.fit(X)
    _log("fit done", start)

    # score_samples: mayor = más normal. Se invierte el signo para que mayor = más
    # anómalo, más intuitivo para el top-N% de la evaluación.
    anomaly_score = -model.score_samples(X)
    _log("scoring done", start)

    y_excluded = sample[TARGET_COLUMN].reset_index(drop=True)
    y_name_match = sample["is_excluded_name_match"].reset_index(drop=True)

    results = {
        "is_excluded": _evaluate_anomaly_ranking(anomaly_score, y_excluded),
        "is_excluded_name_match": _evaluate_anomaly_ranking(anomaly_score, y_name_match),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_model(model, MODELS_DIR / "isolation_forest.pkl")
    save_model(artifacts, MODELS_DIR / "unsupervised_preprocessing.pkl")
    _log("models saved", start)

    return {
        "sample_rows": int(len(sample)),
        "sample_positives_is_excluded": int(y_excluded.sum()),
        "sample_positives_name_match": int(y_name_match.sum()),
        "feature_columns": artifacts["feature_columns"],
        "min_category_count": MIN_CATEGORY_COUNT,
        "n_estimators": N_ESTIMATORS,
        "n_jobs": N_JOBS,
        "runtime_seconds": time.time() - start,
        "results": results,
    }


if __name__ == "__main__":
    summary = train_and_evaluate()
    print(f"\nsample_rows: {summary['sample_rows']:,}")
    print(f"sample_positives_is_excluded: {summary['sample_positives_is_excluded']:,}")
    print(f"sample_positives_name_match: {summary['sample_positives_name_match']:,}")
    print(f"runtime_seconds: {summary['runtime_seconds']:.1f}")
    print(f"min_category_count: {summary['min_category_count']}")
    print(f"n_estimators: {summary['n_estimators']}  n_jobs: {summary['n_jobs']}")
    print(f"feature_columns ({len(summary['feature_columns'])}): {summary['feature_columns']}")
    for eval_name, metrics in summary["results"].items():
        print(f"\n=== {eval_name} ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
