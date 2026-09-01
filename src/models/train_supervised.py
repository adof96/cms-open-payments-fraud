"""Entrenamiento supervisado (Random Forest y XGBoost) sobre TARGET_COLUMN (is_excluded).

Implementa las decisiones ya tomadas en la EDA (notebooks/eda_fraud_features.ipynb),
no las vuelve a derivar:

- **Features**: usa `payment_amount_log` (no `payment_amount`, crudo y de cola muy
  pesada - EDA sección 5); descarta `manufacturer_id` (redundante con
  `manufacturer_name` - EDA sección 10) y `Record_ID` (identificador, no feature - EDA
  sección 11); descarta `is_disputed`, `is_ownership_interest`, `is_charity` (lift 0.00x
  confirmado contra `is_excluded` - EDA, sección agregada tras la sección 7); conserva
  `is_related_product` y `is_third_party_payment` (lift 2.78x y 0.46x respectivamente).
- **`payment_form` / `payment_nature`**: categorías con menos de `MIN_CATEGORY_COUNT`
  filas en el split de entrenamiento se agrupan en "Other" antes de one-hot (EDA
  sección 8: varias categorías con <~1000 filas tenían 0% de exclusión no confiable).
- **`recipient_specialty`**: target encoding (tasa media de `is_excluded` por
  categoría, con suavizado bayesiano hacia la media global) en vez de one-hot o
  frequency encoding - alta cardinalidad y un spread real de ~16x en tasa de exclusión
  entre categorías que one-hot diluiría y frequency encoding ignoraría (EDA sección 9).
  Se ajusta ÚNICAMENTE sobre el fold de entrenamiento y se aplica al de test con esas
  estadísticas - nunca se ajusta sobre el dataset completo antes de separar train/test.
- **`manufacturer_name`**: el enunciado de esta tarea no especifica su tratamiento.
  Tiene la misma alta cardinalidad y el mismo problema que `recipient_specialty` (EDA
  sección 4 ya mostraba spread de tasa de exclusión entre fabricantes), así que por
  analogía se le aplica el mismo target encoding - **esto es una inferencia mía, no una
  decisión explícita de la EDA ni del usuario**, señalada aquí y en el reporte final.
- **`is_excluded_name_match` y `match_confidence`**: nunca se usan como feature de
  entrada (son etiquetas alternativas/exploratorias, no predictores) - solo
  `is_excluded_name_match` se usa después, como target secundario de evaluación.
- **Nulos estructurales**: `recipient_specialty` nulo (hospitales docentes sin NPI) se
  imputa como su propia categoría `"Missing"` antes del target encoding (así el nulo
  queda flageado implícitamente vía su propia tasa de exclusión estimada, sin
  descartar filas). `payment_frequency` nulo se imputa con la mediana del fold de
  entrenamiento y se agrega una columna `payment_frequency_missing` (0/1).

## Desbalance

`class_weight="balanced"` para Random Forest; `scale_pos_weight` para XGBoost
calculado como negativos/positivos **del set de entrenamiento realmente usado**
(no hardcodeado en 4070 - ver estrategia de memoria abajo, que sí usa una muestra).
No se usa SMOTE ni undersampling manual: el desbalance se corrige por peso, no por
resampleo.

## Estrategia de memoria

Esta máquina llegó a tener ~0.6GB de RAM libre durante este proyecto (ver EDA). Un
Random Forest y XGBoost necesitan la matriz de features completa en memoria (a
diferencia de las pasadas streaming de clean_data.py/build_features.py/la EDA, que
nunca necesitaron el dataset completo a la vez). Con 15,498,687 filas, incluso con
dtypes optimizados, la matriz de entrenamiento más el overhead interno de ambos
modelos supera con margen la RAM libre observada. Por eso este módulo entrena sobre
una **muestra estratificada, no el dataset completo**: TODAS las filas
`is_excluded=1` (son pocas, 3,809, no pesan nada) + una muestra aleatoria de filas
`is_excluded=0` a una fracción fija (`NEGATIVE_SAMPLE_FRAC`, ver constante abajo),
tomada con una sola pasada en chunks sobre fraud_features.csv (mismo patrón que la
EDA). Esto se declara explícitamente aquí y se reporta en el resumen final - no es un
submuestreo silencioso.

## Sin CV para hiperparámetros

Se usa un único split estratificado train/test (no k-fold) y valores de
hiperparámetros razonables sin grid search, para mantener el alcance manejable. Si en
el futuro se agrega tuning o CV sobre el target encoding, debe usarse
`StratifiedKFold` (no `KFold`), dado el desbalance severo de `is_excluded`.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE, TARGET_COLUMN
from src.features.build_features import DEFAULT_OUTPUT_FILENAME as FRAUD_FEATURES_FILENAME
from src.models.model_io import save_model

DEFAULT_CHUNKSIZE = 300_000
TEST_SIZE = 0.2

# Fracción de filas is_excluded=0 a retener en la muestra de entrenamiento (además de
# TODAS las is_excluded=1). 2.5% de ~15.49M negativos da ~387k negativos + 3,809
# positivos: suficiente diversidad para entrenar sin arriesgar la RAM libre de esta
# máquina (la matriz resultante, incluso sin optimizar dtypes, pesa unas pocas
# decenas de MB - muy por debajo de los ~0.6GB libres observados).
NEGATIVE_SAMPLE_FRAC = 0.025

# Categorías de payment_form / payment_nature con menos filas que esto (en el fold de
# entrenamiento) se agrupan en "Other" antes de one-hot. La EDA (sección 8) mostró que
# varias categorías con <1000 filas tenían 0% de exclusión, no confiable a ese tamaño
# de muestra: 1000 es un umbral simple y conservador acorde a esa observación.
MIN_CATEGORY_COUNT = 1000

# Fuerza del suavizado bayesiano del target encoding: equivale a "pseudo-conteos" de
# la media global. Con smoothing=50, una categoría con ~50 filas en train queda a
# medio camino entre su propia tasa observada y la media global; con miles de filas
# el suavizado casi no la mueve. Valor elegido para que categorías pequeñas (varias
# especialidades/fabricantes de cola larga tienen pocas decenas de filas) no obtengan
# una tasa 0% o 100% extrema por puro ruido de muestra chica.
TARGET_ENCODING_SMOOTHING = 50.0

BUCKET_COLUMNS = ["payment_form", "payment_nature"]
TARGET_ENCODED_COLUMNS = ["recipient_specialty", "manufacturer_name"]

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


def _load_stratified_sample(
    features_path: Path,
    negative_sample_frac: float = NEGATIVE_SAMPLE_FRAC,
    chunksize: int = DEFAULT_CHUNKSIZE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Lee fraud_features.csv en chunks (una sola pasada) y arma una muestra
    estratificada: todas las filas is_excluded=1 + una fracción aleatoria fija de las
    is_excluded=0. No carga el dataset completo en memoria en ningún momento.
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
    parts = []
    for chunk in reader:
        excl = chunk[TARGET_COLUMN] == 1
        parts.append(chunk[excl])
        parts.append(chunk[~excl].sample(frac=negative_sample_frac, random_state=random_state))
    sample = pd.concat(parts, ignore_index=True)
    return sample.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def _fit_rare_category_bucket(train_series: pd.Series, min_count: int) -> set:
    counts = train_series.value_counts()
    return set(counts[counts >= min_count].index)


def _apply_rare_category_bucket(series: pd.Series, keep: set) -> pd.Series:
    return series.where(series.isin(keep), other="Other")


def _fit_target_encoding(
    train_df: pd.DataFrame, column: str, target_col: str, smoothing: float
) -> tuple[dict, float]:
    """Ajusta el target encoding SOLO sobre train_df. Devuelve el mapping por
    categoría y la media global de train (usada como fallback para categorías nunca
    vistas al aplicar sobre test).
    """
    global_mean = float(train_df[target_col].mean())
    stats = train_df.groupby(column)[target_col].agg(["mean", "count"])
    smoothed = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
    return smoothed.to_dict(), global_mean


def _apply_target_encoding(series: pd.Series, mapping: dict, global_mean: float) -> pd.Series:
    return series.map(mapping).fillna(global_mean).astype("float32")


def _one_hot_bucketed(train_series: pd.Series, test_series: pd.Series, prefix: str, min_count: int):
    keep = _fit_rare_category_bucket(train_series, min_count)
    train_bucketed = _apply_rare_category_bucket(train_series, keep)
    test_bucketed = _apply_rare_category_bucket(test_series, keep)

    train_dummies = pd.get_dummies(train_bucketed, prefix=prefix, dtype="int8")
    test_dummies = pd.get_dummies(test_bucketed, prefix=prefix, dtype="int8")
    test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0)
    return train_dummies, test_dummies, sorted(keep)


def preprocess_features(train_raw: pd.DataFrame, test_raw: pd.DataFrame):
    """Ajusta todo el preprocesamiento (bucketing, target encoding, imputación) SOLO
    sobre train_raw y lo aplica a ambos splits. Devuelve (X_train, X_test, artifacts)
    donde artifacts guarda todo lo necesario para reproducir la transformación sobre
    datos nuevos (p.ej. en un futuro src/inference/predictor.py).
    """
    train_out = pd.DataFrame(index=train_raw.index)
    test_out = pd.DataFrame(index=test_raw.index)
    artifacts: dict = {"bucket_categories": {}, "target_encoding": {}}

    train_out["payment_amount_log"] = train_raw["payment_amount_log"].astype("float32")
    test_out["payment_amount_log"] = test_raw["payment_amount_log"].astype("float32")

    train_out["num_payments_included"] = train_raw["num_payments_included"].astype("float32")
    test_out["num_payments_included"] = test_raw["num_payments_included"].astype("float32")

    train_out["payment_month"] = train_raw["payment_month"].astype("float32")
    test_out["payment_month"] = test_raw["payment_month"].astype("float32")

    train_out["payment_day_of_week"] = train_raw["payment_day_of_week"].astype("float32")
    test_out["payment_day_of_week"] = test_raw["payment_day_of_week"].astype("float32")

    train_out["is_related_product"] = train_raw["is_related_product"].astype("int8")
    test_out["is_related_product"] = test_raw["is_related_product"].astype("int8")

    train_out["is_third_party_payment"] = train_raw["is_third_party_payment"].astype("int8")
    test_out["is_third_party_payment"] = test_raw["is_third_party_payment"].astype("int8")

    # payment_frequency: nulo estructural (hospitales docentes sin NPI) -> flag +
    # imputación con la mediana de TRAIN.
    freq_missing_train = train_raw["payment_frequency"].isna()
    freq_missing_test = test_raw["payment_frequency"].isna()
    freq_median = float(train_raw["payment_frequency"].median())
    train_out["payment_frequency_missing"] = freq_missing_train.astype("int8")
    test_out["payment_frequency_missing"] = freq_missing_test.astype("int8")
    train_out["payment_frequency"] = train_raw["payment_frequency"].fillna(freq_median).astype("float32")
    test_out["payment_frequency"] = test_raw["payment_frequency"].fillna(freq_median).astype("float32")
    artifacts["payment_frequency_median"] = freq_median

    # recipient_specialty / manufacturer_name: alta cardinalidad -> target encoding.
    # Nulo estructural de recipient_specialty se trata como su propia categoría
    # "Missing" antes de codificar (no se descartan filas, y su propia tasa de
    # exclusión estimada actúa como flag implícito).
    for col in TARGET_ENCODED_COLUMNS:
        train_filled = train_raw[col].fillna("Missing")
        test_filled = test_raw[col].fillna("Missing")
        mapping, global_mean = _fit_target_encoding(
            pd.DataFrame({col: train_filled, TARGET_COLUMN: train_raw[TARGET_COLUMN]}),
            col,
            TARGET_COLUMN,
            TARGET_ENCODING_SMOOTHING,
        )
        train_out[f"{col}_te"] = _apply_target_encoding(train_filled, mapping, global_mean)
        test_out[f"{col}_te"] = _apply_target_encoding(test_filled, mapping, global_mean)
        artifacts["target_encoding"][col] = {"mapping": mapping, "global_mean": global_mean}

    # payment_form / payment_nature: baja cardinalidad -> bucket de categorías raras
    # (fit en train) + one-hot, con las columnas de test alineadas a las de train.
    for col in BUCKET_COLUMNS:
        train_dummies, test_dummies, kept = _one_hot_bucketed(
            train_raw[col], test_raw[col], prefix=col, min_count=MIN_CATEGORY_COUNT
        )
        train_out = pd.concat([train_out, train_dummies], axis=1)
        test_out = pd.concat([test_out, test_dummies], axis=1)
        artifacts["bucket_categories"][col] = kept

    artifacts["feature_columns"] = list(train_out.columns)
    return train_out, test_out, artifacts


def _evaluate(model, X_test: pd.DataFrame, y_test: pd.Series, label: str) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "label": label,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_test, y_proba),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_test.nunique() > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_positive": int(y_test.sum()),
        "n_total": int(len(y_test)),
    }


def train_and_evaluate(
    features_path: Path | None = None,
    negative_sample_frac: float = NEGATIVE_SAMPLE_FRAC,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict:
    features_path = features_path or (PROCESSED_DATA_DIR / FRAUD_FEATURES_FILENAME)

    sample = _load_stratified_sample(features_path, negative_sample_frac=negative_sample_frac)

    train_raw, test_raw = train_test_split(
        sample, test_size=test_size, stratify=sample[TARGET_COLUMN], random_state=random_state
    )

    X_train, X_test, artifacts = preprocess_features(train_raw, test_raw)
    y_train = train_raw[TARGET_COLUMN].reset_index(drop=True)
    y_test = test_raw[TARGET_COLUMN].reset_index(drop=True)
    y_test_name_match = test_raw["is_excluded_name_match"].reset_index(drop=True)

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    neg_pos_ratio = n_neg / n_pos  # calculado del set de entrenamiento real, no hardcodeado

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=neg_pos_ratio,
        eval_metric="aucpr",
        random_state=random_state,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)

    results = {
        "random_forest": {
            "primary_is_excluded": _evaluate(rf, X_test, y_test, "is_excluded"),
            "secondary_is_excluded_name_match": _evaluate(
                rf, X_test, y_test_name_match, "is_excluded_name_match"
            ),
        },
        "xgboost": {
            "primary_is_excluded": _evaluate(xgb, X_test, y_test, "is_excluded"),
            "secondary_is_excluded_name_match": _evaluate(
                xgb, X_test, y_test_name_match, "is_excluded_name_match"
            ),
        },
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_model(rf, MODELS_DIR / "random_forest_is_excluded.pkl")
    save_model(xgb, MODELS_DIR / "xgboost_is_excluded.pkl")
    save_model(artifacts, MODELS_DIR / "supervised_preprocessing.pkl")

    return {
        "sample_rows": int(len(sample)),
        "sample_positives": int(sample[TARGET_COLUMN].sum()),
        "sample_negatives": int(len(sample) - sample[TARGET_COLUMN].sum()),
        "train_rows": int(len(train_raw)),
        "test_rows": int(len(test_raw)),
        "train_neg_pos_ratio": neg_pos_ratio,
        "feature_columns": artifacts["feature_columns"],
        "min_category_count": MIN_CATEGORY_COUNT,
        "target_encoding_smoothing": TARGET_ENCODING_SMOOTHING,
        "negative_sample_frac": negative_sample_frac,
        "results": results,
    }


if __name__ == "__main__":
    summary = train_and_evaluate()
    print(f"sample_rows: {summary['sample_rows']:,}")
    print(f"sample_positives: {summary['sample_positives']:,}")
    print(f"sample_negatives: {summary['sample_negatives']:,}")
    print(f"train_rows: {summary['train_rows']:,}  test_rows: {summary['test_rows']:,}")
    print(f"train_neg_pos_ratio: {summary['train_neg_pos_ratio']:.2f}")
    print(f"min_category_count: {summary['min_category_count']}")
    print(f"target_encoding_smoothing: {summary['target_encoding_smoothing']}")
    print(f"feature_columns ({len(summary['feature_columns'])}): {summary['feature_columns']}")
    for model_name, model_results in summary["results"].items():
        print(f"\n=== {model_name} ===")
        for eval_name, metrics in model_results.items():
            print(f"  -- {eval_name} --")
            for k, v in metrics.items():
                print(f"     {k}: {v}")
