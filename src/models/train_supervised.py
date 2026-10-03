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
  Además, dentro del propio fold de entrenamiento el valor codificado de cada fila se
  calcula **out-of-fold** (`TARGET_ENCODING_N_FOLDS` folds de `StratifiedKFold`): el
  mapping que codifica una fila se ajusta sobre los OTROS folds, así la etiqueta de una
  fila nunca contribuye a su propio feature. Sin esto (versión anterior), una categoría
  chica con 1 positivo recibía un encoding inflado por la etiqueta de esa misma fila, y
  el modelo podía aprender ese atajo en train (fuga severa en `manufacturer_name`,
  ver EDA sección 9). El mapping que se guarda y se aplica a test / inferencia sigue
  ajustado sobre TODO el fold de entrenamiento - el out-of-fold solo cambia cómo se
  generan los valores de las filas de entrenamiento.
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

## Split a nivel proveedor (NPI)

`is_excluded` es un dato por PROVEEDOR (el cruce con la LEIE es por NPI en
clean_data.py), pero los 3,809 pagos excluidos vienen de solo 381 proveedores. Con un
split por fila, ~90% de los pagos excluidos del test pertenecían a un proveedor que
también tenía pagos en train: el modelo se evaluaba en parte en reconocer proveedores ya
vistos, no en generalizar a proveedores nuevos. Por eso `train_and_evaluate` separa
PROVEEDORES (estratificando por si el proveedor tiene algún pago excluido) y manda cada
fila al lado de su proveedor, y los folds del target encoding out-of-fold usan
`StratifiedGroupKFold` por proveedor. El NPI no está en fraud_features.csv: se recupera
por Record_ID desde fraud_labels.csv (`attach_provider_ids`), sin regenerar el archivo
de 3.39GB.

## Sin CV para hiperparámetros

Se usa un único split estratificado train/test (no k-fold) y valores de
hiperparámetros razonables sin grid search, para mantener el alcance manejable. El único
k-fold del módulo es el del target encoding out-of-fold (arriba), y usa
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
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE, TARGET_COLUMN
from src.data.clean_data import DEFAULT_OUTPUT_FILENAME as FRAUD_LABELS_FILENAME
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

# Folds del target encoding out-of-fold. 5 es el default estándar: cada mapping se
# ajusta sobre el 80% del fold de entrenamiento (casi tan estable como el mapping
# completo), y con ~3,047 positivos de train quedan ~610 positivos por fold, suficiente
# para que StratifiedKFold reparta positivos parejo. Más folds acercarían los mappings
# out-of-fold al completo a cambio de más cómputo, sin cambiar el punto (evitar que la
# etiqueta de una fila entre en su propio feature).
TARGET_ENCODING_N_FOLDS = 5

PROVIDER_COLUMN = "Covered_Recipient_NPI"

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
    usecols: list[str] = USECOLS,
) -> pd.DataFrame:
    """Lee fraud_features.csv en chunks (una sola pasada) y arma una muestra
    estratificada: todas las filas is_excluded=1 + una fracción aleatoria fija de las
    is_excluded=0. No carga el dataset completo en memoria en ningún momento.

    `usecols` permite sumar columnas (p.ej. Record_ID para recuperar el NPI) sin cambiar
    QUÉ filas se muestrean: el muestreo depende solo del orden/largo de cada chunk, no de
    sus columnas, así que la muestra es la misma fila por fila con o sin columnas extra.
    """
    dtype = {
        "Record_ID": "int64",
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
    dtype = {col: t for col, t in dtype.items() if col in usecols}
    reader = pd.read_csv(
        features_path, usecols=usecols, dtype=dtype, chunksize=chunksize, low_memory=False
    )
    parts = []
    for chunk in reader:
        excl = chunk[TARGET_COLUMN] == 1
        parts.append(chunk[excl])
        parts.append(chunk[~excl].sample(frac=negative_sample_frac, random_state=random_state))
    sample = pd.concat(parts, ignore_index=True)
    return sample.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def attach_provider_ids(
    sample: pd.DataFrame, labels_path: Path | None = None, chunksize: int = DEFAULT_CHUNKSIZE
) -> pd.Series:
    """Devuelve el NPI (Covered_Recipient_NPI) de cada fila de `sample`, buscándolo por
    Record_ID en fraud_labels.csv.

    fraud_features.csv no trae el NPI, y regenerarlo (3.39GB) solo para sumar una columna
    no vale la pena: fraud_labels.csv sí lo trae, así que se lee en chunks quedándose solo
    con las filas cuyos Record_ID están en la muestra (~391k de 15.5M) - nunca se carga el
    archivo completo. Falla si algún Record_ID de la muestra no aparece (NPI nulo sí es
    válido: hospitales docentes, ver `provider_groups`).
    """
    labels_path = labels_path or (PROCESSED_DATA_DIR / FRAUD_LABELS_FILENAME)
    wanted = pd.Index(sample["Record_ID"].unique())
    parts = []
    reader = pd.read_csv(
        labels_path,
        usecols=["Record_ID", PROVIDER_COLUMN],
        dtype={"Record_ID": "int64", PROVIDER_COLUMN: "Int64"},
        chunksize=chunksize,
    )
    for chunk in reader:
        parts.append(chunk[chunk["Record_ID"].isin(wanted)])
    lookup = pd.concat(parts).set_index("Record_ID")[PROVIDER_COLUMN]

    if lookup.index.has_duplicates:
        raise RuntimeError("fraud_labels.csv tiene Record_ID duplicados - el lookup de NPI sería ambiguo.")
    missing = wanted.difference(lookup.index)
    if len(missing):
        raise RuntimeError(f"{len(missing):,} Record_ID de la muestra no están en {labels_path.name}.")
    return sample["Record_ID"].map(lookup).astype("Int64").rename(PROVIDER_COLUMN)


def provider_groups(npi: pd.Series) -> pd.Series:
    """Id de grupo por proveedor para los splits. Las filas sin NPI (hospitales docentes,
    ~0.3%) reciben cada una su propio grupo negativo: no son un proveedor individual, y
    como su is_excluded es siempre 0 (el cruce estricto exige NPI válido) no pueden filtrar
    una etiqueta positiva entre lados. Agruparlas todas en un único grupo, en cambio,
    mandaría todos los hospitales docentes a un solo lado del split.
    """
    groups = npi.astype("float64")
    missing = groups.isna().to_numpy()
    groups[missing] = -1 - np.arange(missing.sum())
    return groups.astype("int64").rename("provider_group")


def provider_level_split(
    sample: pd.DataFrame, groups: pd.Series, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa PROVEEDORES (no filas) en train/test, estratificando por si el proveedor
    tiene algún pago excluido, y manda cada fila al lado de su proveedor. Ningún
    proveedor queda en ambos lados. `test_size` es la fracción de proveedores, no de filas.
    """
    provider_label = sample[TARGET_COLUMN].groupby(groups).max()
    _, test_providers = train_test_split(
        provider_label.index, test_size=test_size, stratify=provider_label, random_state=random_state
    )
    in_test = groups.isin(test_providers).to_numpy()
    return sample[~in_test], sample[in_test]


def load_provider_split(
    features_path: Path | None = None,
    labels_path: Path | None = None,
    negative_sample_frac: float = NEGATIVE_SAMPLE_FRAC,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Misma muestra estratificada que siempre (mismas filas), con el NPI recuperado y
    separada a nivel proveedor. Devuelve (train_raw, test_raw, train_groups, test_groups).
    """
    features_path = features_path or (PROCESSED_DATA_DIR / FRAUD_FEATURES_FILENAME)
    sample = _load_stratified_sample(
        features_path,
        negative_sample_frac=negative_sample_frac,
        random_state=random_state,
        usecols=USECOLS + ["Record_ID"],
    )
    groups = provider_groups(attach_provider_ids(sample, labels_path))
    train_raw, test_raw = provider_level_split(sample, groups, test_size=test_size, random_state=random_state)
    return train_raw, test_raw, groups.loc[train_raw.index], groups.loc[test_raw.index]


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


def _out_of_fold_target_encoding(
    categories: pd.Series,
    target: pd.Series,
    smoothing: float = TARGET_ENCODING_SMOOTHING,
    n_splits: int = TARGET_ENCODING_N_FOLDS,
    random_state: int = RANDOM_STATE,
    groups: pd.Series | None = None,
) -> pd.Series:
    """Codifica las filas de ENTRENAMIENTO sin que la etiqueta de una fila entre en su
    propio valor: para cada fold, ajusta el mapping sobre los otros folds y lo aplica a las
    filas de este fold. Una categoría que solo aparece en el fold propio cae a la media
    global de los otros folds.

    Con `groups` (ids de proveedor) usa StratifiedGroupKFold: un proveedor nunca está a la
    vez del lado que ajusta y del que se codifica, así que tampoco filtran las etiquetas de
    OTRAS filas del mismo proveedor (is_excluded es un dato por proveedor). Sin `groups`
    usa StratifiedKFold por fila, como el candidato `_oof` anterior.
    """
    encoded = np.empty(len(categories), dtype="float32")
    if groups is None:
        folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(categories, target)
    else:
        folds = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(
            categories, target, groups
        )
    for fit_idx, apply_idx in folds:
        fit_df = pd.DataFrame({"cat": categories.iloc[fit_idx].to_numpy(), "y": target.iloc[fit_idx].to_numpy()})
        mapping, global_mean = _fit_target_encoding(fit_df, "cat", "y", smoothing)
        encoded[apply_idx] = _apply_target_encoding(categories.iloc[apply_idx], mapping, global_mean).to_numpy()
    return pd.Series(encoded, index=categories.index)


def _one_hot_bucketed(train_series: pd.Series, test_series: pd.Series, prefix: str, min_count: int):
    keep = _fit_rare_category_bucket(train_series, min_count)
    train_bucketed = _apply_rare_category_bucket(train_series, keep)
    test_bucketed = _apply_rare_category_bucket(test_series, keep)

    train_dummies = pd.get_dummies(train_bucketed, prefix=prefix, dtype="int8")
    test_dummies = pd.get_dummies(test_bucketed, prefix=prefix, dtype="int8")
    test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0)
    return train_dummies, test_dummies, sorted(keep)


def preprocess_features(
    train_raw: pd.DataFrame, test_raw: pd.DataFrame, train_groups: pd.Series | None = None
):
    """Ajusta todo el preprocesamiento (bucketing, target encoding, imputación) SOLO
    sobre train_raw y lo aplica a ambos splits. Devuelve (X_train, X_test, artifacts)
    donde artifacts guarda todo lo necesario para reproducir la transformación sobre
    datos nuevos (p.ej. en un futuro src/inference/predictor.py).

    `train_groups` (ids de proveedor de las filas de train) activa el target encoding
    out-of-fold agrupado por proveedor; sin él, los folds son por fila.
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
        # Mapping sobre TODO el fold de entrenamiento: es el que se aplica a test y el que
        # se guarda para inferencia (apply_preprocessing).
        mapping, global_mean = _fit_target_encoding(
            pd.DataFrame({col: train_filled, TARGET_COLUMN: train_raw[TARGET_COLUMN]}),
            col,
            TARGET_COLUMN,
            TARGET_ENCODING_SMOOTHING,
        )
        # Las filas de entrenamiento en sí se codifican out-of-fold (ver docstring del
        # módulo) - aplicarles el mapping completo filtraría su propia etiqueta.
        train_out[f"{col}_te"] = _out_of_fold_target_encoding(
            train_filled, train_raw[TARGET_COLUMN], groups=train_groups
        )
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


def apply_preprocessing(raw_df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Complemento de solo-transform de preprocess_features: aplica un preprocesamiento
    YA AJUSTADO (p.ej. cargado desde models/supervised_preprocessing.pkl) a un
    DataFrame crudo, sin volver a ajustar nada. Para reutilizar un modelo ya entrenado
    sobre datos nuevos, o para reconstruir X_test sin reentrenar (ver
    src/models/tune_threshold.py).
    """
    out = pd.DataFrame(index=raw_df.index)

    out["payment_amount_log"] = raw_df["payment_amount_log"].astype("float32")
    out["num_payments_included"] = raw_df["num_payments_included"].astype("float32")
    out["payment_month"] = raw_df["payment_month"].astype("float32")
    out["payment_day_of_week"] = raw_df["payment_day_of_week"].astype("float32")
    out["is_related_product"] = raw_df["is_related_product"].astype("int8")
    out["is_third_party_payment"] = raw_df["is_third_party_payment"].astype("int8")

    freq_median = artifacts["payment_frequency_median"]
    out["payment_frequency_missing"] = raw_df["payment_frequency"].isna().astype("int8")
    out["payment_frequency"] = raw_df["payment_frequency"].fillna(freq_median).astype("float32")

    for col in TARGET_ENCODED_COLUMNS:
        mapping = artifacts["target_encoding"][col]["mapping"]
        global_mean = artifacts["target_encoding"][col]["global_mean"]
        filled = raw_df[col].fillna("Missing")
        out[f"{col}_te"] = _apply_target_encoding(filled, mapping, global_mean)

    for col in BUCKET_COLUMNS:
        keep = set(artifacts["bucket_categories"][col])
        bucketed = _apply_rare_category_bucket(raw_df[col], keep)
        dummies = pd.get_dummies(bucketed, prefix=col, dtype="int8")
        expected_cols = [c for c in artifacts["feature_columns"] if c.startswith(f"{col}_")]
        dummies = dummies.reindex(columns=expected_cols, fill_value=0)
        out = pd.concat([out, dummies], axis=1)

    return out[artifacts["feature_columns"]]


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


def build_xgboost(neg_pos_ratio: float, random_state: int = RANDOM_STATE) -> XGBClassifier:
    """XGBoost con los hiperparámetros de este módulo - compartido con
    src/models/ablation_target_encoding.py para que las variantes del ablation difieran
    solo en las features, no en el modelo."""
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=neg_pos_ratio,
        eval_metric="aucpr",
        random_state=random_state,
        n_jobs=-1,
    )


def feature_gain_shares(model: XGBClassifier) -> pd.Series:
    """Participación de cada feature en el total_gain de un XGBoost, de mayor a menor."""
    gains = pd.Series(model.get_booster().get_score(importance_type="total_gain"))
    return (gains / gains.sum()).sort_values(ascending=False)


def _split_stats(raw: pd.DataFrame, groups: pd.Series) -> dict:
    positive_providers = raw[TARGET_COLUMN].groupby(groups).max()
    return {
        "rows": int(len(raw)),
        "providers": int(groups[groups >= 0].nunique()),
        "rows_without_npi": int((groups < 0).sum()),
        "excluded_providers": int(positive_providers[positive_providers.index >= 0].sum()),
        "excluded_rows": int(raw[TARGET_COLUMN].sum()),
        "row_positive_rate": float(raw[TARGET_COLUMN].mean()),
    }


DEFAULT_CANDIDATE_SUFFIX = "_candidate"
PRODUCTION_WARNING = (
    "ATENCIÓN: --promote sobrescribe los artefactos de PRODUCCIÓN en models/ "
    "(xgboost_is_excluded.pkl, random_forest_is_excluded.pkl, supervised_preprocessing.pkl), "
    "que usan predictor.py, tune_threshold.py y la app de Streamlit. tune_threshold.py tiene "
    "hardcodeadas las métricas del modelo actual y el umbral de predictor.py se ajustó para él: "
    "ambos quedan desactualizados hasta regenerarlos."
)


def resolve_artifact_suffix(artifact_suffix: str, promote: bool) -> str:
    """Decide el sufijo de los artefactos a escribir, con la regla de seguridad del CLI:
    los nombres de producción (sufijo "") solo se escriben con un sufijo explícitamente
    vacío Y `promote=True`. Sin promote, un sufijo vacío cae a DEFAULT_CANDIDATE_SUFFIX; un
    sufijo con nombre (p.ej. "_provider_split") se respeta como candidato. Promote con un
    sufijo no vacío es una contradicción y se rechaza.
    """
    if promote:
        if artifact_suffix != "":
            raise ValueError(
                f"promote=True exige un sufijo vacío (nombres de producción), no {artifact_suffix!r}."
            )
        return ""
    return artifact_suffix or DEFAULT_CANDIDATE_SUFFIX


def train_and_evaluate(
    features_path: Path | None = None,
    negative_sample_frac: float = NEGATIVE_SAMPLE_FRAC,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    artifact_suffix: str = DEFAULT_CANDIDATE_SUFFIX,
    include_random_forest: bool = True,
    promote: bool = False,
) -> dict:
    """Entrena y evalúa XGBoost (y RF si `include_random_forest`) y guarda los artefactos
    en models/.

    El split train/test y los folds del target encoding son a nivel PROVEEDOR (NPI), no
    por fila: is_excluded es un dato por proveedor, y con un split por fila ~90% de los
    pagos excluidos del test pertenecían a un proveedor que también estaba en train - el
    modelo se evaluaba en parte en reconocer proveedores ya vistos. `test_size` es la
    fracción de proveedores; las filas siguen a su proveedor.

    `artifact_suffix` (por defecto "_candidate") guarda un candidato sin pisar los
    artefactos en producción: predictor.py, tune_threshold.py y la app de Streamlit leen
    los nombres sin sufijo, así que un candidato no les cambia nada. Escribir los nombres
    de producción exige `artifact_suffix=""` Y `promote=True` (ver resolve_artifact_suffix);
    `artifact_suffix=""` sin promote falla antes de cargar ningún dato.
    """
    if artifact_suffix == "" and not promote:
        raise ValueError('artifact_suffix="" escribe los nombres de PRODUCCIÓN: requiere promote=True.')
    artifact_suffix = resolve_artifact_suffix(artifact_suffix, promote)
    if promote:
        print(PRODUCTION_WARNING)

    train_raw, test_raw, train_groups, test_groups = load_provider_split(
        features_path,
        negative_sample_frac=negative_sample_frac,
        test_size=test_size,
        random_state=random_state,
    )
    shared_providers = set(train_groups[train_groups >= 0]) & set(test_groups[test_groups >= 0])
    if shared_providers:
        raise RuntimeError(f"{len(shared_providers)} proveedores quedaron en train y test a la vez.")

    X_train, X_test, artifacts = preprocess_features(train_raw, test_raw, train_groups=train_groups)
    y_train = train_raw[TARGET_COLUMN].reset_index(drop=True)
    y_test = test_raw[TARGET_COLUMN].reset_index(drop=True)
    y_test_name_match = test_raw["is_excluded_name_match"].reset_index(drop=True)

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    neg_pos_ratio = n_neg / n_pos  # calculado del set de entrenamiento real, no hardcodeado

    xgb = build_xgboost(neg_pos_ratio, random_state=random_state)
    xgb.fit(X_train, y_train)
    results = {
        "xgboost": {
            "primary_is_excluded": _evaluate(xgb, X_test, y_test, "is_excluded"),
            "secondary_is_excluded_name_match": _evaluate(
                xgb, X_test, y_test_name_match, "is_excluded_name_match"
            ),
        },
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if include_random_forest:
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        results["random_forest"] = {
            "primary_is_excluded": _evaluate(rf, X_test, y_test, "is_excluded"),
            "secondary_is_excluded_name_match": _evaluate(
                rf, X_test, y_test_name_match, "is_excluded_name_match"
            ),
        }
        save_model(rf, MODELS_DIR / f"random_forest_is_excluded{artifact_suffix}.pkl")
    save_model(xgb, MODELS_DIR / f"xgboost_is_excluded{artifact_suffix}.pkl")
    save_model(artifacts, MODELS_DIR / f"supervised_preprocessing{artifact_suffix}.pkl")

    return {
        "split": {"train": _split_stats(train_raw, train_groups), "test": _split_stats(test_raw, test_groups)},
        "train_neg_pos_ratio": neg_pos_ratio,
        "feature_columns": artifacts["feature_columns"],
        "xgboost_gain_shares": feature_gain_shares(xgb),
        "min_category_count": MIN_CATEGORY_COUNT,
        "target_encoding_smoothing": TARGET_ENCODING_SMOOTHING,
        "negative_sample_frac": negative_sample_frac,
        "results": results,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Sin --promote, nunca escribe los nombres de producción. Para promover: "" --promote'
    )
    parser.add_argument(
        "artifact_suffix", nargs="?", default=None,
        help=f'sufijo del candidato, p.ej. "_provider_split" (default: {DEFAULT_CANDIDATE_SUFFIX})',
    )
    parser.add_argument("--xgboost-only", action="store_true", help="no entrenar Random Forest")
    parser.add_argument(
        "--promote", action="store_true",
        help='escribe los nombres de PRODUCCIÓN; exige además pasar el sufijo vacío explícito ""',
    )
    args = parser.parse_args()

    if args.promote and args.artifact_suffix is None:
        parser.error('--promote exige pasar explícitamente el sufijo vacío: python -m src.models.train_supervised "" --promote')
    try:
        suffix = resolve_artifact_suffix(args.artifact_suffix or "", args.promote)
    except ValueError as e:
        parser.error(str(e))
    if args.artifact_suffix == "" and not args.promote:
        print(f'Sufijo vacío sin --promote: se guarda como candidato "{DEFAULT_CANDIDATE_SUFFIX}", no en producción.')
    print(f"Escribiendo artefactos con sufijo {suffix!r}" + (" (PRODUCCIÓN)" if suffix == "" else " (candidato)"))

    summary = train_and_evaluate(
        artifact_suffix=suffix, include_random_forest=not args.xgboost_only, promote=args.promote
    )
    for side, stats in summary["split"].items():
        print(f"{side}: {stats}")
    print(f"train_neg_pos_ratio: {summary['train_neg_pos_ratio']:.2f}")
    print(f"xgboost gain shares:\n{summary['xgboost_gain_shares'].round(4).to_string()}")
    print(f"min_category_count: {summary['min_category_count']}")
    print(f"target_encoding_smoothing: {summary['target_encoding_smoothing']}")
    print(f"feature_columns ({len(summary['feature_columns'])}): {summary['feature_columns']}")
    for model_name, model_results in summary["results"].items():
        print(f"\n=== {model_name} ===")
        for eval_name, metrics in model_results.items():
            print(f"  -- {eval_name} --")
            for k, v in metrics.items():
                print(f"     {k}: {v}")
