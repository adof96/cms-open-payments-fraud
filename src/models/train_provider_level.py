"""XGBoost a nivel PROVEEDOR: una fila = un proveedor = un valor de is_excluded.

El modelado por pago (train_supervised.py) dejó que el modelo memorizara proveedores
excluidos concretos (PR-AUC 0.0165 en proveedores nuevos, ver
diagnose_provider_generalization.py). Acá la unidad de análisis coincide con la de la
etiqueta, así que un split estratificado común (train_test_split con stratify=is_excluded)
ya garantiza que ningún proveedor esté en train y test a la vez.

Entrada: data/processed/provider_features.csv y provider_manufacturer_amounts.parquet
(src/features/build_provider_features.py).

## Features que dependen de la etiqueta - sin fuga

- `specialty_te`: target encoding de recipient_specialty a nivel proveedor (tasa de
  proveedores excluidos por especialidad). Train: out-of-fold (StratifiedKFold sobre filas
  = proveedores); test: mapping ajustado sobre todos los proveedores de train.
- `manufacturer_weighted_risk`: riesgo del fabricante de cada pago (target encoding por
  pago), promediado por proveedor ponderando por el monto de cada pago. Train: el encoding
  de cada pago sale de folds AGRUPADOS por proveedor (StratifiedGroupKFold), así la
  etiqueta de un proveedor nunca entra en su propio feature; test: mapping de todos los
  pagos de proveedores de train. Ambos con las funciones de train_supervised.py
  (_out_of_fold_target_encoding / _fit_target_encoding / _apply_target_encoding), mismo
  suavizado (TARGET_ENCODING_SMOOTHING) y mismos folds.

## Muestreo

Todos los proveedores excluidos + una fracción aleatoria fija (NEGATIVE_PROVIDER_SAMPLE_FRAC)
de los no excluidos. La tabla de proveedores es chica; el costo real está en el encoding del
fabricante, que expande los pagos de los proveedores de train - por eso la fracción se fija
para que esa expansión quede en el orden de los ~390k pagos del modelado por pago.

## Evaluación

Además de precision/recall/F1/PR-AUC/ROC-AUC (umbral 0.5, igual que train_supervised.py),
compara contra la especialidad sola como score sobre EL MISMO test de proveedores (bootstrap
pareado del ROC-AUC) y repite el split con varias semillas: con ~76 proveedores excluidos en
test un único split es muy ruidoso. El benchmark 0.78 de la investigación se midió por PAGO
(cada proveedor pesa según su cantidad de pagos), así que no es exactamente la misma unidad
que este ROC-AUC por proveedor - la comparación justa es la de la especialidad sola acá.

Guarda solo nombres de candidato (xgboost_provider_level.pkl,
provider_level_preprocessing.pkl): no hay modelo de producción a nivel proveedor y este
script nunca escribe los nombres de producción del modelado por pago.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE, TARGET_COLUMN
from src.features.build_provider_features import (
    NATURE_GROUP_NAMES,
    PROVIDER_FEATURES_FILENAME,
    PROVIDER_MANUFACTURER_FILENAME,
)
from src.models.diagnose_provider_generalization import memorization_check, single_feature_baseline
from src.models.model_io import save_model
from src.models.train_supervised import (
    TARGET_ENCODING_SMOOTHING,
    TEST_SIZE,
    _apply_target_encoding,
    _evaluate,
    _fit_target_encoding,
    _out_of_fold_target_encoding,
    build_xgboost,
    feature_gain_shares,
)

# Fracción de proveedores NO excluidos a muestrear (además de los 381 excluidos). Fijada tras
# ver el total real (build_provider_features.py: 984,069 proveedores con NPI válido, 15.7
# pagos promedio): 2.5% da ~24.6k proveedores no excluidos y ~390k pagos en la muestra - la
# misma fracción y el mismo presupuesto de ~390k pagos que el modelado por pago
# (NEGATIVE_SAMPLE_FRAC en train_supervised.py), así ambos enfoques son comparables.
NEGATIVE_PROVIDER_SAMPLE_FRAC = 0.025

EVAL_SEEDS = (RANDOM_STATE, 1, 2, 3, 4)
N_BOOTSTRAP = 2000

CANDIDATE_MODEL_PATH = MODELS_DIR / "xgboost_provider_level.pkl"
CANDIDATE_PREPROCESSING_PATH = MODELS_DIR / "provider_level_preprocessing.pkl"

NUMERIC_FEATURES = [
    "total_payment_count",
    "total_payment_amount",
    "mean_payment_amount_log",
    "median_payment_amount_log",
    "n_distinct_manufacturers",
    "top_manufacturer_amount_share",
    *[f"share_nature_{n}" for n in NATURE_GROUP_NAMES],
    "share_related_product",
    "share_third_party_payment",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + ["specialty_te", "manufacturer_weighted_risk"]


def load_provider_sample(
    providers_path: Path | None = None,
    negative_frac: float = NEGATIVE_PROVIDER_SAMPLE_FRAC,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Todos los proveedores excluidos + una fracción aleatoria fija de los demás."""
    providers = pd.read_csv(providers_path or (PROCESSED_DATA_DIR / PROVIDER_FEATURES_FILENAME), dtype={"npi": "int64"})
    excluded = providers[TARGET_COLUMN] == 1
    sample = pd.concat([providers[excluded], providers[~excluded].sample(frac=negative_frac, random_state=random_state)])
    return sample.reset_index(drop=True)


def load_manufacturer_pairs(npis, pairs_path: Path | None = None) -> pd.DataFrame:
    """Filas (npi, manufacturer_name, payment_count, amount) solo de los proveedores pedidos -
    el filtro se aplica al leer el parquet, sin cargar la tabla completa."""
    table = pq.read_table(
        pairs_path or (PROCESSED_DATA_DIR / PROVIDER_MANUFACTURER_FILENAME),
        filters=[("npi", "in", list(map(int, npis)))],
    )
    pairs = table.to_pandas()
    pairs["manufacturer_name"] = pairs["manufacturer_name"].astype(str)
    return pairs


def _weighted_mean_by_provider(pairs: pd.DataFrame, value_col: str) -> pd.Series:
    weighted = (pairs[value_col] * pairs["amount"]).groupby(pairs["npi"]).sum()
    total = pairs.groupby("npi")["amount"].sum()
    unweighted = pairs.groupby("npi")[value_col].mean()  # fallback si un proveedor suma monto 0
    return (weighted / total).where(total > 0, unweighted)


def manufacturer_weighted_risk(train_pairs: pd.DataFrame, test_pairs: pd.DataFrame, train_labels: pd.Series):
    """Riesgo del fabricante por pago (target encoding), promediado por proveedor ponderando
    por monto. `train_labels` es is_excluded indexado por npi (solo proveedores de train).

    Devuelve (risk_train, risk_test, mapping, global_mean). Las filas de train usan folds
    agrupados por proveedor; test usa el mapping ajustado sobre todos los pagos de train."""
    # Una fila por pago: la tabla (npi, fabricante) se expande por payment_count. El encoding
    # cuenta pagos, igual que en el modelado por pago.
    repeat = train_pairs["payment_count"].to_numpy()
    payments = pd.DataFrame({
        "npi": np.repeat(train_pairs["npi"].to_numpy(), repeat),
        "manufacturer_name": np.repeat(train_pairs["manufacturer_name"].to_numpy(), repeat),
    })
    payments[TARGET_COLUMN] = payments["npi"].map(train_labels).to_numpy()

    payments["encoded"] = _out_of_fold_target_encoding(
        payments["manufacturer_name"], payments[TARGET_COLUMN], groups=payments["npi"]
    ).to_numpy()
    per_pair = payments.groupby(["npi", "manufacturer_name"])["encoded"]
    if (per_pair.nunique() > 1).any():
        raise RuntimeError("El encoding out-of-fold no es constante dentro de un proveedor - los folds no están agrupados por npi.")
    train_enc = train_pairs.join(per_pair.first(), on=["npi", "manufacturer_name"])

    mapping, global_mean = _fit_target_encoding(payments, "manufacturer_name", TARGET_COLUMN, smoothing=TARGET_ENCODING_SMOOTHING)
    test_enc = test_pairs.assign(
        encoded=_apply_target_encoding(test_pairs["manufacturer_name"], mapping, global_mean).to_numpy()
    )
    return (
        _weighted_mean_by_provider(train_enc, "encoded"),
        _weighted_mean_by_provider(test_enc, "encoded"),
        mapping,
        global_mean,
    )


def build_split_features(train: pd.DataFrame, test: pd.DataFrame, pairs: pd.DataFrame):
    """Matrices de features de un split de proveedores. Todo lo que usa la etiqueta se ajusta
    solo con proveedores de train. Devuelve (X_train, X_test, artifacts)."""
    X_train = train.set_index("npi")[NUMERIC_FEATURES].astype("float32")
    X_test = test.set_index("npi")[NUMERIC_FEATURES].astype("float32")
    y_train = train.set_index("npi")[TARGET_COLUMN]

    specialty_train = train["recipient_specialty"].fillna("Missing").reset_index(drop=True)
    X_train["specialty_te"] = _out_of_fold_target_encoding(specialty_train, train[TARGET_COLUMN].reset_index(drop=True)).to_numpy()
    spec_mapping, spec_mean = _fit_target_encoding(
        pd.DataFrame({"s": specialty_train, "y": train[TARGET_COLUMN].to_numpy()}), "s", "y", smoothing=TARGET_ENCODING_SMOOTHING
    )
    X_test["specialty_te"] = _apply_target_encoding(
        test["recipient_specialty"].fillna("Missing"), spec_mapping, spec_mean
    ).to_numpy()

    train_pairs = pairs[pairs["npi"].isin(X_train.index)]
    test_pairs = pairs[pairs["npi"].isin(X_test.index)]
    risk_train, risk_test, mfr_mapping, mfr_mean = manufacturer_weighted_risk(train_pairs, test_pairs, y_train)
    X_train["manufacturer_weighted_risk"] = risk_train.reindex(X_train.index).astype("float32")
    X_test["manufacturer_weighted_risk"] = risk_test.reindex(X_test.index).astype("float32")

    artifacts = {
        "feature_columns": FEATURE_COLUMNS,
        "specialty_encoding": {"mapping": spec_mapping, "global_mean": spec_mean},
        "manufacturer_encoding": {"mapping": mfr_mapping, "global_mean": mfr_mean},
    }
    return X_train[FEATURE_COLUMNS], X_test[FEATURE_COLUMNS], artifacts


def paired_bootstrap_auc_diff(y, score_a, score_b, n_boot: int = N_BOOTSTRAP, random_state: int = RANDOM_STATE) -> dict:
    """ROC-AUC(a) - ROC-AUC(b) sobre los MISMOS remuestreos del test (bootstrap pareado,
    estratificado: positivos y negativos se remuestrean por separado para que cada réplica
    tenga los mismos ~76 positivos). Devuelve la diferencia observada, IC 95% y la fracción
    de réplicas en que a NO supera a b."""
    y, score_a, score_b = np.asarray(y), np.asarray(score_a), np.asarray(score_b)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(random_state)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        diffs[i] = roc_auc_score(y[idx], score_a[idx]) - roc_auc_score(y[idx], score_b[idx])
    return {
        "observed_diff": float(roc_auc_score(y, score_a) - roc_auc_score(y, score_b)),
        "ci95_low": float(np.percentile(diffs, 2.5)),
        "ci95_high": float(np.percentile(diffs, 97.5)),
        "share_not_better": float((diffs <= 0).mean()),
    }


def fit_and_evaluate_split(sample: pd.DataFrame, pairs: pd.DataFrame, random_state: int, test_size: float = TEST_SIZE):
    train, test = train_test_split(sample, test_size=test_size, stratify=sample[TARGET_COLUMN], random_state=random_state)
    X_train, X_test, artifacts = build_split_features(train, test, pairs)
    y_train = train.set_index("npi")[TARGET_COLUMN].reindex(X_train.index)
    y_test = test.set_index("npi")[TARGET_COLUMN].reindex(X_test.index)

    neg_pos_ratio = (len(y_train) - y_train.sum()) / y_train.sum()  # del split real, no hardcodeado
    model = build_xgboost(neg_pos_ratio, random_state=RANDOM_STATE).fit(X_train, y_train)
    return model, artifacts, X_train, y_train, X_test, y_test, neg_pos_ratio


def run(providers_path=None, pairs_path=None, negative_frac: float = NEGATIVE_PROVIDER_SAMPLE_FRAC) -> dict:
    sample = load_provider_sample(providers_path, negative_frac=negative_frac)
    pairs = load_manufacturer_pairs(sample["npi"], pairs_path)

    model, artifacts, X_train, y_train, X_test, y_test, ratio = fit_and_evaluate_split(sample, pairs, RANDOM_STATE)
    proba = model.predict_proba(X_test)[:, 1]
    primary = _evaluate(model, X_test, y_test, "is_excluded")
    report = {
        "sample": {
            "providers": len(sample),
            "excluded_providers": int(sample[TARGET_COLUMN].sum()),
            "expanded_train_payments": int(pairs.loc[pairs["npi"].isin(X_train.index), "payment_count"].sum()),
            "train_providers": len(X_train), "test_providers": len(X_test),
            "train_excluded": int(y_train.sum()), "test_excluded": int(y_test.sum()),
            "train_neg_pos_ratio": float(ratio),
        },
        "primary": primary,
        "gain_shares": feature_gain_shares(model),
        "memorization": memorization_check(model, X_train, y_train, X_test, y_test),
        "baselines": single_feature_baseline(
            X_test, y_test, features=["specialty_te", "manufacturer_weighted_risk", "total_payment_count"]
        ),
        "vs_specialty_bootstrap": paired_bootstrap_auc_diff(y_test, proba, X_test["specialty_te"]),
    }

    seed_rows = []
    for seed in EVAL_SEEDS:
        m, _, _, _, Xte, yte, _ = fit_and_evaluate_split(sample, pairs, seed)
        p = m.predict_proba(Xte)[:, 1]
        seed_rows.append({
            "seed": seed,
            "test_excluded": int(yte.sum()),
            "model_roc_auc": roc_auc_score(yte, p),
            "specialty_roc_auc": roc_auc_score(yte, Xte["specialty_te"]),
            "model_pr_auc": average_precision_score(yte, p),
            "specialty_pr_auc": average_precision_score(yte, Xte["specialty_te"]),
            "random_pr_auc": float(yte.mean()),
        })
    report["seeds"] = pd.DataFrame(seed_rows)

    save_model(model, CANDIDATE_MODEL_PATH)
    save_model(artifacts, CANDIDATE_PREPROCESSING_PATH)
    return report


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    r = run()
    print("=== sample / split ===")
    for k, v in r["sample"].items():
        print(f"  {k}: {v:,.2f}" if isinstance(v, float) else f"  {k}: {v:,}")
    print("\n=== XGBoost, provider-level test set (threshold 0.5) ===")
    for k in ("precision", "recall", "f1", "pr_auc", "roc_auc", "confusion_matrix"):
        print(f"  {k}: {r['primary'][k]}")
    print(f"\n=== gain shares ===\n{r['gain_shares'].round(4).to_string()}")
    mem = r["memorization"]
    print(f"\n=== memorization: train PR-AUC {mem['train']['pr_auc']:.4f} vs test {mem['test']['pr_auc']:.4f} (gap {mem['pr_auc_gap']:.1f}x)")
    print(f"\n=== single-feature baselines on the same test providers ===\n{r['baselines'].round(4).to_string()}")
    b = r["vs_specialty_bootstrap"]
    print(f"\n=== model ROC-AUC minus specialty-only ROC-AUC (paired bootstrap, {N_BOOTSTRAP} reps) ===")
    print(f"  observed {b['observed_diff']:+.4f}  95% CI [{b['ci95_low']:+.4f}, {b['ci95_high']:+.4f}]  "
          f"share of reps where model is NOT better: {b['share_not_better']:.3f}")
    print(f"\n=== repeated stratified splits ===\n{r['seeds'].round(4).to_string(index=False)}")
    print(f"\nsaved candidates: {CANDIDATE_MODEL_PATH.name}, {CANDIDATE_PREPROCESSING_PATH.name}")
