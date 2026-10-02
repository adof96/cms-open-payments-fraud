"""Diagnósticos de generalización a proveedores NUEVOS para un modelo supervisado de
is_excluded.

`is_excluded` es un dato por proveedor (NPI) y los pagos excluidos vienen de muy pocos
proveedores (381), así que un modelo puede "aprender" a reconocer proveedores vistos en
train en vez de patrones que generalicen. Estas tres verificaciones, que antes se corrieron
como código descartable durante la investigación de fuga a nivel proveedor, quedan acá como
funciones reutilizables:

1. `memorization_check`: PR-AUC/ROC-AUC del mismo modelo en train vs. test. Un gap enorme
   (p.ej. 0.90 en train vs. 0.02 en test) indica memorización de proveedores de train.
2. `single_feature_baseline`: cada feature por sí solo como score sobre el test. Si un
   único feature (p.ej. recipient_specialty_te) rankea mejor que el modelo completo, el
   modelo está gastando su capacidad en memorizar en vez de generalizar.
3. `seed_variance_check`: rehace el split a nivel proveedor con varias semillas y reentrena
   un XGBoost (descartable, no se guarda) por variante de features. Con pocos proveedores
   excluidos por test (~76) un único split es muy ruidoso; esto mide cuánto.

Todas las métricas se informan junto a la tasa de positivos del test, que es el PR-AUC que
daría un ranking al azar - sin esa referencia un PR-AUC de 0.016 no dice nada.

Uso:
    python -m src.models.diagnose_provider_generalization                 # candidato _provider_split
    python -m src.models.diagnose_provider_generalization _provider_split --skip-seed-variance

Solo lee artefactos de models/ - no guarda ni modifica ningún .pkl (los modelos del chequeo
de semillas viven solo en memoria).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE, TARGET_COLUMN
from src.features.build_features import DEFAULT_OUTPUT_FILENAME as FRAUD_FEATURES_FILENAME
from src.models.model_io import load_model
from src.models.train_supervised import (
    USECOLS,
    _load_stratified_sample,
    attach_provider_ids,
    build_xgboost,
    preprocess_features,
    provider_groups,
    provider_level_split,
)

DEFAULT_ARTIFACT_SUFFIX = "_provider_split"
DEFAULT_SEEDS = (42, 1, 2, 3, 4)
DEFAULT_BASELINE_FEATURES = ("recipient_specialty_te", "manufacturer_name_te", "payment_frequency", "payment_amount_log")
DEFAULT_VARIANTS = {
    "full": [],
    "no payment_frequency": ["payment_frequency", "payment_frequency_missing"],
}


def _scores(y: pd.Series, score: np.ndarray) -> dict:
    y = np.asarray(y)
    positive_rate = float(y.mean())
    pr_auc = float(average_precision_score(y, score))
    return {
        "pr_auc": pr_auc,
        "roc_auc": float(roc_auc_score(y, score)),
        "positive_rate": positive_rate,  # = PR-AUC de un ranking al azar
        "pr_auc_lift": pr_auc / positive_rate if positive_rate else float("nan"),
    }


def memorization_check(model, X_train: pd.DataFrame, y_train, X_test: pd.DataFrame, y_test) -> dict:
    """Mismo modelo evaluado en train y en test. `pr_auc_gap` = PR-AUC train / PR-AUC test."""
    train = _scores(y_train, model.predict_proba(X_train)[:, 1])
    test = _scores(y_test, model.predict_proba(X_test)[:, 1])
    return {"train": train, "test": test, "pr_auc_gap": train["pr_auc"] / test["pr_auc"]}


def single_feature_baseline(X_test: pd.DataFrame, y_test, features=DEFAULT_BASELINE_FEATURES) -> pd.DataFrame:
    """Cada feature usado directamente como score sobre el test (sin modelo). Un ROC-AUC < 0.5
    significa que el feature rankea al revés (mayor valor = menos riesgo), no que no sirva."""
    rows = {f: _scores(y_test, X_test[f].to_numpy()) for f in features if f in X_test.columns}
    return pd.DataFrame(rows).T[["pr_auc", "roc_auc", "pr_auc_lift"]]


def seed_variance_check(
    sample: pd.DataFrame,
    groups: pd.Series,
    seeds=DEFAULT_SEEDS,
    variants: dict[str, list[str]] | None = None,
    model_random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Para cada semilla: split a nivel proveedor + target encoding out-of-fold agrupado
    (igual que train_and_evaluate) y un XGBoost por variante de features. Los modelos son
    descartables - no se guardan. Devuelve una fila por (semilla, variante)."""
    variants = variants or DEFAULT_VARIANTS
    rows = []
    for seed in seeds:
        train_raw, test_raw = provider_level_split(sample, groups, random_state=seed)
        X_train, X_test, _ = preprocess_features(train_raw, test_raw, train_groups=groups.loc[train_raw.index])
        y_train, y_test = train_raw[TARGET_COLUMN].to_numpy(), test_raw[TARGET_COLUMN].to_numpy()
        neg_pos_ratio = (len(y_train) - y_train.sum()) / y_train.sum()
        test_groups = groups.loc[test_raw.index]
        excluded_providers = int(test_raw[TARGET_COLUMN].groupby(test_groups).max().loc[lambda s: s.index >= 0].sum())

        for name, dropped in variants.items():
            cols = [c for c in X_train.columns if c not in dropped]
            model = build_xgboost(neg_pos_ratio, random_state=model_random_state).fit(X_train[cols], y_train)
            rows.append({
                "seed": seed,
                "variant": name,
                "test_excluded_providers": excluded_providers,
                **_scores(y_test, model.predict_proba(X_test[cols])[:, 1]),
            })
    return pd.DataFrame(rows)


def summarize_seed_variance(results: pd.DataFrame) -> pd.DataFrame:
    return results.groupby("variant")[["pr_auc", "roc_auc", "pr_auc_lift"]].agg(["mean", "min", "max"])


def run_diagnostics(model, artifacts: dict, X_train: pd.DataFrame, y_train, X_test: pd.DataFrame, y_test) -> dict:
    """Corre memorización + baseline de un solo feature para un modelo ya entrenado. Las
    columnas se ordenan según artifacts["feature_columns"] (el orden con el que se entrenó)."""
    cols = artifacts["feature_columns"]
    return {
        "memorization": memorization_check(model, X_train[cols], y_train, X_test[cols], y_test),
        "single_feature_baseline": single_feature_baseline(X_test, y_test),
    }


def _load_candidate_inputs(artifact_suffix: str, random_state: int = RANDOM_STATE):
    """Carga el candidato y reconstruye EXACTAMENTE sus datos de train/test (misma muestra,
    mismo split por proveedor, mismo out-of-fold). Verifica - no asume - que el
    preprocesamiento recalculado coincide con el guardado junto al modelo."""
    model = load_model(MODELS_DIR / f"xgboost_is_excluded{artifact_suffix}.pkl")
    artifacts = load_model(MODELS_DIR / f"supervised_preprocessing{artifact_suffix}.pkl")

    sample = _load_stratified_sample(PROCESSED_DATA_DIR / FRAUD_FEATURES_FILENAME, usecols=USECOLS + ["Record_ID"])
    groups = provider_groups(attach_provider_ids(sample))
    train_raw, test_raw = provider_level_split(sample, groups, random_state=random_state)
    X_train, X_test, rebuilt = preprocess_features(train_raw, test_raw, train_groups=groups.loc[train_raw.index])
    if rebuilt != artifacts:
        raise RuntimeError(
            f"El preprocesamiento recalculado no coincide con supervised_preprocessing{artifact_suffix}.pkl - "
            "los datos reconstruidos no son los que vio este modelo (¿candidato de split por fila?)."
        )
    return model, artifacts, sample, groups, X_train, train_raw[TARGET_COLUMN], X_test, test_raw[TARGET_COLUMN]


def _print_report(diagnostics: dict, seed_results: pd.DataFrame | None) -> None:
    mem = diagnostics["memorization"]
    print("=== 1. Memorization: same model on train vs. test ===")
    for side in ("train", "test"):
        s = mem[side]
        print(f"  {side:5s} PR-AUC={s['pr_auc']:.4f}  ROC-AUC={s['roc_auc']:.4f}  "
              f"(random PR-AUC = positive rate {s['positive_rate']:.4f}; lift {s['pr_auc_lift']:.1f}x)")
    print(f"  train/test PR-AUC gap: {mem['pr_auc_gap']:.1f}x")

    test_roc = mem["test"]["roc_auc"]
    print(f"\n=== 2. Single-feature baselines on the test set (full model ROC-AUC = {test_roc:.4f}) ===")
    baseline = diagnostics["single_feature_baseline"].copy()
    baseline["beats_full_model_roc"] = baseline["roc_auc"] > test_roc
    print(baseline.round(4).to_string())

    if seed_results is not None:
        print("\n=== 3. Split variance across seeds (throwaway models, nothing saved) ===")
        print(seed_results.round(4).to_string(index=False))
        print("\nsummary:")
        print(summarize_seed_variance(seed_results).round(4).to_string())


if __name__ == "__main__":
    import argparse

    pd.set_option("display.width", 200)
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_suffix", nargs="?", default=DEFAULT_ARTIFACT_SUFFIX)
    parser.add_argument("--skip-seed-variance", action="store_true", help="no reentrenar modelos descartables")
    args = parser.parse_args()

    model, artifacts, sample, groups, X_train, y_train, X_test, y_test = _load_candidate_inputs(args.artifact_suffix)
    print(f"candidate: xgboost_is_excluded{args.artifact_suffix}.pkl (inputs verified against its saved preprocessing)\n")
    diagnostics = run_diagnostics(model, artifacts, X_train, y_train, X_test, y_test)
    seed_results = None if args.skip_seed_variance else seed_variance_check(sample, groups)
    _print_report(diagnostics, seed_results)
