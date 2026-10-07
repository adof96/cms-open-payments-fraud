"""¿Un XGBoost con menos capacidad memoriza menos y generaliza mejor a nivel proveedor?

El candidato de train_provider_level.py (config de train_supervised.build_xgboost: 300
árboles, max_depth=6, learning_rate=0.1, min_child_weight=1, reg_lambda=1) llega a PR-AUC
0.98 en train vs. 0.0375 en test (26x) con solo 305 proveedores excluidos en train. Este
script reentrena variantes más restringidas sobre EXACTAMENTE la misma muestra, split y
features (funciones de train_provider_level.py, mismo random_state) - no reconstruye
provider_features.csv ni vuelve a muestrear, y no toca la config ni los artefactos del
candidato original.

Antes de comparar, verifica (no asume) que reajustar la config original reproduce las
predicciones del candidato guardado, y que el preprocesamiento coincide con el guardado.

Nota sobre min_child_weight: limita el hessiano sumado de una hoja, no la cantidad de
filas. Con scale_pos_weight ~64.5 cada proveedor excluido aporta ~64.5 * p(1-p) (~16 al
inicio del boosting), así que un valor "moderado" como 10 todavía permite una hoja armada
alrededor de UN solo proveedor excluido. Para obligar a que haya varios por hoja hace falta
del orden de 50-100.

Guarda cada variante como candidato propio (xgboost_provider_level_<variante>.pkl).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import MODELS_DIR, RANDOM_STATE, TARGET_COLUMN
from src.models.model_io import load_model, save_model
from src.models.train_provider_level import (
    CANDIDATE_MODEL_PATH,
    CANDIDATE_PREPROCESSING_PATH,
    EVAL_SEEDS,
    build_split_features,
    load_manufacturer_pairs,
    load_provider_sample,
    paired_bootstrap_auc_diff,
)
from src.models.train_supervised import TEST_SIZE, build_xgboost

# Cada variante = overrides sobre build_xgboost (mismo scale_pos_weight, learning_rate,
# eval_metric y random_state). "original" no cambia nada: es la referencia.
VARIANTS: dict[str, dict] = {
    "original": {},
    "depth4_mcw10": {"max_depth": 4, "min_child_weight": 10},
    "depth3_mcw50": {"max_depth": 3, "min_child_weight": 50},
    "depth2_mcw100_l2": {"max_depth": 2, "min_child_weight": 100, "reg_lambda": 10, "n_estimators": 150},
}


def build_variant(neg_pos_ratio: float, overrides: dict, random_state: int = RANDOM_STATE):
    return build_xgboost(neg_pos_ratio, random_state=random_state).set_params(**overrides)


def _split(sample: pd.DataFrame, pairs: pd.DataFrame, seed: int):
    """Mismo split y features que train_provider_level.fit_and_evaluate_split."""
    train, test = train_test_split(sample, test_size=TEST_SIZE, stratify=sample[TARGET_COLUMN], random_state=seed)
    X_train, X_test, artifacts = build_split_features(train, test, pairs)
    y_train = train.set_index("npi")[TARGET_COLUMN].reindex(X_train.index)
    y_test = test.set_index("npi")[TARGET_COLUMN].reindex(X_test.index)
    return X_train, y_train, X_test, y_test, artifacts


def _neg_pos_ratio(y) -> float:
    return (len(y) - y.sum()) / y.sum()


def capacity_row(name: str, model, X_train, y_train, X_test, y_test) -> dict:
    train_pr = average_precision_score(y_train, model.predict_proba(X_train)[:, 1])
    test_proba = model.predict_proba(X_test)[:, 1]
    test_pr = average_precision_score(y_test, test_proba)
    return {
        "variant": name,
        "train_pr_auc": train_pr,
        "test_pr_auc": test_pr,
        "test_roc_auc": roc_auc_score(y_test, test_proba),
        "train_test_pr_ratio": train_pr / test_pr,
    }


def run() -> dict:
    sample = load_provider_sample()
    pairs = load_manufacturer_pairs(sample["npi"])
    X_train, y_train, X_test, y_test, artifacts = _split(sample, pairs, RANDOM_STATE)
    ratio = _neg_pos_ratio(y_train)

    if artifacts != load_model(CANDIDATE_PREPROCESSING_PATH):
        raise RuntimeError("El preprocesamiento recalculado no coincide con provider_level_preprocessing.pkl.")

    rows, models = [], {}
    for name, overrides in VARIANTS.items():
        model = build_variant(ratio, overrides).fit(X_train, y_train)
        models[name] = model
        rows.append(capacity_row(name, model, X_train, y_train, X_test, y_test))

    saved = load_model(CANDIDATE_MODEL_PATH).predict_proba(X_test)[:, 1]
    if not np.allclose(saved, models["original"].predict_proba(X_test)[:, 1]):
        raise RuntimeError("La variante 'original' no reproduce xgboost_provider_level.pkl - el split/features no son los mismos.")

    table = pd.DataFrame(rows).set_index("variant")
    specialty = {
        "test_pr_auc": average_precision_score(y_test, X_test["specialty_te"]),
        "test_roc_auc": roc_auc_score(y_test, X_test["specialty_te"]),
    }

    # La mejor variante se elige por ROC-AUC en ESTE test - es selección sobre el test, así
    # que su número acá es optimista; el chequeo de 5 splits de abajo es la lectura más justa.
    restricted = table.drop(index="original")
    best = restricted["test_roc_auc"].idxmax()
    for name, model in models.items():
        if name != "original":
            save_model(model, MODELS_DIR / f"xgboost_provider_level_{name}.pkl")

    best_proba = models[best].predict_proba(X_test)[:, 1]
    bootstrap = paired_bootstrap_auc_diff(y_test, best_proba, X_test["specialty_te"])

    seed_rows = []
    for seed in EVAL_SEEDS:
        Xtr, ytr, Xte, yte, _ = _split(sample, pairs, seed)
        r = _neg_pos_ratio(ytr)
        row = {"seed": seed, "specialty_roc_auc": roc_auc_score(yte, Xte["specialty_te"]),
               "specialty_pr_auc": average_precision_score(yte, Xte["specialty_te"])}
        for name in ("original", best):
            p = build_variant(r, VARIANTS[name]).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
            row[f"{name}_roc_auc"] = roc_auc_score(yte, p)
            row[f"{name}_pr_auc"] = average_precision_score(yte, p)
        seed_rows.append(row)

    return {"table": table, "specialty": specialty, "best": best, "bootstrap": bootstrap,
            "seeds": pd.DataFrame(seed_rows), "random_pr_auc": float(y_test.mean())}


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    r = run()
    print("original candidate reproduced exactly; preprocessing matches provider_level_preprocessing.pkl\n")
    print("=== capacity variants, seed-42 provider-level split ===")
    print(r["table"].round(4).to_string())
    s = r["specialty"]
    print(f"\nspecialty_te alone on the same test: PR-AUC {s['test_pr_auc']:.4f}  ROC-AUC {s['test_roc_auc']:.4f}  "
          f"(random PR-AUC {r['random_pr_auc']:.4f})")
    b = r["bootstrap"]
    print(f"\nbest restricted variant (by test ROC-AUC): {r['best']}")
    print(f"  ROC-AUC minus specialty-only: {b['observed_diff']:+.4f}  95% CI [{b['ci95_low']:+.4f}, {b['ci95_high']:+.4f}]  "
          f"not better in {b['share_not_better']:.1%} of bootstrap reps")
    print(f"\n=== 5 splits: original vs {r['best']} vs specialty alone ===")
    seeds = r["seeds"]
    print(seeds.round(4).to_string(index=False))
    print("\nmeans:")
    print(seeds.drop(columns="seed").mean().round(4).to_string())
