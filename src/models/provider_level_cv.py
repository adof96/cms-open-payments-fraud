"""Selección de capacidad del XGBoost a nivel proveedor por validación cruzada, usando SOLO
los proveedores de train - los de test no participan de la selección.

provider_level_capacity.py eligió la variante depth2_mcw100_l2 mirando el test del split
principal, y depth 2 era además el punto más restringido probado. Acá:

1. Grilla chica (81 configs: max_depth x min_child_weight x reg_lambda x n_estimators,
   learning_rate 0.1) evaluada con StratifiedKFold de 5 folds sobre los proveedores de
   train del split principal (misma muestra, split y random_state que
   train_provider_level.py / provider_level_capacity.py).
2. Las features que usan la etiqueta (specialty_te, manufacturer_weighted_risk) se ajustan
   DENTRO de cada fold, solo con los proveedores de la parte de ajuste de ese fold, vía
   train_provider_level.build_split_features - la misma lógica sin fuga ya establecida.
   scale_pos_weight sale del ratio de cada fold.
3. Regla de selección: la config MÁS SIMPLE cuyo PR-AUC medio está a menos de un desvío
   estándar (entre folds, ddof=1) del mejor PR-AUC medio. "Más simple" = menor profundidad,
   después menos árboles, después más regularización (min_child_weight y reg_lambda más
   altos) - ver complexity_key.
4. Baselines en los mismos folds: especialidad sola (target encoding como score) y
   regresión logística (features estandarizadas, class_weight="balanced").
5. La config elegida se reentrena con todos los proveedores de train y se evalúa en el test
   held-out y en los mismos 5 splits de proveedores, junto a depth2_mcw100_l2.

Recursos: n_jobs=2 (no -1) y un tope de MAX_RUNTIME_SECONDS - si se supera, se corta con
TimeoutError en vez de seguir. Guarda solo xgboost_provider_level_cv.pkl (usa el mismo
preprocesamiento que provider_level_preprocessing.pkl, que se verifica); no toca ningún
otro .pkl.
"""

import itertools
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, RANDOM_STATE, TARGET_COLUMN
from src.models.model_io import load_model, save_model
from src.models.provider_level_capacity import VARIANTS
from src.models.train_provider_level import (
    CANDIDATE_PREPROCESSING_PATH,
    EVAL_SEEDS,
    build_split_features,
    load_manufacturer_pairs,
    load_provider_sample,
)
from src.models.train_supervised import TEST_SIZE, build_xgboost

GRID = {
    "max_depth": [1, 2, 3],
    "min_child_weight": [50, 100, 200],
    "reg_lambda": [1, 10, 50],
    "n_estimators": [100, 150, 300],
}
LEARNING_RATE = 0.1
N_FOLDS = 5
N_JOBS = 2
MAX_RUNTIME_SECONDS = 14 * 60

DEPTH2_VARIANT = "depth2_mcw100_l2"
DEPTH2_MODEL_PATH = MODELS_DIR / f"xgboost_provider_level_{DEPTH2_VARIANT}.pkl"
CV_MODEL_PATH = MODELS_DIR / "xgboost_provider_level_cv.pkl"


def grid_configs() -> list[dict]:
    return [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]


def complexity_key(config) -> tuple:
    """Orden de más simple a más complejo: profundidad (hojas por árbol) primero, después
    cantidad de árboles, después regularización (min_child_weight y reg_lambda más altos =
    más simple)."""
    return (
        int(config["max_depth"]),
        int(config["n_estimators"]),
        -int(config["min_child_weight"]),
        -int(config["reg_lambda"]),
    )


def build_model(config: dict, neg_pos_ratio: float, random_state: int = RANDOM_STATE):
    return build_xgboost(neg_pos_ratio, random_state=random_state).set_params(
        learning_rate=LEARNING_RATE, n_jobs=N_JOBS, **config
    )


def build_depth2(neg_pos_ratio: float, random_state: int = RANDOM_STATE):
    """La variante de provider_level_capacity.py, solo que con n_jobs=2."""
    return build_xgboost(neg_pos_ratio, random_state=random_state).set_params(
        n_jobs=N_JOBS, **VARIANTS[DEPTH2_VARIANT]
    )


def logistic_baseline():
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000))


def neg_pos_ratio(y) -> float:
    return float((len(y) - y.sum()) / y.sum())


def split_providers(sample: pd.DataFrame, seed: int):
    """Mismo split que train_provider_level.fit_and_evaluate_split."""
    return train_test_split(sample, test_size=TEST_SIZE, stratify=sample[TARGET_COLUMN], random_state=seed)


def _labels(providers: pd.DataFrame, index) -> pd.Series:
    return providers.set_index("npi")[TARGET_COLUMN].reindex(index)


def fold_features(train: pd.DataFrame, fit_idx, val_idx, pairs: pd.DataFrame):
    """Features de un fold: todo lo que usa la etiqueta se ajusta solo con los proveedores
    de ajuste (fit_idx); los de validación solo reciben el mapping."""
    fit, val = train.iloc[fit_idx], train.iloc[val_idx]
    X_fit, X_val, _ = build_split_features(fit, val, pairs)
    return X_fit, _labels(fit, X_fit.index), X_val, _labels(val, X_val.index)


def cv_folds(train: pd.DataFrame, pairs: pd.DataFrame, n_folds: int = N_FOLDS, random_state: int = RANDOM_STATE):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    return [fold_features(train, fit_idx, val_idx, pairs) for fit_idx, val_idx in skf.split(train, train[TARGET_COLUMN])]


def _summary(pr: list, roc: list) -> dict:
    return {
        "mean_pr_auc": float(np.mean(pr)), "std_pr_auc": float(np.std(pr, ddof=1)),
        "mean_roc_auc": float(np.mean(roc)), "std_roc_auc": float(np.std(roc, ddof=1)),
        "fold_pr_auc": list(pr),
    }


def evaluate_grid(folds, configs: list[dict], deadline: float | None = None) -> pd.DataFrame:
    rows = []
    for i, config in enumerate(configs):
        pr, roc = [], []
        for X_fit, y_fit, X_val, y_val in folds:
            if deadline is not None and time.time() > deadline:
                raise TimeoutError(f"Se superó el tiempo máximo tras {i} de {len(configs)} configuraciones.")
            p = build_model(config, neg_pos_ratio(y_fit)).fit(X_fit, y_fit).predict_proba(X_val)[:, 1]
            pr.append(average_precision_score(y_val, p))
            roc.append(roc_auc_score(y_val, p))
        rows.append({**config, **_summary(pr, roc)})
    return pd.DataFrame(rows)


def select_config(results: pd.DataFrame):
    """Devuelve (índice del mejor PR-AUC medio, índice elegido por la regla, umbral)."""
    best = results["mean_pr_auc"].idxmax()
    threshold = results.loc[best, "mean_pr_auc"] - results.loc[best, "std_pr_auc"]
    eligible = results.index[results["mean_pr_auc"] >= threshold]
    selected = min(eligible, key=lambda i: complexity_key(results.loc[i]))
    return best, selected, float(threshold)


def config_of(results: pd.DataFrame, idx) -> dict:
    return {k: int(results.loc[idx, k]) for k in GRID}


def _specialty_score(X_fit, y_fit, X_val):
    return X_val["specialty_te"].to_numpy()


def _logistic_score(X_fit, y_fit, X_val):
    return logistic_baseline().fit(X_fit, y_fit).predict_proba(X_val)[:, 1]


BASELINES = {"specialty_only": _specialty_score, "logistic_regression": _logistic_score}


def baseline_cv(folds) -> pd.DataFrame:
    rows = []
    for name, score in BASELINES.items():
        pr, roc = [], []
        for X_fit, y_fit, X_val, y_val in folds:
            s = score(X_fit, y_fit, X_val)
            pr.append(average_precision_score(y_val, s))
            roc.append(roc_auc_score(y_val, s))
        rows.append({"baseline": name, **_summary(pr, roc)})
    return pd.DataFrame(rows).set_index("baseline")


def separability(results: pd.DataFrame, best, selected, threshold: float) -> dict:
    """¿Las diferencias entre configs cercanas superan la variación entre folds?"""
    paired = np.array(results.loc[best, "fold_pr_auc"]) - np.array(results.loc[selected, "fold_pr_auc"])
    top10 = results["mean_pr_auc"].nlargest(10)
    marginal = {
        k: results.groupby(k)["mean_pr_auc"].mean().round(4).to_dict() for k in GRID
    }
    return {
        "median_fold_std_pr_auc": float(results["std_pr_auc"].median()),
        "spread_all_81_means": float(results["mean_pr_auc"].max() - results["mean_pr_auc"].min()),
        "spread_top10_means": float(top10.max() - top10.min()),
        "n_within_one_sd_of_best": int((results["mean_pr_auc"] >= threshold).sum()),
        "best_minus_selected_per_fold": paired.round(4).tolist(),
        "best_minus_selected_mean": float(paired.mean()),
        "best_minus_selected_std": float(paired.std(ddof=1)) if len(paired) > 1 else float("nan"),
        "folds_where_best_beats_selected": int((paired > 0).sum()),
        "marginal_mean_pr_auc": marginal,
    }


def _scores(y, p) -> dict:
    return {"pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p)}


def compare_on_splits(sample: pd.DataFrame, pairs: pd.DataFrame, selected_config: dict, seeds=EVAL_SEEDS):
    """Reentrena la config elegida, depth2_mcw100_l2 y la logística en cada split de
    proveedores; la especialidad sola se usa como score. Devuelve (tabla larga, modelo CV del
    split principal, chequeos)."""
    rows, checks, main_cv_model = [], {}, None
    for seed in seeds:
        train, test = split_providers(sample, seed)
        X_tr, X_te, artifacts = build_split_features(train, test, pairs)
        y_tr, y_te = _labels(train, X_tr.index), _labels(test, X_te.index)
        ratio = neg_pos_ratio(y_tr)
        models = {
            "cv_selected": build_model(selected_config, ratio).fit(X_tr, y_tr),
            DEPTH2_VARIANT: build_depth2(ratio).fit(X_tr, y_tr),
            "logistic_regression": logistic_baseline().fit(X_tr, y_tr),
        }
        for name, model in models.items():
            rows.append({
                "seed": seed, "model": name,
                "train_pr_auc": average_precision_score(y_tr, model.predict_proba(X_tr)[:, 1]),
                **_scores(y_te, model.predict_proba(X_te)[:, 1]),
            })
        rows.append({"seed": seed, "model": "specialty_only", "train_pr_auc": np.nan, **_scores(y_te, X_te["specialty_te"])})

        if seed == RANDOM_STATE:
            main_cv_model = models["cv_selected"]
            checks["preprocessing_matches_saved"] = artifacts == load_model(CANDIDATE_PREPROCESSING_PATH)
            saved = load_model(DEPTH2_MODEL_PATH).predict_proba(X_te)[:, 1]
            checks["depth2_max_abs_diff_vs_saved"] = float(
                np.abs(saved - models[DEPTH2_VARIANT].predict_proba(X_te)[:, 1]).max()
            )
            checks["test_positives"] = int(y_te.sum())
    return pd.DataFrame(rows), main_cv_model, checks


def summarize_comparison(long: pd.DataFrame) -> pd.DataFrame:
    held_out = long[long["seed"] == RANDOM_STATE].set_index("model")
    by_model = long.groupby("model")
    specialty = long[long["model"] == "specialty_only"].set_index("seed")
    out = pd.DataFrame({
        "heldout_train_pr_auc": held_out["train_pr_auc"],
        "heldout_test_pr_auc": held_out["pr_auc"],
        "heldout_test_roc_auc": held_out["roc_auc"],
        "5split_mean_pr_auc": by_model["pr_auc"].mean(),
        "5split_min_pr_auc": by_model["pr_auc"].min(),
        "5split_mean_roc_auc": by_model["roc_auc"].mean(),
        "5split_min_roc_auc": by_model["roc_auc"].min(),
    })
    wins = {}
    for model, g in long.groupby("model"):
        g = g.set_index("seed")
        wins[model] = int(((g["pr_auc"] > specialty["pr_auc"]) & (g["roc_auc"] > specialty["roc_auc"])).sum())
    out["splits_beating_specialty_on_both"] = pd.Series(wins)
    return out.loc[["cv_selected", DEPTH2_VARIANT, "logistic_regression", "specialty_only"]]


def run() -> dict:
    start = time.time()
    deadline = start + MAX_RUNTIME_SECONDS
    timings = {}

    sample = load_provider_sample()
    pairs = load_manufacturer_pairs(sample["npi"])
    train, _ = split_providers(sample, RANDOM_STATE)
    folds = cv_folds(train, pairs)
    timings["fold_features"] = time.time() - start

    results = evaluate_grid(folds, grid_configs(), deadline=deadline)
    timings["grid"] = time.time() - start - timings["fold_features"]

    best, selected, threshold = select_config(results)
    baselines = baseline_cv(folds)
    selected_config = config_of(results, selected)

    long, cv_model, checks = compare_on_splits(sample, pairs, selected_config)
    save_model(cv_model, CV_MODEL_PATH)
    timings["total"] = time.time() - start

    return {
        "results": results, "best": best, "selected": selected, "threshold": threshold,
        "baselines": baselines, "separability": separability(results, best, selected, threshold),
        "comparison": summarize_comparison(long), "comparison_long": long, "checks": checks,
        "fold_positives": [int(y_val.sum()) for _, _, _, y_val in folds],
        "fold_sizes": [len(y_val) for _, _, _, y_val in folds],
        "timings": timings,
    }


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    r = run()
    res = r["results"]
    cols = list(GRID) + ["mean_pr_auc", "std_pr_auc", "mean_roc_auc", "std_roc_auc"]

    print(f"CV on training providers only: {N_FOLDS} folds, validation positives per fold {r['fold_positives']}, "
          f"validation providers per fold {r['fold_sizes']}\n")
    print("=== all 81 configs, sorted by mean CV PR-AUC ===")
    print(res.sort_values("mean_pr_auc", ascending=False)[cols].round(4).to_string(index=False))

    print(f"\nraw best:      {config_of(res, r['best'])}  mean PR-AUC {res.loc[r['best'], 'mean_pr_auc']:.4f} "
          f"± {res.loc[r['best'], 'std_pr_auc']:.4f}")
    print(f"rule-selected: {config_of(res, r['selected'])}  mean PR-AUC {res.loc[r['selected'], 'mean_pr_auc']:.4f} "
          f"± {res.loc[r['selected'], 'std_pr_auc']:.4f}  (threshold = best mean - 1 SD = {r['threshold']:.4f})")

    print("\n=== baselines in the same folds ===")
    print(r["baselines"].drop(columns="fold_pr_auc").round(4).to_string())

    print("\n=== separability ===")
    for k, v in r["separability"].items():
        print(f"  {k}: {v}")

    c = r["checks"]
    print(f"\nchecks: preprocessing matches provider_level_preprocessing.pkl = {c['preprocessing_matches_saved']}; "
          f"depth2 refit (n_jobs=2) vs saved depth2 candidate, max |diff| = {c['depth2_max_abs_diff_vs_saved']:.2e}; "
          f"held-out test positives = {c['test_positives']}")
    print("\n=== final comparison: held-out test (seed 42) + same 5 provider splits ===")
    print(r["comparison"].round(4).to_string())
    print("\nper split:")
    print(r["comparison_long"].pivot(index="seed", columns="model", values=["pr_auc", "roc_auc"]).round(4).to_string())

    t = r["timings"]
    print(f"\nruntime: fold features {t['fold_features']:.0f}s, grid {t['grid']:.0f}s, total {t['total']:.0f}s")
    print(f"saved candidate: {CV_MODEL_PATH.name}")
