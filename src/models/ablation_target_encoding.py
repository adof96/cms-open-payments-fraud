"""Ablation de los dos features target-encoded (recipient_specialty_te y
manufacturer_name_te) sobre el pipeline YA corregido con target encoding out-of-fold
(ver train_supervised.py), más un chequeo de solapamiento especialidad/fabricante.

Reconstruye el mismo split train/test que train_and_evaluate() (vía
tune_threshold.rebuild_test_set: misma muestra, mismo random_state) y entrena tres
XGBoost idénticos salvo por las features:

- full:                 todas las features (debe reproducir exactamente el candidato
                        models/xgboost_is_excluded_oof.pkl - se verifica, no se asume)
- no_manufacturer (a):  sin manufacturer_name_te
- no_specialty (b):     sin recipient_specialty_te

Chequeo de solapamiento (¿el "riesgo" de un fabricante es solo la mezcla de
especialidades a las que les paga?), por tres vías:
1. Correlación fila a fila entre los dos features target-encoded.
2. Cuánto cambia la importancia (total_gain) de cada feature cuando el otro está ausente.
3. Estandarización indirecta sobre el dataset completo (15.5M filas, lectura en chunks)
   para los top-20 fabricantes por volumen: tasa observada vs. la tasa esperada si cada
   pago tuviera la tasa de exclusión de su especialidad (SMR = observada / esperada). SMR
   ~1 = el fabricante no agrega nada sobre su mezcla de especialidades.

Solo reporta - no guarda ni modifica ningún .pkl.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE, TARGET_COLUMN
from src.features.build_features import DEFAULT_OUTPUT_FILENAME as FRAUD_FEATURES_FILENAME
from src.models.model_io import load_model
from src.models.train_supervised import build_xgboost, preprocess_features
from src.models.tune_threshold import rebuild_test_set

CANDIDATE_XGB_PATH = MODELS_DIR / "xgboost_is_excluded_oof.pkl"
TE_COLUMNS = ["recipient_specialty_te", "manufacturer_name_te"]
VARIANTS = {
    "full": [],
    "no_manufacturer (a)": ["manufacturer_name_te"],
    "no_specialty (b)": ["recipient_specialty_te"],
}
TOP_N_MANUFACTURERS = 20
CHUNKSIZE = 300_000


def _gain_shares(model, columns) -> dict:
    gains = model.get_booster().get_score(importance_type="total_gain")
    total = sum(gains.values())
    return {c: gains.get(c, 0.0) / total for c in columns}


def run_ablation() -> dict:
    train_raw, test_raw = rebuild_test_set()
    X_train, X_test, _ = preprocess_features(train_raw, test_raw)
    y_train = train_raw[TARGET_COLUMN].reset_index(drop=True)
    y_test = test_raw[TARGET_COLUMN].reset_index(drop=True)
    neg_pos_ratio = (len(y_train) - y_train.sum()) / y_train.sum()

    results = {}
    for name, dropped in VARIANTS.items():
        cols = [c for c in X_train.columns if c not in dropped]
        model = build_xgboost(neg_pos_ratio, random_state=RANDOM_STATE)
        model.fit(X_train[cols], y_train)
        proba = model.predict_proba(X_test[cols])[:, 1]
        results[name] = {
            "pr_auc": average_precision_score(y_test, proba),
            "roc_auc": roc_auc_score(y_test, proba),
            "gain_share": _gain_shares(model, [c for c in TE_COLUMNS if c in cols]),
            "proba": proba,
        }

    candidate_proba = load_model(CANDIDATE_XGB_PATH).predict_proba(X_test)[:, 1]
    if not np.allclose(candidate_proba, results["full"]["proba"]):
        raise RuntimeError(
            "El XGBoost 'full' del ablation no reproduce el candidato xgboost_is_excluded_oof.pkl "
            "- el ablation no estaría usando el mismo pipeline/split que la parte 1."
        )

    correlation = {
        "train_pearson": X_train[TE_COLUMNS].corr().iloc[0, 1],
        "train_spearman": spearmanr(X_train[TE_COLUMNS[0]], X_train[TE_COLUMNS[1]]).statistic,
        "test_pearson": X_test[TE_COLUMNS].corr().iloc[0, 1],
    }
    return {"variants": results, "correlation": correlation, "neg_pos_ratio": neg_pos_ratio}


def _poisson_ci(k: int, alpha: float = 0.05) -> tuple[float, float]:
    low = chi2.ppf(alpha / 2, 2 * k) / 2 if k > 0 else 0.0
    high = chi2.ppf(1 - alpha / 2, 2 * (k + 1)) / 2
    return low, high


def specialty_standardized_rates(top_n: int = TOP_N_MANUFACTURERS, features_path=None) -> pd.DataFrame:
    """SMR por fabricante sobre el dataset completo: pagos observados excluidos vs. los
    esperados si cada pago tuviera la tasa de exclusión de su especialidad."""
    parts = []
    reader = pd.read_csv(
        features_path or (PROCESSED_DATA_DIR / FRAUD_FEATURES_FILENAME),
        usecols=["manufacturer_name", "recipient_specialty", TARGET_COLUMN],
        dtype={TARGET_COLUMN: "int8"},
        chunksize=CHUNKSIZE,
    )
    for chunk in reader:
        keys = [chunk["manufacturer_name"].fillna("Missing"), chunk["recipient_specialty"].fillna("Missing")]
        parts.append(chunk[TARGET_COLUMN].groupby(keys).agg(["count", "sum"]))
    pairs = pd.concat(parts).groupby(level=[0, 1]).sum()
    pairs.index.names = ["manufacturer", "specialty"]

    by_specialty = pairs.groupby(level="specialty").sum()
    specialty_rate = by_specialty["sum"] / by_specialty["count"]
    overall_rate = pairs["sum"].sum() / pairs["count"].sum()

    pairs = pairs.join(specialty_rate.rename("specialty_rate"), on="specialty")
    pairs["expected"] = pairs["count"] * pairs["specialty_rate"]
    by_manufacturer = pairs.groupby(level="manufacturer")[["count", "sum", "expected"]].sum()
    top = by_manufacturer.sort_values("count", ascending=False).head(top_n)

    rows = []
    for name, r in top.iterrows():
        low, high = _poisson_ci(int(r["sum"]))
        rows.append({
            "manufacturer": name,
            "rows": int(r["count"]),
            "excluded": int(r["sum"]),
            "raw_rate_vs_overall": (r["sum"] / r["count"]) / overall_rate,
            "specialty_mix_rate_vs_overall": (r["expected"] / r["count"]) / overall_rate,
            "smr": r["sum"] / r["expected"],
            "smr_ci95_low": low / r["expected"],
            "smr_ci95_high": high / r["expected"],
        })
    table = pd.DataFrame(rows).set_index("manufacturer")
    table["smr_verdict"] = np.where(
        table["smr_ci95_low"] > 1, "above its specialty mix",
        np.where(table["smr_ci95_high"] < 1, "below its specialty mix", "explained by specialty mix"),
    )
    return table


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    ablation = run_ablation()
    print("full variant reproduces xgboost_is_excluded_oof.pkl exactly: OK")
    print(f"train neg:pos ratio: {ablation['neg_pos_ratio']:.2f}\n")
    for name, r in ablation["variants"].items():
        shares = ", ".join(f"{k}={v:.1%}" for k, v in r["gain_share"].items())
        print(f"{name:22s} PR-AUC={r['pr_auc']:.4f}  ROC-AUC={r['roc_auc']:.4f}  gain share: {shares}")
    print("\ncorrelation between the two target-encoded features:", {k: round(v, 4) for k, v in ablation["correlation"].items()})

    smr = specialty_standardized_rates()
    print(f"\n=== top {TOP_N_MANUFACTURERS} manufacturers: raw rate vs. specialty-mix-expected rate (full data) ===")
    print(smr.round(3).to_string())
    print("\nverdict counts:", smr["smr_verdict"].value_counts().to_dict())
