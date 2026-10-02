import numpy as np
import pandas as pd
import pytest

from src.config import TARGET_COLUMN
from src.models.diagnose_provider_generalization import (
    memorization_check,
    run_diagnostics,
    seed_variance_check,
    single_feature_baseline,
    summarize_seed_variance,
)
from src.models.train_supervised import provider_groups


class _ColumnModel:
    """Modelo falso: su probabilidad es una columna del input. Registra el orden de columnas."""

    def __init__(self, column: str):
        self.column = column
        self.seen_columns = None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.seen_columns = list(X.columns)
        p = X[self.column].to_numpy(dtype=float)
        return np.column_stack([1 - p, p])


def _memorizing_setup(seed: int = 0):
    # En train, "fingerprint" ES la etiqueta (memorización perfecta); en test es ruido.
    rng = np.random.default_rng(seed)
    y_train = pd.Series((rng.random(500) < 0.05).astype(int))
    y_test = pd.Series((rng.random(500) < 0.05).astype(int))
    X_train = pd.DataFrame({"fingerprint": y_train.astype(float), "signal": rng.random(500)})
    X_test = pd.DataFrame({"fingerprint": rng.random(500), "signal": y_test * 0.5 + rng.random(500) * 0.5})
    return X_train, y_train, X_test, y_test


def test_memorization_check_flags_a_large_train_test_gap():
    X_train, y_train, X_test, y_test = _memorizing_setup()

    result = memorization_check(_ColumnModel("fingerprint"), X_train, y_train, X_test, y_test)

    assert result["train"]["pr_auc"] == pytest.approx(1.0)
    assert result["test"]["pr_auc"] < 0.3
    assert result["pr_auc_gap"] > 3
    assert result["test"]["positive_rate"] == pytest.approx(y_test.mean())


def test_single_feature_baseline_scores_each_feature_directly():
    y = pd.Series([0] * 90 + [1] * 10)
    X = pd.DataFrame({"perfect": y.astype(float), "reversed": 1.0 - y, "other": np.arange(100.0)})

    table = single_feature_baseline(X, y, features=["perfect", "reversed", "not_in_X"])

    assert list(table.index) == ["perfect", "reversed"]  # los features ausentes se omiten
    assert table.loc["perfect", "roc_auc"] == pytest.approx(1.0)
    assert table.loc["reversed", "roc_auc"] == pytest.approx(0.0)
    assert table.loc["perfect", "pr_auc_lift"] == pytest.approx(1.0 / 0.1)


def test_run_diagnostics_feeds_the_model_columns_in_training_order():
    X_train, y_train, X_test, y_test = _memorizing_setup()
    model = _ColumnModel("signal")
    artifacts = {"feature_columns": ["signal", "fingerprint"]}  # orden distinto al de X

    result = run_diagnostics(model, artifacts, X_train, y_train, X_test.assign(recipient_specialty_te=0.5), y_test)

    assert model.seen_columns == ["signal", "fingerprint"]
    assert set(result) == {"memorization", "single_feature_baseline"}


def _provider_raw_sample(n_providers: int = 120, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows, npis = [], []
    for provider in range(n_providers):
        excluded = int(provider < 25)
        for _ in range(int(rng.integers(1, 5))):
            npis.append(2_000_000_000 + provider)
            rows.append({
                "payment_amount_log": rng.normal(3, 1),
                "num_payments_included": 1.0,
                "payment_month": float(rng.integers(1, 13)),
                "payment_day_of_week": float(rng.integers(0, 7)),
                "payment_frequency": float(rng.integers(1, 50)),
                "is_related_product": int(rng.integers(0, 2)),
                "is_third_party_payment": 0,
                "recipient_specialty": "Psychiatry" if excluded and rng.random() < 0.7 else rng.choice(["Psychiatry", "Dermatology"]),
                "manufacturer_name": rng.choice(["Big Pharma", "Small Co"]),
                "payment_form": "In-kind items and services",
                "payment_nature": "Food and Beverage",
                TARGET_COLUMN: excluded,
                "is_excluded_name_match": 0,
            })
    sample = pd.DataFrame(rows)
    return sample, provider_groups(pd.Series(npis, dtype="Int64"))


def test_seed_variance_check_returns_one_row_per_seed_and_variant():
    sample, groups = _provider_raw_sample()

    results = seed_variance_check(sample, groups, seeds=(0, 1))

    assert len(results) == 4  # 2 semillas x 2 variantes por defecto
    assert set(results["variant"]) == {"full", "no payment_frequency"}
    assert results[["pr_auc", "roc_auc", "positive_rate"]].notna().all().all()
    assert (results["test_excluded_providers"] > 0).all()
    summary = summarize_seed_variance(results)
    assert ("pr_auc", "mean") in summary.columns
