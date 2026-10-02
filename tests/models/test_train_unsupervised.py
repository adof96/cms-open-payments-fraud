import numpy as np
import pandas as pd
import pytest

from src.config import TARGET_COLUMN
from src.models.train_unsupervised import _evaluate_anomaly_ranking, preprocess_features


def _raw_frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frequency = rng.integers(1, 200, n).astype("float32")
    frequency[:3] = np.nan
    return pd.DataFrame({
        "payment_amount_log": rng.normal(3, 1, n).astype("float32"),
        "num_payments_included": np.ones(n, dtype="float32"),
        "payment_month": rng.integers(1, 13, n).astype("float32"),
        "payment_day_of_week": rng.integers(0, 7, n).astype("float32"),
        "payment_frequency": frequency,
        "is_related_product": rng.integers(0, 2, n).astype("int8"),
        "is_third_party_payment": rng.integers(0, 2, n).astype("int8"),
        "recipient_specialty": rng.choice(["Psychiatry", "Family Medicine", None], n),
        "manufacturer_name": rng.choice(["Big Pharma", "Small Co"], n, p=[0.8, 0.2]),
        "payment_form": rng.choice(["Cash or cash equivalent", "In-kind items and services"], n),
        "payment_nature": rng.choice(["Food and Beverage", "Consulting Fee"], n),
        TARGET_COLUMN: (rng.random(n) < 0.1).astype("int8"),
        "is_excluded_name_match": (rng.random(n) < 0.02).astype("int8"),
    })


def test_preprocess_features_never_uses_the_labels():
    # El enfoque es no supervisado: sin las columnas de etiqueta el resultado debe ser idéntico.
    df = _raw_frame()

    X_with_labels, _ = preprocess_features(df)
    X_without_labels, _ = preprocess_features(df.drop(columns=[TARGET_COLUMN, "is_excluded_name_match"]))

    pd.testing.assert_frame_equal(X_with_labels, X_without_labels)
    assert not any(c.startswith(TARGET_COLUMN) or "name_match" in c for c in X_with_labels.columns)


def test_frequency_encoding_is_the_category_share_and_handles_missing():
    df = _raw_frame()

    X, artifacts = preprocess_features(df)

    expected = df["manufacturer_name"].map(df["manufacturer_name"].value_counts(normalize=True))
    np.testing.assert_allclose(X["manufacturer_name_freq"], expected, rtol=1e-6)
    assert "Missing" in artifacts["frequency_encoding"]["recipient_specialty"]
    assert not X.isna().any().any()
    assert artifacts["feature_columns"] == list(X.columns)


def test_anomaly_ranking_capture_rates_and_ranks():
    # 100 pagos con score 0..99; los dos positivos tienen score 99 (el más anómalo) y 50.
    scores = np.arange(100, dtype=float)
    y = pd.Series(np.zeros(100, dtype=int))
    y[[99, 50]] = 1

    result = _evaluate_anomaly_ranking(scores, y)

    assert result["n_positive"] == 2
    assert result["top_1pct_n_captured"] == 1 and result["top_1pct_capture_rate"] == pytest.approx(0.5)
    assert result["top_10pct_n_captured"] == 1
    assert result["mean_anomaly_score_positive"] == pytest.approx(74.5)
    assert result["mean_percentile_rank_positive"] > result["mean_percentile_rank_negative"]


def test_anomaly_ranking_with_no_positives_returns_nan_rates():
    result = _evaluate_anomaly_ranking(np.linspace(0, 1, 50), pd.Series(np.zeros(50, dtype=int)))

    assert result["n_positive"] == 0
    assert np.isnan(result["top_5pct_capture_rate"])
