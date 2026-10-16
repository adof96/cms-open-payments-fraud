import numpy as np
import pandas as pd
import pytest

from src.config import MODELS_DIR, TARGET_COLUMN
from src.models.train_unsupervised import (
    _evaluate_anomaly_ranking,
    parse_cli_args,
    preprocess_features,
    train_and_evaluate,
)


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


# --- Compuerta de seguridad: nombres de producción solo con promote explícito ----------
# Datos y guardado mockeados: no se lee fraud_features.csv ni se escribe nada en models/.

PRODUCTION_NAMES = {MODELS_DIR / "isolation_forest.pkl", MODELS_DIR / "unsupervised_preprocessing.pkl"}


def _forbid_data_loading(monkeypatch):
    def _must_not_load(*args, **kwargs):
        raise AssertionError("cargó datos antes de validar el sufijo")

    monkeypatch.setattr("src.models.train_unsupervised._load_random_sample", _must_not_load)


def _mock_io(monkeypatch) -> list:
    saved = []
    monkeypatch.setattr("src.models.train_unsupervised._load_random_sample", lambda *a, **k: _raw_frame())
    monkeypatch.setattr("src.models.train_unsupervised.save_model", lambda obj, path: saved.append(path))
    return saved


def test_empty_suffix_without_promote_raises_before_loading_data(monkeypatch):
    _forbid_data_loading(monkeypatch)

    with pytest.raises(ValueError, match="promote"):
        train_and_evaluate(artifact_suffix="")


def test_promote_with_a_named_suffix_raises_before_loading_data(monkeypatch):
    _forbid_data_loading(monkeypatch)

    with pytest.raises(ValueError, match="sufijo vacío"):
        train_and_evaluate(artifact_suffix="_v2", promote=True)


def test_default_call_writes_only_candidate_names(monkeypatch):
    saved = _mock_io(monkeypatch)

    summary = train_and_evaluate()

    assert saved == [MODELS_DIR / "isolation_forest_candidate.pkl", MODELS_DIR / "unsupervised_preprocessing_candidate.pkl"]
    assert PRODUCTION_NAMES.isdisjoint(saved)
    assert set(summary) == {
        "sample_rows", "sample_positives_is_excluded", "sample_positives_name_match", "feature_columns",
        "min_category_count", "n_estimators", "n_jobs", "runtime_seconds", "results",
    }


def test_named_suffix_writes_that_candidate(monkeypatch):
    saved = _mock_io(monkeypatch)

    train_and_evaluate(artifact_suffix="_v2")

    assert saved == [MODELS_DIR / "isolation_forest_v2.pkl", MODELS_DIR / "unsupervised_preprocessing_v2.pkl"]


def test_empty_suffix_with_promote_writes_production_names_and_warns(monkeypatch, capsys):
    saved = _mock_io(monkeypatch)

    train_and_evaluate(artifact_suffix="", promote=True)

    assert set(saved) == PRODUCTION_NAMES
    assert "PRODUCCIÓN" in capsys.readouterr().out


@pytest.mark.parametrize("argv, expected", [
    ([], ("_candidate", False)),
    (["_v2"], ("_v2", False)),
    ([""], ("_candidate", False)),
    (["", "--promote"], ("", True)),
])
def test_cli_suffix_rules(argv, expected, capsys):
    assert parse_cli_args(argv) == expected
    out = capsys.readouterr().out
    if argv == [""]:
        assert "sin --promote" in out
    if expected[0] == "":
        assert "(PRODUCCIÓN)" in out


@pytest.mark.parametrize("argv", [["--promote"], ["_v2", "--promote"]])
def test_cli_promote_without_explicit_empty_suffix_exits_with_error(argv):
    with pytest.raises(SystemExit) as exc:
        parse_cli_args(argv)
    assert exc.value.code == 2
