import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from src.config import TARGET_COLUMN
from src.models.train_supervised import (
    TARGET_ENCODING_SMOOTHING,
    _fit_target_encoding,
    _out_of_fold_target_encoding,
    apply_preprocessing,
    preprocess_features,
)


def _categories_and_target():
    # 'SOLO' aparece en una sola fila, y es positiva: el caso exacto de la fuga que corrige
    # el out-of-fold (un fabricante chico cuyo encoding refleja la etiqueta de esa fila).
    cats = ["A"] * 100 + ["B"] * 99 + ["SOLO"]
    y = [1] * 10 + [0] * 90 + [1] * 5 + [0] * 94 + [1]
    return pd.Series(cats, name="cat"), pd.Series(y, name="y")


def test_oof_encoding_never_uses_a_rows_own_label():
    cats, y = _categories_and_target()
    encoded = _out_of_fold_target_encoding(cats, y, smoothing=TARGET_ENCODING_SMOOTHING, n_splits=5, random_state=0)

    solo_pos = int(np.flatnonzero(cats == "SOLO")[0])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    fit_idx = next(fit for fit, apply in skf.split(cats, y) if solo_pos in apply)

    # 'SOLO' no existe en los otros folds -> cae a su media global, sin rastro de su propio 1.
    assert encoded.iloc[solo_pos] == pytest.approx(y.iloc[fit_idx].mean(), rel=1e-5)
    leaky_value = (1 + TARGET_ENCODING_SMOOTHING * y.mean()) / (1 + TARGET_ENCODING_SMOOTHING)
    assert encoded.iloc[solo_pos] < leaky_value


def test_oof_encoding_keeps_index_and_has_no_nans():
    cats, y = _categories_and_target()
    cats.index = y.index = pd.RangeIndex(1000, 1200)  # índice no contiguo desde 0, como tras train_test_split

    encoded = _out_of_fold_target_encoding(cats, y, n_splits=5, random_state=0)

    assert encoded.index.equals(cats.index)
    assert not encoded.isna().any()


def _raw_frame(n: int, seed: int) -> pd.DataFrame:
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
        "manufacturer_name": rng.choice(["Big Pharma", "Small Co", "Tiny Co"], n),
        "payment_form": rng.choice(["Cash or cash equivalent", "In-kind items and services"], n),
        "payment_nature": rng.choice(["Food and Beverage", "Consulting Fee"], n),
        TARGET_COLUMN: (rng.random(n) < 0.1).astype("int8"),
    })


def test_test_set_encoding_matches_inference_path():
    # El out-of-fold solo debe afectar a las filas de entrenamiento: test (y por lo tanto
    # predictor.py / la app vía apply_preprocessing) sigue usando el mapping de todo el train.
    train_raw, test_raw = _raw_frame(300, seed=1), _raw_frame(80, seed=2)

    _, X_test, artifacts = preprocess_features(train_raw, test_raw)

    pd.testing.assert_frame_equal(X_test, apply_preprocessing(test_raw, artifacts))


def test_train_rows_are_encoded_out_of_fold_not_with_full_mapping():
    train_raw, test_raw = _raw_frame(300, seed=1), _raw_frame(80, seed=2)

    X_train, _, artifacts = preprocess_features(train_raw, test_raw)

    filled = train_raw["manufacturer_name"].fillna("Missing")
    full = artifacts["target_encoding"]["manufacturer_name"]
    full_mapping_values = filled.map(full["mapping"]).astype("float32")
    assert not np.allclose(X_train["manufacturer_name_te"], full_mapping_values)
