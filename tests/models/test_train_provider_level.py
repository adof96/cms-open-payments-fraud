import numpy as np
import pandas as pd
import pytest

from src.config import TARGET_COLUMN
from src.models.train_provider_level import (
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    build_split_features,
    load_provider_sample,
    manufacturer_weighted_risk,
    paired_bootstrap_auc_diff,
)
from src.models.train_supervised import TARGET_ENCODING_SMOOTHING


def _train_pairs_with_solo_manufacturer():
    # 200 proveedores pagados por BigCo (20 excluidos); el proveedor 999 (excluido) es el ÚNICO
    # pagado por SoloCo. Si su etiqueta se filtrara, SoloCo lo marcaría como riesgoso.
    npis = list(range(200)) + [999]
    pairs = pd.DataFrame({
        "npi": npis,
        "manufacturer_name": ["BigCo"] * 200 + ["SoloCo"],
        "payment_count": [3] * 200 + [5],
        "amount": [100.0] * 201,
    })
    labels = pd.Series([1] * 20 + [0] * 180 + [1], index=npis)
    return pairs, labels


def test_training_provider_risk_does_not_see_its_own_label():
    pairs, labels = _train_pairs_with_solo_manufacturer()

    risk_train, _, _, _ = manufacturer_weighted_risk(pairs, pairs.iloc[:0], labels)

    # Sin fuga, SoloCo no existe en los folds que codifican al 999 -> cae a la media de esos folds.
    payment_rate = (20 * 3 + 5) / (200 * 3 + 5)
    leaky = (5 * 1 + TARGET_ENCODING_SMOOTHING * payment_rate) / (5 + TARGET_ENCODING_SMOOTHING)
    assert risk_train[999] < leaky
    assert risk_train[999] < 2 * payment_rate


def test_test_provider_risk_is_amount_weighted_with_full_training_mapping():
    pairs, labels = _train_pairs_with_solo_manufacturer()
    test_pairs = pd.DataFrame({
        "npi": [5000, 5000], "manufacturer_name": ["SoloCo", "BigCo"],
        "payment_count": [1, 1], "amount": [300.0, 100.0],
    })

    _, risk_test, mapping, _ = manufacturer_weighted_risk(pairs, test_pairs, labels)

    expected = (300 * mapping["SoloCo"] + 100 * mapping["BigCo"]) / 400
    assert risk_test[5000] == pytest.approx(expected, rel=1e-6)
    assert mapping["SoloCo"] > mapping["BigCo"]  # en test SÍ se usa el mapping completo de train


def test_load_provider_sample_keeps_every_excluded_provider(tmp_path):
    providers = pd.DataFrame({"npi": range(1020), TARGET_COLUMN: [1] * 20 + [0] * 1000})
    providers.to_csv(tmp_path / "providers.csv", index=False)

    sample = load_provider_sample(tmp_path / "providers.csv", negative_frac=0.1, random_state=0)
    again = load_provider_sample(tmp_path / "providers.csv", negative_frac=0.1, random_state=0)

    assert sample[TARGET_COLUMN].sum() == 20
    assert (sample[TARGET_COLUMN] == 0).sum() == 100
    assert sample["npi"].is_unique
    assert sample["npi"].tolist() == again["npi"].tolist()


def _synthetic_providers(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    providers = pd.DataFrame({c: rng.random(n) for c in NUMERIC_FEATURES})
    providers["npi"] = np.arange(n) + 10_000
    providers[TARGET_COLUMN] = (np.arange(n) < 30).astype(int)
    providers["recipient_specialty"] = rng.choice(["Psychiatry", "Dermatology", None], n)
    pairs = pd.DataFrame({
        "npi": np.repeat(providers["npi"].to_numpy(), 2),
        "manufacturer_name": np.tile(["BigCo", "SmallCo"], n),
        "payment_count": rng.integers(1, 4, 2 * n),
        "amount": rng.random(2 * n) * 100 + 1,
    })
    return providers, pairs


def test_build_split_features_has_fixed_columns_no_nans_and_disjoint_providers():
    providers, pairs = _synthetic_providers()
    train, test = providers.iloc[:240], providers.iloc[240:]

    X_train, X_test, artifacts = build_split_features(train, test, pairs)

    assert list(X_train.columns) == list(X_test.columns) == FEATURE_COLUMNS == artifacts["feature_columns"]
    assert not X_train.isna().any().any() and not X_test.isna().any().any()
    assert set(X_train.index).isdisjoint(X_test.index)
    assert len(X_train) == 240 and len(X_test) == 60


def test_paired_bootstrap_detects_a_clearly_better_score():
    rng = np.random.default_rng(0)
    y = np.array([1] * 30 + [0] * 300)
    perfect = y + rng.random(len(y)) * 0.1
    noise = rng.random(len(y))

    better = paired_bootstrap_auc_diff(y, perfect, noise, n_boot=300)
    same = paired_bootstrap_auc_diff(y, noise, noise, n_boot=50)

    assert better["ci95_low"] > 0 and better["share_not_better"] == 0
    assert same["observed_diff"] == 0 and same["share_not_better"] == 1
