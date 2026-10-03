import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from src.config import TARGET_COLUMN
from src.models.train_supervised import (
    DEFAULT_CANDIDATE_SUFFIX,
    PROVIDER_COLUMN,
    resolve_artifact_suffix,
    train_and_evaluate,
    TARGET_ENCODING_SMOOTHING,
    _out_of_fold_target_encoding,
    apply_preprocessing,
    attach_provider_ids,
    preprocess_features,
    provider_groups,
    provider_level_split,
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


# --- Split y folds a nivel proveedor ---------------------------------------------------

def _provider_sample(seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    # 200 proveedores con 1-8 pagos cada uno; 30 de ellos excluidos (todas sus filas = 1,
    # porque is_excluded es un dato por proveedor). Más 10 filas sin NPI (hospitales).
    rng = np.random.default_rng(seed)
    npis, labels = [], []
    for provider in range(200):
        k = int(rng.integers(1, 9))
        npis += [1_000_000_000 + provider] * k
        labels += [int(provider < 30)] * k
    npis += [None] * 10
    labels += [0] * 10
    sample = pd.DataFrame({TARGET_COLUMN: labels}, index=pd.RangeIndex(len(labels)))
    npi = pd.Series(npis, index=sample.index, dtype="Int64")
    return sample, provider_groups(npi)


def test_provider_level_split_never_puts_a_provider_on_both_sides():
    sample, groups = _provider_sample()

    train_raw, test_raw = provider_level_split(sample, groups, test_size=0.2, random_state=42)

    train_providers, test_providers = set(groups[train_raw.index]), set(groups[test_raw.index])
    assert train_providers.isdisjoint(test_providers)
    assert len(train_raw) + len(test_raw) == len(sample)
    # Estratificado por proveedor: hay proveedores excluidos en ambos lados.
    assert train_raw[TARGET_COLUMN].sum() > 0 and test_raw[TARGET_COLUMN].sum() > 0


def test_provider_groups_gives_each_missing_npi_its_own_group():
    npi = pd.Series([111, None, 111, None, 222], dtype="Int64")

    groups = provider_groups(npi)

    assert groups[0] == groups[2] == 111 and groups[4] == 222
    assert groups[1] < 0 and groups[3] < 0 and groups[1] != groups[3]


def test_attach_provider_ids_looks_up_npi_by_record_id(tmp_path):
    labels = tmp_path / "labels.csv"
    pd.DataFrame({
        "Record_ID": [10, 11, 12, 13],
        PROVIDER_COLUMN: [555, None, 777, 999],
        TARGET_COLUMN: [0, 0, 1, 0],
    }).to_csv(labels, index=False)

    npi = attach_provider_ids(pd.DataFrame({"Record_ID": [12, 10, 11]}), labels_path=labels)

    assert npi.tolist()[:2] == [777, 555] and pd.isna(npi.iloc[2])

    with pytest.raises(RuntimeError):
        attach_provider_ids(pd.DataFrame({"Record_ID": [10, 404]}), labels_path=labels)


def test_group_folds_hide_other_rows_of_the_same_provider():
    # El proveedor P (excluido, 6 pagos) es el único al que le paga 'RareCo'. Con folds
    # agrupados, cuando se codifica una fila de P ninguna otra fila de P está del lado que
    # ajusta, así que 'RareCo' no existe ahí y cae a la media global. Con folds por fila,
    # las otras filas de P sí filtran su etiqueta y el encoding sube por encima de la media.
    n_other = 300
    cats = pd.Series(["RareCo"] * 6 + ["BigCo"] * n_other)
    y = pd.Series([1] * 6 + [1] * 15 + [0] * (n_other - 15))
    groups = pd.Series([-999] * 6 + list(range(n_other)))

    grouped = _out_of_fold_target_encoding(cats, y, n_splits=5, random_state=0, groups=groups)
    row_level = _out_of_fold_target_encoding(cats, y, n_splits=5, random_state=0)

    global_mean = y.mean()
    assert (grouped.iloc[:6] < global_mean * 1.5).all()
    assert (row_level.iloc[:6] > grouped.iloc[:6]).all()


# --- Compuerta de seguridad: nombres de producción solo con promote explícito ----------

def test_suffix_resolution_never_yields_production_names_without_promote():
    assert resolve_artifact_suffix("", promote=False) == DEFAULT_CANDIDATE_SUFFIX
    assert resolve_artifact_suffix("_provider_split", promote=False) == "_provider_split"
    assert resolve_artifact_suffix("", promote=True) == ""


def test_promote_with_a_named_suffix_is_rejected():
    with pytest.raises(ValueError):
        resolve_artifact_suffix("_provider_split", promote=True)


def test_train_and_evaluate_refuses_production_names_without_promote(monkeypatch):
    # El chequeo corre antes de cargar datos: si no frenara, load_provider_split fallaría acá.
    def _must_not_load(*args, **kwargs):
        raise AssertionError("cargó datos antes de validar el sufijo")

    monkeypatch.setattr("src.models.train_supervised.load_provider_split", _must_not_load)

    with pytest.raises(ValueError, match="promote"):
        train_and_evaluate(artifact_suffix="")


def test_train_and_evaluate_defaults_to_the_candidate_suffix():
    import inspect

    default = inspect.signature(train_and_evaluate).parameters["artifact_suffix"].default
    assert default == DEFAULT_CANDIDATE_SUFFIX != ""
