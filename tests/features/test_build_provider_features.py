import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.config import TARGET_COLUMN
from src.features.build_provider_features import (
    NATURE_GROUP_NAMES,
    NPI_COLUMN,
    aggregate_provider_rows,
    build_provider_table,
)


def _payment(npi, specialty="Psychiatry", manufacturer="Big Pharma", nature="food_beverage",
             amount=10.0, related=1, third_party=0, excluded=0):
    return {
        "npi": npi, "recipient_specialty": specialty, "manufacturer": manufacturer, "nature_group": nature,
        "payment_amount": amount, "payment_amount_log": float(np.log1p(amount)),
        "is_related_product": related, "is_third_party_payment": third_party, TARGET_COLUMN: excluded,
    }


def test_aggregation_produces_exactly_one_row_per_npi_with_max_label():
    payments = pd.DataFrame(
        [_payment(1, excluded=1)] * 3 + [_payment(2)] * 5 + [_payment(3, amount=50.0)]
    )

    providers, _, stats = aggregate_provider_rows(payments)

    assert providers["npi"].is_unique
    assert set(providers["npi"]) == {1, 2, 3} and stats["providers"] == 3
    labels = providers.set_index("npi")[TARGET_COLUMN]
    assert labels.to_dict() == {1: 1, 2: 0, 3: 0}
    assert providers.set_index("npi").loc[2, "total_payment_count"] == 5


def test_aggregation_rejects_a_provider_with_mixed_labels():
    payments = pd.DataFrame([_payment(7, excluded=1), _payment(7, excluded=0)])

    with pytest.raises(RuntimeError, match="is_excluded"):
        aggregate_provider_rows(payments)


def test_specialty_uses_the_mode_and_counts_multi_specialty_providers():
    payments = pd.DataFrame(
        [_payment(1, specialty="Psychiatry")] * 3 + [_payment(1, specialty="Neurology")] + [_payment(2)]
    )

    providers, _, stats = aggregate_provider_rows(payments)

    assert providers.set_index("npi").loc[1, "recipient_specialty"] == "Psychiatry"
    assert stats["providers_with_multiple_specialties"] == 1


def test_manufacturer_concentration_and_payment_mix():
    payments = pd.DataFrame([
        _payment(1, manufacturer="A", amount=60.0, nature="consulting_fee", third_party=1),
        _payment(1, manufacturer="A", amount=20.0),
        _payment(1, manufacturer="B", amount=20.0, related=0),
        _payment(1, manufacturer="C", amount=0.0, nature="debt_forgiveness"),
    ])

    providers, pairs, _ = aggregate_provider_rows(payments)
    row = providers.set_index("npi").loc[1]

    assert row["n_distinct_manufacturers"] == 3
    assert row["top_manufacturer_amount_share"] == pytest.approx(80.0 / 100.0)
    assert row["share_nature_consulting_fee"] == pytest.approx(0.25)
    assert row["share_nature_debt_forgiveness"] == pytest.approx(0.25)
    assert row[[f"share_nature_{n}" for n in NATURE_GROUP_NAMES]].sum() == pytest.approx(1.0)
    assert row["share_third_party_payment"] == pytest.approx(0.25)
    assert row["share_related_product"] == pytest.approx(0.75)
    assert pairs.set_index(["npi", "manufacturer"]).loc[(1, "A"), "payment_count"] == 2


def _write_inputs(tmp_path, rows):
    features = pd.DataFrame([{
        "Record_ID": r["record_id"], "payment_amount": r["amount"], "payment_amount_log": 1.0,
        "payment_nature": r["nature"], "recipient_specialty": r["specialty"], "manufacturer_name": r["manufacturer"],
        "is_related_product": 1, "is_third_party_payment": 0, TARGET_COLUMN: r["excluded"],
    } for r in rows])
    labels = pd.DataFrame({
        "Record_ID": [r["record_id"] for r in rows],
        NPI_COLUMN: pd.array([r["npi"] for r in rows], dtype="Int64"),
    })
    features.to_csv(tmp_path / "features.csv", index=False)
    labels.to_csv(tmp_path / "labels.csv", index=False)


def _row(record_id, npi, excluded=0, manufacturer="Big Pharma", amount=10.0):
    return {"record_id": record_id, "npi": npi, "excluded": excluded, "manufacturer": manufacturer,
            "amount": amount, "nature": "Consulting Fee", "specialty": "Psychiatry"}


def test_build_end_to_end_one_row_per_npi_across_buckets_and_drops_rows_without_npi(tmp_path):
    # Los pagos de cada NPI están intercalados en el archivo y en chunks distintos (chunksize=2);
    # con 3 buckets, NPIs distintos caen en buckets distintos.
    rows = [_row(1, 101, excluded=1), _row(2, 202), _row(3, None), _row(4, 101, excluded=1),
            _row(5, 303), _row(6, 202, manufacturer="Small Co"), _row(7, None), _row(8, 101, excluded=1)]
    _write_inputs(tmp_path, rows)

    stats = build_provider_table(
        features_path=tmp_path / "features.csv", labels_path=tmp_path / "labels.csv",
        output_path=tmp_path / "providers.csv", pairs_path=tmp_path / "pairs.parquet",
        chunksize=2, n_buckets=3,
    )
    providers = pd.read_csv(tmp_path / "providers.csv")

    assert providers["npi"].is_unique and sorted(providers["npi"]) == [101, 202, 303]
    assert stats["providers"] == 3 and stats["excluded_providers"] == 1
    assert stats["rows_without_npi"] == 2
    assert providers.set_index("npi").loc[101, "total_payment_count"] == 3
    pairs = pq.read_table(tmp_path / "pairs.parquet").to_pandas()
    assert set(pairs.loc[pairs["npi"] == 202, "manufacturer_name"]) == {"Big Pharma", "Small Co"}


def test_build_fails_if_features_and_labels_are_not_row_aligned(tmp_path):
    rows = [_row(1, 101), _row(2, 202)]
    _write_inputs(tmp_path, rows)
    labels = pd.read_csv(tmp_path / "labels.csv")
    labels.iloc[::-1].to_csv(tmp_path / "labels.csv", index=False)  # mismo contenido, otro orden

    with pytest.raises(RuntimeError, match="alineados"):
        build_provider_table(
            features_path=tmp_path / "features.csv", labels_path=tmp_path / "labels.csv",
            output_path=tmp_path / "providers.csv", pairs_path=tmp_path / "pairs.parquet",
        )
