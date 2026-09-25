import pandas as pd
import pytest

from src.config import TARGET_COLUMN
from src.models.ablation_target_encoding import specialty_standardized_rates


def _write_features(tmp_path, rows):
    path = tmp_path / "features.csv"
    pd.DataFrame(rows, columns=["manufacturer_name", "recipient_specialty", TARGET_COLUMN]).to_csv(path, index=False)
    return path


def _block(manufacturer, specialty, n, excluded):
    return [(manufacturer, specialty, 1)] * excluded + [(manufacturer, specialty, 0)] * (n - excluded)


def test_manufacturer_that_only_pays_a_risky_specialty_is_explained_by_its_mix(tmp_path):
    # PsychPharma tiene tasa cruda alta, pero solo porque le paga a Psychiatry (riesgosa):
    # su tasa es exactamente la de la especialidad -> SMR = 1.
    rows = (
        _block("PsychPharma", "Psychiatry", 1000, 20)
        + _block("OtherCo", "Psychiatry", 1000, 20)
        + _block("OtherCo", "Dermatology", 2000, 2)
    )
    table = specialty_standardized_rates(top_n=2, features_path=_write_features(tmp_path, rows))

    psych = table.loc["PsychPharma"]
    assert psych["raw_rate_vs_overall"] > 1.5
    assert psych["smr"] == pytest.approx(1.0)
    assert psych["smr_verdict"] == "explained by specialty mix"


def test_manufacturer_riskier_than_its_specialty_mix_is_flagged_above(tmp_path):
    # RiskyCo le paga a la misma especialidad que CleanCo pero concentra las exclusiones.
    rows = _block("RiskyCo", "Family Medicine", 2000, 60) + _block("CleanCo", "Family Medicine", 2000, 2)
    table = specialty_standardized_rates(top_n=2, features_path=_write_features(tmp_path, rows))

    assert table.loc["RiskyCo", "smr"] > 1
    assert table.loc["RiskyCo", "smr_verdict"] == "above its specialty mix"
    assert table.loc["CleanCo", "smr_verdict"] == "below its specialty mix"
