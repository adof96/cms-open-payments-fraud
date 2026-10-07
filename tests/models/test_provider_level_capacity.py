import numpy as np
import pandas as pd
import pytest

from src.models.provider_level_capacity import VARIANTS, build_variant, capacity_row
from src.models.train_supervised import build_xgboost


def test_original_variant_is_exactly_the_shared_config():
    assert VARIANTS["original"] == {}
    assert build_variant(64.5, {}).get_params() == build_xgboost(64.5).get_params()


def _same(a, b) -> bool:
    # `missing` vale NaN por defecto, y NaN != NaN.
    return a == b or (isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b))


def test_variants_only_change_capacity_knobs():
    base = build_xgboost(64.5).get_params()
    for name, overrides in VARIANTS.items():
        params = build_variant(64.5, overrides).get_params()
        changed = {k for k in params if not _same(params[k], base[k])}
        assert changed == set(overrides), name
        assert params["scale_pos_weight"] == 64.5  # el manejo del desbalance no cambia


def test_restricted_variants_actually_reduce_capacity():
    base = build_xgboost(64.5).get_params()
    for name, overrides in VARIANTS.items():
        if name == "original":
            continue
        params = build_variant(64.5, overrides).get_params()
        assert params["max_depth"] < base["max_depth"]
        assert params["min_child_weight"] > (base["min_child_weight"] or 1)


class _FixedModel:
    def __init__(self, train_scores, test_scores):
        self.scores = {"train": train_scores, "test": test_scores}

    def predict_proba(self, X):
        p = self.scores[X.attrs["side"]]
        return np.column_stack([1 - p, p])


def test_capacity_row_reports_the_train_test_pr_ratio():
    y_train = pd.Series([1, 0, 0, 0])
    y_test = pd.Series([1, 0, 0, 0])
    X_train, X_test = pd.DataFrame(index=range(4)), pd.DataFrame(index=range(4))
    X_train.attrs["side"], X_test.attrs["side"] = "train", "test"
    model = _FixedModel(np.array([0.9, 0.1, 0.1, 0.1]), np.array([0.1, 0.9, 0.5, 0.2]))

    row = capacity_row("x", model, X_train, y_train, X_test, y_test)

    assert row["train_pr_auc"] == pytest.approx(1.0)
    assert row["test_pr_auc"] == pytest.approx(0.25)
    assert row["train_test_pr_ratio"] == pytest.approx(4.0)
