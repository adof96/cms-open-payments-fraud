import time

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from src.config import TARGET_COLUMN
from src.models.provider_level_cv import (
    GRID,
    baseline_cv,
    build_model,
    complexity_key,
    cv_folds,
    evaluate_grid,
    fold_features,
    grid_configs,
    select_config,
)
from src.models.train_provider_level import NUMERIC_FEATURES


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


def test_grid_has_81_configs_and_fixed_learning_rate_and_threads():
    configs = grid_configs()

    assert len(configs) == 81 and len({tuple(c.values()) for c in configs}) == 81
    params = build_model(configs[0], 12.5).get_params()
    assert params["learning_rate"] == 0.1 and params["n_jobs"] == 2 and params["scale_pos_weight"] == 12.5


def test_selection_picks_simplest_config_within_one_sd_of_best():
    rows = [
        {"max_depth": 3, "n_estimators": 300, "min_child_weight": 50, "reg_lambda": 1, "mean_pr_auc": 0.100, "std_pr_auc": 0.020},
        {"max_depth": 2, "n_estimators": 150, "min_child_weight": 100, "reg_lambda": 10, "mean_pr_auc": 0.085, "std_pr_auc": 0.030},
        {"max_depth": 2, "n_estimators": 100, "min_child_weight": 100, "reg_lambda": 10, "mean_pr_auc": 0.081, "std_pr_auc": 0.030},
        # La más simple de todas, pero fuera del umbral (0.100 - 0.020 = 0.080).
        {"max_depth": 1, "n_estimators": 100, "min_child_weight": 200, "reg_lambda": 50, "mean_pr_auc": 0.070, "std_pr_auc": 0.010},
    ]
    results = pd.DataFrame(rows)

    best, selected, threshold = select_config(results)

    assert best == 0 and threshold == pytest.approx(0.080)
    assert selected == 2


def test_complexity_key_orders_depth_then_trees_then_regularization():
    simple = {"max_depth": 1, "n_estimators": 300, "min_child_weight": 50, "reg_lambda": 1}
    deeper = {"max_depth": 2, "n_estimators": 100, "min_child_weight": 200, "reg_lambda": 50}
    more_reg = {"max_depth": 1, "n_estimators": 300, "min_child_weight": 200, "reg_lambda": 1}

    assert complexity_key(simple) < complexity_key(deeper)
    assert complexity_key(more_reg) < complexity_key(simple)


def test_fold_features_do_not_depend_on_validation_labels():
    # Si las features de un fold usaran las etiquetas de validación, cambiarlas cambiaría X.
    providers, pairs = _synthetic_providers()
    fit_idx, val_idx = next(StratifiedKFold(5, shuffle=True, random_state=0).split(providers, providers[TARGET_COLUMN]))

    X_fit, _, X_val, y_val = fold_features(providers, fit_idx, val_idx, pairs)
    flipped = providers.copy()
    col = flipped.columns.get_loc(TARGET_COLUMN)
    flipped.iloc[val_idx, col] = 1 - flipped.iloc[val_idx, col]
    X_fit_f, _, X_val_f, y_val_f = fold_features(flipped, fit_idx, val_idx, pairs)

    pd.testing.assert_frame_equal(X_fit, X_fit_f)
    pd.testing.assert_frame_equal(X_val, X_val_f)
    assert not y_val.equals(y_val_f)


def test_cv_folds_partition_the_training_providers():
    providers, pairs = _synthetic_providers()

    folds = cv_folds(providers, pairs, n_folds=3, random_state=0)

    val_npis = [npi for _, _, X_val, _ in folds for npi in X_val.index]
    assert sorted(val_npis) == sorted(providers["npi"])
    for X_fit, _, X_val, _ in folds:
        assert set(X_fit.index).isdisjoint(X_val.index)


def _tiny_folds(n_folds: int = 2):
    rng = np.random.default_rng(0)
    folds = []
    for _ in range(n_folds):
        y = pd.Series([1] * 20 + [0] * 180)
        X = pd.DataFrame({"specialty_te": y * 0.3 + rng.random(200), "x": rng.random(200)})
        folds.append((X, y, X, y))
    return folds


def test_evaluate_grid_stops_when_past_the_deadline():
    with pytest.raises(TimeoutError):
        evaluate_grid(_tiny_folds(), grid_configs()[:2], deadline=time.time() - 1)


def test_evaluate_grid_and_baselines_report_mean_and_std_per_entry():
    folds = _tiny_folds()

    results = evaluate_grid(folds, grid_configs()[:2])
    baselines = baseline_cv(folds)

    assert list(results.columns[: len(GRID)]) == list(GRID)
    assert results[["mean_pr_auc", "std_pr_auc", "mean_roc_auc"]].notna().all().all()
    assert list(baselines.index) == ["specialty_only", "logistic_regression"]
    assert (baselines["mean_roc_auc"] > 0.5).all()
